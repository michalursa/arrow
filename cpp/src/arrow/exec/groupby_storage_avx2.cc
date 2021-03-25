// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

#include <immintrin.h>

#include "arrow/exec/common.h"
#include "arrow/exec/groupby_storage.h"
#include "arrow/util/bit_util.h"

namespace arrow {
namespace exec {

#if defined(ARROW_HAVE_AVX2)

inline __m256i set_first_n_bytes_avx2(int n) {
  return _mm256_cmpgt_epi8(
      _mm256_set1_epi8(n),
      _mm256_setr_epi64x(0x0706050403020100ULL, 0x0f0e0d0c0b0a0908ULL,
                         0x1716151413121110ULL, 0x1f1e1d1c1b1a1918ULL));
}

// Input batch cannot be larger than 4GB due to 32-bit offset arithmetic.
// Add offsets horizontally.
// Subtract lengths of values marked as nulls.
void KeyLength::compute_offsets_avx2(int num_columns, int num_rows,
                                     uint32_t row_fixed_len, const uint32_t** offsets,
                                     const uint8_t** non_nulls,
                                     const uint8_t* any_nulls_bitvector,
                                     uint32_t* row_offsets) {
  if (num_rows == 0) {
    return;
  }
  ARROW_DCHECK(num_columns > 0);
  constexpr int unroll = 8;

  // Sum of all fixed length value lengths from rows visited so far.
  uint32_t sum_offset_first = 0;
  for (int col = 0; col < num_columns; ++col) {
    bool is_varlen_col = (offsets[col] != nullptr);
    if (is_varlen_col) {
      sum_offset_first += offsets[col][0];
    }
  }
  __m256i offset_adjustment = _mm256_sub_epi32(
      _mm256_setr_epi32(row_fixed_len, row_fixed_len * 2, row_fixed_len * 3,
                        row_fixed_len * 4, row_fixed_len * 5, row_fixed_len * 6,
                        row_fixed_len * 7, row_fixed_len * 8),
      _mm256_set1_epi32(sum_offset_first));
  __m256i offset_adjustment_fixed_incr = _mm256_set1_epi32(row_fixed_len * 8);

  row_offsets[0] = 0;
  for (int i = 0; i < num_rows / unroll; ++i) {
    // Add all column lengths together for each rows.
    // Replace column length with zero for nulls.
    __m256i row_offset;
    // We have no nulls
    if (any_nulls_bitvector[i] == 0) {
      row_offset =
          _mm256_loadu_si256(reinterpret_cast<const __m256i*>(offsets[0] + 1) + i);
      for (int col = 1; col < num_columns; ++col) {
        row_offset = _mm256_add_epi32(
            row_offset,
            _mm256_loadu_si256(reinterpret_cast<const __m256i*>(offsets[col] + 1) + i));
      }
      row_offset = _mm256_add_epi32(row_offset, offset_adjustment);
    } else {
      row_offset = _mm256_setzero_si256();
      // Total lengths per row of values marked as null
      __m256i row_null_len = _mm256_setzero_si256();
      for (int col = 0; col < num_columns; ++col) {
        __m256i offset =
            _mm256_loadu_si256(reinterpret_cast<const __m256i*>(offsets[col] + 1) + i);
        row_offset = _mm256_add_epi32(row_offset, offset);
        __m256i null_len = _mm256_sub_epi32(
            offset,
            _mm256_loadu_si256(reinterpret_cast<const __m256i*>(offsets[col]) + i));
        // Zero out lengths for values that are not null
        const __m256i individual_bits =
            _mm256_setr_epi32(0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80);
        __m256i null_mask = _mm256_cmpeq_epi32(
            _mm256_setzero_si256(),
            _mm256_and_si256(_mm256_set1_epi32(non_nulls[col][i]), individual_bits));
        row_null_len =
            _mm256_add_epi32(row_null_len, _mm256_and_si256(null_len, null_mask));
      }
      // Inclusive prefix sum of 32-bit elements
      row_null_len = _mm256_add_epi32(
          row_null_len,
          _mm256_permutevar8x32_epi32(
              _mm256_andnot_si256(_mm256_setr_epi32(0, 0, 0, 0, 0, 0, 0, 0xffffffff),
                                  row_null_len),
              _mm256_setr_epi32(7, 0, 1, 2, 3, 4, 5, 6)));
      row_null_len = _mm256_add_epi32(
          row_null_len, _mm256_permute4x64_epi64(
                            _mm256_andnot_si256(_mm256_setr_epi32(0, 0, 0, 0, 0, 0,
                                                                  0xffffffff, 0xffffffff),
                                                row_null_len),
                            0x93));  // 0b10010011
      row_null_len = _mm256_add_epi32(
          row_null_len, _mm256_permute4x64_epi64(
                            _mm256_andnot_si256(_mm256_setr_epi32(0, 0, 0, 0, 0, 0,
                                                                  0xffffffff, 0xffffffff),
                                                row_null_len),
                            0x4f));  // 0b01001111

      row_offset = _mm256_sub_epi32(row_offset, row_null_len);
      row_offset = _mm256_add_epi32(row_offset, offset_adjustment);
      offset_adjustment = _mm256_sub_epi32(
          offset_adjustment,
          _mm256_permutevar8x32_epi32(row_null_len, _mm256_set1_epi32(7)));
    }
    offset_adjustment = _mm256_add_epi32(offset_adjustment, offset_adjustment_fixed_incr);
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(row_offsets + 1) + i, row_offset);
  }
}

void KeyCompare::compare_varlen_avx2(uint32_t num_rows, const uint32_t* offsets_left,
                                     const uint8_t* concatenated_keys_left,
                                     const uint32_t* ids_right,
                                     const uint32_t* offsets_right,
                                     const uint8_t* concatenated_keys_right,
                                     uint8_t* match_bitvector) {
  uint64_t bits = 0ULL;
  for (uint32_t irow = 0; irow < num_rows; ++irow) {
    uint32_t begin_left = offsets_left[irow];
    uint32_t begin_right = offsets_right[ids_right[irow]];
    uint32_t length = offsets_left[irow + 1] - begin_left;
    uint64_t result = 1;
    uint32_t istripe;
    for (istripe = 0; istripe < (length - 1) / 32; ++istripe) {
      __m256i key_stripe_left = _mm256_loadu_si256(
          reinterpret_cast<const __m256i*>(concatenated_keys_left + begin_left) +
          istripe);
      __m256i key_stripe_right = _mm256_loadu_si256(
          reinterpret_cast<const __m256i*>(concatenated_keys_right + begin_right) +
          istripe);
      uint32_t cmp_bits =
          _mm256_movemask_epi8(_mm256_cmpeq_epi8(key_stripe_left, key_stripe_right));
      if (cmp_bits != static_cast<uint32_t>(~0UL)) {
        result = 0;
        break;
      }
    }
    if (result) {
      __m256i key_stripe_left = _mm256_loadu_si256(
          reinterpret_cast<const __m256i*>(concatenated_keys_left + begin_left) +
          istripe);
      __m256i key_stripe_right = _mm256_loadu_si256(
          reinterpret_cast<const __m256i*>(concatenated_keys_right + begin_right) +
          istripe);
      uint32_t cmp_bits =
          _mm256_movemask_epi8(_mm256_cmpeq_epi8(key_stripe_left, key_stripe_right));
      uint32_t mask = (static_cast<uint32_t>(~0UL) >> (32 - (((length - 1) % 32) + 1)));
      if ((cmp_bits & mask) != mask) {
        result = 0;
      }
    }
    bits |= (result << (irow & 63));
    if ((irow & 63) == 63) {
      reinterpret_cast<uint64_t*>(match_bitvector)[irow / 64] = bits;
      bits = 0ULL;
    }
  }
  if ((num_rows % 64) != 0) {
    reinterpret_cast<uint64_t*>(match_bitvector)[num_rows / 64] = bits;
  }
}

void KeyCompare::compare_fixedlen_avx2(uint32_t num_rows, uint32_t length,
                                       const uint8_t* concatenated_keys_left,
                                       const uint32_t* ids_right,
                                       const uint8_t* concatenated_keys_right,
                                       uint8_t* match_bitvector) {
  if (length > 16) {
    uint32_t mask_last = ~(~0UL >> ((length % 32) == 0 ? 0 : 32 - (length % 32)));
    uint64_t match = 0ULL;
    for (uint32_t irow = 0; irow < num_rows; ++irow) {
      const uint8_t* base_left = concatenated_keys_left + length * irow;
      const uint8_t* base_right = concatenated_keys_right + length * ids_right[irow];
      uint32_t cmp = static_cast<uint32_t>(~0UL);
      uint32_t istripe;
      for (istripe = 0; istripe < (length - 1) / 32; ++istripe) {
        __m256i key_stripe_left =
            _mm256_loadu_si256(reinterpret_cast<const __m256i*>(base_left) + istripe);
        __m256i key_stripe_right =
            _mm256_loadu_si256(reinterpret_cast<const __m256i*>(base_right) + istripe);
        cmp &= _mm256_movemask_epi8(_mm256_cmpeq_epi8(key_stripe_left, key_stripe_right));
      }
      __m256i key_stripe_left =
          _mm256_loadu_si256(reinterpret_cast<const __m256i*>(base_left) + istripe);
      __m256i key_stripe_right =
          _mm256_loadu_si256(reinterpret_cast<const __m256i*>(base_right) + istripe);
      cmp &= (mask_last |
              _mm256_movemask_epi8(_mm256_cmpeq_epi8(key_stripe_left, key_stripe_right)));
      match |= (static_cast<uint64_t>(arrow::BitUtil::PopCount(cmp) >> 5) << (irow & 63));
      if ((irow & 63) == 63) {
        reinterpret_cast<uint64_t*>(match_bitvector)[irow / 64] = match;
        match = 0ULL;
      }
    }
    if ((num_rows % 64) > 0) {
      reinterpret_cast<uint64_t*>(match_bitvector)[num_rows / 64] = match;
    }
  } else if (length > 8) {
    __m256i mask = _mm256_cmpgt_epi8(
        _mm256_set1_epi8(length),
        _mm256_setr_epi64x(0ULL, 0x0f0e0d0c0b0a0908ULL, 0ULL, 0x0f0e0d0c0b0a0908ULL));
    uint64_t match = 0ULL;
    match_bitvector[(num_rows - 1) / 8] = 0;
    for (uint32_t i = 0; i < num_rows / 2; ++i) {
      const uint8_t* base_left = concatenated_keys_left + length * (i * 2 + 0);
      __m256i key_left = _mm256_inserti128_si256(
          _mm256_castsi128_si256(
              _mm_loadu_si128(reinterpret_cast<const __m128i*>(base_left))),
          _mm_loadu_si128(reinterpret_cast<const __m128i*>(base_left + length)), 1);
      __m256i key_right = _mm256_inserti128_si256(
          _mm256_castsi128_si256(_mm_loadu_si128(reinterpret_cast<const __m128i*>(
              concatenated_keys_right + length * ids_right[2 * i + 0]))),
          _mm_loadu_si128(reinterpret_cast<const __m128i*>(
              concatenated_keys_right + length * ids_right[2 * i + 1])),
          1);
      uint32_t cmp = _mm256_movemask_epi8(_mm256_cmpeq_epi64(
          _mm256_and_si256(key_left, mask), _mm256_and_si256(key_right, mask)));
      cmp &= (cmp >> 8);
      cmp = (cmp & 1) | ((cmp >> 15) & 2);
      match |= static_cast<uint64_t>(cmp) << (2 * (i & 31));
      if ((i & 31) == 31) {
        reinterpret_cast<uint64_t*>(match_bitvector)[i / 32] = match;
        match = 0ULL;
      }
    }
    if ((num_rows / 2 % 32) > 0) {
      reinterpret_cast<uint64_t*>(match_bitvector)[num_rows / 64] = match;
    }
    // Process the last row if there is an odd number of rows
    for (uint32_t i = num_rows - (num_rows % 2); i < num_rows; ++i) {
      bool cmp = (memcmp(concatenated_keys_left + length * i,
                         concatenated_keys_right + length * ids_right[i], length) == 0);
      if (cmp) {
        match_bitvector[i / 8] |= 1 << (i & 7);
      } else {
        match_bitvector[i / 8] &= ~(1 << (i & 7));
      }
    }
  } else {
    __m256i offset_left = _mm256_setr_epi64x(0, length, length * 2, length * 3);
    __m256i mask = _mm256_set1_epi64x(~0ULL >> (8 * (8 - length)));
    memset(match_bitvector, 0, (num_rows + 7) / 8);
    for (uint32_t i = 0; i < num_rows / 4; ++i) {
      __m256i key_left = _mm256_i64gather_epi64((const long long*)concatenated_keys_left,
                                                offset_left, 1);
      __m256i key_right = _mm256_i32gather_epi64(
          (const long long*)concatenated_keys_right,
          _mm_mullo_epi32(
              _mm_loadu_si128(reinterpret_cast<const __m128i*>(ids_right) + i),
              _mm_set1_epi32(length)),
          1);
      uint32_t cmp = _mm256_movemask_epi8(_mm256_cmpeq_epi64(
          _mm256_and_si256(key_left, mask), _mm256_and_si256(key_right, mask)));
      cmp = _pext_u32(cmp, 0x01010101UL);
      match_bitvector[i / 2] |= cmp << ((i & 1) * 4);
      offset_left = _mm256_add_epi64(offset_left, _mm256_set1_epi64x(length * 4));
    }
    for (uint32_t i = num_rows - (num_rows % 4); i < num_rows; ++i) {
      bool cmp = (memcmp(concatenated_keys_left + length * i,
                         concatenated_keys_right + length * ids_right[i], length) == 0);
      if (cmp) {
        match_bitvector[i / 8] |= 1 << (i & 7);
      }
    }
  }
}

template <bool is_row_fixedlen>
void KeyTranspose::col2row_shortpair_avx2(uint32_t num_rows, const uint32_t col_length,
                                          const uint8_t* col_vals_A,
                                          const uint8_t* col_vals_B,
                                          const uint32_t* row_offsets,
                                          uint8_t* row_vals) {
  ARROW_DCHECK(col_length == 1 || col_length == 2 || col_length == 4 || col_length == 8);
  if (num_rows == 0) {
    return;
  }

  uint32_t fixed_len = row_offsets[0];

  if (col_length == 1) {
    // 1-1
    constexpr int unroll = 32;
    for (uint32_t i = 0; i < num_rows / unroll; ++i) {
      __m256i col_A =
          _mm256_loadu_si256(reinterpret_cast<const __m256i*>(col_vals_A) + i);
      __m256i col_B =
          _mm256_loadu_si256(reinterpret_cast<const __m256i*>(col_vals_B) + i);
      __m256i r0 = _mm256_unpacklo_epi8(
          col_A, col_B);  // 16-bit outputs in the order: 0..7, 16..23
      __m256i r1 = _mm256_unpackhi_epi8(
          col_A, col_B);  // 16-bit outputs in the order: 8..15, 24..31
      col_A = _mm256_permute2x128_si256(r0, r1, 0x20);
      col_B = _mm256_permute2x128_si256(r0, r1, 0x31);
      uint16_t buffer[unroll];
      _mm256_storeu_si256(reinterpret_cast<__m256i*>(buffer), col_A);
      _mm256_storeu_si256(reinterpret_cast<__m256i*>(buffer) + 1, col_B);
      uint8_t* dst_base = is_row_fixedlen ? row_vals + i * unroll * fixed_len : row_vals;
      uint32_t offsets_base = i * unroll;
      for (int j = 0; j < unroll; ++j) {
        uint8_t* dst = (is_row_fixedlen ? dst_base + j * fixed_len
                                        : dst_base + row_offsets[offsets_base + j]);
        *reinterpret_cast<uint16_t*>(dst) = buffer[j];
      }
    }
  } else if (col_length == 2) {
    // 2-2
    constexpr int unroll = 16;
    for (uint32_t i = 0; i < num_rows / unroll; ++i) {
      __m256i col_A =
          _mm256_loadu_si256(reinterpret_cast<const __m256i*>(col_vals_A) + i);
      __m256i col_B =
          _mm256_loadu_si256(reinterpret_cast<const __m256i*>(col_vals_B) + i);
      __m256i r0 = _mm256_unpacklo_epi16(
          col_A, col_B);  // 32-bit outputs in the order: 0..3, 8..11
      __m256i r1 = _mm256_unpackhi_epi16(
          col_A, col_B);  // 32-bit outputs in the order: 4..7, 12..15
      col_A = _mm256_permute2x128_si256(r0, r1, 0x20);
      col_B = _mm256_permute2x128_si256(r0, r1, 0x31);
      uint32_t buffer[unroll];
      _mm256_storeu_si256(reinterpret_cast<__m256i*>(buffer), col_A);
      _mm256_storeu_si256(reinterpret_cast<__m256i*>(buffer) + 1, col_B);
      uint8_t* dst_base = is_row_fixedlen ? row_vals + i * unroll * fixed_len : row_vals;
      uint32_t offsets_base = i * unroll;
      for (int j = 0; j < unroll; ++j) {
        uint8_t* dst = (is_row_fixedlen ? dst_base + j * fixed_len
                                        : dst_base + row_offsets[offsets_base + j]);
        *reinterpret_cast<uint32_t*>(dst) = buffer[j];
      }
    }
  } else if (col_length == 4) {
    // 4-4
    constexpr int unroll = 8;
    for (uint32_t i = 0; i < num_rows / unroll; ++i) {
      __m256i col_A =
          _mm256_loadu_si256(reinterpret_cast<const __m256i*>(col_vals_A) + i);
      __m256i col_B =
          _mm256_loadu_si256(reinterpret_cast<const __m256i*>(col_vals_B) + i);
      __m256i r0 =
          _mm256_unpacklo_epi32(col_A, col_B);  // 64-bit outputs in the order: 0..1, 4..5
      __m256i r1 =
          _mm256_unpackhi_epi32(col_A, col_B);  // 64-bit outputs in the order: 2..3, 6..7
      col_A = _mm256_permute2x128_si256(r0, r1, 0x20);
      col_B = _mm256_permute2x128_si256(r0, r1, 0x31);
      uint64_t buffer[unroll];
      _mm256_storeu_si256(reinterpret_cast<__m256i*>(buffer), col_A);
      _mm256_storeu_si256(reinterpret_cast<__m256i*>(buffer) + 1, col_B);
      uint8_t* dst_base = is_row_fixedlen ? row_vals + i * unroll * fixed_len : row_vals;
      uint32_t offsets_base = i * unroll;
      for (int j = 0; j < unroll; ++j) {
        uint8_t* dst = (is_row_fixedlen ? dst_base + j * fixed_len
                                        : dst_base + row_offsets[offsets_base + j]);
        *reinterpret_cast<uint64_t*>(dst) = buffer[j];
      }
    }
  } else if (col_length == 8) {
    // 8-8
    constexpr int unroll = 4;
    for (uint32_t i = 0; i < num_rows / unroll; ++i) {
      __m256i col_A =
          _mm256_loadu_si256(reinterpret_cast<const __m256i*>(col_vals_A) + i);
      __m256i col_B =
          _mm256_loadu_si256(reinterpret_cast<const __m256i*>(col_vals_B) + i);
      __m256i r0 =
          _mm256_unpacklo_epi64(col_A, col_B);  // 128-bit outputs in the order: 0, 2
      __m256i r1 =
          _mm256_unpackhi_epi64(col_A, col_B);  // 128-bit outputs in the order: 1, 3
      col_A = _mm256_permute2x128_si256(r0, r1, 0x20);
      col_B = _mm256_permute2x128_si256(r0, r1, 0x31);
      __m128i* dst0 = reinterpret_cast<__m128i*>(
          row_vals + (is_row_fixedlen ? row_offsets[0] * (i * unroll + 0)
                                      : row_offsets[i * unroll + 0]));
      __m128i* dst1 = reinterpret_cast<__m128i*>(
          row_vals + (is_row_fixedlen ? row_offsets[0] * (i * unroll + 1)
                                      : row_offsets[i * unroll + 1]));
      __m128i* dst2 = reinterpret_cast<__m128i*>(
          row_vals + (is_row_fixedlen ? row_offsets[0] * (i * unroll + 2)
                                      : row_offsets[i * unroll + 2]));
      __m128i* dst3 = reinterpret_cast<__m128i*>(
          row_vals + (is_row_fixedlen ? row_offsets[0] * (i * unroll + 3)
                                      : row_offsets[i * unroll + 3]));
      _mm_storeu_si128(dst0, _mm256_castsi256_si128(r0));
      _mm_storeu_si128(dst1, _mm256_castsi256_si128(r1));
      _mm_storeu_si128(dst2, _mm256_extracti128_si256(r0, 1));
      _mm_storeu_si128(dst3, _mm256_extracti128_si256(r1, 1));
    }
  }
  // Process tail
  uint32_t unroll = 32 / col_length;
  uint32_t processed = num_rows / unroll * unroll;
  void (*col2row_shortpair_fn[])(uint32_t, const uint8_t*, const uint8_t*,
                                 const uint32_t*, uint8_t*) = {
      col2row_shortpair_impl<false, 1, 1>, col2row_shortpair_impl<false, 2, 2>,
      col2row_shortpair_impl<false, 4, 4>, col2row_shortpair_impl<false, 8, 8>,
      col2row_shortpair_impl<true, 1, 1>,  col2row_shortpair_impl<true, 2, 2>,
      col2row_shortpair_impl<true, 4, 4>,  col2row_shortpair_impl<true, 8, 8>,
  };

  int dispatch_const =
      (col_length == 1 ? 0 : col_length == 2 ? 1 : col_length == 4 ? 2 : 3) +
      (is_row_fixedlen ? 4 : 0);
  col2row_shortpair_fn[dispatch_const](
      num_rows - processed, col_vals_A + col_length * processed,
      col_vals_B + col_length * processed,
      (is_row_fixedlen ? row_offsets : row_offsets + processed),
      row_vals + (is_row_fixedlen ? row_offsets[0] * processed : 0));
}
template void KeyTranspose::col2row_shortpair_avx2<false>(uint32_t, const uint32_t,
                                                          const uint8_t*, const uint8_t*,
                                                          const uint32_t*, uint8_t*);
template void KeyTranspose::col2row_shortpair_avx2<true>(uint32_t, const uint32_t,
                                                         const uint8_t*, const uint8_t*,
                                                         const uint32_t*, uint8_t*);

template <bool is_row_fixedlen>
void KeyTranspose::row2col_shortpair_avx2(uint32_t num_rows, const uint32_t col_length,
                                          uint8_t* col_vals_A, uint8_t* col_vals_B,
                                          const uint32_t* row_offsets,
                                          const uint8_t* row_vals) {
  ARROW_DCHECK(col_length == 1 || col_length == 2 || col_length == 4 || col_length == 8);
  if (num_rows == 0) {
    return;
  }

  uint32_t fixed_len = row_offsets[0];

  if (col_length == 1) {
    // 1-1
    constexpr int unroll = 32;
    for (uint32_t i = 0; i < num_rows / unroll; ++i) {
      uint16_t buffer[unroll];
      for (int j = 0; j < unroll; ++j) {
        const uint8_t* src =
            row_vals + (is_row_fixedlen ? row_offsets[0] * (i * unroll + j)
                                        : row_offsets[i * unroll + j]);
        buffer[j] = *reinterpret_cast<const uint16_t*>(src);
      }
      __m256i r0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(buffer));
      __m256i r1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(buffer) + 1);
      // Collect every second byte next to each other
      r0 = _mm256_shuffle_epi8(
          r0, _mm256_setr_epi64x(0x0e0c0a0806040200ULL, 0x0f0d0b0907050301ULL,
                                 0x0e0c0a0806040200ULL, 0x0f0d0b0907050301ULL));
      r1 = _mm256_shuffle_epi8(
          r1, _mm256_setr_epi64x(0x0e0c0a0806040200ULL, 0x0f0d0b0907050301ULL,
                                 0x0e0c0a0806040200ULL, 0x0f0d0b0907050301ULL));
      r0 = _mm256_permute4x64_epi64(
          r0, 0xd8);  // 0b11011000 swapping second and third 64-bit lane
      r1 = _mm256_permute4x64_epi64(r1, 0xd8);
      __m256i col_A = _mm256_permute2x128_si256(
          r0, r1, 0x20);  // First 128-bit lanes from both inputs
      __m256i col_B = _mm256_permute2x128_si256(r0, r1, 0x31);
      _mm256_storeu_si256(reinterpret_cast<__m256i*>(col_vals_A) + i, col_A);
      _mm256_storeu_si256(reinterpret_cast<__m256i*>(col_vals_B) + i, col_B);
    }
  } else if (col_length == 2) {
    // 2-2
    constexpr int unroll = 16;
    for (uint32_t i = 0; i < num_rows / unroll; ++i) {
      uint32_t buffer[unroll];
      for (int j = 0; j < unroll; ++j) {
        const uint8_t* src =
            row_vals + (is_row_fixedlen ? row_offsets[0] * (i * unroll + j)
                                        : row_offsets[i * unroll + j]);
        buffer[j] = *reinterpret_cast<const uint32_t*>(src);
      }
      __m256i r0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(buffer));
      __m256i r1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(buffer) + 1);
      // Collect every second 16-bit word next to each other
      r0 = _mm256_shuffle_epi8(
          r0, _mm256_setr_epi64x(0x0d0c090805040100ULL, 0x0f0e0b0a07060302ULL,
                                 0x0d0c090805040100ULL, 0x0f0e0b0a07060302ULL));
      r1 = _mm256_shuffle_epi8(
          r1, _mm256_setr_epi64x(0x0d0c090805040100ULL, 0x0f0e0b0a07060302ULL,
                                 0x0d0c090805040100ULL, 0x0f0e0b0a07060302ULL));
      r0 = _mm256_permute4x64_epi64(
          r0, 0xd8);  // 0b11011000 swapping second and third 64-bit lane
      r1 = _mm256_permute4x64_epi64(r1, 0xd8);
      __m256i col_A = _mm256_permute2x128_si256(
          r0, r1, 0x20);  // First 128-bit lanes from both inputs
      __m256i col_B = _mm256_permute2x128_si256(r0, r1, 0x31);
      _mm256_storeu_si256(reinterpret_cast<__m256i*>(col_vals_A) + i, col_A);
      _mm256_storeu_si256(reinterpret_cast<__m256i*>(col_vals_B) + i, col_B);
    }
  } else if (col_length == 4) {
    // 4-4
    constexpr int unroll = 8;
    for (uint32_t i = 0; i < num_rows / unroll; ++i) {
      uint64_t buffer[unroll];
      for (int j = 0; j < 8; ++j) {
        const uint8_t* src =
            row_vals + (is_row_fixedlen ? row_offsets[0] * (i * unroll + j)
                                        : row_offsets[i * unroll + j]);
        buffer[j] = *reinterpret_cast<const uint64_t*>(src);
      }
      __m256i r0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(buffer));
      __m256i r1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(buffer) + 1);
      // Collect every second 32-bit word next to each other
      r0 = _mm256_permutevar8x32_epi32(r0, _mm256_setr_epi32(0, 2, 4, 6, 1, 3, 5, 7));
      r1 = _mm256_permutevar8x32_epi32(r1, _mm256_setr_epi32(0, 2, 4, 6, 1, 3, 5, 7));
      __m256i col_A = _mm256_permute2x128_si256(
          r0, r1, 0x20);  // First 128-bit lanes from both inputs
      __m256i col_B = _mm256_permute2x128_si256(r0, r1, 0x31);
      _mm256_storeu_si256(reinterpret_cast<__m256i*>(col_vals_A) + i, col_A);
      _mm256_storeu_si256(reinterpret_cast<__m256i*>(col_vals_B) + i, col_B);
    }
  } else if (col_length == 8) {
    // 8-8
    constexpr int unroll = 4;
    for (uint32_t i = 0; i < num_rows / unroll; ++i) {
      const __m128i* src0 = reinterpret_cast<const __m128i*>(
          row_vals + (is_row_fixedlen ? row_offsets[0] * (i * unroll + 0)
                                      : row_offsets[i * unroll + 0]));
      const __m128i* src1 = reinterpret_cast<const __m128i*>(
          row_vals + (is_row_fixedlen ? row_offsets[0] * (i * unroll + 1)
                                      : row_offsets[i * unroll + 1]));
      const __m128i* src2 = reinterpret_cast<const __m128i*>(
          row_vals + (is_row_fixedlen ? row_offsets[0] * (i * unroll + 2)
                                      : row_offsets[i * unroll + 2]));
      const __m128i* src3 = reinterpret_cast<const __m128i*>(
          row_vals + (is_row_fixedlen ? row_offsets[0] * (i * unroll + 3)
                                      : row_offsets[i * unroll + 3]));
      __m256i r0 = _mm256_inserti128_si256(_mm256_castsi128_si256(_mm_loadu_si128(src0)),
                                           _mm_loadu_si128(src1), 1);
      __m256i r1 = _mm256_inserti128_si256(_mm256_castsi128_si256(_mm_loadu_si128(src2)),
                                           _mm_loadu_si128(src3), 1);
      r0 = _mm256_permute4x64_epi64(r0, 0xd8);  // 0b11011000
      r1 = _mm256_permute4x64_epi64(r1, 0xd8);
      __m256i col_A = _mm256_permute2x128_si256(
          r0, r1, 0x20);  // First 128-bit lanes from both inputs
      __m256i col_B = _mm256_permute2x128_si256(r0, r1, 0x31);
      _mm256_storeu_si256(reinterpret_cast<__m256i*>(col_vals_A) + i, col_A);
      _mm256_storeu_si256(reinterpret_cast<__m256i*>(col_vals_B) + i, col_B);
    }
  }
  // Process tail
  int unroll = 32 / col_length;
  uint32_t processed = num_rows / unroll * unroll;
  void (*process_pair_fn[])(uint32_t, uint8_t*, uint8_t*, const uint32_t*,
                            const uint8_t*) = {
      row2col_shortpair_impl<false, 1, 1>, row2col_shortpair_impl<false, 2, 2>,
      row2col_shortpair_impl<false, 4, 4>, row2col_shortpair_impl<false, 8, 8>,
      row2col_shortpair_impl<true, 1, 1>,  row2col_shortpair_impl<true, 2, 2>,
      row2col_shortpair_impl<true, 4, 4>,  row2col_shortpair_impl<true, 8, 8>};
  int dispatch_const =
      (col_length == 1 ? 0 : col_length == 2 ? 1 : col_length == 4 ? 2 : 3) +
      (is_row_fixedlen ? 4 : 0);
  process_pair_fn[dispatch_const](
      num_rows - processed, col_vals_A + col_length * processed,
      col_vals_B + col_length * processed,
      is_row_fixedlen ? row_offsets : row_offsets + processed,
      row_vals + (is_row_fixedlen ? processed * fixed_len : 0));
}
template void KeyTranspose::row2col_shortpair_avx2<false>(uint32_t, const uint32_t,
                                                          uint8_t*, uint8_t*,
                                                          const uint32_t*,
                                                          const uint8_t*);
template void KeyTranspose::row2col_shortpair_avx2<true>(uint32_t, const uint32_t,
                                                         uint8_t*, uint8_t*,
                                                         const uint32_t*, const uint8_t*);

template <bool is_col_fixedlen, bool is_row_fixedlen, bool update_row_offsets>
void KeyTranspose::col2row_long_avx2(uint32_t num_rows, const uint8_t* col_non_nulls,
                                     const uint32_t* col_offsets, const uint8_t* col_vals,
                                     const uint32_t* row_offsets, uint8_t* row_vals,
                                     uint32_t* row_offsets_updated) {
  ARROW_DCHECK(!is_row_fixedlen || !update_row_offsets);

  // If the column is fixed length, then col_offsets array has only one element, which
  // specifies the length of the column values. If the row is fixed length, then
  // row_offsets array has only one element, which specifies the length of the row values.
  uint32_t col_fixed_len = col_offsets[0];
  uint32_t row_fixed_len = row_offsets[0];
  uint32_t col_offset_next = col_offsets[0];
  uint64_t col_non_nulls_word = 0;
  for (uint32_t i = 0; i < num_rows; ++i) {
    uint32_t col_offset;
    uint32_t length;
    if (is_col_fixedlen) {
      col_offset = col_fixed_len * i;
      length = col_fixed_len;
    } else {
      col_offset = col_offset_next;
      col_offset_next = col_offsets[i + 1];
      length = col_offset_next - col_offset;
      if ((i & 63) == 0) {
        col_non_nulls_word = reinterpret_cast<const uint64_t*>(col_non_nulls)[i / 64];
      }
      uint32_t is_non_null = (col_non_nulls_word & 1);
      col_non_nulls_word >>= 1;
      length *= is_non_null;
    }
    const __m256i* src = reinterpret_cast<const __m256i*>(col_vals + col_offset);
    uint32_t row_offset;
    if (is_row_fixedlen) {
      row_offset = row_fixed_len * i;
    } else {
      row_offset = row_offsets[i];
    }
    __m256i* dst = reinterpret_cast<__m256i*>(row_vals + row_offset);
    uint32_t istripe;
    for (istripe = 0; istripe < length / 32; ++istripe) {
      _mm256_storeu_si256(dst + istripe, _mm256_loadu_si256(src + istripe));
    }
    if ((length % 32) > 0) {
      __m256i mask = set_first_n_bytes_avx2(length % 32);
      _mm256_storeu_si256(dst + istripe,
                          _mm256_blendv_epi8(_mm256_loadu_si256(dst + istripe),
                                             _mm256_loadu_si256(src + istripe), mask));
    }
    if (update_row_offsets) {
      row_offsets_updated[i] = row_offset + length;
    }
  }
}
template void KeyTranspose::col2row_long_avx2<false, false, false>(
    uint32_t, const uint8_t*, const uint32_t*, const uint8_t*, const uint32_t*, uint8_t*,
    uint32_t*);
template void KeyTranspose::col2row_long_avx2<false, false, true>(
    uint32_t, const uint8_t*, const uint32_t*, const uint8_t*, const uint32_t*, uint8_t*,
    uint32_t*);
template void KeyTranspose::col2row_long_avx2<false, true, false>(
    uint32_t, const uint8_t*, const uint32_t*, const uint8_t*, const uint32_t*, uint8_t*,
    uint32_t*);
template void KeyTranspose::col2row_long_avx2<true, false, false>(
    uint32_t, const uint8_t*, const uint32_t*, const uint8_t*, const uint32_t*, uint8_t*,
    uint32_t*);
template void KeyTranspose::col2row_long_avx2<true, false, true>(uint32_t, const uint8_t*,
                                                                 const uint32_t*,
                                                                 const uint8_t*,
                                                                 const uint32_t*,
                                                                 uint8_t*, uint32_t*);
template void KeyTranspose::col2row_long_avx2<true, true, false>(uint32_t, const uint8_t*,
                                                                 const uint32_t*,
                                                                 const uint8_t*,
                                                                 const uint32_t*,
                                                                 uint8_t*, uint32_t*);

template <bool is_col_fixedlen, bool is_row_fixedlen, bool update_row_offsets>
void KeyTranspose::row2col_long_avx2(uint32_t num_rows, const uint32_t* col_offsets,
                                     uint8_t* col_vals, const uint32_t* row_offsets,
                                     const uint8_t* row_vals,
                                     uint32_t* row_offsets_updated) {
  ARROW_DCHECK(!is_row_fixedlen || !update_row_offsets);

  // If the column is fixed length, then col_offsets array has only one element, which
  // specifies the length of the column values. If the row is fixed length, then
  // row_offsets array has only one element, which specifies the length of the row values.
  uint32_t col_fixed_len = col_offsets[0];
  uint32_t row_fixed_len = row_offsets[0];
  uint32_t col_offset_next = col_offsets[0];
  for (uint32_t i = 0; i < num_rows; ++i) {
    uint32_t col_offset;
    uint32_t length;
    if (is_col_fixedlen) {
      col_offset = col_fixed_len * i;
      length = col_fixed_len;
    } else {
      col_offset = col_offset_next;
      col_offset_next = col_offsets[i + 1];
      length = col_offset_next - col_offset;
    }
    __m256i* dst = reinterpret_cast<__m256i*>(col_vals + col_offset);
    uint32_t row_offset;
    if (is_row_fixedlen) {
      row_offset = row_fixed_len * i;
    } else {
      row_offset = row_offsets[i];
    }
    const __m256i* src = reinterpret_cast<const __m256i*>(row_vals + row_offset);
    uint32_t istripe;
    for (istripe = 0; istripe < (length + 31) / 32; ++istripe) {
      _mm256_storeu_si256(dst + istripe, _mm256_loadu_si256(src + istripe));
    }
    if (update_row_offsets) {
      row_offsets_updated[i] = row_offset + length;
    }
  }
}
template void KeyTranspose::row2col_long_avx2<false, false, false>(
    uint32_t, const uint32_t*, uint8_t*, const uint32_t*, const uint8_t*, uint32_t*);
template void KeyTranspose::row2col_long_avx2<false, false, true>(
    uint32_t, const uint32_t*, uint8_t*, const uint32_t*, const uint8_t*, uint32_t*);
template void KeyTranspose::row2col_long_avx2<false, true, false>(
    uint32_t, const uint32_t*, uint8_t*, const uint32_t*, const uint8_t*, uint32_t*);
template void KeyTranspose::row2col_long_avx2<true, false, false>(
    uint32_t, const uint32_t*, uint8_t*, const uint32_t*, const uint8_t*, uint32_t*);
template void KeyTranspose::row2col_long_avx2<true, false, true>(
    uint32_t, const uint32_t*, uint8_t*, const uint32_t*, const uint8_t*, uint32_t*);
template void KeyTranspose::row2col_long_avx2<true, true, false>(
    uint32_t, const uint32_t*, uint8_t*, const uint32_t*, const uint8_t*, uint32_t*);

void KeyTranspose::offsets_to_lengths_avx2(uint32_t num_rows, const uint32_t* offsets,
                                           uint32_t* lengths) {
  constexpr int unroll = 8;
  for (uint32_t i = 0; i < num_rows / unroll; ++i) {
    __m256i offs_begin =
        _mm256_loadu_si256(reinterpret_cast<const __m256i*>(offsets) + i);
    __m256i offs_end =
        _mm256_loadu_si256(reinterpret_cast<const __m256i*>(offsets + 1) + i);
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(lengths) + i,
                        _mm256_sub_epi32(offs_end, offs_begin));
  }
}

#endif

}  // namespace exec
}  // namespace arrow
