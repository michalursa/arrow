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

#include "arrow/engine/key_encode.h"

namespace arrow {
namespace compute {

#if defined(ARROW_HAVE_AVX2)

inline __m256i set_first_n_bytes_avx2(int n) {
  return _mm256_cmpgt_epi8(
      _mm256_set1_epi8(n),
      _mm256_setr_epi64x(0x0706050403020100ULL, 0x0f0e0d0c0b0a0908ULL,
                         0x1716151413121110ULL, 0x1f1e1d1c1b1a1918ULL));
}

inline __m256i inclusive_prefix_sum_32bit_avx2(__m256i x) {
  x = _mm256_add_epi32(
      x, _mm256_permutevar8x32_epi32(
             _mm256_andnot_si256(_mm256_setr_epi32(0, 0, 0, 0, 0, 0, 0, 0xffffffff), x),
             _mm256_setr_epi32(7, 0, 1, 2, 3, 4, 5, 6)));
  x = _mm256_add_epi32(
      x, _mm256_permute4x64_epi64(
             _mm256_andnot_si256(
                 _mm256_setr_epi32(0, 0, 0, 0, 0, 0, 0xffffffff, 0xffffffff), x),
             0x93));  // 0b10010011
  x = _mm256_add_epi32(
      x, _mm256_permute4x64_epi64(
             _mm256_andnot_si256(
                 _mm256_setr_epi32(0, 0, 0, 0, 0, 0, 0xffffffff, 0xffffffff), x),
             0x4f));  // 0b01001111
  return x;
}

void KeyEncoder::EncoderBinary::EncodeHelper_avx2(bool is_row_fixed_length,
                                                  uint32_t offset_within_row,
                                                  KeyRowArray* rows,
                                                  const KeyColumnArray& col) {
  if (is_row_fixed_length) {
    EncodeImp_avx2<true>(offset_within_row, rows, col);
  } else {
    EncodeImp_avx2<false>(offset_within_row, rows, col);
  }
}

template <bool is_row_fixed_length>
void KeyEncoder::EncoderBinary::EncodeImp_avx2(uint32_t offset_within_row,
                                               KeyRowArray* rows,
                                               const KeyColumnArray& col) {
  EncodeDecodeHelper<is_row_fixed_length, true>(
      0, static_cast<uint32_t>(col.get_length()), offset_within_row, rows, rows, &col,
      nullptr, [](uint8_t* dst, const uint8_t* src, int64_t length) {
        __m256i* dst256 = reinterpret_cast<__m256i*>(dst);
        const __m256i* src256 = reinterpret_cast<const __m256i*>(src);
        uint32_t istripe;
        for (istripe = 0; istripe < length / 32; ++istripe) {
          _mm256_storeu_si256(dst256 + istripe, _mm256_loadu_si256(src256 + istripe));
        }
        if ((length % 32) > 0) {
          __m256i mask = set_first_n_bytes_avx2(length % 32);
          _mm256_storeu_si256(
              dst256 + istripe,
              _mm256_blendv_epi8(_mm256_loadu_si256(dst256 + istripe),
                                 _mm256_loadu_si256(src256 + istripe), mask));
        }
      });
}

void KeyEncoder::EncoderBinary::DecodeHelper_avx2(bool is_row_fixed_length,
                                                  uint32_t start_row, uint32_t num_rows,
                                                  uint32_t offset_within_row,
                                                  const KeyRowArray& rows,
                                                  KeyColumnArray* col) {
  if (is_row_fixed_length) {
    DecodeImp_avx2<true>(start_row, num_rows, offset_within_row, rows, col);
  } else {
    DecodeImp_avx2<false>(start_row, num_rows, offset_within_row, rows, col);
  }
}

template <bool is_row_fixed_length>
void KeyEncoder::EncoderBinary::DecodeImp_avx2(uint32_t start_row, uint32_t num_rows,
                                               uint32_t offset_within_row,
                                               const KeyRowArray& rows,
                                               KeyColumnArray* col) {
  EncodeDecodeHelper<is_row_fixed_length, false>(
      start_row, num_rows, offset_within_row, &rows, nullptr, col, col,
      [](uint8_t* dst, const uint8_t* src, int64_t length) {
        for (uint32_t istripe = 0; istripe < (length + 31) / 32; ++istripe) {
          __m256i* dst256 = reinterpret_cast<__m256i*>(dst);
          const __m256i* src256 = reinterpret_cast<const __m256i*>(src);
          _mm256_storeu_si256(dst256 + istripe, _mm256_loadu_si256(src256 + istripe));
        }
      });
}

uint32_t KeyEncoder::EncoderBinaryPair::EncodeHelper_avx2(
    bool is_row_fixed_length, uint32_t col_width, uint32_t offset_within_row,
    KeyRowArray* rows, const KeyColumnArray& col1, const KeyColumnArray& col2) {
  typedef uint32_t (*EncodeImp_avx2_t)(uint32_t, KeyRowArray&, const KeyColumnArray&,
                                       const KeyColumnArray&);
  static const EncodeImp_avx2_t EncodeImp_avx2_fn[] = {
      EncodeImp_avx2<false, 1>, EncodeImp_avx2<false, 2>, EncodeImp_avx2<false, 4>,
      EncodeImp_avx2<false, 8>, EncodeImp_avx2<true, 1>,  EncodeImp_avx2<true, 2>,
      EncodeImp_avx2<true, 4>,  EncodeImp_avx2<true, 8>};
  int log_col_width = col_width == 8 ? 3 : col_width == 4 ? 2 : col_width == 2 ? 1 : 0;
  int dispatch_const = (is_row_fixed_length ? 4 : 0) + log_col_width;
  return EncodeImp_avx2_fn[dispatch_const](offset_within_row, rows, col1, col2);
}

template <bool is_row_fixed_length, uint32_t col_width>
uint32_t KeyEncoder::EncoderBinaryPair::EncodeImp_avx2(uint32_t offset_within_row,
                                                       KeyRowArray* rows,
                                                       const KeyColumnArray& col1,
                                                       const KeyColumnArray& col2) {
  uint32_t num_rows = static_cast<uint32_t>(col1.get_length());
  ARROW_DCHECK(col_width == 1 || col_width == 2 || col_width == 4 || col_width == 8);

  const uint8_t* col_vals_A = col1.data(1);
  const uint8_t* col_vals_B = col2.data(1);
  uint8_t* row_vals = is_row_fixed_length ? rows->mutable_data(1) : rows->mutable_data(2);

  constexpr int unroll = 32 / col_width;

  uint32_t num_processed = num_rows / unroll * unroll;

  for (uint32_t i = 0; i < num_rows / unroll; ++i) {
    __m256i col_A = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(col_vals_A) + i);
    __m256i col_B = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(col_vals_B) + i);
    __m256i r0, r1;
    if (col_width == 1) {
      // results in 16-bit outputs in the order: 0..7, 16..23
      r0 = _mm256_unpacklo_epi8(col_A, col_B);
      // results in 16-bit outputs in the order: 8..15, 24..31
      r1 = _mm256_unpackhi_epi8(col_A, col_B);
    } else if (col_width == 2) {
      // results in 32-bit outputs in the order: 0..3, 8..11
      r0 = _mm256_unpacklo_epi16(col_A, col_B);
      // results in 32-bit outputs in the order: 4..7, 12..15
      r1 = _mm256_unpackhi_epi16(col_A, col_B);
    } else if (col_width == 4) {
      // results in 64-bit outputs in the order: 0..1, 4..5
      r0 = _mm256_unpacklo_epi32(col_A, col_B);
      // results in 64-bit outputs in the order: 2..3, 6..7
      r1 = _mm256_unpackhi_epi32(col_A, col_B);
    } else if (col_width == 8) {
      // results in 128-bit outputs in the order: 0, 2
      r0 = _mm256_unpacklo_epi64(col_A, col_B);
      // results in 128-bit outputs in the order: 1, 3
      r1 = _mm256_unpackhi_epi64(col_A, col_B);
    }
    col_A = _mm256_permute2x128_si256(r0, r1, 0x20);
    col_B = _mm256_permute2x128_si256(r0, r1, 0x31);
    if (col_width == 8) {
      __m128i *dst0, *dst1, *dst2, *dst3;
      if (is_row_fixed_length) {
        uint32_t fixed_length = rows->get_metadata().fixed_length;
        uint8_t* dst = row_vals + offset_within_row + fixed_length * i * unroll;
        dst0 = reinterpret_cast<__m128i*>(dst);
        dst1 = reinterpret_cast<__m128i*>(dst + fixed_length);
        dst2 = reinterpret_cast<__m128i*>(dst + fixed_length * 2);
        dst3 = reinterpret_cast<__m128i*>(dst + fixed_length * 3);
      } else {
        const uint32_t* row_offsets = rows->get_offsets() + i * unroll;
        uint8_t* dst = row_vals + offset_within_row;
        dst0 = reinterpret_cast<__m128i*>(dst + row_offsets[0]);
        dst1 = reinterpret_cast<__m128i*>(dst + row_offsets[1]);
        dst2 = reinterpret_cast<__m128i*>(dst + row_offsets[2]);
        dst3 = reinterpret_cast<__m128i*>(dst + row_offsets[3]);
      }
      _mm_storeu_si128(dst0, _mm256_castsi256_si128(r0));
      _mm_storeu_si128(dst1, _mm256_castsi256_si128(r1));
      _mm_storeu_si128(dst2, _mm256_extracti128_si256(r0, 1));
      _mm_storeu_si128(dst3, _mm256_extracti128_si256(r1, 1));

    } else {
      uint8_t buffer[64];
      _mm256_storeu_si256(reinterpret_cast<__m256i*>(buffer), col_A);
      _mm256_storeu_si256(reinterpret_cast<__m256i*>(buffer) + 1, col_B);

      if (is_row_fixed_length) {
        uint32_t fixed_length = rows->get_metadata().fixed_length;
        uint8_t* dst = row_vals + offset_within_row + fixed_length * i * unroll;
        for (int j = 0; j < unroll; ++j) {
          if (col_width == 1) {
            *reinterpret_cast<uint16_t*>(dst + fixed_length * j) =
                reinterpret_cast<const uint16_t*>(buffer)[j];
          } else if (col_width == 2) {
            *reinterpret_cast<uint32_t*>(dst + fixed_length * j) =
                reinterpret_cast<const uint32_t*>(buffer)[j];
          } else if (col_width == 4) {
            *reinterpret_cast<uint64_t*>(dst + fixed_length * j) =
                reinterpret_cast<const uint64_t*>(buffer)[j];
          }
        }
      } else {
        const uint32_t* row_offsets = rows->get_offsets() + i * unroll;
        uint8_t* dst = row_vals + offset_within_row;
        for (int j = 0; j < unroll; ++j) {
          if (col_width == 1) {
            *reinterpret_cast<uint16_t*>(dst + row_offsets[j]) =
                reinterpret_cast<const uint16_t*>(buffer)[j];
          } else if (col_width == 2) {
            *reinterpret_cast<uint32_t*>(dst + row_offsets[j]) =
                reinterpret_cast<const uint32_t*>(buffer)[j];
          } else if (col_width == 4) {
            *reinterpret_cast<uint64_t*>(dst + row_offsets[j]) =
                reinterpret_cast<const uint64_t*>(buffer)[j];
          }
        }
      }
    }
  }

  return num_processed;
}

uint32_t KeyEncoder::EncoderBinaryPair::DecodeHelper_avx2(
    bool is_row_fixed_length, uint32_t col_width, uint32_t start_row, uint32_t num_rows,
    uint32_t offset_within_row, const KeyRowArray& rows, KeyColumnArray* col1,
    KeyColumnArray* col2) {
  typedef uint32_t (*DecodeImp_avx2_t)(uint32_t, uint32_t, uint32_t, const KeyRowArray&,
                                       KeyColumnArray&, KeyColumnArray&);
  static const DecodeImp_avx2_t DecodeImp_avx2_fn[] = {
      DecodeImp_avx2<false, 1>, DecodeImp_avx2<false, 2>, DecodeImp_avx2<false, 4>,
      DecodeImp_avx2<false, 8>, DecodeImp_avx2<true, 1>,  DecodeImp_avx2<true, 2>,
      DecodeImp_avx2<true, 4>,  DecodeImp_avx2<true, 8>};
  int log_col_width = col_width == 8 ? 3 : col_width == 4 ? 2 : col_width == 2 ? 1 : 0;
  int dispatch_const = log_col_width | (is_row_fixed_length ? 4 : 0);
  return DecodeImp_avx2_fn[dispatch_const](start_row, num_processed, offset_within_row,
                                           rows, col1, col2);
}

template <bool is_row_fixed_length, uint32_t col_width>
uint32_t KeyEncoder::EncoderBinaryPair::DecodeImp_avx2(
    uint32_t start_row, uint32_t num_rows, uint32_t offset_within_row,
    const KeyRowArray& rows, KeyColumnArray* col1, KeyColumnArray* col2) {
  ARROW_DCHECK(col_width == 1 || col_width == 2 || col_width == 4 || col_width == 8);

  uint8_t* col_vals_A = col1->mutable_data(1);
  uint8_t* col_vals_B = col2->mutable_data(1);
  const uint8_t* row_vals = is_row_fixed_length ? rows.data(1) : rows.data(2);

  constexpr int unroll = 32 / col_width;

  uint32_t num_processed = num_rows / unroll * unroll;

  if (col_width == 8) {
    for (uint32_t i = 0; i < num_rows / unroll; ++i) {
      const __m128i *src0, *src1, *src2, *src3;
      if (is_row_fixed_length) {
        uint32_t fixed_length = rows.get_metadata().fixed_length;
        const uint8_t* src = row_vals + offset_within_row + fixed_length * i * unroll;
        src0 = reinterpret_cast<const __m128i*>(src);
        src1 = reinterpret_cast<const __m128i*>(src + fixed_length);
        src2 = reinterpret_cast<const __m128i*>(src + fixed_length * 2);
        src3 = reinterpret_cast<const __m128i*>(src + fixed_length * 3);
      } else {
        const uint32_t* row_offsets = rows.get_offsets() + i * unroll;
        const uint8_t* src = row_vals + offset_within_row;
        src0 = reinterpret_cast<const __m128i*>(src + row_offsets[0]);
        src1 = reinterpret_cast<const __m128i*>(src + row_offsets[1]);
        src2 = reinterpret_cast<const __m128i*>(src + row_offsets[2]);
        src3 = reinterpret_cast<const __m128i*>(src + row_offsets[3]);
      }

      __m256i r0 = _mm256_inserti128_si256(_mm256_castsi128_si256(_mm_loadu_si128(src0)),
                                           _mm_loadu_si128(src1), 1);
      __m256i r1 = _mm256_inserti128_si256(_mm256_castsi128_si256(_mm_loadu_si128(src2)),
                                           _mm_loadu_si128(src3), 1);

      r0 = _mm256_permute4x64_epi64(r0, 0xd8);  // 0b11011000
      r1 = _mm256_permute4x64_epi64(r1, 0xd8);

      // First 128-bit lanes from both inputs
      __m256i c1 = _mm256_permute2x128_si256(r0, r1, 0x20);
      // Second 128-bit lanes from both inputs
      __m256i c2 = _mm256_permute2x128_si256(r0, r1, 0x31);
      _mm256_storeu_si256(reinterpret_cast<__m256i*>(col_vals_A) + i, c1);
      _mm256_storeu_si256(reinterpret_cast<__m256i*>(col_vals_B) + i, c2);
    }
  } else {
    uint8_t buffer[64];
    for (uint32_t i = 0; i < num_rows / unroll; ++i) {
      if (is_row_fixed_length) {
        uint32_t fixed_length = rows.get_metadata().fixed_length;
        const uint8_t* src = row_vals + offset_within_row + fixed_length * i * unroll;
        for (int j = 0; j < unroll; ++j) {
          if (col_width == 1) {
            reinterpret_cast<uint16_t*>(buffer)[j] =
                *reinterpret_cast<const uint16_t*>(src + fixed_length * j);
          } else if (col_width == 2) {
            reinterpret_cast<uint32_t*>(buffer)[j] =
                *reinterpret_cast<const uint32_t*>(src + fixed_length * j);
          } else if (col_width == 4) {
            reinterpret_cast<uint64_t*>(buffer)[j] =
                *reinterpret_cast<const uint64_t*>(src + fixed_length * j);
          }
        }
      } else {
        const uint32_t* row_offsets = rows.get_offsets() + i * unroll;
        const uint8_t* src = row_vals + offset_within_row;
        for (int j = 0; j < unroll; ++j) {
          if (col_width == 1) {
            reinterpret_cast<uint16_t*>(buffer)[j] =
                *reinterpret_cast<const uint16_t*>(src + row_offsets[j]);
          } else if (col_width == 2) {
            reinterpret_cast<uint32_t*>(buffer)[j] =
                *reinterpret_cast<const uint32_t*>(src + row_offsets[j]);
          } else if (col_width == 4) {
            reinterpret_cast<uint64_t*>(buffer)[j] =
                *reinterpret_cast<const uint64_t*>(src + row_offsets[j]);
          }
        }
      }

      __m256i r0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(buffer));
      __m256i r1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(buffer) + 1);

      if (col_width == 1) {
        // Collect every second byte next to each other
        const __m256i shuffle_const =
            _mm256_setr_epi64x(0x0e0c0a0806040200ULL, 0x0f0d0b0907050301ULL,
                               0x0e0c0a0806040200ULL, 0x0f0d0b0907050301ULL);
        r0 = _mm256_shuffle_epi8(r0, shuffle_const);
        r1 = _mm256_shuffle_epi8(r1, shuffle_const);
        // 0b11011000 swapping second and third 64-bit lane
        r0 = _mm256_permute4x64_epi64(r0, 0xd8);
        r1 = _mm256_permute4x64_epi64(r1, 0xd8);
      } else if (col_width == 2) {
        // Collect every second 16-bit word next to each other
        const __m256i shuffle_const =
            _mm256_setr_epi64x(0x0d0c090805040100ULL, 0x0f0e0b0a07060302ULL,
                               0x0d0c090805040100ULL, 0x0f0e0b0a07060302ULL);
        r0 = _mm256_shuffle_epi8(r0, shuffle_const);
        r1 = _mm256_shuffle_epi8(r1, shuffle_const);
        // 0b11011000 swapping second and third 64-bit lane
        r0 = _mm256_permute4x64_epi64(r0, 0xd8);
        r1 = _mm256_permute4x64_epi64(r1, 0xd8);
      } else if (col_width == 4) {
        // Collect every second 32-bit word next to each other
        const __m256i permute_const = _mm256_setr_epi32(0, 2, 4, 6, 1, 3, 5, 7);
        r0 = _mm256_permutevar8x32_epi32(r0, permute_const);
        r1 = _mm256_permutevar8x32_epi32(r1, permute_const);
      }

      // First 128-bit lanes from both inputs
      __m256i c1 = _mm256_permute2x128_si256(r0, r1, 0x20);
      // Second 128-bit lanes from both inputs
      __m256i c2 = _mm256_permute2x128_si256(r0, r1, 0x31);
      _mm256_storeu_si256(reinterpret_cast<__m256i*>(col_vals_A) + i, c1);
      _mm256_storeu_si256(reinterpret_cast<__m256i*>(col_vals_B) + i, c2);
    }
  }

  return num_processed;
}

uint32_t KeyEncoder::EncoderOffsets::EncodeImp_avx2(
    KeyRowArray* rows, const std::vector<KeyColumnArray>& varbinary_cols,
    KeyColumnArray* temp_buffer_32B_per_col) {
  ARROW_DCHECK(temp_buffer_32B_per_col->get_metadata().is_fixed_length &&
               temp_buffer_32B_per_col->get_metadata().fixed_length ==
                   static_cast<uint32_t>(sizeof(uint32_t)) &&
               temp_buffer_32B_per_col->get_length() >=
                   static_cast<int64_t>(varbinary_cols.size()) * 8);

  ARROW_DCHECK(varbinary_cols.size() > 0);

  // Add together first offset in every column.
  // This value will be mapped to zero when
  // computing varying-length component of output row offsets.
  uint32_t sum_offsets_first = 0;
  for (size_t col = 0; col < varbinary_cols.size(); ++col) {
    ARROW_DCHECK(!varbinary_cols[col].get_metadata().is_fixed_length);
    const uint32_t* col_offsets =
        reinterpret_cast<const uint32_t*>(varbinary_cols[col].data(1));
    sum_offsets_first += col_offsets[0];
  }

  // There is a fixed-length part in every row.
  // This needs to be included in calculation of row offsets.
  uint32_t fixed_part =
      rows->get_metadata().fixed_length + rows->get_metadata().cumulative_lengths_length;

  // Difference between output row offset and direct sum of offsets for all columns
  __m256i offset_adjustment = _mm256_sub_epi32(
      _mm256_setr_epi32(fixed_part * 1, fixed_part * 2, fixed_part * 3, fixed_part * 4,
                        fixed_part * 5, fixed_part * 6, fixed_part * 7, fixed_part * 8),
      _mm256_set1_epi32(sum_offsets_first));
  __m256i offset_adjustment_incr = _mm256_set1_epi32(fixed_part * 8);

  uint32_t* row_offsets = reinterpret_cast<uint32_t*>(rows->mutable_data(1));
  uint8_t* row_values = rows->mutable_data(2);
  uint32_t num_rows = static_cast<uint32_t>(varbinary_cols[0].get_length());
  constexpr int unroll = 8;
  uint32_t num_processed = num_rows / unroll * unroll;
  uint32_t* temp_cumulative_lengths =
      reinterpret_cast<uint32_t*>(temp_buffer_32B_per_col->mutable_data(1));

  row_offsets[0] = 0;
  for (uint32_t i = 0; i < num_rows / unroll; ++i) {
    // Zero out lengths for nulls.
    // Add horizontally offsets, after adjustment for nulls, to
    // produce row offsets.
    // Add horizontally and store in temp buffer lengths,
    // after adjustment for nulls,
    // to produce cumulative lengths of individual column values in each row.
    bool any_nulls = false;
    __m256i col_null_length_sum = _mm256_setzero_si256();
    __m256i col_offset_sum = _mm256_setzero_si256();
    __m256i col_length_sum = _mm256_setzero_si256();
    for (size_t col = 0; col < varbinary_cols.size(); ++col) {
      const uint32_t* col_offsets =
          reinterpret_cast<const uint32_t*>(varbinary_cols[col].data(1));
      __m256i col_offset =
          _mm256_loadu_si256(reinterpret_cast<const __m256i*>(col_offsets + 1) + i);
      __m256i col_length = _mm256_sub_epi32(
          col_offset,
          _mm256_loadu_si256(reinterpret_cast<const __m256i*>(col_offsets + 0) + i));
      col_offset_sum = _mm256_add_epi32(col_offset_sum, col_offset);

      const uint8_t* non_nulls = varbinary_cols[col].data(0);
      if (non_nulls && non_nulls[i] != 0xff) {
        // Zero out lengths for values that are not null
        any_nulls = true;
        const __m256i individual_bits =
            _mm256_setr_epi32(0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80);
        __m256i null_mask = _mm256_cmpeq_epi32(
            _mm256_setzero_si256(),
            _mm256_and_si256(_mm256_set1_epi32(non_nulls[i]), individual_bits));
        __m256i null_length = _mm256_and_si256(col_length, null_mask);
        col_null_length_sum = _mm256_add_epi32(col_null_length_sum, null_length);
        col_length = _mm256_sub_epi32(col_length, null_length);
      }

      col_length_sum = _mm256_add_epi32(col_length_sum, col_length);
      _mm256_storeu_si256(reinterpret_cast<__m256i*>(temp_cumulative_lengths) + col,
                          col_length_sum);
    }

    // Add fixed-length part to row offsets
    __m256i row_offset = _mm256_add_epi32(col_offset_sum, offset_adjustment);
    offset_adjustment = _mm256_add_epi32(offset_adjustment, offset_adjustment_incr);

    // Adjustments for nulls
    if (any_nulls) {
      // Inclusive prefix sum of 32-bit elements
      col_null_length_sum = inclusive_prefix_sum_32bit_avx2(col_null_length_sum);
      row_offset = _mm256_sub_epi32(row_offset, col_null_length_sum);
      offset_adjustment = _mm256_sub_epi32(
          offset_adjustment,
          _mm256_permutevar8x32_epi32(col_null_length_sum, _mm256_set1_epi32(7)));
    }

    _mm256_storeu_si256(reinterpret_cast<__m256i*>(row_offsets + 1) + i, row_offset);

    // Output cumulative varbinary key column lengths for each row
    for (size_t col = 0; col < varbinary_cols.size(); ++col) {
      for (uint32_t row = 0; row < unroll; ++row) {
        uint32_t* dst =
            reinterpret_cast<uint32_t*>(row_values + row_offsets[i * unroll + row] +
                                        rows->get_metadata().fixed_length) +
            col;
        const uint32_t* src = temp_cumulative_lengths + (col * unroll + row);
        *dst = *src;
      }
    }
  }

  return num_processed;
}

void KeyEncoder::EncoderVarBinary::EncodeHelper_avx2(uint32_t varbinary_col_id,
                                                     KeyRowArray* rows,
                                                     const KeyColumnArray& col) {
  if (varbinary_col_id == 0) {
    EncodeImp_avx2<true>(varbinary_col_id, rows, col);
  } else {
    EncodeImp_avx2<false>(varbinary_col_id, rows, col);
  }
}

template <bool first_varbinary_col>
void KeyEncoder::EncoderVarBinary::EncodeImp_avx2(uint32_t varbinary_col_id,
                                                  KeyRowArray* rows,
                                                  const KeyColumnArray& col) {
  EncodeDecodeHelper<first_varbinary_col, true>(
      0, static_cast<uint32_t>(col.get_length()), varbinary_col_id, rows, rows, &col,
      nullptr, [](uint8_t* dst, const uint8_t* src, int64_t length) {
        __m256i* dst256 = reinterpret_cast<__m256i*>(dst);
        const __m256i* src256 = reinterpret_cast<const __m256i*>(src);
        uint32_t istripe;
        for (istripe = 0; istripe < length / 32; ++istripe) {
          _mm256_storeu_si256(dst256 + istripe, _mm256_loadu_si256(src256 + istripe));
        }
        if ((length % 32) > 0) {
          __m256i mask = set_first_n_bytes_avx2(length % 32);
          _mm256_storeu_si256(
              dst256 + istripe,
              _mm256_blendv_epi8(_mm256_loadu_si256(dst256 + istripe),
                                 _mm256_loadu_si256(src256 + istripe), mask));
        }
      });
}

void KeyEncoder::EncoderVarBinary::DecodeHelper_avx2(uint32_t start_row,
                                                     uint32_t num_rows,
                                                     uint32_t varbinary_col_id,
                                                     const KeyRowArray& rows,
                                                     KeyColumnArray* col) {
  if (varbinary_col_id == 0) {
    DecodeImp_avx2<true>(start_row, num_rows, varbinary_col_id, rows, col);
  } else {
    DecodeImp_avx2<false>(start_row, num_rows, varbinary_col_id, rows, col);
  }
}

template <bool first_varbinary_col>
void KeyEncoder::EncoderVarBinary::DecodeImp_avx2(uint32_t start_row, uint32_t num_rows,
                                                  uint32_t varbinary_col_id,
                                                  const KeyRowArray& rows,
                                                  KeyColumnArray* col) {
  EncodeDecodeHelper<first_varbinary_col, false>(
      start_row, num_rows, varbinary_col_id, &rows, nullptr, col, col,
      [](uint8_t* dst, const uint8_t* src, int64_t length) {
        for (uint32_t istripe = 0; istripe < (length + 31) / 32; ++istripe) {
          __m256i* dst256 = reinterpret_cast<__m256i*>(dst);
          const __m256i* src256 = reinterpret_cast<const __m256i*>(src);
          _mm256_storeu_si256(dst256 + istripe, _mm256_loadu_si256(src256 + istripe));
        }
      });
}

#endif

}  // namespace compute
}  // namespace arrow
