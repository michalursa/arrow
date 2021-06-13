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
#include <memory.h>

#include <algorithm>
#include <cstdint>

#include "arrow/compute/exec/key_compare.h"
#include "arrow/compute/exec/util.h"
#include "arrow/util/bit_util.h"

namespace arrow {
namespace compute {

template <bool use_selection>
void KeyCompare::NullUpdateColumnToRow(uint32_t id_col, uint32_t num_rows_to_compare,
                                       const uint16_t* sel_left_maybe_null,
                                       const uint32_t* left_to_right_map,
                                       KeyEncoder::KeyEncoderContext* ctx,
                                       const KeyEncoder::KeyColumnArray& col,
                                       const KeyEncoder::KeyRowArray& rows,
                                       uint8_t* match_bytevector) {
  if (rows.has_any_nulls(ctx) || col.data(0)) {
    if (!col.data(0)) {
      // Remove rows from the result for which the column value is a null
      const uint8_t* null_masks = rows.null_masks();
      uint32_t null_mask_num_bytes = rows.metadata().null_masks_bytes_per_row;

      uint32_t num_processed = 0;
      if (ctx->has_avx2()) {
        constexpr uint32_t unroll = 8;
        for (uint32_t i = 0; i < num_rows_to_compare / unroll; ++i) {
          __m256i irow_right;
          if (use_selection) {
            __m256i irow_left = _mm256_cvtepu16_epi32(_mm_loadu_si128(
                reinterpret_cast<const __m128i*>(sel_left_maybe_null) + i));
            irow_right =
                _mm256_i32gather_epi32((const int*)left_to_right_map, irow_left, 4);
          } else {
            irow_right = _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(left_to_right_map) + i);
          }
          __m256i bitid =
              _mm256_mullo_epi32(irow_right, _mm256_set1_epi32(null_mask_num_bytes * 8));
          bitid = _mm256_add_epi32(bitid, _mm256_set1_epi32(id_col));
          __m256i right = _mm256_i32gather_epi32((const int*)null_masks,
                                                 _mm256_srli_epi32(bitid, 3), 1);
          right = _mm256_and_si256(
              _mm256_set1_epi32(1),
              _mm256_srlv_epi32(right, _mm256_and_si256(bitid, _mm256_set1_epi32(7))));
          __m256i cmp = _mm256_cmpeq_epi32(right, _mm256_setzero_si256());
          uint32_t result_lo =
              _mm256_movemask_epi8(_mm256_cvtepi32_epi64(_mm256_castsi256_si128(cmp)));
          uint32_t result_hi = _mm256_movemask_epi8(
              _mm256_cvtepi32_epi64(_mm256_extracti128_si256(cmp, 1)));
          reinterpret_cast<uint64_t*>(match_bytevector)[i] &=
              result_lo | (static_cast<uint64_t>(result_hi) << 32);
        }
        num_processed = num_rows_to_compare / unroll * unroll;
      }
      for (uint32_t i = num_processed; i < num_rows_to_compare; ++i) {
        uint32_t irow_left = use_selection ? sel_left_maybe_null[i] : i;
        uint32_t irow_right = left_to_right_map[irow_left];
        int64_t bitid = irow_right * null_mask_num_bytes * 8 + id_col;
        match_bytevector[i] &= (BitUtil::GetBit(null_masks, bitid) ? 0 : 0xff);
      }
    } else if (!rows.has_any_nulls(ctx)) {
      // Remove rows from the result for which the column value on left side is null
      const uint8_t* non_nulls = col.data(0);
      ARROW_DCHECK(non_nulls);
      uint32_t num_processed = 0;
      if (ctx->has_avx2()) {
        constexpr uint32_t unroll = 8;
        for (uint32_t i = 0; i < num_rows_to_compare / unroll; ++i) {
          __m256i cmp;
          if (use_selection) {
            __m256i irow_left = _mm256_cvtepu16_epi32(_mm_loadu_si128(
                reinterpret_cast<const __m128i*>(sel_left_maybe_null) + i));
            __m256i left = _mm256_i32gather_epi32((const int*)non_nulls,
                                                  _mm256_srli_epi32(irow_left, 3), 1);
            left = _mm256_and_si256(
                _mm256_set1_epi32(1),
                _mm256_srlv_epi32(left,
                                  _mm256_and_si256(irow_left, _mm256_set1_epi32(7))));
            cmp = _mm256_cmpeq_epi32(left, _mm256_set1_epi32(1));
          } else {
            __m256i left = _mm256_cvtepu8_epi32(_mm_set1_epi8(non_nulls[i]));
            __m256i bits = _mm256_setr_epi32(1, 2, 4, 8, 16, 32, 64, 128);
            cmp = _mm256_cmpeq_epi32(_mm256_and_si256(left, bits), bits);
          }
          uint32_t result_lo =
              _mm256_movemask_epi8(_mm256_cvtepi32_epi64(_mm256_castsi256_si128(cmp)));
          uint32_t result_hi = _mm256_movemask_epi8(
              _mm256_cvtepi32_epi64(_mm256_extracti128_si256(cmp, 1)));
          reinterpret_cast<uint64_t*>(match_bytevector)[i] &=
              result_lo | (static_cast<uint64_t>(result_hi) << 32);
        }
        num_processed = num_rows_to_compare / unroll * unroll;
      }
      for (uint32_t i = num_processed; i < num_rows_to_compare; ++i) {
        uint32_t irow_left = use_selection ? sel_left_maybe_null[i] : i;
        match_bytevector[i] &= BitUtil::GetBit(non_nulls, irow_left) ? 0xff : 0;
      }
    } else {
      const uint8_t* null_masks = rows.null_masks();
      uint32_t null_mask_num_bytes = rows.metadata().null_masks_bytes_per_row;
      const uint8_t* non_nulls = col.data(0);
      ARROW_DCHECK(non_nulls);

      uint32_t num_processed = 0;
      if (ctx->has_avx2()) {
        constexpr uint32_t unroll = 8;
        for (uint32_t i = 0; i < num_rows_to_compare / unroll; ++i) {
          __m256i left_null;
          __m256i irow_right;
          if (use_selection) {
            __m256i irow_left = _mm256_cvtepu16_epi32(_mm_loadu_si128(
                reinterpret_cast<const __m128i*>(sel_left_maybe_null) + i));
            irow_right =
                _mm256_i32gather_epi32((const int*)left_to_right_map, irow_left, 4);

            __m256i left = _mm256_i32gather_epi32((const int*)non_nulls,
                                                  _mm256_srli_epi32(irow_left, 3), 1);
            left = _mm256_and_si256(
                _mm256_set1_epi32(1),
                _mm256_srlv_epi32(left,
                                  _mm256_and_si256(irow_left, _mm256_set1_epi32(7))));
            left_null = _mm256_cmpeq_epi32(left, _mm256_setzero_si256());
          } else {
            irow_right = _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(left_to_right_map) + i);

            __m256i left = _mm256_cvtepu8_epi32(_mm_set1_epi8(non_nulls[i]));
            __m256i bits = _mm256_setr_epi32(1, 2, 4, 8, 16, 32, 64, 128);
            left_null =
                _mm256_cmpeq_epi32(_mm256_and_si256(left, bits), _mm256_setzero_si256());
          }
          __m256i bitid =
              _mm256_mullo_epi32(irow_right, _mm256_set1_epi32(null_mask_num_bytes * 8));
          bitid = _mm256_add_epi32(bitid, _mm256_set1_epi32(id_col));
          __m256i right = _mm256_i32gather_epi32((const int*)null_masks,
                                                 _mm256_srli_epi32(bitid, 3), 1);
          right = _mm256_and_si256(
              _mm256_set1_epi32(1),
              _mm256_srlv_epi32(right, _mm256_and_si256(bitid, _mm256_set1_epi32(7))));
          __m256i right_null = _mm256_cmpeq_epi32(right, _mm256_set1_epi32(1));

          uint64_t left_null_64 =
              static_cast<uint32_t>(_mm256_movemask_epi8(
                  _mm256_cvtepi32_epi64(_mm256_castsi256_si128(left_null)))) |
              (static_cast<uint64_t>(static_cast<uint32_t>(_mm256_movemask_epi8(
                   _mm256_cvtepi32_epi64(_mm256_extracti128_si256(left_null, 1)))))
               << 32);

          uint64_t right_null_64 =
              static_cast<uint32_t>(_mm256_movemask_epi8(
                  _mm256_cvtepi32_epi64(_mm256_castsi256_si128(right_null)))) |
              (static_cast<uint64_t>(static_cast<uint32_t>(_mm256_movemask_epi8(
                   _mm256_cvtepi32_epi64(_mm256_extracti128_si256(right_null, 1)))))
               << 32);

          reinterpret_cast<uint64_t*>(match_bytevector)[i] |=
              left_null_64 & right_null_64;
          reinterpret_cast<uint64_t*>(match_bytevector)[i] &=
              ~(left_null_64 ^ right_null_64);
        }
        num_processed = num_rows_to_compare / unroll * unroll;
      }
      for (uint32_t i = num_processed; i < num_rows_to_compare; ++i) {
        uint32_t irow_left = use_selection ? sel_left_maybe_null[i] : i;
        uint32_t irow_right = left_to_right_map[irow_left];
        int64_t bitid_right = irow_right * null_mask_num_bytes * 8 + id_col;
        int right_null = BitUtil::GetBit(null_masks, bitid_right) ? 0xff : 0;
        int left_null = BitUtil::GetBit(non_nulls, irow_left) ? 0 : 0xff;
        match_bytevector[i] |= left_null & right_null;
        match_bytevector[i] &= ~(left_null ^ right_null);
      }
    }
  }
}

template <bool use_selection, class COMPARE8_FN, class COMPARE_FN>
void KeyCompare::CompareBinaryColumnToRowImp(
    uint32_t offset_within_row, uint32_t num_rows_to_compare,
    const uint16_t* sel_left_maybe_null, const uint32_t* left_to_right_map,
    KeyEncoder::KeyEncoderContext* ctx, const KeyEncoder::KeyColumnArray& col,
    const KeyEncoder::KeyRowArray& rows, uint8_t* match_bytevector,
    COMPARE8_FN compare8_fn, COMPARE_FN compare_fn) {
  bool is_fixed_length = rows.metadata().is_fixed_length;
  if (is_fixed_length) {
    uint32_t fixed_length = rows.metadata().fixed_length;
    const uint8_t* rows_left = col.data(1);
    const uint8_t* rows_right = rows.data(1);
    constexpr uint32_t unroll = 8;
    __m256i irow_left = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    for (uint32_t i = 0; i < num_rows_to_compare / unroll; ++i) {
      if (use_selection) {
        irow_left = _mm256_cvtepu16_epi32(
            _mm_loadu_si128(reinterpret_cast<const __m128i*>(sel_left_maybe_null) + i));
      }
      __m256i irow_right;
      if (use_selection) {
        irow_right = _mm256_i32gather_epi32((const int*)left_to_right_map, irow_left, 4);
      } else {
        irow_right =
            _mm256_loadu_si256(reinterpret_cast<const __m256i*>(left_to_right_map) + i);
      }

      __m256i offset_right =
          _mm256_mullo_epi32(irow_right, _mm256_set1_epi32(fixed_length));
      offset_right = _mm256_add_epi32(offset_right, _mm256_set1_epi32(offset_within_row));

      reinterpret_cast<uint64_t*>(match_bytevector)[i] =
          compare8_fn(rows_left, rows_right, i * unroll, irow_left, offset_right);

      if (!use_selection) {
        irow_left = _mm256_add_epi32(irow_left, _mm256_set1_epi32(8));
      }
    }
    for (uint32_t i = num_rows_to_compare - (num_rows_to_compare % unroll);
         i < num_rows_to_compare; ++i) {
      uint32_t irow_left = use_selection ? sel_left_maybe_null[i] : i;
      uint32_t irow_right = left_to_right_map[irow_left];
      uint32_t offset_right = irow_right * fixed_length + offset_within_row;
      match_bytevector[i] = compare_fn(rows_left, rows_right, irow_left, offset_right);
    }
  } else {
    const uint8_t* rows_left = col.data(1);
    const uint32_t* offsets_right = rows.offsets();
    const uint8_t* rows_right = rows.data(2);
    constexpr uint32_t unroll = 8;
    __m256i irow_left = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    for (uint32_t i = 0; i < num_rows_to_compare / unroll; ++i) {
      if (use_selection) {
        irow_left = _mm256_cvtepu16_epi32(
            _mm_loadu_si128(reinterpret_cast<const __m128i*>(sel_left_maybe_null) + i));
      }
      __m256i irow_right;
      if (use_selection) {
        irow_right = _mm256_i32gather_epi32((const int*)left_to_right_map, irow_left, 4);
      } else {
        irow_right =
            _mm256_loadu_si256(reinterpret_cast<const __m256i*>(left_to_right_map) + i);
      }
      __m256i offset_right =
          _mm256_i32gather_epi32((const int*)offsets_right, irow_right, 4);
      offset_right = _mm256_add_epi32(offset_right, _mm256_set1_epi32(offset_within_row));

      reinterpret_cast<uint64_t*>(match_bytevector)[i] =
          compare8_fn(rows_left, rows_right, i * unroll, irow_left, offset_right);

      if (!use_selection) {
        irow_left = _mm256_add_epi32(irow_left, _mm256_set1_epi32(8));
      }
    }
    for (uint32_t i = num_rows_to_compare - (num_rows_to_compare % unroll);
         i < num_rows_to_compare; ++i) {
      uint32_t irow_left = use_selection ? sel_left_maybe_null[i] : i;
      uint32_t irow_right = left_to_right_map[irow_left];
      uint32_t offset_right = offsets_right[irow_right] + offset_within_row;
      match_bytevector[i] = compare_fn(rows_left, rows_right, irow_left, offset_right);
    }
  }
  uint32_t col_width = col.metadata().fixed_length;
  if (col_width == 0) {
    col_width = 1;
  }
}

template <bool use_selection>
void KeyCompare::CompareBinaryColumnToRow(
    uint32_t offset_within_row, uint32_t num_rows_to_compare,
    const uint16_t* sel_left_maybe_null, const uint32_t* left_to_right_map,
    KeyEncoder::KeyEncoderContext* ctx, const KeyEncoder::KeyColumnArray& col,
    const KeyEncoder::KeyRowArray& rows, uint8_t* match_bytevector) {
  uint32_t col_width = col.metadata().fixed_length;
  if (col_width == 0) {
    CompareBinaryColumnToRowImp<use_selection>(
        offset_within_row, num_rows_to_compare, sel_left_maybe_null, left_to_right_map,
        ctx, col, rows, match_bytevector,
        [](const uint8_t* left_base, const uint8_t* right_base, uint32_t irow_left_base,
           __m256i irow_left, __m256i offset_right) {
          __m256i left;
          if (use_selection) {
            left = _mm256_i32gather_epi32((const int*)left_base,
                                          _mm256_srli_epi32(irow_left, 3), 1);
            left = _mm256_and_si256(
                _mm256_set1_epi32(1),
                _mm256_srlv_epi32(left,
                                  _mm256_and_si256(irow_left, _mm256_set1_epi32(7))));
            left = _mm256_mullo_epi32(left, _mm256_set1_epi32(0xff));
          } else {
            __m256i bits = _mm256_setr_epi32(1, 2, 4, 8, 16, 32, 64, 128);
            left = _mm256_cmpeq_epi32(
                _mm256_and_si256(bits, _mm256_set1_epi8(left_base[irow_left_base / 8])),
                bits);
            left = _mm256_and_si256(left, _mm256_set1_epi32(0xff));
          }
          __m256i right = _mm256_i32gather_epi32((const int*)right_base, offset_right, 1);
          right = _mm256_and_si256(right, _mm256_set1_epi32(0xff));
          __m256i cmp = _mm256_cmpeq_epi32(left, right);
          uint32_t result_lo =
              _mm256_movemask_epi8(_mm256_cvtepi32_epi64(_mm256_castsi256_si128(cmp)));
          uint32_t result_hi = _mm256_movemask_epi8(
              _mm256_cvtepi32_epi64(_mm256_extracti128_si256(cmp, 1)));
          return result_lo | (static_cast<uint64_t>(result_hi) << 32);
        },
        [](const uint8_t* left_base, const uint8_t* right_base, uint32_t irow_left,
           uint32_t offset_right) {
          uint8_t left = BitUtil::GetBit(left_base, irow_left) ? 0xff : 0x00;
          uint8_t right = right_base[offset_right];
          return left == right ? 0xff : 0;
        });
  } else if (col_width == 1) {
    CompareBinaryColumnToRowImp<use_selection>(
        offset_within_row, num_rows_to_compare, sel_left_maybe_null, left_to_right_map,
        ctx, col, rows, match_bytevector,
        [](const uint8_t* left_base, const uint8_t* right_base, uint32_t irow_left_base,
           __m256i irow_left, __m256i offset_right) {
          __m256i left;
          if (use_selection) {
            left = _mm256_i32gather_epi32((const int*)left_base, irow_left, 1);
            left = _mm256_and_si256(left, _mm256_set1_epi32(0xff));
          } else {
            left = _mm256_cvtepu8_epi32(_mm_set1_epi64x(
                reinterpret_cast<const uint64_t*>(left_base)[irow_left_base / 8]));
          }
          __m256i right = _mm256_i32gather_epi32((const int*)right_base, offset_right, 1);
          right = _mm256_and_si256(right, _mm256_set1_epi32(0xff));
          __m256i cmp = _mm256_cmpeq_epi32(left, right);
          uint32_t result_lo =
              _mm256_movemask_epi8(_mm256_cvtepi32_epi64(_mm256_castsi256_si128(cmp)));
          uint32_t result_hi = _mm256_movemask_epi8(
              _mm256_cvtepi32_epi64(_mm256_extracti128_si256(cmp, 1)));
          return result_lo | (static_cast<uint64_t>(result_hi) << 32);
        },
        [](const uint8_t* left_base, const uint8_t* right_base, uint32_t irow_left,
           uint32_t offset_right) {
          uint8_t left = left_base[irow_left];
          uint8_t right = right_base[offset_right];
          return left == right ? 0xff : 0;
        });
  } else if (col_width == 2) {
    CompareBinaryColumnToRowImp<use_selection>(
        offset_within_row, num_rows_to_compare, sel_left_maybe_null, left_to_right_map,
        ctx, col, rows, match_bytevector,
        [](const uint8_t* left_base, const uint8_t* right_base, uint32_t irow_left_base,
           __m256i irow_left, __m256i offset_right) {
          __m256i left;
          if (use_selection) {
            left = _mm256_i32gather_epi32((const int*)left_base, irow_left, 2);
            left = _mm256_and_si256(left, _mm256_set1_epi32(0xffff));
          } else {
            left = _mm256_cvtepu16_epi32(_mm_loadu_si128(
                reinterpret_cast<const __m128i*>(left_base) + irow_left_base / 8));
          }
          __m256i right = _mm256_i32gather_epi32((const int*)right_base, offset_right, 1);
          right = _mm256_and_si256(right, _mm256_set1_epi32(0xffff));
          __m256i cmp = _mm256_cmpeq_epi32(left, right);
          uint32_t result_lo =
              _mm256_movemask_epi8(_mm256_cvtepi32_epi64(_mm256_castsi256_si128(cmp)));
          uint32_t result_hi = _mm256_movemask_epi8(
              _mm256_cvtepi32_epi64(_mm256_extracti128_si256(cmp, 1)));
          return result_lo | (static_cast<uint64_t>(result_hi) << 32);
        },
        [](const uint8_t* left_base, const uint8_t* right_base, uint32_t irow_left,
           uint32_t offset_right) {
          uint16_t left = reinterpret_cast<const uint16_t*>(left_base)[irow_left];
          uint16_t right = *reinterpret_cast<const uint16_t*>(right_base + offset_right);
          return left == right ? 0xff : 0;
        });
  } else if (col_width == 4) {
    CompareBinaryColumnToRowImp<use_selection>(
        offset_within_row, num_rows_to_compare, sel_left_maybe_null, left_to_right_map,
        ctx, col, rows, match_bytevector,
        [](const uint8_t* left_base, const uint8_t* right_base, uint32_t irow_left_base,
           __m256i irow_left, __m256i offset_right) {
          __m256i left;
          if (use_selection) {
            left = _mm256_i32gather_epi32((const int*)left_base, irow_left, 4);
          } else {
            left = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(left_base) +
                                      irow_left_base / 8);
          }
          __m256i right = _mm256_i32gather_epi32((const int*)right_base, offset_right, 1);
          __m256i cmp = _mm256_cmpeq_epi32(left, right);
          uint32_t result_lo =
              _mm256_movemask_epi8(_mm256_cvtepi32_epi64(_mm256_castsi256_si128(cmp)));
          uint32_t result_hi = _mm256_movemask_epi8(
              _mm256_cvtepi32_epi64(_mm256_extracti128_si256(cmp, 1)));
          return result_lo | (static_cast<uint64_t>(result_hi) << 32);
        },
        [](const uint8_t* left_base, const uint8_t* right_base, uint32_t irow_left,
           uint32_t offset_right) {
          uint32_t left = reinterpret_cast<const uint32_t*>(left_base)[irow_left];
          uint32_t right = *reinterpret_cast<const uint32_t*>(right_base + offset_right);
          return left == right ? 0xff : 0;
        });
  } else if (col_width == 8) {
    CompareBinaryColumnToRowImp<use_selection>(
        offset_within_row, num_rows_to_compare, sel_left_maybe_null, left_to_right_map,
        ctx, col, rows, match_bytevector,
        [](const uint8_t* left_base, const uint8_t* right_base, uint32_t irow_left_base,
           __m256i irow_left, __m256i offset_right) {
          __m256i left_lo = _mm256_i32gather_epi64((const long long*)left_base,
                                                   _mm256_castsi256_si128(irow_left), 8);
          __m256i left_hi = _mm256_i32gather_epi64(
              (const long long*)left_base, _mm256_extracti128_si256(irow_left, 1), 8);
          if (use_selection) {
            left_lo = _mm256_i32gather_epi64((const long long*)left_base,
                                             _mm256_castsi256_si128(irow_left), 8);
            left_hi = _mm256_i32gather_epi64((const long long*)left_base,
                                             _mm256_extracti128_si256(irow_left, 1), 8);
          } else {
            left_lo = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(left_base) +
                                         irow_left_base / 4);
            left_hi = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(left_base) +
                                         irow_left_base / 4 + 1);
          }
          __m256i right_lo = _mm256_i32gather_epi64(
              (const long long*)right_base, _mm256_castsi256_si128(offset_right), 1);
          __m256i right_hi = _mm256_i32gather_epi64(
              (const long long*)right_base, _mm256_extracti128_si256(offset_right, 1), 1);
          uint32_t result_lo =
              _mm256_movemask_epi8(_mm256_cmpeq_epi64(left_lo, right_lo));
          uint32_t result_hi =
              _mm256_movemask_epi8(_mm256_cmpeq_epi64(left_hi, right_hi));
          return result_lo | (static_cast<uint64_t>(result_hi) << 32);
        },
        [](const uint8_t* left_base, const uint8_t* right_base, uint32_t irow_left,
           uint32_t offset_right) {
          uint64_t left = reinterpret_cast<const uint64_t*>(left_base)[irow_left];
          uint64_t right = *reinterpret_cast<const uint64_t*>(right_base + offset_right);
          return left == right ? 0xff : 0;
        });
  } else {
    CompareBinaryColumnToRowImp<use_selection>(
        offset_within_row, num_rows_to_compare, sel_left_maybe_null, left_to_right_map,
        ctx, col, rows, match_bytevector,
        [&col](const uint8_t* left_base, const uint8_t* right_base,
               uint32_t irow_left_base, __m256i irow_left, __m256i offset_right) {
          uint32_t irow_left_array[8];
          uint32_t offset_right_array[8];
          if (use_selection) {
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(irow_left_array), irow_left);
          }
          _mm256_storeu_si256(reinterpret_cast<__m256i*>(offset_right_array),
                              offset_right);
          uint32_t length = col.metadata().fixed_length;

          // Non-zero length guarantees no underflow
          int32_t num_loops_less_one = (static_cast<int32_t>(length) + 31) / 32 - 1;

          __m256i tail_mask = _mm256_cmpgt_epi8(
              _mm256_set1_epi8(length - num_loops_less_one * 32),
              _mm256_setr_epi64x(0x0706050403020100ULL, 0x0f0e0d0c0b0a0908ULL,
                                 0x1716151413121110ULL, 0x1f1e1d1c1b1a1918ULL));

          uint64_t result = 0;
          for (uint32_t irow = 0; irow < 8; ++irow) {
            const __m256i* key_left_ptr = reinterpret_cast<const __m256i*>(
                left_base +
                (use_selection ? irow_left_array[irow] : irow_left_base + irow) * length);
            const __m256i* key_right_ptr =
                reinterpret_cast<const __m256i*>(right_base + offset_right_array[irow]);
            __m256i result_or = _mm256_setzero_si256();
            int32_t i;
            // length cannot be zero
            for (i = 0; i < num_loops_less_one; ++i) {
              __m256i key_left = _mm256_loadu_si256(key_left_ptr + i);
              __m256i key_right = _mm256_loadu_si256(key_right_ptr + i);
              result_or =
                  _mm256_or_si256(result_or, _mm256_xor_si256(key_left, key_right));
            }
            __m256i key_left = _mm256_loadu_si256(key_left_ptr + i);
            __m256i key_right = _mm256_loadu_si256(key_right_ptr + i);
            result_or = _mm256_or_si256(
                result_or,
                _mm256_and_si256(tail_mask, _mm256_xor_si256(key_left, key_right)));
            uint64_t result_single = _mm256_testz_si256(result_or, result_or) * 0xff;
            result |= result_single << (8 * irow);
          }
          return result;
        },
        [&col](const uint8_t* left_base, const uint8_t* right_base, uint32_t irow_left,
               uint32_t offset_right) {
          uint32_t length = col.metadata().fixed_length;

          // Non-zero length guarantees no underflow
          int32_t num_loops_less_one = (static_cast<int32_t>(length) + 31) / 32 - 1;

          __m256i tail_mask = _mm256_cmpgt_epi8(
              _mm256_set1_epi8(length - num_loops_less_one * 32),
              _mm256_setr_epi64x(0x0706050403020100ULL, 0x0f0e0d0c0b0a0908ULL,
                                 0x1716151413121110ULL, 0x1f1e1d1c1b1a1918ULL));

          const __m256i* key_left_ptr =
              reinterpret_cast<const __m256i*>(left_base + irow_left * length);
          const __m256i* key_right_ptr =
              reinterpret_cast<const __m256i*>(right_base + offset_right);
          __m256i result_or = _mm256_setzero_si256();
          int32_t i;
          // length cannot be zero
          for (i = 0; i < num_loops_less_one; ++i) {
            __m256i key_left = _mm256_loadu_si256(key_left_ptr + i);
            __m256i key_right = _mm256_loadu_si256(key_right_ptr + i);
            result_or = _mm256_or_si256(result_or, _mm256_xor_si256(key_left, key_right));
          }
          __m256i key_left = _mm256_loadu_si256(key_left_ptr + i);
          __m256i key_right = _mm256_loadu_si256(key_right_ptr + i);
          result_or = _mm256_or_si256(
              result_or,
              _mm256_and_si256(tail_mask, _mm256_xor_si256(key_left, key_right)));
          return _mm256_testz_si256(result_or, result_or) * 0xff;
        });
  }
}

// Overwrites the match_bytevector instead of updating it
template <bool use_selection, bool is_first_varbinary_col>
void KeyCompare::CompareVarBinaryColumnToRow(
    uint32_t id_varbinary_col, uint32_t num_rows_to_compare,
    const uint16_t* sel_left_maybe_null, const uint32_t* left_to_right_map,
    KeyEncoder::KeyEncoderContext* ctx, const KeyEncoder::KeyColumnArray& col,
    const KeyEncoder::KeyRowArray& rows, uint8_t* match_bytevector) {
  const uint32_t* offsets_left = col.offsets();
  const uint32_t* offsets_right = rows.offsets();
  const uint8_t* rows_left = col.data(2);
  const uint8_t* rows_right = rows.data(2);
  for (uint32_t i = 0; i < num_rows_to_compare; ++i) {
    uint32_t irow_left = use_selection ? sel_left_maybe_null[i] : i;
    uint32_t irow_right = left_to_right_map[irow_left];
    uint32_t begin_left = offsets_left[irow_left];
    uint32_t length_left = offsets_left[irow_left + 1] - begin_left;
    uint32_t begin_right = offsets_right[irow_right];
    uint32_t length_right;
    uint32_t offset_within_row;
    if (!is_first_varbinary_col) {
      rows.metadata().nth_varbinary_offset_and_length(
          rows_right + begin_right, id_varbinary_col, &offset_within_row, &length_right);
    } else {
      rows.metadata().first_varbinary_offset_and_length(
          rows_right + begin_right, &offset_within_row, &length_right);
    }
    begin_right += offset_within_row;
    uint32_t length = std::min(length_left, length_right);
    const __m256i* key_left_ptr =
        reinterpret_cast<const __m256i*>(rows_left + begin_left);
    const __m256i* key_right_ptr =
        reinterpret_cast<const __m256i*>(rows_right + begin_right);
    __m256i result_or = _mm256_setzero_si256();
    int32_t j;
    // length can be zero
    for (j = 0; j < (static_cast<int32_t>(length) + 31) / 32 - 1; ++j) {
      __m256i key_left = _mm256_loadu_si256(key_left_ptr + j);
      __m256i key_right = _mm256_loadu_si256(key_right_ptr + j);
      result_or = _mm256_or_si256(result_or, _mm256_xor_si256(key_left, key_right));
    }

    __m256i tail_mask = _mm256_cmpgt_epi8(
        _mm256_set1_epi8(length - j * 32),
        _mm256_setr_epi64x(0x0706050403020100ULL, 0x0f0e0d0c0b0a0908ULL,
                           0x1716151413121110ULL, 0x1f1e1d1c1b1a1918ULL));

    __m256i key_left = _mm256_loadu_si256(key_left_ptr + j);
    __m256i key_right = _mm256_loadu_si256(key_right_ptr + j);
    result_or = _mm256_or_si256(
        result_or, _mm256_and_si256(tail_mask, _mm256_xor_si256(key_left, key_right)));
    int result = _mm256_testz_si256(result_or, result_or) * 0xff;
    result *= (length_left == length_right ? 1 : 0);
    match_bytevector[i] = result;
  }
}

void KeyCompare::AndByteVectors(uint32_t num_elements, uint8_t* bytevector_A,
                                const uint8_t* bytevector_B) {
  constexpr int unroll = 32;
  for (uint32_t i = 0; i < num_elements / unroll; ++i) {
    __m256i result = _mm256_and_si256(
        _mm256_loadu_si256(reinterpret_cast<const __m256i*>(bytevector_A) + i),
        _mm256_loadu_si256(reinterpret_cast<const __m256i*>(bytevector_B) + i));
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(bytevector_A) + i, result);
  }
  for (uint32_t i = (num_elements - (num_elements % unroll)) / 8;
       i < (num_elements + 7) / 8; ++i) {
    uint64_t* a = reinterpret_cast<uint64_t*>(bytevector_A);
    const uint64_t* b = reinterpret_cast<const uint64_t*>(bytevector_B);
    a[i] &= b[i];
  }
}

void KeyCompare::CompareColumnsToRows(uint32_t num_rows_to_compare,
                                      const uint16_t* sel_left_maybe_null,
                                      const uint32_t* left_to_right_map,
                                      KeyEncoder::KeyEncoderContext* ctx,
                                      uint32_t* out_num_rows,
                                      uint16_t* out_sel_left_maybe_same,
                                      const std::vector<KeyEncoder::KeyColumnArray>& cols,
                                      const KeyEncoder::KeyRowArray& rows) {
  if (num_rows_to_compare == 0) {
    *out_num_rows = 0;
    return;
  }

  // Allocate temporary byte and bit vectors
  auto bytevector_A_holder =
      util::TempVectorHolder<uint8_t>(ctx->stack, num_rows_to_compare);
  auto bytevector_B_holder =
      util::TempVectorHolder<uint8_t>(ctx->stack, num_rows_to_compare);
  auto bitvector_holder =
      util::TempVectorHolder<uint8_t>(ctx->stack, num_rows_to_compare);

  uint8_t* match_bytevector_A = bytevector_A_holder.mutable_data();
  uint8_t* match_bytevector_B = bytevector_B_holder.mutable_data();
  uint8_t* match_bitvector = bitvector_holder.mutable_data();

  bool is_first_column = true;
  for (size_t icol = 0; icol < cols.size(); ++icol) {
    const KeyEncoder::KeyColumnArray& col = cols[icol];
    uint32_t offset_within_row = rows.metadata().encoded_field_offset(icol);
    if (col.metadata().is_fixed_length) {
      if (sel_left_maybe_null) {
        CompareBinaryColumnToRow<true>(
            offset_within_row, num_rows_to_compare, sel_left_maybe_null,
            left_to_right_map, ctx, col, rows,
            is_first_column ? match_bytevector_A : match_bytevector_B);
        NullUpdateColumnToRow<true>(
            static_cast<uint32_t>(icol), num_rows_to_compare, sel_left_maybe_null,
            left_to_right_map, ctx, col, rows,
            is_first_column ? match_bytevector_A : match_bytevector_B);
      } else {
        // Version without using selection vector
        CompareBinaryColumnToRow<false>(
            offset_within_row, num_rows_to_compare, sel_left_maybe_null,
            left_to_right_map, ctx, col, rows,
            is_first_column ? match_bytevector_A : match_bytevector_B);
        NullUpdateColumnToRow<false>(
            static_cast<uint32_t>(icol), num_rows_to_compare, sel_left_maybe_null,
            left_to_right_map, ctx, col, rows,
            is_first_column ? match_bytevector_A : match_bytevector_B);
      }
      if (!is_first_column) {
        AndByteVectors(num_rows_to_compare, match_bytevector_A, match_bytevector_B);
      }
      is_first_column = false;
    }
  }

  uint32_t ivarbinary = 0;
  for (size_t icol = 0; icol < cols.size(); ++icol) {
    const KeyEncoder::KeyColumnArray& col = cols[icol];
    if (!col.metadata().is_fixed_length) {
      // Process varbinary and nulls
      if (sel_left_maybe_null) {
        if (ivarbinary == 0) {
          CompareVarBinaryColumnToRow<true, true>(
              ivarbinary, num_rows_to_compare, sel_left_maybe_null, left_to_right_map,
              ctx, col, rows, is_first_column ? match_bytevector_A : match_bytevector_B);
        } else {
          CompareVarBinaryColumnToRow<true, false>(ivarbinary, num_rows_to_compare,
                                                   sel_left_maybe_null, left_to_right_map,
                                                   ctx, col, rows, match_bytevector_B);
        }
        NullUpdateColumnToRow<true>(
            static_cast<uint32_t>(icol), num_rows_to_compare, sel_left_maybe_null,
            left_to_right_map, ctx, col, rows,
            is_first_column ? match_bytevector_A : match_bytevector_B);
      } else {
        if (ivarbinary == 0) {
          CompareVarBinaryColumnToRow<false, true>(
              ivarbinary, num_rows_to_compare, sel_left_maybe_null, left_to_right_map,
              ctx, col, rows, is_first_column ? match_bytevector_A : match_bytevector_B);
        } else {
          CompareVarBinaryColumnToRow<false, false>(
              ivarbinary, num_rows_to_compare, sel_left_maybe_null, left_to_right_map,
              ctx, col, rows, match_bytevector_B);
        }
        NullUpdateColumnToRow<false>(
            static_cast<uint32_t>(icol), num_rows_to_compare, sel_left_maybe_null,
            left_to_right_map, ctx, col, rows,
            is_first_column ? match_bytevector_A : match_bytevector_B);
      }
      if (!is_first_column) {
        AndByteVectors(num_rows_to_compare, match_bytevector_A, match_bytevector_B);
      }
      is_first_column = false;
      ++ivarbinary;
    }
  }

  util::BitUtil::bytes_to_bits(ctx->hardware_flags, num_rows_to_compare, match_bytevector_A,
                               match_bitvector);
  if (sel_left_maybe_null) {
    int out_num_rows_int;
    util::BitUtil::bits_filter_indexes(0, ctx->hardware_flags, num_rows_to_compare,
                                       match_bitvector, sel_left_maybe_null,
                                       &out_num_rows_int, out_sel_left_maybe_same);
    *out_num_rows = out_num_rows_int;
  } else {
    int out_num_rows_int;
    util::BitUtil::bits_to_indexes(0, ctx->hardware_flags, num_rows_to_compare, match_bitvector,
                                   &out_num_rows_int, out_sel_left_maybe_same);
    *out_num_rows = out_num_rows_int;
  }
}

}  // namespace compute
}  // namespace arrow
