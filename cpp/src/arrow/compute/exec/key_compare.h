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

#pragma once

#include <cstdint>

#include "arrow/compute/exec/key_encode.h"
#include "arrow/compute/exec/util.h"
#include "arrow/memory_pool.h"
#include "arrow/result.h"
#include "arrow/status.h"

namespace arrow {
namespace compute {

class KeyCompare {
 public:
  // Returns a single 16-bit selection vector of rows that failed comparison.
  // If there is input selection on the left, the resulting selection is a filtered image
  // of input selection.
  static void CompareRows(uint32_t num_rows_to_compare,
                          const uint16_t* sel_left_maybe_null,
                          const uint32_t* left_to_right_map,
                          KeyEncoder::KeyEncoderContext* ctx, uint32_t* out_num_rows,
                          uint16_t* out_sel_left_maybe_same,
                          const KeyEncoder::KeyRowArray& rows_left,
                          const KeyEncoder::KeyRowArray& rows_right);

  static void CompareColumnsToRows(uint32_t num_rows_to_compare,
                                   const uint16_t* sel_left_maybe_null,
                                   const uint32_t* left_to_right_map,
                                   KeyEncoder::KeyEncoderContext* ctx,
                                   uint32_t* out_num_rows,
                                   uint16_t* out_sel_left_maybe_same,
                                   const std::vector<KeyEncoder::KeyColumnArray>& cols,
                                   const KeyEncoder::KeyRowArray& rows);

 private:

  static void AndByteVectors(uint32_t num_elements, uint8_t* bytevector_A,
                             const uint8_t* bytevector_B);

  template <bool use_selection>
  static void NullUpdateColumnToRow(uint32_t id_col, uint32_t num_rows_to_compare,
                                    const uint16_t* sel_left_maybe_null,
                                    const uint32_t* left_to_right_map,
                                    KeyEncoder::KeyEncoderContext* ctx,
                                    const KeyEncoder::KeyColumnArray& col,
                                    const KeyEncoder::KeyRowArray& rows,
                                    uint8_t* match_bytevector);
  template <bool use_selection, class COMPARE8_FN, class COMPARE_FN>
  static void CompareBinaryColumnToRowImp(
      uint32_t offset_within_row, uint32_t num_rows_to_compare,
      const uint16_t* sel_left_maybe_null, const uint32_t* left_to_right_map,
      KeyEncoder::KeyEncoderContext* ctx, const KeyEncoder::KeyColumnArray& col,
      const KeyEncoder::KeyRowArray& rows, uint8_t* match_bytevector,
      COMPARE8_FN compare8_fn, COMPARE_FN compare_fn);
  template <bool use_selection>
  static void CompareBinaryColumnToRow(
      uint32_t offset_within_row, uint32_t num_rows_to_compare,
      const uint16_t* sel_left_maybe_null, const uint32_t* left_to_right_map,
      KeyEncoder::KeyEncoderContext* ctx, const KeyEncoder::KeyColumnArray& col,
      const KeyEncoder::KeyRowArray& rows, uint8_t* match_bytevector);

  template <bool use_selection, bool is_first_varbinary_col>
  static void CompareVarBinaryColumnToRow(
      uint32_t id_varlen_col, uint32_t num_rows_to_compare,
      const uint16_t* sel_left_maybe_null, const uint32_t* left_to_right_map,
      KeyEncoder::KeyEncoderContext* ctx, const KeyEncoder::KeyColumnArray& col,
      const KeyEncoder::KeyRowArray& rows, uint8_t* match_bytevector);

  static void CompareFixedLength(uint32_t num_rows_to_compare,
                                 const uint16_t* sel_left_maybe_null,
                                 const uint32_t* left_to_right_map,
                                 uint8_t* match_bytevector,
                                 KeyEncoder::KeyEncoderContext* ctx,
                                 uint32_t fixed_length, const uint8_t* rows_left,
                                 const uint8_t* rows_right);
  static void CompareVaryingLength(uint32_t num_rows_to_compare,
                                   const uint16_t* sel_left_maybe_null,
                                   const uint32_t* left_to_right_map,
                                   uint8_t* match_bytevector,
                                   KeyEncoder::KeyEncoderContext* ctx,
                                   const uint8_t* rows_left, const uint8_t* rows_right,
                                   const uint32_t* offsets_left,
                                   const uint32_t* offsets_right);

  // Second template argument is 0, 1 or 2.
  // 0 means arbitrarily many 64-bit words, 1 means up to 1 and 2 means up to 2.
  template <bool use_selection, int num_64bit_words>
  static void CompareFixedLengthImp(uint32_t num_rows_already_processed,
                                    uint32_t num_rows,
                                    const uint16_t* sel_left_maybe_null,
                                    const uint32_t* left_to_right_map,
                                    uint8_t* match_bytevector, uint32_t length,
                                    const uint8_t* rows_left, const uint8_t* rows_right);
  template <bool use_selection>
  static void CompareVaryingLengthImp(uint32_t num_rows,
                                      const uint16_t* sel_left_maybe_null,
                                      const uint32_t* left_to_right_map,
                                      uint8_t* match_bytevector, const uint8_t* rows_left,
                                      const uint8_t* rows_right,
                                      const uint32_t* offsets_left,
                                      const uint32_t* offsets_right);

#if defined(ARROW_HAVE_AVX2)

  static uint32_t CompareFixedLength_UpTo8B_avx2(
      uint32_t num_rows, const uint32_t* left_to_right_map, uint8_t* match_bytevector,
      uint32_t length, const uint8_t* rows_left, const uint8_t* rows_right);
  static uint32_t CompareFixedLength_UpTo16B_avx2(
      uint32_t num_rows, const uint32_t* left_to_right_map, uint8_t* match_bytevector,
      uint32_t length, const uint8_t* rows_left, const uint8_t* rows_right);
  static uint32_t CompareFixedLength_avx2(uint32_t num_rows,
                                          const uint32_t* left_to_right_map,
                                          uint8_t* match_bytevector, uint32_t length,
                                          const uint8_t* rows_left,
                                          const uint8_t* rows_right);
  static void CompareVaryingLength_avx2(
      uint32_t num_rows, const uint32_t* left_to_right_map, uint8_t* match_bytevector,
      const uint8_t* rows_left, const uint8_t* rows_right, const uint32_t* offsets_left,
      const uint32_t* offsets_right);

#endif
};

}  // namespace compute
}  // namespace arrow
