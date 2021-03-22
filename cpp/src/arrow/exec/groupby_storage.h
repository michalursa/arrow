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

#include "arrow/memory_pool.h"
#include "arrow/result.h"
#include "arrow/status.h"
#include "arrow/exec/common.h"
#include "arrow/exec/util.h"

#define AGGREGATES_STORED_SEPARATELY

namespace arrow {
namespace exec {

class KeyStore {
 public:
  KeyStore()
      : num_cols_(0),
        is_row_fixedlen_(true),
        row_length_(0),
        instruction_set_(util::CPUInstructionSet::scalar),
        pool_(),
        num_rows_(0),
        row_offsets_size_(0),
        row_nulls_size_(0),
        row_any_nulls_size_(0),
        num_rows_allocated_(0),
        num_value_bytes_allocated_(0),
        row_offsets_(),
        row_vals_(),
        row_nulls_(),
        row_any_nulls_() {}
  Status init(util::CPUInstructionSet instruction_set, MemoryPool* pool,
              uint32_t num_cols, bool is_row_fixedlen, uint32_t row_length);
  void cleanup();
  uint64_t get_num_keys() const { return num_rows_; }
  uint32_t get_num_cols() const { return num_cols_; }
  bool is_row_fixedlen() const { return is_row_fixedlen_; }
  uint32_t get_row_length() const { return row_length_; }
  uint64_t get_size_keys() const;
  void get_rows(uint64_t& num_rows, const uint32_t*& row_offsets,
                const uint8_t*& row_vals, const uint8_t*& row_nulls,
                const uint8_t*& row_any_nulls);
  Status append(uint32_t num_rows, const uint16_t* selection, const uint32_t* row_offsets,
                const uint8_t* row_vals, const uint8_t* row_nulls);
  static uint32_t get_log_row_null_bits(uint32_t num_cols) {
    uint32_t log_row_null_bits = 0;
    while (num_cols > (1UL << log_row_null_bits)) {
      ++log_row_null_bits;
    }
    return log_row_null_bits;
  }
  static constexpr uint64_t padding_for_SIMD = 32;

 private:
  Status resize_buffers_if_needed(uint64_t num_extra_rows_, uint64_t num_extra_bytes_);

  int num_cols_;
  bool is_row_fixedlen_;
  uint32_t row_length_;
  util::CPUInstructionSet instruction_set_;
  MemoryPool* pool_;

  uint64_t num_rows_;

  // Size in bytes of allocated buffers.
  // This information will be passed to free,
  // so that after unsuccessful allocation cleanup can still be called.
  uint64_t row_offsets_size_;
  uint64_t row_nulls_size_;
  uint64_t row_any_nulls_size_;

  uint64_t num_rows_allocated_;
  uint64_t num_value_bytes_allocated_;

  uint32_t* row_offsets_;
  uint8_t* row_vals_;
  uint8_t* row_nulls_;
  uint8_t* row_any_nulls_;
};

class KeyCompare {
 public:
  static void compare_fixedlen(
      util::CPUInstructionSet instruction_set, uint32_t num_rows,
      // Selection should be set to nullptr if no selection is used
      const uint16_t* selection_maybe_null, uint32_t length,
      const uint8_t* concatenated_keys_left, const uint32_t* ids_right,
      const uint8_t* concatenated_keys_right, uint8_t* match_bitvector);

  static void compare_varlen(util::CPUInstructionSet instruction_set, uint32_t num_rows,
                             // Selection should be set to nullptr if no selection is used
                             const uint16_t* selection_maybe_null,
                             const uint32_t* offsets_left,
                             const uint8_t* concatenated_keys_left,
                             const uint32_t* ids_right, const uint32_t* offsets_right,
                             const uint8_t* concatenated_keys_right,
                             uint8_t* match_bitvector);

  // Update match bit vector eliminating rows that have mismatched null values
  //
  static void update_comparison_for_nulls(
      util::CPUInstructionSet instruction_set, uint32_t num_rows,
      const uint16_t* selection_maybe_null, uint32_t num_cols,
      const uint8_t* any_nulls_left, const uint8_t* nulls_left, const uint32_t* ids_right,
      const uint8_t* any_nulls_right, const uint8_t* nulls_right,
      // Input match bit vector should already be initialized, possibly as a result of key
      // value comparison. Appropriate bits will be cleared if null information does not
      // match for a pair of input rows.
      uint8_t* match_bitvector, uint8_t* temp_vector_8bit, uint16_t* temp_vector_16bit);

 private:
  template <bool use_selection>
  static void compare_varlen_imp(uint32_t num_rows, const uint16_t* selection,
                                 const uint32_t* offsets_left,
                                 const uint8_t* concatenated_keys_left,
                                 const uint32_t* ids_right, const uint32_t* offsets_right,
                                 const uint8_t* concatenated_keys_right,
                                 uint8_t* match_bitvector);

  template <bool use_selection, int num_64bit_words>
  static void compare_fixedlen_imp(uint32_t num_rows, const uint16_t* selection,
                                   uint32_t length, const uint8_t* concatenated_keys_left,
                                   const uint32_t* ids_right,
                                   const uint8_t* concatenated_keys_right,
                                   uint8_t* match_bitvector);

#if defined(ARROW_HAVE_AVX2)
  static void compare_varlen_avx2(uint32_t num_rows, const uint32_t* offsets_left,
                                  const uint8_t* concatenated_keys_left,
                                  const uint32_t* ids_right,
                                  const uint32_t* offsets_right,
                                  const uint8_t* concatenated_keys_right,
                                  uint8_t* match_bitvector);

  static void compare_fixedlen_avx2(uint32_t num_rows, uint32_t length,
                                    const uint8_t* concatenated_keys_left,
                                    const uint32_t* ids_right,
                                    const uint8_t* concatenated_keys_right,
                                    uint8_t* match_bitvector);

#endif
};

class KeyLength {
 public:
  static void compute_offsets(util::CPUInstructionSet instruction_set, int num_columns,
                              int num_rows, uint32_t row_fixed_len,
                              const uint32_t** offsets, const uint8_t** non_nulls,
                              const uint8_t* any_nulls_bitvector, uint32_t* row_offsets);

 private:
#if defined(ARROW_HAVE_AVX2)
  static void compute_offsets_avx2(int num_columns, int num_rows, uint32_t row_fixed_len,
                                   const uint32_t** offsets, const uint8_t** non_nulls,
                                   const uint8_t* any_nulls_bitvector,
                                   uint32_t* row_offsets);
#endif
};

class KeyTranspose {
 public:
  static void col2row(
      util::CPUInstructionSet instruction_set, uint32_t num_cols, uint32_t num_rows,
      // Column widths for all columns.
      // Undefined for varying length column (identified by offset vector not being null).
      // Column width of 0 means 1 bit per row.
      // Column widths are not restricted to powers of 2.
      const uint32_t* col_widths,
      // Bit vectors marking non-null values in columns cannot be null.
      // A constant bit vector with all bits set to 1 needs to be provided, if the
      // input column does not specify one.
      const uint8_t** col_non_nulls, const uint32_t** col_offsets_or_null,
      const uint8_t** col_vals,
      // In case of varying length rows - byte offsets for output rows with extra element
      // indicating total size of rows. In case of fixed length rows - there is only one
      // element that indicates row size in bytes. If any of the input columns has non
      // null offsets array pointer the row is varying length and otherwise it is fixed
      // length.
      const uint32_t* row_offsets_or_length, uint8_t* row_vals,
      // Bit vector marking null values in any of the columns with 1.
      // Bits are organized in row oriented way (all null bits for the first row are
      // followed by all null bits for the second). It uses ceil(log2(num_cols)) bits per
      // row.
      uint8_t* row_nulls,
      // Temporary vectors with at least num_rows 32-bit elements with additional reserved
      // at least 32 bytes at the end, allocated and provided by the caller.
      uint32_t* temp_vector_32_A, uint32_t* temp_vector_32_B);

  static void row2col(
      util::CPUInstructionSet instruction_set, bool only_varlen_buffers,
      uint32_t num_cols, uint64_t num_rows,
      // Column widths for all columns.
      // Undefined for varying length column (identified by offset vector not being null).
      // Column width of 0 means 1 bit per row.
      // Column widths are not restricted to powers of 2.
      const uint32_t* col_widths,
      // Output bit vectors marking non-null values in columns cannot be null.
      uint8_t** col_non_nulls, uint32_t** col_offsets_maybe_missing, uint8_t** col_vals,
      // In case of varying length rows - byte offsets for input rows with extra element
      // indicating total size of rows. In case of fixed length rows - there is only one
      // element that indicates row size in bytes. If any of the columns has non null
      // offsets array pointer the row is varying length and otherwise it is fixed length.
      const uint32_t* row_offsets_or_length, const uint8_t* row_vals,
      const uint8_t* row_nulls,
      // Temporary vectors with at least num_rows 32-bit elements with additional reserved
      // at least 32 bytes at the end, allocated and provided by the caller.
      uint32_t temp_vector_length, uint32_t* temp_vector_32_A,
      uint32_t* temp_vector_32_B);

 private:
  static void col2row_short(uint32_t num_rows, const uint32_t col_length,
                            const uint8_t* col_vals, bool is_row_fixedlen,
                            const uint32_t* row_offsets_or_length_ptr, uint8_t* row_vals);
  static void row2col_short(uint32_t num_rows, const uint32_t col_length,
                            uint8_t* col_vals, bool is_row_fixedlen,
                            const uint32_t* row_offsets, const uint8_t* row_vals);
  template <bool is_row_fixedlen, int col_len_A, int col_len_B>
  static void col2row_shortpair_impl(uint32_t num_rows, const uint8_t* col_vals_A,
                                     const uint8_t* col_vals_B,
                                     const uint32_t* row_offsets, uint8_t* row_vals);
  template <bool is_row_fixedlen, int col_len_A, int col_len_B>
  static void row2col_shortpair_impl(uint32_t num_rows, uint8_t* col_vals_A,
                                     uint8_t* col_vals_B, const uint32_t* row_offsets,
                                     const uint8_t* row_vals);
  static void col2row_shortpair(util::CPUInstructionSet instruction_set,
                                bool is_row_fixedlen, int col_len_A, int col_len_B,
                                uint32_t num_rows, const uint8_t* col_vals_A,
                                const uint8_t* col_vals_B, const uint32_t* row_offsets,
                                uint8_t* row_vals);
  static void row2col_shortpair(util::CPUInstructionSet instruction_set,
                                bool is_row_fixedlen, int col_len_A, int col_len_B,
                                uint32_t num_rows, uint8_t* col_vals_A,
                                uint8_t* col_vals_B, const uint32_t* row_offsets,
                                const uint8_t* row_vals);
  template <bool is_col_fixedlen, bool is_row_fixedlen, bool update_row_offsets>
  static void col2row_long(uint32_t num_rows, const uint8_t* col_non_nulls,
                           const uint32_t* col_offsets, const uint8_t* col_vals,
                           const uint32_t* row_offsets, uint8_t* row_vals,
                           uint32_t* row_offsets_updated);
  template <bool is_col_fixedlen, bool is_row_fixedlen, bool update_row_offsets>
  static void row2col_long(uint32_t num_rows, const uint32_t* col_offsets,
                           uint8_t* col_vals, const uint32_t* row_offsets,
                           const uint8_t* row_vals, uint32_t* row_offsets_updated);
  static void offsets_to_lengths(util::CPUInstructionSet instruction_set,
                                 uint32_t num_rows, const uint32_t* offsets,
                                 uint32_t* lengths);
  static void lengths_to_offsets(uint32_t num_rows, uint32_t first_offset,
                                 const uint32_t* lengths, uint32_t* offsets);
  template <bool is_row_fixedlen>
  static void memset_selection_of_values(int col_len, uint32_t num_rows,
                                         const uint16_t* row_ids,
                                         const uint32_t* row_offsets, uint8_t* rows,
                                         uint8_t byte_value);
  static void col2row_null_bitvector(int col_id, int num_cols, uint32_t col_num_bits_set,
                                     const uint16_t* col_bits_set_ids,
                                     uint8_t* row_oriented_bits);
  static void row2col_null_bitvector(util::CPUInstructionSet instruction_set,
                                     uint64_t num_rows, int num_cols,
                                     uint8_t** col_non_null_bitvectors,
                                     const uint8_t* row_oriented_bits,
                                     int temp_vector_length, uint16_t* temp_vector_16);
#if defined(ARROW_HAVE_AVX2)
  template <bool is_row_fixedlen>
  static void col2row_shortpair_avx2(uint32_t num_rows, const uint32_t col_length,
                                     const uint8_t* col_vals_A, const uint8_t* col_vals_B,
                                     const uint32_t* row_offsets, uint8_t* row_vals);
  template <bool is_row_fixedlen>
  static void row2col_shortpair_avx2(uint32_t num_rows, const uint32_t col_length,
                                     uint8_t* col_vals_A, uint8_t* col_vals_B,
                                     const uint32_t* row_offsets,
                                     const uint8_t* row_vals);
  template <bool is_col_fixedlen, bool is_row_fixedlen, bool update_row_offsets>
  static void col2row_long_avx2(uint32_t num_rows, const uint8_t* col_non_nulls,
                                const uint32_t* col_offsets, const uint8_t* col_vals,
                                const uint32_t* row_offsets, uint8_t* row_vals,
                                uint32_t* row_offsets_updated);
  template <bool is_col_fixedlen, bool is_row_fixedlen, bool update_row_offsets>
  static void row2col_long_avx2(uint32_t num_rows, const uint32_t* col_offsets,
                                uint8_t* col_vals, const uint32_t* row_offsets,
                                const uint8_t* row_vals, uint32_t* row_offsets_updated);
  static void offsets_to_lengths_avx2(uint32_t num_rows, const uint32_t* offsets,
                                      uint32_t* lengths);
#endif
};

}  // namespace exec
}  // namespace arrow
