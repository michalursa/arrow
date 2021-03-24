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

#include <memory.h>

#include <cstdint>

#include "arrow/exec/common.h"

#ifdef _MSC_VER
#include <intrin.h>
#else
#include <x86intrin.h>
#endif
#include <immintrin.h>

#include <algorithm>

#include "arrow/exec/groupby.h"
#include "arrow/exec/groupby_map.h"
#include "arrow/exec/groupby_storage.h"

namespace arrow {
namespace exec {

GroupMap::~GroupMap() {
  group_id_map.cleanup();
  key_store.cleanup();
}

void GroupMap::equal_callback(int num_keys_to_compare,
                              const uint16_t* selection_may_be_null,
                              const uint32_t* group_ids, uint8_t* match_bitvector) {
  uint64_t right_num_rows;
  const uint32_t* right_offsets;
  const uint8_t* right_vals;
  const uint8_t* right_nulls;
  const uint8_t* right_any_nulls;
  key_store.get_rows(right_num_rows, right_offsets, right_vals, right_nulls,
                     right_any_nulls);

  if (key_store.is_row_fixedlen()) {
    KeyCompare::compare_fixedlen(instruction_set, num_keys_to_compare,
                                 selection_may_be_null, key_store.get_row_length(),
                                 minibatch_rows.data(), group_ids, right_vals,
                                 match_bitvector);
  } else {
    KeyCompare::compare_varlen(instruction_set, num_keys_to_compare,
                               selection_may_be_null, minibatch_offsets.data(),
                               minibatch_rows.data(), group_ids, right_offsets,
                               right_vals, match_bitvector);
  }

  KeyCompare::update_comparison_for_nulls(
      instruction_set, num_keys_to_compare, selection_may_be_null,
      key_store.get_num_cols(), minibatch_any_nulls, minibatch_nulls.data(), group_ids,
      right_any_nulls, right_nulls, match_bitvector,
      reinterpret_cast<uint8_t*>(minibatch_temp.data()),
      reinterpret_cast<uint16_t*>(reinterpret_cast<uint8_t*>(minibatch_temp.data()) +
                                  minibatch_size_max + KeyStore::padding_for_SIMD));
}

Status GroupMap::append_callback(int num_keys, const uint16_t* selection) {
  uint32_t row_length = key_store.get_row_length();
  return key_store.append(
      num_keys, selection,
      key_store.is_row_fixedlen() ? &row_length : minibatch_offsets.data(),
      minibatch_rows.data(), minibatch_nulls.data());
}

Status GroupMap::init(util::CPUInstructionSet instruction_set_in, MemoryPool* pool,
                      uint32_t num_columns, const std::vector<bool>& is_fixed_len_in,
                      const uint32_t* col_widths_in) {
  memory_pool = pool;
  minibatch_size = minibatch_size_min;
  RETURN_NOT_OK(temp_buffers.init(memory_pool, minibatch_size_max));
  instruction_set = instruction_set_in;

  bool is_row_fixed_len = true;
  uint32_t fixed_len_size = 0;
  is_col_fixed_len.resize(num_columns);
  col_widths.resize(num_columns);
  for (uint32_t icol = 0; icol < num_columns; ++icol) {
    col_widths[icol] = col_widths_in[icol];
    fixed_len_size += col_widths[icol] == 0 ? 1 : col_widths[icol];
  }
  for (uint32_t icol = 0; icol < num_columns; ++icol) {
    is_col_fixed_len[icol] = is_fixed_len_in[icol];
    if (!is_fixed_len_in[icol]) {
      is_row_fixed_len = false;
    }
  }

  RETURN_NOT_OK(key_store.init(instruction_set, memory_pool, num_columns,
                               is_row_fixed_len, fixed_len_size));
  auto equal_func = [this](int num_keys_to_compare,
                           const uint16_t* selection_may_be_null /* may be null */,
                           const uint32_t* group_ids, uint8_t* match_bitvector) {
    this->equal_callback(num_keys_to_compare, selection_may_be_null, group_ids,
                         match_bitvector);
  };
  auto append_func = [this](int num_keys, const uint16_t* selection) {
    return this->append_callback(num_keys, selection);
  };

  RETURN_NOT_OK(group_id_map.init(instruction_set, memory_pool, &temp_buffers,
                                  log_minibatch_max, equal_func, append_func));

  col_non_nulls.resize(key_store.get_num_cols());
  col_offsets.resize(key_store.get_num_cols());
  col_values.resize(key_store.get_num_cols());
  out_col_non_nulls.resize(key_store.get_num_cols());
  out_col_offsets.resize(key_store.get_num_cols());
  out_col_values.resize(key_store.get_num_cols());
  varlen_col_non_nulls.resize(key_store.get_num_cols());
  varlen_col_offsets.resize(key_store.get_num_cols());

  bit_vector_all_0.resize((minibatch_size_max + 7) / 8 + KeyStore::padding_for_SIMD);
  memset(bit_vector_all_0.data(), 0, (minibatch_size_max + 7) / 8);
  bit_vector_all_1.resize((minibatch_size_max + 7) / 8 + KeyStore::padding_for_SIMD);
  memset(bit_vector_all_1.data(), 0xff, (minibatch_size_max + 7) / 8);

  minibatch_hashes.resize(minibatch_size_max +
                          KeyStore::padding_for_SIMD / sizeof(uint32_t));
  minibatch_temp.resize((minibatch_size_max + KeyStore::padding_for_SIMD) * 4);

  minibatch_any_nulls_buffer.resize((minibatch_size_max + 7) / 8 +
                                    KeyStore::padding_for_SIMD);
  memset(minibatch_any_nulls_buffer.data(), 0, (minibatch_size_max + 7) / 8);
  minibatch_nulls.resize(
      ((minibatch_size_max << KeyStore::get_log_row_null_bits(key_store.get_num_cols())) +
       7) /
          8 +
      KeyStore::padding_for_SIMD);
  minibatch_offsets.resize((minibatch_size_max + 1) +
                           KeyStore::padding_for_SIMD / sizeof(uint32_t));
  return Status::OK();
}

Status GroupMap::push_input(uint32_t num_rows,
                            const uint8_t** non_null_buffers_maybe_null,
                            const uint8_t** fixedlen_values_buffers,
                            const uint8_t** varlen_buffers_maybe_null,
                            uint32_t* group_ids) {
  uint32_t num_columns = key_store.get_num_cols();

  bool fixed_len_row = true;
  uint32_t fixed_len_part = 0;
  for (uint32_t icol = 0; icol < num_columns; ++icol) {
    fixed_len_part += col_widths[icol] == 0
                          ? 1
                          : is_col_fixed_len[icol] ? col_widths[icol] : sizeof(uint32_t);
    if (!is_col_fixed_len[icol]) {
      fixed_len_row = false;
    }
  }

  for (uint64_t irow0 = 0; irow0 < num_rows;) {
    uint32_t curr_minibatch_size = static_cast<uint32_t>(
        std::min(static_cast<uint64_t>(minibatch_size), num_rows - irow0));

    // All minibatches except for the last one should be multiples of 8 elements,
    // in order to not split bit vectors within a single byte on minibatch boundaries.
    //
    ARROW_DCHECK((curr_minibatch_size % 8) == 0 ||
                 irow0 + curr_minibatch_size == num_rows);

    uint32_t null_cols = 0;
    for (uint32_t icol = 0; icol < num_columns; ++icol) {
      if (non_null_buffers_maybe_null[icol]) {
        col_non_nulls[null_cols++] = non_null_buffers_maybe_null[icol] + irow0 / 8;
      }
    }

    // Compute any nulls bit vector.
    // It is used in two places:
    // - computing offsets for varying length rows for a minibatch
    // - comparison callback
    if (num_columns > 1 && null_cols > 0) {
      for (uint64_t iword = 0; iword < curr_minibatch_size / 64; ++iword) {
        uint64_t anded = ~0ULL;
        for (uint32_t icol = 0; icol < null_cols; ++icol) {
          anded &= reinterpret_cast<const uint64_t*>(col_non_nulls[icol])[iword];
        }
        reinterpret_cast<uint64_t*>(minibatch_any_nulls_buffer.data())[iword] = ~anded;
      }
      for (uint64_t ibyte = (curr_minibatch_size - (curr_minibatch_size % 64)) / 8;
           ibyte < (curr_minibatch_size + 7) / 8; ++ibyte) {
        uint8_t anded = ~0;
        for (uint32_t icol = 0; icol < null_cols; ++icol) {
          anded &= col_non_nulls[icol][ibyte];
        }
        minibatch_any_nulls_buffer[ibyte] = ~anded;
      }
      minibatch_any_nulls = minibatch_any_nulls_buffer.data();
    } else if (null_cols == 0) {
      minibatch_any_nulls = bit_vector_all_0.data();
    } else {
      for (uint64_t iword = 0; iword < curr_minibatch_size / 64; ++iword) {
        uint64_t anded = reinterpret_cast<const uint64_t*>(col_non_nulls[0])[iword];
        reinterpret_cast<uint64_t*>(minibatch_any_nulls_buffer.data())[iword] = ~anded;
      }
      for (uint64_t ibyte = (curr_minibatch_size - (curr_minibatch_size % 64)) / 8;
           ibyte < (curr_minibatch_size + 7) / 8; ++ibyte) {
        uint8_t anded = col_non_nulls[0][ibyte];
        minibatch_any_nulls_buffer[ibyte] = ~anded;
      }
      minibatch_any_nulls = minibatch_any_nulls_buffer.data();
    }

    for (uint32_t icol = 0; icol < num_columns; ++icol) {
      col_non_nulls[icol] = non_null_buffers_maybe_null[icol]
                                ? non_null_buffers_maybe_null[icol] + irow0 / 8
                                : bit_vector_all_1.data();
      col_offsets[icol] =
          is_col_fixed_len[icol]
              ? nullptr
              : reinterpret_cast<const uint32_t*>(fixedlen_values_buffers[icol]) + irow0;
      col_values[icol] =
          is_col_fixed_len[icol]
              ? fixedlen_values_buffers[icol] +
                    (col_widths[icol] == 0 ? irow0 / 8 : col_widths[icol] * irow0)
              : varlen_buffers_maybe_null[icol];
    }

    // Compute the size of row buffers for mini batch
    if (fixed_len_row) {
      minibatch_rows.resize(fixed_len_part * curr_minibatch_size +
                            KeyStore::padding_for_SIMD);
      minibatch_offsets[0] = fixed_len_part;
    } else {
      // col_offsets below and col_non_nulls below should only be concerned with variable
      // length columns.
      uint32_t num_varlen_cols = 0;
      for (uint32_t icol = 0; icol < num_columns; ++icol) {
        bool is_varlen_col = (col_offsets[icol] != nullptr);
        if (is_varlen_col) {
          varlen_col_offsets[num_varlen_cols] = col_offsets[icol];
          varlen_col_non_nulls[num_varlen_cols] = col_non_nulls[icol];
          ++num_varlen_cols;
        }
      }
      KeyLength::compute_offsets(instruction_set, num_varlen_cols, curr_minibatch_size,
                                 fixed_len_part, varlen_col_offsets.data(),
                                 varlen_col_non_nulls.data(), minibatch_any_nulls,
                                 minibatch_offsets.data());
      minibatch_rows.resize(minibatch_offsets[curr_minibatch_size] +
                            KeyStore::padding_for_SIMD);
    }

    // Convert input columns to rows
    KeyTranspose::col2row(
        instruction_set, num_columns, curr_minibatch_size, col_widths.data(),
        col_non_nulls.data(), col_offsets.data(), col_values.data(),
        fixed_len_row ? &fixed_len_part : minibatch_offsets.data(), minibatch_rows.data(),
        minibatch_nulls.data(), minibatch_temp.data(),
        minibatch_temp.data() + minibatch_size_max +
            KeyStore::padding_for_SIMD / sizeof(uint32_t));

    // Compute hash
    if (fixed_len_row) {
      Hashing::hash_fixed(instruction_set, curr_minibatch_size, fixed_len_part,
                          minibatch_rows.data(), minibatch_hashes.data());
    } else {
      Hashing::hash_varlen(instruction_set, curr_minibatch_size, minibatch_offsets.data(),
                           minibatch_rows.data(), minibatch_temp.data(),
                           minibatch_hashes.data());
    }

    // Map
    RETURN_NOT_OK(group_id_map.map(curr_minibatch_size, minibatch_hashes.data(),
                                   group_ids + irow0));

    irow0 += curr_minibatch_size;
    if (minibatch_size * 2 <= minibatch_size_max) {
      minibatch_size *= 2;
    }
  }

  return Status::OK();
}

void GroupMap::pull_output_prepare(uint64_t* out_num_rows, bool* out_is_row_fixedlen) {
  *out_num_rows = key_store.get_num_keys();
  *out_is_row_fixedlen = key_store.is_row_fixedlen();
}

void GroupMap::pull_output_fixedlen_and_nulls(uint8_t** non_null_buffers,
                                              uint8_t** fixedlen_buffers,
                                              uint64_t* out_varlen_buffer_sizes) {
  uint32_t log_row_null_bits = KeyStore::get_log_row_null_bits(key_store.get_num_cols());
  uint64_t num_rows;
  const uint32_t* row_offsets;
  const uint8_t* row_vals;
  const uint8_t* row_nulls;
  const uint8_t* row_any_nulls;
  key_store.get_rows(num_rows, row_offsets, row_vals, row_nulls, row_any_nulls);

  for (uint64_t irow0 = 0; irow0 < num_rows;) {
    int curr_minibatch_size = static_cast<int>(
        std::min(static_cast<uint64_t>(minibatch_size_max), num_rows - irow0));
    uint32_t row_length = key_store.get_row_length();

    ARROW_DCHECK((irow0 % 8) == 0);

    for (uint32_t icol = 0; icol < key_store.get_num_cols(); ++icol) {
      out_col_non_nulls[icol] = non_null_buffers[icol] + irow0 / 8;
      out_col_offsets[icol] = reinterpret_cast<uint32_t*>(
          is_col_fixed_len[icol] ? nullptr
                                 : fixedlen_buffers[icol] + sizeof(uint32_t) * irow0);
      if (irow0 == 0 && out_col_offsets[icol]) {
        // First batch initializes first offset to 0
        out_col_offsets[icol][0] = 0;
      }
      out_col_values[icol] =
          !is_col_fixed_len[icol]
              ? nullptr
              : (col_widths[icol] == 0
                     ? fixedlen_buffers[icol] + irow0 / 8
                     : fixedlen_buffers[icol] + col_widths[icol] * irow0);
    }

    arrow::exec::KeyTranspose::row2col(
        instruction_set, false, key_store.get_num_cols(), curr_minibatch_size,
        col_widths.data(), out_col_non_nulls.data(), out_col_offsets.data(),
        out_col_values.data(),
        key_store.is_row_fixedlen() ? &row_length : row_offsets + irow0,
        key_store.is_row_fixedlen() ? row_vals + row_length * irow0 : row_vals,
        row_nulls + (irow0 << log_row_null_bits) / 8, minibatch_size_max,
        minibatch_temp.data(),
        minibatch_temp.data() + minibatch_size_max +
            KeyStore::padding_for_SIMD / sizeof(uint32_t));

    irow0 += curr_minibatch_size;
    if (minibatch_size * 2 <= minibatch_size_max) {
      minibatch_size *= 2;
    }
  }

  for (uint32_t icol = 0; icol < key_store.get_num_cols(); ++icol) {
    if (is_col_fixed_len[icol]) {
      out_varlen_buffer_sizes[icol] = 0;
    } else {
      out_varlen_buffer_sizes[icol] =
          reinterpret_cast<const uint32_t*>(fixedlen_buffers[icol])[num_rows];
    }
  }
}

void GroupMap::pull_output_varlen(uint8_t** non_null_buffers, uint8_t** fixedlen_buffers,
                                  uint8_t** varlen_buffers_maybe_null) {
  uint32_t log_row_null_bits = KeyStore::get_log_row_null_bits(key_store.get_num_cols());
  uint64_t num_rows;
  const uint32_t* row_offsets;
  const uint8_t* row_vals;
  const uint8_t* row_nulls;
  const uint8_t* row_any_nulls;
  key_store.get_rows(num_rows, row_offsets, row_vals, row_nulls, row_any_nulls);

  for (uint64_t irow0 = 0; irow0 < num_rows;) {
    int curr_minibatch_size = static_cast<int>(
        std::min(static_cast<uint64_t>(minibatch_size_max), num_rows - irow0));
    uint32_t row_length = key_store.get_row_length();

    for (uint32_t icol = 0; icol < key_store.get_num_cols(); ++icol) {
      out_col_non_nulls[icol] = non_null_buffers[icol] + irow0 / 8;
      out_col_offsets[icol] = reinterpret_cast<uint32_t*>(
          is_col_fixed_len[icol] ? nullptr
                                 : fixedlen_buffers[icol] + sizeof(uint32_t) * irow0);
      out_col_values[icol] =
          is_col_fixed_len[icol] ? nullptr : varlen_buffers_maybe_null[icol];
    }

    arrow::exec::KeyTranspose::row2col(
        instruction_set, true, key_store.get_num_cols(), curr_minibatch_size,
        col_widths.data(), out_col_non_nulls.data(), out_col_offsets.data(),
        out_col_values.data(),
        key_store.is_row_fixedlen() ? &row_length : row_offsets + irow0,
        key_store.is_row_fixedlen() ? row_vals + row_length * irow0 : row_vals,
        row_nulls + (irow0 << log_row_null_bits) / 8, minibatch_size_max,
        minibatch_temp.data(),
        minibatch_temp.data() + minibatch_size_max +
            KeyStore::padding_for_SIMD / sizeof(uint32_t));

    irow0 += curr_minibatch_size;
    if (minibatch_size * 2 <= minibatch_size_max) {
      minibatch_size *= 2;
    }
  }
}

}  // namespace exec
}  // namespace arrow
