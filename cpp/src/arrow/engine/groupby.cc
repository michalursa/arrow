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

#include "arrow/engine/groupby.h"

#ifdef _MSC_VER
#include <intrin.h>
#else
#include <x86intrin.h>
#endif
#include <memory.h>

#include <algorithm>
#include <cstdint>

namespace arrow {
namespace compute {

GroupMap::~GroupMap() { map_.cleanup(); }

Status GroupMap::init(util::CPUInstructionSet instruction_set_in, MemoryPool* pool,
                      uint32_t num_columns, const std::vector<bool>& is_fixed_len_in,
                      const uint32_t* col_widths_in) {
  memory_pool_ = pool;
  RETURN_NOT_OK(temp_stack_.Init(memory_pool_, 64 * minibatch_size_max));

  ctx_.instr = instruction_set_in;
  ctx_.stack = &temp_stack_;

  col_metadata_.resize(num_columns);
  for (uint32_t icol = 0; icol < num_columns; ++icol) {
    col_metadata_[icol] =
        KeyEncoder::KeyColumnMetadata(is_fixed_len_in[icol], col_widths_in[icol]);
  }

  encoder_.Init(col_metadata_, &ctx_);

  RETURN_NOT_OK(rows_.Init(memory_pool_, encoder_.get_row_metadata()));
  RETURN_NOT_OK(rows_minibatch_.Init(memory_pool_, encoder_.get_row_metadata()));

  minibatch_size = minibatch_size_min;

  auto equal_func = [this](int num_keys_to_compare,
                           const uint16_t* selection_may_be_null /* may be null */,
                           const uint32_t* group_ids, uint32_t* out_num_keys_mismatch,
                           uint16_t* out_selection_mismatch) {
    uint64_t c0 = __rdtsc();
    KeyCompare::CompareRows(num_keys_to_compare, selection_may_be_null, group_ids, &ctx_,
                            *out_num_keys_mismatch, out_selection_mismatch,
                            rows_minibatch_, rows_);
    uint64_t c1 = __rdtsc();
    cycles_cmp_ += c1 - c0;
    stat_num_rows_cmp_ += num_keys_to_compare;
  };
  auto append_func = [this](int num_keys, const uint16_t* selection) {
    return rows_.AppendSelectionFrom(rows_minibatch_, num_keys, selection);
  };

  RETURN_NOT_OK(map_.init(ctx_.instr, memory_pool_, ctx_.stack, log_minibatch_max,
                          equal_func, append_func));

  cols_.resize(num_columns);

  constexpr int padding_for_SIMD = 32;
  minibatch_hashes_.resize(minibatch_size_max + padding_for_SIMD / sizeof(uint32_t));

  cycles_cmp_ = 0;
  cycles_map_ = 0;
  cycles_hash_ = 0;
  cycles_encode_ = 0;
  stat_num_rows_ = 0;
  stat_num_rows_cmp_ = 0;

  return Status::OK();
}

Status GroupMap::push_input(uint32_t num_rows,
                            const uint8_t** non_null_buffers_maybe_null,
                            const uint8_t** fixedlen_values_buffers,
                            const uint8_t** varlen_buffers_maybe_null,
                            uint32_t* group_ids) {
  uint32_t num_cols = static_cast<uint32_t>(cols_.size());
  for (uint32_t i = 0; i < num_cols; ++i) {
    cols_[i] = KeyEncoder::KeyColumnArray(
        col_metadata_[i], num_rows, non_null_buffers_maybe_null[i],
        fixedlen_values_buffers[i], varlen_buffers_maybe_null[i]);
  }

  for (uint32_t start_row = 0; start_row < num_rows;) {
    uint32_t batch_size_next =
        std::min(static_cast<uint32_t>(minibatch_size), num_rows - start_row);
    uint64_t c0 = __rdtsc();

    // Encode
    rows_minibatch_.Clean();
    RETURN_NOT_OK(encoder_.PrepareOutputForEncode(start_row, batch_size_next,
                                                  rows_minibatch_, cols_));
    encoder_.Encode(start_row, batch_size_next, rows_minibatch_, cols_);
    uint64_t c1 = __rdtsc();

    // Compute hash
    if (encoder_.get_row_metadata().is_fixed_length) {
      Hashing::hash_fixed(ctx_.instr, batch_size_next,
                          encoder_.get_row_metadata().fixed_length,
                          rows_minibatch_.data(1), minibatch_hashes_.data());
    } else {
      auto hash_temp_buf =
          util::TempVectorHolder<uint32_t>(&temp_stack_, 4 * batch_size_next);
      Hashing::hash_varlen(ctx_.instr, batch_size_next, rows_minibatch_.get_offsets(),
                           rows_minibatch_.data(2), hash_temp_buf.mutable_data(),
                           minibatch_hashes_.data());
    }
    uint64_t c2 = __rdtsc();

    // Map
    RETURN_NOT_OK(
        map_.map(batch_size_next, minibatch_hashes_.data(), group_ids + start_row));
    uint64_t c3 = __rdtsc();

    cycles_encode_ += c1 - c0;
    cycles_hash_ += c2 - c1;
    cycles_map_ += c3 - c2;
    stat_num_rows_ += batch_size_next;

    start_row += batch_size_next;

    if (minibatch_size * 2 <= minibatch_size_max) {
      minibatch_size *= 2;
    }
  }
  return Status::OK();
}

void GroupMap::pull_output_prepare(uint64_t* out_num_rows, bool* out_is_row_fixedlen) {
  *out_num_rows = rows_.get_length();
  *out_is_row_fixedlen = rows_.get_metadata().is_fixed_length;

  // printf("(enc, hash, map, cmp) = (%I64d, %I64d, %I64d, %I64d) lookup_ratio %.1f",
  //       cycles_encode_ / stat_num_rows_, cycles_hash_ / stat_num_rows_,
  //       (cycles_map_ - cycles_cmp_) / stat_num_rows_, cycles_cmp_ / stat_num_rows_cmp_,
  //       static_cast<float>(stat_num_rows_cmp_) / static_cast<float>(stat_num_rows_));
}

void GroupMap::pull_output_fixedlen_and_nulls(uint8_t** non_null_buffers,
                                              uint8_t** fixedlen_buffers,
                                              uint64_t* out_varlen_buffer_sizes) {
  int64_t num_rows = rows_.get_length();
  uint32_t num_cols = static_cast<uint32_t>(cols_.size());
  for (uint32_t i = 0; i < num_cols; ++i) {
    cols_[i] = KeyEncoder::KeyColumnArray(col_metadata_[i], num_rows, non_null_buffers[i],
                                          fixedlen_buffers[i], nullptr);
  }

  for (int64_t start_row = 0; start_row < num_rows;) {
    int64_t batch_size_next =
        std::min(num_rows - start_row, static_cast<int64_t>(minibatch_size_max));
    encoder_.DecodeFixedLengthBuffers(start_row, start_row, batch_size_next, rows_,
                                      cols_);
    start_row += batch_size_next;
  }

  for (uint32_t icol = 0; icol < num_cols; ++icol) {
    if (col_metadata_[icol].is_fixed_length) {
      out_varlen_buffer_sizes[icol] = 0;
    } else {
      out_varlen_buffer_sizes[icol] =
          reinterpret_cast<const uint32_t*>(fixedlen_buffers[icol])[num_rows];
    }
  }
}

void GroupMap::pull_output_varlen(uint8_t** non_null_buffers, uint8_t** fixedlen_buffers,
                                  uint8_t** varlen_buffers_maybe_null) {
  int64_t num_rows = rows_.get_length();
  uint32_t num_cols = static_cast<uint32_t>(cols_.size());
  for (uint32_t i = 0; i < num_cols; ++i) {
    if (!col_metadata_[i].is_fixed_length) {
      cols_[i] =
          KeyEncoder::KeyColumnArray(col_metadata_[i], num_rows, non_null_buffers[i],
                                     fixedlen_buffers[i], varlen_buffers_maybe_null[i]);
    }
  }

  if (!rows_.get_metadata().is_fixed_length) {
    for (int64_t start_row = 0; start_row < num_rows;) {
      int64_t batch_size_next =
          std::min(num_rows - start_row, static_cast<int64_t>(minibatch_size_max));
      encoder_.DecodeVaryingLengthBuffers(start_row, start_row, batch_size_next, rows_,
                                          cols_);
      start_row += batch_size_next;
    }
  }
}

}  // namespace compute
}  // namespace arrow
