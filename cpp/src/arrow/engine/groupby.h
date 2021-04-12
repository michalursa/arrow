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

#include "arrow/engine/key_compare.h"
#include "arrow/engine/key_encode.h"
#include "arrow/engine/key_hash.h"
#include "arrow/engine/key_map.h"
#include "arrow/engine/util.h"
#include "arrow/memory_pool.h"
#include "arrow/result.h"
#include "arrow/status.h"

namespace arrow {
namespace compute {

class GroupMap {
 public:
  ~GroupMap();

  Status init(util::CPUInstructionSet instruction_set_in, MemoryPool* pool,
              uint32_t num_columns, const std::vector<bool>& is_fixed_len_in,
              const uint32_t* col_widths_in);

  Status push_input(uint32_t num_rows, const uint8_t** non_null_buffers_maybe_null,
                    const uint8_t** fixedlen_buffers,
                    const uint8_t** varlen_buffers_maybe_null, uint32_t* group_ids);

  void pull_output_prepare(uint64_t* out_num_rows, bool* is_row_fixedlen);

  void pull_output_fixedlen_and_nulls(uint8_t** non_null_buffers,
                                      uint8_t** fixedlen_buffers,
                                      uint64_t* out_varlen_buffer_sizes);

  void pull_output_varlen(uint8_t** non_null_buffers, uint8_t** fixedlen_buffers,
                          uint8_t** varlen_buffers_maybe_null);

  uint64_t get_num_keys() const { return rows_.get_length(); }

 private:
  static constexpr int log_minibatch_max = 10;
  static constexpr int minibatch_size_max = 1 << log_minibatch_max;
  static constexpr int minibatch_size_min = 128;

  int minibatch_size;

  MemoryPool* memory_pool_;
  util::TempVectorStack temp_stack_;
  KeyEncoder::KeyEncoderContext ctx_;
  KeyEncoder::KeyRowArray rows_;
  KeyEncoder::KeyRowArray rows_minibatch_;
  KeyEncoder encoder_;
  SwissTable map_;

  std::vector<KeyEncoder::KeyColumnMetadata> col_metadata_;
  std::vector<KeyEncoder::KeyColumnArray> cols_;

  std::vector<uint32_t> minibatch_hashes_;

  uint64_t cycles_encode_;
  uint64_t cycles_hash_;
  uint64_t cycles_map_;
  uint64_t cycles_cmp_;
  uint64_t stat_num_rows_;
  uint64_t stat_num_rows_cmp_;
};

}  // namespace compute
}  // namespace arrow
