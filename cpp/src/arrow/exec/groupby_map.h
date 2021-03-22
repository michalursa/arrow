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

#include <functional>

#include "arrow/memory_pool.h"
#include "arrow/result.h"
#include "arrow/status.h"
#include "arrow/exec/common.h"
#include "arrow/exec/util.h"

class SwissTableSimple;

namespace arrow {
namespace exec {

//
// 0 byte - 7 bucket | 1. byte - 6 bucket | ...
// ---------------------------------------------------
// |     Empty bit*   |    Empty bit       |
// ---------------------------------------------------
// |   7-bit hash    |    7-bit hash      |
// ---------------------------------------------------
// * Empty bucket has value 0x80. Non-empty bucket has highest bit set to 0.
// ** The order of bytes is reversed - highest byte represents 0th bucket.
// No other part of data structure uses this reversed order.
//
class SwissTable {
  friend class ::SwissTableSimple;

 public:
  SwissTable()
      : log_blocks_(0),
        num_inserted_(0),
        blocks_(),
        hashes_(),
        pool_(),
        temp_buffers_() {}
  ~SwissTable() { cleanup(); }

  using EqualImpl =
      std::function<void(int num_keys, const uint16_t* selection /* may be null */,
                         const uint32_t* group_ids, uint8_t* match_bitvector)>;
  using AppendImpl = std::function<Status(int num_keys, const uint16_t* selection)>;

  Status init(util::CPUInstructionSet cpu_instruction_set, MemoryPool* pool,
              util::TempBufferAlloc* temp_buffers, int log_minibatch,
              EqualImpl equal_impl, AppendImpl append_impl);
  void cleanup();

  Status map(const int ckeys, const uint32_t* hashes, uint32_t* outgroupids);

 private:
  // Lookup helpers
  template <bool use_start_slot>
  inline void search_block(uint64_t block, int stamp, int start_slot, int& out_slot,
                           int& out_match_found);
  inline uint64_t extract_group_id(const uint8_t* block_ptr, int slot,
                                   uint64_t group_id_mask);
  inline uint64_t next_slot_to_visit(uint64_t block_index, int slot, int match_found);
  inline void insert(uint8_t* block_base, uint64_t slot_id, uint32_t hash, uint8_t stamp,
                     uint32_t group_id);

  //
  // First hash table access
  // Find first match in the first block.
  // Possible cases:
  // 1. Stamp match in a block
  // 2. No stamp match in a block, no empty buckets in a block
  // 3. No stamp match in a block, empty buckets in a block
  //
  template <bool use_selection>
  void lookup_1(const uint16_t* selection, const int num_keys, const uint32_t* hashes,
                uint8_t* out_match_bitvector, uint32_t* out_group_ids,
                uint32_t* out_slot_ids);
#if defined(ARROW_HAVE_AVX2)
  void lookup_1_avx2_x8(const int num_hashes, const uint32_t* hashes,
                        uint8_t* out_match_bitvector, uint32_t* out_group_ids,
                        uint32_t* out_next_slot_ids);
  void lookup_1_avx2_x32(const int num_hashes, const uint32_t* hashes,
                         uint8_t* out_match_bitvector, uint32_t* out_group_ids,
                         uint32_t* out_next_slot_ids);
#endif

  // Completing hash table lookup post first access
  Status lookup_2(const uint32_t* hashes, int& inout_num_selected,
                  uint16_t* inout_selection, bool& out_need_resize,
                  uint32_t* out_group_ids, uint32_t* out_next_slot_ids);

  // Resize small hash tables when 50% full (up to 8KB).
  // Resize large hash tables when 75% full.
  Status grow_double();

  // Use 32-bit hash for now
  static constexpr int bits_hash_ = 32;

  // Number of hash bits stored in slots in a block.
  // The highest bits of hash determine block id.
  // The next set of highest bits is a "stamp" stored in a slot in a block.
  static constexpr int bits_stamp_ = 7;

  // Padding bytes added at the end of buffers for ease of SIMD access
  static constexpr int padding_ = 64;

  int log_minibatch_;
  // Base 2 log of the number of blocks
  int log_blocks_;
  // Number of keys inserted into hash table
  uint32_t num_inserted_;

  // Data for blocks.
  // Each block has 8x of onse byte stamp slots, followed by 8x of bit packed group ids.
  // In 8B stamp word, the order of bytes is reversed. Group ids are in normal order.
  // There is 64B padding at the end.
  uint8_t* blocks_;

  // Array of hashes of values inserted into slots.
  // Undefined if the corresponding slot is empty.
  // There is 64B padding at the end.
  uint32_t* hashes_;

  MemoryPool* pool_;
  util::TempBufferAlloc* temp_buffers_;

  EqualImpl equal_impl_;
  AppendImpl append_impl_;

  util::CPUInstructionSet cpu_instruction_set_;

  bool precomputed_available() const;
  void precomputed_insert(uint32_t hash, uint32_t group_id, uint64_t slot_id);
  void precomputed_clear();
  void precomputed_make();
  void precomputed_lookup(int num_rows, const uint32_t* hashes, uint32_t* group_ids,
                          uint32_t* slot_ids, int& num_mismatch_ids,
                          uint16_t* mismatch_ids, uint8_t* temp_bitvector);
  std::vector<uint8_t> precomputed_group_ids;
  std::vector<uint8_t> precomputed_slot_ids;
};

}  // namespace exec
}  // namespace arrow
