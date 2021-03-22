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

#include "arrow/exec/groupby_map.h"

#include <immintrin.h>
#include <memory.h>

#include <algorithm>
#include <cstdint>

#include "arrow/exec/common.h"

namespace arrow {
namespace exec {

// Scan bytes in block in reverse and stop as soon
// as a position of interest is found, which is either of:
// a) slot with a matching stamp is encountered,
// b) first empty slot is encountered,
// c) we reach the end of the block.
// Return an index corresponding to this position (8 represents end of block).
// Return also an integer flag (0 or 1) indicating if we reached case a)
// (if we found a stamp match).
//
template <bool use_start_slot>
inline void SwissTable::search_block(
    uint64_t block,  // 8B block of hash table
    int stamp,       // 7-bits of hash used as a stamp
    int start_slot,  // Index of the first slot in the block to start search from.
                     // We assume that this index always points to a non-empty slot
                     // (comes before any empty slots).
                     // Used only by one template variant.
    int& out_slot,   // Returned index of a slot
    int& out_match_found) {  // Returned integer flag indicating match found
  // Filled slot bytes have the highest bit set to 0 and empty slots are equal to 0x80.
  // Replicate 7-bit stamp to all non-empty slots:
  uint64_t block_high_bits = block & UINT64_C(0x8080808080808080);
  uint64_t stamp_pattern =
      stamp * ((block_high_bits ^ UINT64_C(0x8080808080808080)) >> 7);
  // If we xor this pattern with block bytes we get:
  // a) 0x00, for filled slots matching the stamp,
  // b) 0x00 < x < 0x80, for filled slots not matching the stamp,
  // c) 0x80, for empty slots.
  // If we then add 0x7f to every byte, negate the result and leave only the highest bits
  // in each byte, we get 0x00 for non-match slot and 0x80 for match slot.
  uint64_t matches = ~((block ^ stamp_pattern) + UINT64_C(0x7f7f7f7f7f7f7f7f));
  if (use_start_slot) {
    matches &= UINT64_C(0x8080808080808080) >> (8 * start_slot);
  } else {
    matches &= UINT64_C(0x8080808080808080);
  }
  // We get 0 if there are no matches
  out_match_found = (matches == 0 ? 0 : 1);
  // Now if we or with the highest bits of the block and scan zero bits in reverse,
  // we get 8x slot index that we were looking for.
  out_slot = static_cast<int>(LZCNT64(matches | block_high_bits) >> 3);
}

// Extract group id for a given slot in a given block.
// Group ids follow in memory after 64-bit block data.
// Maximum number of groups inserted is equal to the number
// of all slots in all blocks, which is 8 * the number of blocks.
// Group ids are bit packed using that maximum to determine the necessary number of bits.
//
inline uint64_t SwissTable::extract_group_id(const uint8_t* block_ptr, int slot,
                                             uint64_t group_id_mask) {
  int num_bits_group_id = log_blocks_ + 3;
  // TODO: Explain why slot can be from 0 to 8 (inclusive) as input and in case of 8 we
  // just need to output any valid group id, so we take the one from slot 0 in the block.
  int bit_offset = (slot & 7) * num_bits_group_id;
  const uint8_t* group_id_bytes = (block_ptr + sizeof(uint64_t) + (bit_offset >> 3));
  uint64_t group_id =
      (*reinterpret_cast<const uint64_t*>(group_id_bytes) >> (bit_offset & 7)) &
      group_id_mask;
  return group_id;
}

inline uint64_t SwissTable::next_slot_to_visit(uint64_t block_index, int slot,
                                               int match_found) {
  return block_index * 8 + slot + match_found;
}

template <bool use_selection>
void SwissTable::lookup_1(const uint16_t* selection, const int num_keys,
                          const uint32_t* hashes, uint8_t* out_match_bitvector,
                          uint32_t* out_groupids, uint32_t* out_slot_ids) {
  memset(out_match_bitvector, 0, (num_keys + 7) / 8);

  const int num_groupid_bits = log_blocks_ + 3;
  uint32_t stamp_mask = (1 << bits_stamp_) - 1;
  uint32_t groupid_mask = (1 << num_groupid_bits) - 1;

  for (int i = 0; i < num_keys; ++i) {
    int id;
    if (use_selection) {
      id = selection[i];
    } else {
      id = i;
    }

    // Calculate block index and hash stamp for a byte in a block
    //
    uint32_t hash = hashes[id];
    uint32_t iblock = hash >> (bits_hash_ - bits_stamp_ - log_blocks_);
    uint32_t stamp = iblock & stamp_mask;
    iblock >>= bits_stamp_;

    const uint8_t* blockbase = reinterpret_cast<const uint8_t*>(blocks_) +
                               static_cast<uint64_t>(iblock) * (num_groupid_bits + 8);
    uint64_t block = *reinterpret_cast<const uint64_t*>(blockbase);

    int match_found;
    int islot_in_block;
    search_block<false>(block, stamp, 0, islot_in_block, match_found);
    uint64_t groupid = extract_group_id(blockbase, islot_in_block, groupid_mask);
    ARROW_DCHECK(groupid < num_inserted_ || num_inserted_ == 0);
    uint64_t islot = next_slot_to_visit(iblock, islot_in_block, match_found);

    out_match_bitvector[id / 8] |= match_found << (id & 7);

    out_groupids[id] = static_cast<uint32_t>(groupid);
    out_slot_ids[id] = static_cast<uint32_t>(islot);
  }
}
template void SwissTable::lookup_1<true>(const uint16_t*, const int, const uint32_t*,
                                         uint8_t*, uint32_t*, uint32_t*);
template void SwissTable::lookup_1<false>(const uint16_t*, const int, const uint32_t*,
                                          uint8_t*, uint32_t*, uint32_t*);

// Run a single round of slot search - comparison / insert - filter unprocessed.
// Update selection vector to reflect which items have been processed.
// Ids in selection vector do not have to be sorted.
//
Status SwissTable::lookup_2(const uint32_t* hashes, int& inout_num_selected,
                            uint16_t* inout_selection, bool& out_need_resize,
                            uint32_t* out_group_ids, uint32_t* inout_next_slot_ids) {
  // How many groups we can keep in hash table without resizing.
  // When we reach this limit, we need to break processing of any further rows.
  // Resize small hash tables when 50% full (up to 8KB).
  // Resize large hash tables when 75% full.
  constexpr int log_blocks_small_ = 9;
  uint32_t max_groupid = (log_blocks_ <= log_blocks_small_) ? (8 << log_blocks_) / 2
                                                            : 3 * (8 << log_blocks_) / 4;
  ARROW_DCHECK(num_inserted_ <= max_groupid);

  // Temporary arrays are of limited size.
  // The input needs to be split into smaller portions if it exceeds that limit.
  //
  ARROW_DCHECK(inout_num_selected <= (1 << log_minibatch_));

  // We will split input row ids into three categories:
  // - needing to visit next block [0]
  // - needing comparison [1]
  // - inserted [2]
  //
  // TODO: Move away from temp buffers to pre-allocated buffers
  util::TempBuffer<uint16_t> ids_inserted_buf(temp_buffers_);
  RETURN_NOT_OK(ids_inserted_buf.alloc());
  util::TempBuffer<uint16_t> ids_for_comparison_buf(temp_buffers_);
  RETURN_NOT_OK(ids_for_comparison_buf.alloc());
  constexpr int category_nomatch = 0;
  constexpr int category_cmp = 1;
  constexpr int category_inserted = 2;
  int num_ids[3];
  num_ids[0] = num_ids[1] = num_ids[2] = 0;
  uint16_t* ids[3]{inout_selection, ids_for_comparison_buf.mutable_data(),
                   ids_inserted_buf.mutable_data()};
  auto push_id = [&num_ids, &ids](int category, int id) {
    ids[category][num_ids[category]++] = static_cast<uint16_t>(id);
  };

  uint64_t slot_id_mask = (1 << (log_blocks_ + 3)) - 1;
  uint64_t groupid_mask = slot_id_mask;
  uint64_t num_groupid_bits = log_blocks_ + 3;
  constexpr uint64_t stamp_mask = 0x7f;
  uint64_t num_block_bytes = (8 + num_groupid_bits);

  int num_processed;
  for (num_processed = 0;
       // Second condition in for loop:
       // We need to break processing and have the caller of this function
       // resize hash table if we reach max_groupid groups.
       num_processed < inout_num_selected &&
       num_inserted_ + num_ids[category_inserted] < max_groupid;
       ++num_processed) {
    // row id in original batch
    int id = inout_selection[num_processed];

    uint64_t slot_id = (inout_next_slot_ids[id] & slot_id_mask);
    uint64_t block_id = slot_id >> 3;
    uint32_t hash = hashes[id];
    uint8_t* blockbase = blocks_ + num_block_bytes * block_id;
    uint64_t block = *reinterpret_cast<uint64_t*>(blockbase);
    uint64_t stamp = (hash >> (bits_hash_ - log_blocks_ - bits_stamp_)) & stamp_mask;
    int start_slot = (slot_id & 7);

    bool isempty = (blockbase[7 - start_slot] == 0x80);
    if (isempty) {
      blockbase[7 - start_slot] = static_cast<uint8_t>(stamp);
      int groupid_bit_offset = static_cast<int>(start_slot * num_groupid_bits);
      uint32_t group_id = num_inserted_ + num_ids[category_inserted];
      *reinterpret_cast<uint64_t*>(blockbase + 8 + (groupid_bit_offset >> 3)) |=
          (group_id << (groupid_bit_offset & 7));
      hashes_[slot_id] = hash;
      out_group_ids[id] = group_id;
      push_id(category_inserted, id);
      precomputed_insert(hash, group_id, slot_id);
    } else {
      int new_match_found;
      int new_slot;
      search_block<true>(block, static_cast<int>(stamp), start_slot, new_slot,
                         new_match_found);
      uint32_t new_groupid =
          static_cast<uint32_t>(extract_group_id(blockbase, new_slot, groupid_mask));
      ARROW_DCHECK(new_groupid < num_inserted_ + num_ids[category_inserted]);
      new_slot =
          static_cast<int>(next_slot_to_visit(block_id, new_slot, new_match_found));
      inout_next_slot_ids[id] = new_slot;
      out_group_ids[id] = new_groupid;
      push_id(new_match_found, id);
    }
  }

  // Copy keys for newly inserted rows
  RETURN_NOT_OK(append_impl_(num_ids[category_inserted], ids[category_inserted]));
  num_inserted_ += num_ids[category_inserted];

  // Evaluate comparisons and push ids of rows that failed
  // Add 3 copies of the first id, so that SIMD processing 4 elements at a time can work.
  //
  {
    util::TempBuffer<uint8_t> cmp_result_buf(temp_buffers_);
    RETURN_NOT_OK(cmp_result_buf.alloc());
    uint8_t* cmp_result = cmp_result_buf.mutable_data();
    equal_impl_(num_ids[category_cmp], ids[category_cmp], out_group_ids, cmp_result);
    int num_not_equal;
    util::BitUtil::bits_filter_indexes<0>(
        cpu_instruction_set_, num_ids[category_cmp], cmp_result, ids[category_cmp],
        num_not_equal, ids[category_nomatch] + num_ids[category_nomatch]);
    num_ids[category_nomatch] += num_not_equal;
  }

  // Append any unprocessed entries
  if (num_processed < inout_num_selected) {
    memcpy(ids[category_nomatch] + num_ids[category_nomatch],
           inout_selection + num_processed,
           sizeof(uint16_t) * (inout_num_selected - num_processed));
    num_ids[category_nomatch] += (inout_num_selected - num_processed);
  }

  out_need_resize = num_processed < inout_num_selected;
  inout_num_selected = num_ids[category_nomatch];
  return Status::OK();
}

Status SwissTable::map(const int num_keys, const uint32_t* hashes,
                       uint32_t* out_groupids) {
  // Temporary buffers have limited size.
  // Caller is responsible for splitting larger input arrays into smaller chunks.
  ARROW_DCHECK(num_keys <= (1 << log_minibatch_));

  // Allocate temporary buffers
  util::TempBuffer<uint8_t> match_bitvector_buf(temp_buffers_);
  RETURN_NOT_OK(match_bitvector_buf.alloc());
  uint8_t* match_bitvector = match_bitvector_buf.mutable_data();

  util::TempBuffer<uint32_t> slot_ids_buf(temp_buffers_);
  RETURN_NOT_OK(slot_ids_buf.alloc());
  uint32_t* slot_ids = slot_ids_buf.mutable_data();

  util::TempBuffer<uint16_t> ids_buf(temp_buffers_);
  RETURN_NOT_OK(ids_buf.alloc());
  uint16_t* ids = ids_buf.mutable_data();
  int num_ids;

  if (num_inserted_ > 0 && precomputed_available()) {
    precomputed_lookup(num_keys, hashes, out_groupids, slot_ids, num_ids, ids,
                       match_bitvector);
  } else {
    switch (cpu_instruction_set_) {
      case util::CPUInstructionSet::scalar:
        lookup_1<false>(nullptr, num_keys, hashes, match_bitvector, out_groupids,
                        slot_ids);
        break;
#if defined(ARROW_HAVE_AVX2)
      case util::CPUInstructionSet::avx2:
        if (log_blocks_ <= 4) {
          int tail = num_keys % 32;
          int delta = num_keys - tail;
          lookup_1_avx2_x32(num_keys - tail, hashes, match_bitvector, out_groupids,
                            slot_ids);
          lookup_1_avx2_x8(tail, hashes + delta, match_bitvector + delta / 8,
                           out_groupids + delta, slot_ids + delta);
        } else {
          lookup_1_avx2_x8(num_keys, hashes, match_bitvector, out_groupids, slot_ids);
        }
        break;
#endif
      default:
        break;
    }

    int num_matches = util::BitUtil::popcnt_bitvector(num_keys, match_bitvector);

    // after first pass count rows with matches and decide based on their percentage
    // whether to call dense or sparse comparison function
    //

    // TODO: explain num_inserted_ > 0 condition below
    if (num_inserted_ > 0 && num_matches > 0 && num_matches > 3 * num_keys / 4) {
      equal_impl_(num_keys, nullptr, out_groupids, match_bitvector);
      util::BitUtil::bits_to_indexes<0>(cpu_instruction_set_, num_keys, match_bitvector,
                                        num_ids, ids);
    } else {
      util::TempBuffer<uint16_t> ids_cmp_buf(temp_buffers_);
      RETURN_NOT_OK(ids_cmp_buf.alloc());
      uint16_t* ids_cmp = ids_cmp_buf.mutable_data();
      util::BitUtil::bits_split_indexes(cpu_instruction_set_, num_keys, match_bitvector,
                                        num_ids, ids, ids_cmp);
      equal_impl_(num_keys - num_ids, ids_cmp, out_groupids, match_bitvector);
      int num_not_equal;
      util::BitUtil::bits_filter_indexes<0>(cpu_instruction_set_, num_keys - num_ids,
                                            match_bitvector, ids_cmp, num_not_equal,
                                            ids + num_ids);
      num_ids += num_not_equal;
    }
  }  // precomputed_available()

  do {
    bool out_of_capacity;
    RETURN_NOT_OK(lookup_2(hashes, num_ids, ids, out_of_capacity, out_groupids, slot_ids));
    if (out_of_capacity) {
      RETURN_NOT_OK(grow_double());
      // Set slot_ids for selected vectors to first slot in new initial block.
      for (int i = 0; i < num_ids; ++i) {
        slot_ids[ids[i]] = (hashes[ids[i]] >> (bits_hash_ - log_blocks_)) * 8;
      }
    }
  } while (num_ids > 0);

  return Status::OK();
}

Status SwissTable::grow_double() {
  // Before and after metadata
  int num_group_id_bits_before = log_blocks_ + 3;
  int num_group_id_bits_after = num_group_id_bits_before + 1;
  uint64_t group_id_mask_before = ~0ULL >> (64 - num_group_id_bits_before);
  int log_blocks_before = log_blocks_;
  int log_blocks_after = log_blocks_ + 1;
  uint64_t block_size_before = (8 + num_group_id_bits_before);
  uint64_t block_size_after = (8 + num_group_id_bits_after);
  uint64_t block_size_total_before = (block_size_before << log_blocks_before) + padding_;
  uint64_t block_size_total_after = (block_size_after << log_blocks_after) + padding_;
  uint64_t hashes_size_total_before =
      (bits_hash_ / 8 * (1 << (log_blocks_before + 3))) + padding_;
  uint64_t hashes_size_total_after =
      (bits_hash_ / 8 * (1 << (log_blocks_after + 3))) + padding_;
  constexpr uint32_t stamp_mask = (1 << bits_stamp_) - 1;

  // Allocate new buffers
  uint8_t* blocks_new;
  RETURN_NOT_OK(pool_->Allocate(block_size_total_after, &blocks_new));
  memset(blocks_new, 0, block_size_total_after);
  uint8_t* hashes_new_8B;
  uint32_t* hashes_new;
  RETURN_NOT_OK(pool_->Allocate(hashes_size_total_after, &hashes_new_8B));
  hashes_new = reinterpret_cast<uint32_t*>(hashes_new_8B);

  // First pass over all old blocks.
  // Reinsert entries that were not in the overflow block
  // (block other than selected by hash bits corresponding to the entry).
  for (int i = 0; i < (1 << log_blocks_); ++i) {
    // How many full slots in this block
    uint8_t* block_base = blocks_ + i * block_size_before;
    uint8_t* double_block_base_new = blocks_new + 2 * i * block_size_after;
    uint64_t block = *reinterpret_cast<const uint64_t*>(block_base);
    int full_slots = static_cast<int>(LZCNT64(block & 0x8080808080808080ULL) >> 3);
    int full_slots_new[2];
    full_slots_new[0] = full_slots_new[1] = 0;
    *reinterpret_cast<uint64_t*>(double_block_base_new) = 0x8080808080808080ULL;
    *reinterpret_cast<uint64_t*>(double_block_base_new + block_size_after) =
        0x8080808080808080ULL;

    for (int j = 0; j < full_slots; ++j) {
      uint64_t slot_id = i * 8 + j;
      uint32_t hash = hashes_[slot_id];
      uint64_t block_id_new = hash >> (bits_hash_ - log_blocks_after);
      bool is_overflow_entry = ((block_id_new >> 1) != static_cast<uint64_t>(i));
      if (is_overflow_entry) {
        continue;
      }

      int ihalf = block_id_new & 1;
      uint8_t stamp_new =
          hash >> ((bits_hash_ - log_blocks_after - bits_stamp_)) & stamp_mask;
      uint64_t group_id_bit_offs = j * num_group_id_bits_before;
      uint64_t group_id = (*reinterpret_cast<const uint64_t*>(block_base + 8 +
                                                              (group_id_bit_offs >> 3)) >>
                           (group_id_bit_offs & 7)) &
                          group_id_mask_before;

      uint64_t slot_id_new = i * 16 + ihalf * 8 + full_slots_new[ihalf];
      hashes_new[slot_id_new] = hash;
      uint8_t* block_base_new = double_block_base_new + ihalf * block_size_after;
      block_base_new[7 - full_slots_new[ihalf]] = stamp_new;
      int group_id_bit_offs_new = full_slots_new[ihalf] * num_group_id_bits_after;
      *reinterpret_cast<uint64_t*>(block_base_new + 8 + (group_id_bit_offs_new >> 3)) |=
          (group_id << (group_id_bit_offs_new & 7));
      full_slots_new[ihalf]++;
    }
  }

  // Second pass over all old blocks.
  // Reinsert entries that were in an overflow block.
  for (int i = 0; i < (1 << log_blocks_); ++i) {
    // How many full slots in this block
    uint8_t* block_base = blocks_ + i * block_size_before;
    uint64_t block = *reinterpret_cast<const uint64_t*>(block_base);
    int full_slots = static_cast<int>(LZCNT64(block & 0x8080808080808080ULL) >> 3);

    for (int j = 0; j < full_slots; ++j) {
      uint64_t slot_id = i * 8 + j;
      uint32_t hash = hashes_[slot_id];
      uint64_t block_id_new = hash >> (bits_hash_ - log_blocks_after);
      bool is_overflow_entry = ((block_id_new >> 1) != static_cast<uint64_t>(i));
      if (!is_overflow_entry) {
        continue;
      }

      uint64_t group_id_bit_offs = j * num_group_id_bits_before;
      uint64_t group_id = (*reinterpret_cast<const uint64_t*>(block_base + 8 +
                                                              (group_id_bit_offs >> 3)) >>
                           (group_id_bit_offs & 7)) &
                          group_id_mask_before;
      uint8_t stamp_new =
          hash >> ((bits_hash_ - log_blocks_after - bits_stamp_)) & stamp_mask;

      uint8_t* block_base_new = blocks_new + block_id_new * block_size_after;
      uint64_t block_new = *reinterpret_cast<const uint64_t*>(block_base_new);
      int full_slots_new =
          static_cast<int>(LZCNT64(block_new & 0x8080808080808080ULL) >> 3);
      while (full_slots_new == 8) {
        block_id_new = (block_id_new + 1) & ((1 << log_blocks_after) - 1);
        block_base_new = blocks_new + block_id_new * block_size_after;
        block_new = *reinterpret_cast<const uint64_t*>(block_base_new);
        full_slots_new =
            static_cast<int>(LZCNT64(block_new & 0x8080808080808080ULL) >> 3);
      }

      hashes_new[block_id_new * 8 + full_slots_new] = hash;
      block_base_new[7 - full_slots_new] = stamp_new;
      int group_id_bit_offs_new = full_slots_new * num_group_id_bits_after;
      *reinterpret_cast<uint64_t*>(block_base_new + 8 + (group_id_bit_offs_new >> 3)) |=
          (group_id << (group_id_bit_offs_new & 7));
    }
  }

  pool_->Free(blocks_, block_size_total_before);
  pool_->Free(reinterpret_cast<uint8_t*>(hashes_), hashes_size_total_before);
  log_blocks_ = log_blocks_after;
  blocks_ = blocks_new;
  hashes_ = hashes_new;

  precomputed_make();

  return Status::OK();
}

Status SwissTable::init(util::CPUInstructionSet cpu_instruction_set, MemoryPool* pool,
                        util::TempBufferAlloc* temp_buffers, int log_minibatch,
                        EqualImpl equal_impl, AppendImpl append_impl) {
  cpu_instruction_set_ = cpu_instruction_set;
  pool_ = pool;
  temp_buffers_ = temp_buffers;
  log_minibatch_ = log_minibatch;
  equal_impl_ = equal_impl;
  append_impl_ = append_impl;

  log_blocks_ = 0;
  const int num_groupid_bits = log_blocks_ + 3;
  num_inserted_ = 0;

  const uint64_t cbblocks = ((8 + num_groupid_bits) << log_blocks_) + padding_;
  RETURN_NOT_OK(pool_->Allocate(cbblocks, &blocks_));
  memset(blocks_, 0, cbblocks);

  for (uint64_t i = 0; i < (static_cast<uint64_t>(1) << log_blocks_); ++i) {
    *reinterpret_cast<uint64_t*>(blocks_ + i * (8 + num_groupid_bits)) =
        UINT64_C(0x8080808080808080);
  }

  const uint64_t cbhashes = (sizeof(uint32_t) << num_groupid_bits) + padding_;
  uint8_t* hashes8;
  RETURN_NOT_OK(pool_->Allocate(cbhashes, &hashes8));
  hashes_ = reinterpret_cast<uint32_t*>(hashes8);

  precomputed_clear();

  return Status::OK();
}

void SwissTable::cleanup() {
  const int cgroupidbits = log_blocks_ + 3;
  if (blocks_) {
    const uint64_t cbblocks = ((8 + cgroupidbits) << log_blocks_) + padding_;
    pool_->Free(blocks_, cbblocks);
    blocks_ = nullptr;
  }
  if (hashes_) {
    const uint64_t cbhashes = (sizeof(uint32_t) << cgroupidbits) + padding_;
    pool_->Free(reinterpret_cast<uint8_t*>(hashes_), cbhashes);
    hashes_ = nullptr;
  }
  log_blocks_ = 0;
  num_inserted_ = 0;
}

inline bool SwissTable::precomputed_available() const { return log_blocks_ + 3 <= 8; }

inline void SwissTable::precomputed_insert(uint32_t hash, uint32_t group_id,
                                           uint64_t slot_id) {
  if (precomputed_available()) {
    int pos = hash >> (bits_hash_ - log_blocks_ - 4);
    if (precomputed_group_ids[pos] == 0x80) {
      precomputed_group_ids[pos] = static_cast<uint8_t>(group_id & 0xff);
      precomputed_slot_ids[pos] = static_cast<uint8_t>((slot_id + 1) & 0xff);
    }
  }
}

void SwissTable::precomputed_clear() {
  if (precomputed_available()) {
    precomputed_group_ids.resize(1ULL << (log_blocks_ + 4));
    precomputed_slot_ids.resize(1ULL << (log_blocks_ + 4));
    memset(precomputed_group_ids.data(), 0x80, (1ULL << (log_blocks_ + 4)));
    for (int i = 0; i < (1 << log_blocks_); ++i) {
      for (int j = 0; j < 16; ++j) {
        precomputed_slot_ids[i * 16 + j] = i * 8;
      }
    }
  }
}

void SwissTable::precomputed_make() {
  if (precomputed_available()) {
    precomputed_clear();
    uint32_t num_group_id_bits = log_blocks_ + 3;
    uint64_t group_id_mask = ~0ULL >> (64 - num_group_id_bits);
    for (int i = 0; i < (1 << log_blocks_); ++i) {
      uint8_t* block_base = blocks_ + i * (8 + num_group_id_bits);
      uint64_t block = *reinterpret_cast<const uint64_t*>(block_base);
      int full_slots = static_cast<int>(LZCNT64(block & 0x8080808080808080ULL) >> 3);
      for (int j = 0; j < full_slots; ++j) {
        uint64_t group_id_bit_offs = j * num_group_id_bits;
        uint64_t group_id = (*reinterpret_cast<const uint64_t*>(
                                 block_base + 8 + (group_id_bit_offs >> 3)) >>
                             (group_id_bit_offs & 7)) &
                            group_id_mask;
        uint32_t slot_id = i * 8 + j;
        uint32_t hash = hashes_[slot_id];
        precomputed_insert(hash, static_cast<uint32_t>(group_id), slot_id);
      }
    }
  }
}

void SwissTable::precomputed_lookup(int num_rows, const uint32_t* hashes,
                                    uint32_t* group_ids, uint32_t* slot_ids,
                                    int& num_mismatch_ids, uint16_t* mismatch_ids,
                                    uint8_t* temp_bitvector) {
  if (precomputed_available()) {
    for (int i = 0; i < num_rows; ++i) {
      group_ids[i] =
          precomputed_group_ids[hashes[i] >> (bits_hash_ - log_blocks_ - 4)] & 0x7f;
    }
    equal_impl_(num_rows, nullptr, group_ids, temp_bitvector);
    util::BitUtil::bits_to_indexes<0>(cpu_instruction_set_, num_rows, temp_bitvector,
                                      num_mismatch_ids, mismatch_ids);
    for (int i = 0; i < num_mismatch_ids; ++i) {
      int id = mismatch_ids[i];
      slot_ids[id] = precomputed_slot_ids[hashes[id] >> (bits_hash_ - log_blocks_ - 4)];
    }
  }
}

}  // namespace exec
}  // namespace arrow
