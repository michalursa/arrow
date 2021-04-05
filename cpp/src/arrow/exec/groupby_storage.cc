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

#ifdef _MSC_VER
#include <intrin.h>
#else
#include <x86intrin.h>
#endif
#include <immintrin.h>

#include <algorithm>

#include "arrow/exec/common.h"
#include "arrow/exec/groupby_storage.h"
#include "arrow/exec/util.h"

namespace arrow {
namespace exec {

Status KeyStore::init(util::CPUInstructionSet instruction_set, MemoryPool* pool,
                      uint32_t num_cols, bool is_row_fixedlen, uint32_t row_length) {
  instruction_set_ = instruction_set;
  pool_ = pool;
  num_cols_ = num_cols;
  is_row_fixedlen_ = is_row_fixedlen;
  row_length_ = row_length;

  constexpr uint64_t initial_num_rows = 64;
  constexpr uint64_t initial_num_value_bytes = 1024;
  RETURN_NOT_OK(resize_buffers_if_needed(initial_num_rows, initial_num_value_bytes));
  num_rows_ = 0;
  return Status::OK();
}

uint64_t KeyStore::get_size_keys() const {
  uint64_t num_value_bytes_used;
  if (is_row_fixedlen_) {
    num_value_bytes_used = (static_cast<uint64_t>(row_length_) * num_rows_allocated_);
  } else {
    num_value_bytes_used = ((num_rows_ == 0) ? 0 : row_offsets_[num_rows_]);
  }
  return num_value_bytes_used;
}

Status KeyStore::resize_buffers_if_needed(uint64_t num_extra_rows,
                                          uint64_t num_extra_value_bytes) {
  if (num_rows_ + num_extra_rows > num_rows_allocated_) {
    // We need to reallocate all buffers except the one for value bytes
    uint64_t num_rows_allocated_new =
        std::max(static_cast<uint64_t>(1ULL), 2 * num_rows_allocated_);
    while (num_rows_ + num_extra_rows > num_rows_allocated_new) {
      num_rows_allocated_new *= 2;
    }

    // Nulls bit vector
    uint32_t log_row_nulls = get_log_row_null_bits(num_cols_);
    uint64_t nulls_size_old =
        ((num_rows_allocated_ << log_row_nulls) + 7) / 8 + padding_for_SIMD;
    uint64_t nulls_size_new =
        ((num_rows_allocated_new << log_row_nulls) + 7) / 8 + padding_for_SIMD;
    uint8_t* row_nulls_new;
    RETURN_NOT_OK(pool_->Allocate(nulls_size_new, &row_nulls_new));
    if (num_rows_allocated_ > 0) {
      memcpy(row_nulls_new, row_nulls_, nulls_size_old);
      pool_->Free(row_nulls_, row_nulls_size_);
      row_nulls_ = nullptr;
      memset(row_nulls_new + nulls_size_old, 0, nulls_size_new - nulls_size_old);
    } else {
      memset(row_nulls_new, 0, nulls_size_new);
    }
    row_nulls_size_ = nulls_size_new;
    row_nulls_ = row_nulls_new;

    // Any nulls bit vector
    uint64_t any_nulls_size_old = (num_rows_allocated_ + 7) / 8 + padding_for_SIMD;
    uint64_t any_nulls_size_new = (num_rows_allocated_new + 7) / 8 + padding_for_SIMD;
    uint8_t* any_nulls_new;
    RETURN_NOT_OK(pool_->Allocate(any_nulls_size_new, &any_nulls_new));
    if (num_rows_allocated_) {
      memcpy(any_nulls_new, row_any_nulls_, any_nulls_size_old);
      pool_->Free(row_any_nulls_, row_any_nulls_size_);
      row_any_nulls_ = nullptr;
      memset(any_nulls_new + any_nulls_size_old, 0,
             any_nulls_size_new - any_nulls_size_old);
    } else {
      memset(any_nulls_new, 0, any_nulls_size_new);
    }
    row_any_nulls_ = any_nulls_new;
    row_any_nulls_size_ = any_nulls_size_new;

    // Offsets vector, only used for varying length keys
    if (!is_row_fixedlen_) {
      uint64_t offsets_size_old =
          sizeof(uint32_t) * num_rows_allocated_ + padding_for_SIMD;
      uint64_t offsets_size_new =
          sizeof(uint32_t) * num_rows_allocated_new + padding_for_SIMD;
      uint8_t* offsets_new;
      RETURN_NOT_OK(pool_->Allocate(offsets_size_new, &offsets_new));
      if (num_rows_allocated_ > 0) {
        memcpy(offsets_new, row_offsets_, offsets_size_old);
        pool_->Free(reinterpret_cast<uint8_t*>(row_offsets_), row_offsets_size_);
        row_offsets_ = nullptr;
      }
      memset(offsets_new + offsets_size_old, 0, offsets_size_new - offsets_size_old);
      row_offsets_ = reinterpret_cast<uint32_t*>(offsets_new);
      row_offsets_size_ = offsets_size_new;
    }

    num_rows_allocated_ = num_rows_allocated_new;
  }

  // Value bytes vector
  uint64_t num_value_bytes_used = get_size_keys();
  if (num_value_bytes_used + num_extra_value_bytes > num_value_bytes_allocated_) {
    uint64_t size_new =
        std::max(static_cast<uint64_t>(1ULL), num_value_bytes_allocated_ * 2);
    while (num_value_bytes_used + num_extra_value_bytes > size_new) {
      size_new *= 2;
    }
    uint8_t* values_new;
    RETURN_NOT_OK(pool_->Allocate(size_new + padding_for_SIMD, &values_new));
    if (num_value_bytes_allocated_ > 0) {
      memcpy(values_new, row_vals_, num_value_bytes_allocated_);
      pool_->Free(row_vals_, num_value_bytes_allocated_ + padding_for_SIMD);
      row_vals_ = nullptr;
    }
    row_vals_ = values_new;
    num_value_bytes_allocated_ = size_new;
  }

  return Status::OK();
}

void KeyStore::cleanup() {
  if (row_nulls_) {
    pool_->Free(row_nulls_, row_nulls_size_);
    row_nulls_ = nullptr;
    row_nulls_size_ = 0;
  }
  if (row_any_nulls_) {
    pool_->Free(row_any_nulls_, row_any_nulls_size_);
    row_any_nulls_ = nullptr;
    row_any_nulls_size_ = 0;
  }
  if (row_offsets_) {
    pool_->Free(reinterpret_cast<uint8_t*>(row_offsets_), row_offsets_size_);
    row_offsets_ = nullptr;
    row_offsets_size_ = 0;
  }
  if (row_vals_) {
    pool_->Free(row_vals_, num_value_bytes_allocated_ + padding_for_SIMD);
    row_vals_ = nullptr;
    num_value_bytes_allocated_ = 0;
  }
  num_rows_allocated_ = 0;
  num_rows_ = 0;
}

void KeyStore::get_rows(uint64_t& num_rows, const uint32_t*& row_offsets,
                        const uint8_t*& row_vals, const uint8_t*& row_nulls,
                        const uint8_t*& row_any_nulls) {
  num_rows = num_rows_;
  row_offsets = row_offsets_;
  row_vals = row_vals_;
  row_nulls = row_nulls_;
  row_any_nulls = row_any_nulls_;
}

Status KeyStore::append(uint32_t num_rows, const uint16_t* selection,
                        const uint32_t* row_offsets, const uint8_t* row_vals,
                        const uint8_t* row_nulls) {
  // Currently append always requires providing a list of selected input rows
  ARROW_DCHECK(selection != nullptr);
  uint64_t new_rows_size;
  if (is_row_fixedlen_) {
    // If the row is fixed len, the caller should have passed common row length as
    // the element pointed to by the argument row_offsets
    ARROW_DCHECK(row_offsets[0] == row_length_);
    new_rows_size = num_rows * row_length_;
  } else {
    // Visit all rows to calculate their total size
    new_rows_size = 0;
    for (uint32_t i = 0; i < num_rows; ++i) {
      uint32_t id = selection[i];
      uint32_t length = row_offsets[id + 1] - row_offsets[id];
      new_rows_size += length;
    }
  }
  RETURN_NOT_OK(resize_buffers_if_needed(num_rows, new_rows_size));

  // Copy values
  if (is_row_fixedlen_) {
    uint8_t* base = row_vals_ + num_rows_ * row_length_;
    for (uint32_t i = 0; i < num_rows; ++i) {
      uint32_t id = selection[i];
      const uint64_t* src =
          reinterpret_cast<const uint64_t*>(row_vals + row_length_ * id);
      uint64_t* dst = reinterpret_cast<uint64_t*>(base);
      for (uint32_t j = 0; j < (row_length_ + 7) / 8; ++j) {
        dst[j] = src[j];
      }
      base += row_length_;
    }
  } else {
    if (num_rows_ == 0) {
      row_offsets_[0] = 0;
    }
    uint8_t* base = row_vals_ + row_offsets_[num_rows_];
    for (uint32_t i = 0; i < num_rows; ++i) {
      uint32_t id = selection[i];
      const uint64_t* src = reinterpret_cast<const uint64_t*>(row_vals + row_offsets[id]);
      uint64_t* dst = reinterpret_cast<uint64_t*>(base);
      uint32_t length = row_offsets[id + 1] - row_offsets[id];
      for (uint32_t j = 0; j < (length + 7) / 8; ++j) {
        dst[j] = src[j];
      }
      base += length;
      row_offsets_[num_rows_ + i + 1] = static_cast<uint32_t>(base - row_vals_);
    }
  }

  // Copy null bits
  uint32_t log_row_null_bits = get_log_row_null_bits(num_cols_);
  uint32_t num_row_null_bits = 1 << log_row_null_bits;
  // Copy process is different for less than a byte of null bits per row and for more than
  // that
  if (log_row_null_bits < 3) {
    uint64_t dst_bit_offset = (num_rows_ << log_row_null_bits);
    for (uint32_t i = 0; i < num_rows; ++i) {
      uint32_t id = selection[i];
      uint64_t bit_offset = (id << log_row_null_bits);
      uint8_t mask = ((1 << num_row_null_bits) - 1);
      uint8_t null_bits = ((row_nulls[bit_offset / 8] >> (bit_offset % 8)) & mask);
      uint8_t current_byte =
          row_nulls_[dst_bit_offset / 8] & ~(mask << (dst_bit_offset % 8));
      row_nulls_[dst_bit_offset / 8] =
          (current_byte | (null_bits << (dst_bit_offset % 8)));
      dst_bit_offset += num_row_null_bits;
      row_any_nulls_[(num_rows_ + i) / 8] |=
          ((null_bits == 0 ? 0 : 1) << ((num_rows_ + i) % 8));
    }
  } else {
    uint32_t byte_length = num_row_null_bits / 8;
    uint64_t dst_byte_offset = num_rows_ * byte_length;
    for (uint32_t i = 0; i < num_rows; ++i) {
      uint32_t id = selection[i];
      uint64_t src_byte_offset = id * byte_length;
      const uint8_t* src = row_nulls + src_byte_offset;
      uint8_t* dst = row_nulls_ + dst_byte_offset;
      uint8_t src_or = 0;
      for (uint32_t ibyte = 0; ibyte < byte_length; ++ibyte) {
        dst[ibyte] = src[ibyte];
        src_or |= src[ibyte];
      }
      dst_byte_offset += byte_length;
      row_any_nulls_[(num_rows_ + i) / 8] |=
          ((src_or == 0 ? 0 : 1) << ((num_rows_ + i) % 8));
    }
  }

  num_rows_ += num_rows;

  return Status::OK();
}

template <bool use_selection>
void KeyCompare::compare_varlen_imp(uint32_t num_rows, const uint16_t* selection,
                                    const uint32_t* offsets_left,
                                    const uint8_t* concatenated_keys_left,
                                    const uint32_t* ids_right,
                                    const uint32_t* offsets_right,
                                    const uint8_t* concatenated_keys_right,
                                    uint8_t* match_bitvector) {
  uint64_t bits = 0ULL;
  for (uint32_t irow = 0; irow < num_rows; ++irow) {
    uint32_t id = use_selection ? selection[irow] : irow;
    uint32_t begin_left = offsets_left[id];
    uint32_t begin_right = offsets_right[ids_right[id]];
    uint32_t length = offsets_left[id + 1] - begin_left;
    uint64_t result = 1;
    uint32_t istripe;
    for (istripe = 0; istripe < (length - 1) / 8; ++istripe) {
      uint64_t key_stripe_left =
          *(reinterpret_cast<const uint64_t*>(concatenated_keys_left + begin_left) +
            istripe);
      uint64_t key_stripe_right =
          *(reinterpret_cast<const uint64_t*>(concatenated_keys_right + begin_right) +
            istripe);
      if (key_stripe_left != key_stripe_right) {
        result = 0;
        // We skip remaining comparisons in case of mismatch,
        // that way we do not have to worry about the lengths of compared values not being
        // equal. Lengths are stored in the prefix of the encoding of key columns, so the
        // implicit comparison of lenghts will happen before comparison of varying length
        // components.
        break;
      }
    }
    if (result) {
      uint64_t key_stripe_left =
          *(reinterpret_cast<const uint64_t*>(concatenated_keys_left + begin_left) +
            istripe);
      uint64_t key_stripe_right =
          *(reinterpret_cast<const uint64_t*>(concatenated_keys_right + begin_right) +
            istripe);
      // In the last 64-bit word mask out the bytes that fall beyond boundaries
      uint64_t mask = ~0ULL >> (8 * (((istripe + 1) * 8) - length));
      if (((key_stripe_left ^ key_stripe_right) & mask) != 0) {
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

template <bool use_selection, int num_64bit_words>
void KeyCompare::compare_fixedlen_imp(uint32_t num_rows, const uint16_t* selection,
                                      uint32_t length,
                                      const uint8_t* concatenated_keys_left,
                                      const uint32_t* ids_right,
                                      const uint8_t* concatenated_keys_right,
                                      uint8_t* match_bitvector) {
  // Key length (for encoded key) has to be non-zero
  ARROW_DCHECK(length > 0);

  // Specializations for keys up to 8 bytes and between 9 and 16 bytes to
  // avoid internal loop over words in the value for short values.
  //

  // Mask out bytes outside the length
  uint64_t mask = ~0ULL >> (8 * (8 - (length % 8)));
  uint64_t bits = 0ULL;
  for (uint32_t i = 0; i < num_rows; ++i) {
    uint32_t id = use_selection ? selection[i] : i;
    const uint64_t key_offset_left = static_cast<uint64_t>(id) * length;
    const uint64_t key_offset_right = static_cast<uint64_t>(ids_right[id]) * length;
    const uint64_t* key_base_left =
        reinterpret_cast<const uint64_t*>(concatenated_keys_left + key_offset_left);
    const uint64_t* key_base_right =
        reinterpret_cast<const uint64_t*>(concatenated_keys_right + key_offset_right);
    uint64_t result = 1;
    uint64_t last_word_left;
    uint64_t last_word_right;
    if (num_64bit_words != 1 && num_64bit_words != 2) {
      // Process all except for last word
      uint32_t iword;
      for (iword = 0; iword < (length - 1) / 8; ++iword) {
        if (key_base_left[iword] != key_base_right[iword]) {
          result = 0;
        }
        // No need (in terms of correctness) for early exit in case of fixed length key
        // comparison
      }
      last_word_left = key_base_left[iword];
      last_word_right = key_base_right[iword];
    }
    if (num_64bit_words == 2) {
      // Same as the loop above but with only a single iteration
      if (key_base_left[0] != key_base_right[0]) {
        result = 0;
      }
      last_word_left = key_base_left[1];
      last_word_right = key_base_right[1];
    }
    if (num_64bit_words == 1) {
      last_word_left = key_base_left[0];
      last_word_right = key_base_right[0];
    }
    result &= (((last_word_left ^ last_word_right) & mask) == 0 ? 1 : 0);
    bits |= (result << (i & 63));
    if ((i & 63) == 63) {
      reinterpret_cast<uint64_t*>(match_bitvector)[i / 64] = bits;
      bits = 0ULL;
    }
  }
  if ((num_rows % 64) != 0) {
    reinterpret_cast<uint64_t*>(match_bitvector)[num_rows / 64] = bits;
  }
}

void KeyCompare::compare_fixedlen(
    util::CPUInstructionSet instruction_set, uint32_t num_rows,
    // Selection should be set to nullptr if no selection is used
    const uint16_t* selection_maybe_null, uint32_t length,
    const uint8_t* concatenated_keys_left, const uint32_t* ids_right,
    const uint8_t* concatenated_keys_right, uint8_t* match_bitvector) {
  if (num_rows == 0) {
    return;
  }
#if defined(ARROW_HAVE_AVX2)
  if (selection_maybe_null == nullptr &&
      instruction_set == util::CPUInstructionSet::avx2) {
    compare_fixedlen_avx2(num_rows, length, concatenated_keys_left, ids_right,
                          concatenated_keys_right, match_bitvector);
  } else {
#endif
    void (*compare_fixedlen_fn[])(uint32_t, const uint16_t*, uint32_t, const uint8_t*,
                                  const uint32_t*, const uint8_t*, uint8_t*) = {
        compare_fixedlen_imp<false, 1>, compare_fixedlen_imp<false, 2>,
        compare_fixedlen_imp<false, 0>, compare_fixedlen_imp<true, 1>,
        compare_fixedlen_imp<true, 2>,  compare_fixedlen_imp<true, 0>};
    int dispatch_const =
        (selection_maybe_null ? 3 : 0) + ((length <= 8) ? 0 : ((length <= 16) ? 1 : 2));
    compare_fixedlen_fn[dispatch_const](num_rows, selection_maybe_null, length,
                                        concatenated_keys_left, ids_right,
                                        concatenated_keys_right, match_bitvector);
#if defined(ARROW_HAVE_AVX2)
  }
#endif
}

void KeyCompare::compare_varlen(
    util::CPUInstructionSet instruction_set, uint32_t num_rows,
    // Selection should be set to nullptr if no selection is used
    const uint16_t* selection_maybe_null, const uint32_t* offsets_left,
    const uint8_t* concatenated_keys_left, const uint32_t* ids_right,
    const uint32_t* offsets_right, const uint8_t* concatenated_keys_right,
    uint8_t* match_bitvector) {
  if (num_rows == 0) {
    return;
  }
#if defined(ARROW_HAVE_AVX2)
  if (selection_maybe_null == nullptr &&
      instruction_set == util::CPUInstructionSet::avx2) {
    compare_varlen_avx2(num_rows, offsets_left, concatenated_keys_left, ids_right,
                        offsets_right, concatenated_keys_right, match_bitvector);
  } else {
#endif
    if (selection_maybe_null) {
      compare_varlen_imp<true>(num_rows, selection_maybe_null, offsets_left,
                               concatenated_keys_left, ids_right, offsets_right,
                               concatenated_keys_right, match_bitvector);
    } else {
      compare_varlen_imp<false>(num_rows, nullptr, offsets_left, concatenated_keys_left,
                                ids_right, offsets_right, concatenated_keys_right,
                                match_bitvector);
    }
#if defined(ARROW_HAVE_AVX2)
  }
#endif
}

void KeyCompare::update_comparison_for_nulls(
    util::CPUInstructionSet instruction_set, uint32_t num_rows,
    const uint16_t* selection_maybe_null, uint32_t num_cols,
    const uint8_t* any_nulls_left, const uint8_t* nulls_left, const uint32_t* ids_right,
    const uint8_t* any_nulls_right, const uint8_t* nulls_right,
    // Input match bit vector should already be initialized, possibly as a result of key
    // value comparison. Appropriate bits will be cleared if null information does not
    // match for a pair of input rows.
    uint8_t* match_bitvector, uint8_t* temp_vector_8bit, uint16_t* temp_vector_16bit) {
  if (!selection_maybe_null) {
    util::BitUtil::bit_vector_lookup(instruction_set, num_rows, ids_right,
                                     any_nulls_right, temp_vector_8bit);
    // Clear result bit vector whenever left and right any null bits do not match
    //
    for (uint32_t i = 0; i < num_rows / 64; ++i) {
      uint64_t not_equal = reinterpret_cast<const uint64_t*>(temp_vector_8bit)[i] ^
                           reinterpret_cast<const uint64_t*>(any_nulls_left)[i];
      reinterpret_cast<uint64_t*>(match_bitvector)[i] &= ~not_equal;
      if (num_cols > 1) {
        reinterpret_cast<uint64_t*>(temp_vector_8bit)[i] &= ~not_equal;
      }
    }
  } else {
    uint8_t no_match_bits = 0;
    uint8_t temp_bits = 0;
    for (uint32_t i = 0; i < num_rows; ++i) {
      uint32_t id_left = selection_maybe_null[i];
      uint32_t id_right = ids_right[id_left];
      int bit_left = ((any_nulls_left[id_left / 8] >> (id_left % 8)) & 1);
      int bit_right = ((any_nulls_right[id_right / 8] >> (id_right % 8)) & 1);
      no_match_bits |= ((bit_left ^ bit_right) << (i % 8));
      temp_bits |= ((bit_left & bit_right) << (i % 8));
      if ((i % 8) == 7) {
        match_bitvector[i / 8] &= ~no_match_bits;
        temp_vector_8bit[i / 8] = temp_bits;
        temp_bits = 0;
      }
    }
    if ((num_rows % 8) > 0) {
      match_bitvector[num_rows / 8] &= ~no_match_bits;
      temp_vector_8bit[num_rows / 8] = temp_bits;
    }
  }
  if (num_cols > 1) {
    uint32_t log_row_null_bits = KeyStore::get_log_row_null_bits(num_cols);
    // Compare nulls for all columns for each row that contains any nulls
    int num_ids;
    if (!selection_maybe_null) {
      util::BitUtil::bits_to_indexes<1>(instruction_set, num_rows, temp_vector_8bit,
                                        num_ids, temp_vector_16bit);
    } else {
      util::BitUtil::bits_filter_indexes<1>(instruction_set, num_rows, temp_vector_8bit,
                                            selection_maybe_null, num_ids,
                                            temp_vector_16bit);
    }
    if (log_row_null_bits < 8) {
      const uint8_t mask = ((1 << log_row_null_bits) - 1);
      for (int i = 0; i < num_ids; ++i) {
        uint32_t left_row_id = temp_vector_16bit[i];
        uint32_t right_row_id = ids_right[left_row_id];
        uint8_t right_bits = (nulls_right[(right_row_id << log_row_null_bits) / 8] >>
                              ((right_row_id << log_row_null_bits) & 7)) &
                             mask;
        uint8_t left_bits = ((nulls_left[(left_row_id << log_row_null_bits) / 8] >>
                              ((left_row_id << log_row_null_bits) & 7)) &
                             mask);
        uint8_t no_match = (right_bits == left_bits) ? 0 : 1;
        match_bitvector[left_row_id / 8] &= ~(no_match << (left_row_id & 7));
      }
    } else {
      for (int i = 0; i < num_ids; ++i) {
        uint32_t left_row_id = temp_vector_16bit[i];
        uint32_t right_row_id = ids_right[left_row_id];
        for (uint32_t j = 0; j < (num_cols + 7) / 8; ++j) {
          uint8_t right_bits = nulls_right[(right_row_id << log_row_null_bits) / 8 + j];
          uint8_t left_bits = nulls_left[(left_row_id << log_row_null_bits) / 8 + j];
          uint8_t no_match = (right_bits == left_bits) ? 0 : 1;
          match_bitvector[left_row_id / 8] &= ~(no_match << (left_row_id & 7));
        }
      }
    }
  }
}

void KeyLength::compute_offsets(util::CPUInstructionSet instruction_set, int num_columns,
                                int num_rows, uint32_t row_fixed_len,
                                const uint32_t** offsets, const uint8_t** non_nulls,
                                const uint8_t* any_nulls_bitvector,
                                uint32_t* row_offsets) {
  int64_t offset_max =
      static_cast<int64_t>(row_fixed_len) * static_cast<int64_t>(num_rows);
  for (int col = 0; col < num_columns; ++col) {
    offset_max += static_cast<int64_t>(offsets[col][num_rows]);
  }
  ARROW_DCHECK((offset_max >> 32) == 0);
  // TODO: error out if 32-bit offsets are too small

  // AVX2 version uses 32-bit arithmetic and can only be used if maximum offset fits
  // 32-bits
  int num_processed = 0;
#if defined(ARROW_HAVE_AVX2)
  if (instruction_set == util::CPUInstructionSet::avx2 && (offset_max >> 32) == 0) {
    constexpr int unroll_avx2 = 8;
    int tail = num_rows % unroll_avx2;
    compute_offsets_avx2(num_columns, num_rows - tail, row_fixed_len, offsets, non_nulls,
                         any_nulls_bitvector, row_offsets);
    num_processed = num_rows - tail;
  }
#endif
  if (num_processed < num_rows) {
    int64_t row_offset_adjustment = 0;
    for (int col = 0; col < num_columns; ++col) {
      row_offset_adjustment -= static_cast<int64_t>(offsets[col][num_processed]);
    }
    if (num_processed > 0) {
      row_offset_adjustment += static_cast<int64_t>(row_offsets[num_processed]);
    }
    while (num_processed < num_rows) {
      int64_t row_offset = 0;
      for (int col = 0; col < num_columns; ++col) {
        row_offset += static_cast<int64_t>(offsets[col][num_processed + 1]);
      }
      if (((any_nulls_bitvector[num_processed / 8] >> (num_processed & 7)) & 1) != 0) {
        // Subtract length corresponding to null values
        for (int col = 0; col < num_columns; ++col) {
          int64_t is_null =
              1 - ((non_nulls[col][num_processed / 8] >> (num_processed & 7)) & 1);
          row_offset_adjustment -=
              is_null * (static_cast<int64_t>(offsets[col][num_processed + 1]) -
                         static_cast<int64_t>(offsets[col][num_processed]));
        }
      }
      row_offset_adjustment += row_fixed_len;
      row_offset += row_offset_adjustment;
      // This is where we cast 64-bit row offset to a 32-bit value - output data type
      // needs to change to support batches larger than 4GB.
      ARROW_DCHECK((row_offset >> 32) == 0);
      row_offsets[num_processed + 1] = static_cast<uint32_t>(row_offset);
      ++num_processed;
    }
  }
}

void KeyTranspose::col2row_short(uint32_t num_rows, const uint32_t col_length,
                                 const uint8_t* col_vals, bool is_row_fixedlen,
                                 const uint32_t* row_offsets, uint8_t* row_vals) {
  if (is_row_fixedlen && col_length == row_offsets[0]) {
    memcpy(row_vals, col_vals, col_length * num_rows);
    return;
  }
  if (is_row_fixedlen) {
    uint32_t row_length = row_offsets[0];
    switch (col_length) {
      case 1:
        for (uint32_t i = 0; i < num_rows; ++i) {
          row_vals[i * row_length] = col_vals[i];
        }
        break;
      case 2:
        for (uint32_t i = 0; i < num_rows; ++i) {
          *reinterpret_cast<uint16_t*>(row_vals + i * row_length) =
              reinterpret_cast<const uint16_t*>(col_vals)[i];
        }
        break;
      case 4:
        for (uint32_t i = 0; i < num_rows; ++i) {
          *reinterpret_cast<uint32_t*>(row_vals + i * row_length) =
              reinterpret_cast<const uint32_t*>(col_vals)[i];
        }
        break;
      case 8:
        for (uint32_t i = 0; i < num_rows; ++i) {
          *reinterpret_cast<uint64_t*>(row_vals + i * row_length) =
              reinterpret_cast<const uint64_t*>(col_vals)[i];
        }
        break;
    }
  } else {
    switch (col_length) {
      case 1:
        for (uint32_t i = 0; i < num_rows; ++i) {
          row_vals[row_offsets[i]] = col_vals[i];
        }
        break;
      case 2:
        for (uint32_t i = 0; i < num_rows; ++i) {
          *reinterpret_cast<uint16_t*>(row_vals + row_offsets[i]) =
              reinterpret_cast<const uint16_t*>(col_vals)[i];
        }
        break;
      case 4:
        for (uint32_t i = 0; i < num_rows; ++i) {
          *reinterpret_cast<uint32_t*>(row_vals + row_offsets[i]) =
              reinterpret_cast<const uint32_t*>(col_vals)[i];
        }
        break;
      case 8:
        for (uint32_t i = 0; i < num_rows; ++i) {
          *reinterpret_cast<uint64_t*>(row_vals + row_offsets[i]) =
              reinterpret_cast<const uint64_t*>(col_vals)[i];
        }
        break;
    }
  }
}

void KeyTranspose::row2col_short(uint32_t num_rows, const uint32_t col_length,
                                 uint8_t* col_vals, bool is_row_fixedlen,
                                 const uint32_t* row_offsets, const uint8_t* row_vals) {
  if (is_row_fixedlen && col_length == row_offsets[0]) {
    memcpy(col_vals, row_vals, col_length * num_rows);
    return;
  }
  if (is_row_fixedlen) {
    uint32_t row_length = row_offsets[0];
    switch (col_length) {
      case 1:
        for (uint32_t i = 0; i < num_rows; ++i) {
          col_vals[i] = row_vals[i * row_length];
        }
        break;
      case 2:
        for (uint32_t i = 0; i < num_rows; ++i) {
          reinterpret_cast<uint16_t*>(col_vals)[i] =
              *reinterpret_cast<const uint16_t*>(row_vals + i * row_length);
        }
        break;
      case 4:
        for (uint32_t i = 0; i < num_rows; ++i) {
          reinterpret_cast<uint32_t*>(col_vals)[i] =
              *reinterpret_cast<const uint32_t*>(row_vals + i * row_length);
        }
        break;
      case 8:
        for (uint32_t i = 0; i < num_rows; ++i) {
          reinterpret_cast<uint64_t*>(col_vals)[i] =
              *reinterpret_cast<const uint64_t*>(row_vals + i * row_length);
        }
        break;
    }
  } else {
    switch (col_length) {
      case 1:
        for (uint32_t i = 0; i < num_rows; ++i) {
          col_vals[i] = row_vals[row_offsets[i]];
        }
        break;
      case 2:
        for (uint32_t i = 0; i < num_rows; ++i) {
          reinterpret_cast<uint16_t*>(col_vals)[i] =
              *reinterpret_cast<const uint16_t*>(row_vals + row_offsets[i]);
        }
        break;
      case 4:
        for (uint32_t i = 0; i < num_rows; ++i) {
          reinterpret_cast<uint32_t*>(col_vals)[i] =
              *reinterpret_cast<const uint32_t*>(row_vals + row_offsets[i]);
        }
        break;
      case 8:
        for (uint32_t i = 0; i < num_rows; ++i) {
          reinterpret_cast<uint64_t*>(col_vals)[i] =
              *reinterpret_cast<const uint64_t*>(row_vals + row_offsets[i]);
        }
        break;
    }
  }
}

template <bool is_row_fixedlen, int col_len_A, int col_len_B>
void KeyTranspose::col2row_shortpair_impl(uint32_t num_rows, const uint8_t* col_vals_A,
                                          const uint8_t* col_vals_B,
                                          const uint32_t* row_offsets,
                                          uint8_t* row_vals) {
  for (uint32_t i = 0; i < num_rows; ++i) {
    uint8_t* dst = row_vals + (is_row_fixedlen ? row_offsets[0] * i : row_offsets[i]);
    if (col_len_A == 1) {
      *dst = col_vals_A[i];
    } else if (col_len_A == 2) {
      *reinterpret_cast<uint16_t*>(dst) =
          reinterpret_cast<const uint16_t*>(col_vals_A)[i];
    } else if (col_len_A == 4) {
      *reinterpret_cast<uint32_t*>(dst) =
          reinterpret_cast<const uint32_t*>(col_vals_A)[i];
    } else if (col_len_A == 8) {
      *reinterpret_cast<uint64_t*>(dst) =
          reinterpret_cast<const uint64_t*>(col_vals_A)[i];
    }
    if (col_len_B == 1) {
      *(dst + col_len_A) = col_vals_B[i];
    } else if (col_len_B == 2) {
      *reinterpret_cast<uint16_t*>(dst + col_len_A) =
          reinterpret_cast<const uint16_t*>(col_vals_B)[i];
    } else if (col_len_B == 4) {
      *reinterpret_cast<uint32_t*>(dst + col_len_A) =
          reinterpret_cast<const uint32_t*>(col_vals_B)[i];
    } else if (col_len_B == 8) {
      *reinterpret_cast<uint64_t*>(dst + col_len_A) =
          reinterpret_cast<const uint64_t*>(col_vals_B)[i];
    }
  }
}
// These template specializations are used outside of this file
template void KeyTranspose::col2row_shortpair_impl<false, 1, 1>(uint32_t, const uint8_t*,
                                                                const uint8_t*,
                                                                const uint32_t*,
                                                                uint8_t*);
template void KeyTranspose::col2row_shortpair_impl<false, 2, 2>(uint32_t, const uint8_t*,
                                                                const uint8_t*,
                                                                const uint32_t*,
                                                                uint8_t*);
template void KeyTranspose::col2row_shortpair_impl<false, 4, 4>(uint32_t, const uint8_t*,
                                                                const uint8_t*,
                                                                const uint32_t*,
                                                                uint8_t*);
template void KeyTranspose::col2row_shortpair_impl<false, 8, 8>(uint32_t, const uint8_t*,
                                                                const uint8_t*,
                                                                const uint32_t*,
                                                                uint8_t*);
template void KeyTranspose::col2row_shortpair_impl<true, 1, 1>(uint32_t, const uint8_t*,
                                                               const uint8_t*,
                                                               const uint32_t*, uint8_t*);
template void KeyTranspose::col2row_shortpair_impl<true, 2, 2>(uint32_t, const uint8_t*,
                                                               const uint8_t*,
                                                               const uint32_t*, uint8_t*);
template void KeyTranspose::col2row_shortpair_impl<true, 4, 4>(uint32_t, const uint8_t*,
                                                               const uint8_t*,
                                                               const uint32_t*, uint8_t*);
template void KeyTranspose::col2row_shortpair_impl<true, 8, 8>(uint32_t, const uint8_t*,
                                                               const uint8_t*,
                                                               const uint32_t*, uint8_t*);

template <bool is_row_fixedlen, int col_len_A, int col_len_B>
void KeyTranspose::row2col_shortpair_impl(uint32_t num_rows, uint8_t* col_vals_A,
                                          uint8_t* col_vals_B,
                                          const uint32_t* row_offsets,
                                          const uint8_t* row_vals) {
  for (uint32_t i = 0; i < num_rows; ++i) {
    const uint8_t* src =
        row_vals + (is_row_fixedlen ? row_offsets[0] * i : row_offsets[i]);
    if (col_len_A == 1) {
      col_vals_A[i] = *src;
    } else if (col_len_A == 2) {
      reinterpret_cast<uint16_t*>(col_vals_A)[i] =
          *reinterpret_cast<const uint16_t*>(src);
    } else if (col_len_A == 4) {
      reinterpret_cast<uint32_t*>(col_vals_A)[i] =
          *reinterpret_cast<const uint32_t*>(src);
    } else if (col_len_A == 8) {
      reinterpret_cast<uint64_t*>(col_vals_A)[i] =
          *reinterpret_cast<const uint64_t*>(src);
    }
    if (col_len_B == 1) {
      col_vals_B[i] = *(src + col_len_A);
    } else if (col_len_B == 2) {
      reinterpret_cast<uint16_t*>(col_vals_B)[i] =
          *reinterpret_cast<const uint16_t*>(src + col_len_A);
    } else if (col_len_B == 4) {
      reinterpret_cast<uint32_t*>(col_vals_B)[i] =
          *reinterpret_cast<const uint32_t*>(src + col_len_A);
    } else if (col_len_B == 8) {
      reinterpret_cast<uint64_t*>(col_vals_B)[i] =
          *reinterpret_cast<const uint64_t*>(src + col_len_A);
    }
  }
}
// These template specializations are used outside of this file
template void KeyTranspose::row2col_shortpair_impl<false, 1, 1>(uint32_t, uint8_t*,
                                                                uint8_t*, const uint32_t*,
                                                                const uint8_t*);
template void KeyTranspose::row2col_shortpair_impl<false, 2, 2>(uint32_t, uint8_t*,
                                                                uint8_t*, const uint32_t*,
                                                                const uint8_t*);
template void KeyTranspose::row2col_shortpair_impl<false, 4, 4>(uint32_t, uint8_t*,
                                                                uint8_t*, const uint32_t*,
                                                                const uint8_t*);
template void KeyTranspose::row2col_shortpair_impl<false, 8, 8>(uint32_t, uint8_t*,
                                                                uint8_t*, const uint32_t*,
                                                                const uint8_t*);
template void KeyTranspose::row2col_shortpair_impl<true, 1, 1>(uint32_t, uint8_t*,
                                                               uint8_t*, const uint32_t*,
                                                               const uint8_t*);
template void KeyTranspose::row2col_shortpair_impl<true, 2, 2>(uint32_t, uint8_t*,
                                                               uint8_t*, const uint32_t*,
                                                               const uint8_t*);
template void KeyTranspose::row2col_shortpair_impl<true, 4, 4>(uint32_t, uint8_t*,
                                                               uint8_t*, const uint32_t*,
                                                               const uint8_t*);
template void KeyTranspose::row2col_shortpair_impl<true, 8, 8>(uint32_t, uint8_t*,
                                                               uint8_t*, const uint32_t*,
                                                               const uint8_t*);

void KeyTranspose::col2row_shortpair(util::CPUInstructionSet instruction_set,
                                     bool is_row_fixedlen, int col_len_A, int col_len_B,
                                     uint32_t num_rows, const uint8_t* col_vals_A,
                                     const uint8_t* col_vals_B,
                                     const uint32_t* row_offsets, uint8_t* row_vals) {
  int log_col_len_A = col_len_A;
  log_col_len_A =
      log_col_len_A == 8 ? 3 : log_col_len_A == 4 ? 2 : log_col_len_A == 2 ? 1 : 0;
  int log_col_len_B = col_len_B;
  log_col_len_B =
      log_col_len_B == 8 ? 3 : log_col_len_B == 4 ? 2 : log_col_len_B == 2 ? 1 : 0;
  int dispatch_const = (log_col_len_B << 2) | log_col_len_A;

#if defined(ARROW_HAVE_AVX2)
  if (instruction_set == util::CPUInstructionSet::avx2 && col_len_A == col_len_B) {
    void (*col2row_shortpair_fn[])(uint32_t, const uint32_t, const uint8_t*,
                                   const uint8_t*, const uint32_t*, uint8_t*) = {
        col2row_shortpair_avx2<false>, col2row_shortpair_avx2<true>};
    col2row_shortpair_fn[is_row_fixedlen ? 1 : 0](num_rows, col_len_A, col_vals_A,
                                                  col_vals_B, row_offsets, row_vals);
  } else {
#endif
    void (*col2row_shortpair_fn[])(uint32_t, const uint8_t*, const uint8_t*,
                                   const uint32_t*, uint8_t*) = {
        col2row_shortpair_impl<false, 1, 1>, col2row_shortpair_impl<false, 2, 1>,
        col2row_shortpair_impl<false, 4, 1>, col2row_shortpair_impl<false, 8, 1>,
        col2row_shortpair_impl<false, 1, 2>, col2row_shortpair_impl<false, 2, 2>,
        col2row_shortpair_impl<false, 4, 2>, col2row_shortpair_impl<false, 8, 2>,
        col2row_shortpair_impl<false, 1, 4>, col2row_shortpair_impl<false, 2, 4>,
        col2row_shortpair_impl<false, 4, 4>, col2row_shortpair_impl<false, 8, 4>,
        col2row_shortpair_impl<false, 1, 8>, col2row_shortpair_impl<false, 2, 8>,
        col2row_shortpair_impl<false, 4, 8>, col2row_shortpair_impl<false, 8, 8>,
        col2row_shortpair_impl<true, 1, 1>,  col2row_shortpair_impl<true, 2, 1>,
        col2row_shortpair_impl<true, 4, 1>,  col2row_shortpair_impl<true, 8, 1>,
        col2row_shortpair_impl<true, 1, 2>,  col2row_shortpair_impl<true, 2, 2>,
        col2row_shortpair_impl<true, 4, 2>,  col2row_shortpair_impl<true, 8, 2>,
        col2row_shortpair_impl<true, 1, 4>,  col2row_shortpair_impl<true, 2, 4>,
        col2row_shortpair_impl<true, 4, 4>,  col2row_shortpair_impl<true, 8, 4>,
        col2row_shortpair_impl<true, 1, 8>,  col2row_shortpair_impl<true, 2, 8>,
        col2row_shortpair_impl<true, 4, 8>,  col2row_shortpair_impl<true, 8, 8>};
    col2row_shortpair_fn[dispatch_const + (is_row_fixedlen ? 16 : 0)](
        num_rows, col_vals_A, col_vals_B, row_offsets, row_vals);
#if defined(ARROW_HAVE_AVX2)
  }
#endif
}

void KeyTranspose::row2col_shortpair(util::CPUInstructionSet instruction_set,
                                     bool is_row_fixedlen, int col_len_A, int col_len_B,
                                     uint32_t num_rows, uint8_t* col_vals_A,
                                     uint8_t* col_vals_B, const uint32_t* row_offsets,
                                     const uint8_t* row_vals) {
  int log_col_len_A = col_len_A;
  log_col_len_A =
      log_col_len_A == 8 ? 3 : log_col_len_A == 4 ? 2 : log_col_len_A == 2 ? 1 : 0;
  int log_col_len_B = col_len_B;
  log_col_len_B =
      log_col_len_B == 8 ? 3 : log_col_len_B == 4 ? 2 : log_col_len_B == 2 ? 1 : 0;
  int dispatch_const = (log_col_len_B << 2) | log_col_len_A;

#if defined(ARROW_HAVE_AVX2)
  if (instruction_set == util::CPUInstructionSet::avx2 && col_len_A == col_len_B) {
    void (*row2col_shortpair_fn[])(uint32_t, const uint32_t, uint8_t*, uint8_t*,
                                   const uint32_t*, const uint8_t*) = {
        row2col_shortpair_avx2<false>, row2col_shortpair_avx2<true>};
    row2col_shortpair_fn[is_row_fixedlen ? 1 : 0](num_rows, col_len_A, col_vals_A,
                                                  col_vals_B, row_offsets, row_vals);
  } else {
#endif
    void (*row2col_shortpair_fn[])(uint32_t, uint8_t*, uint8_t*, const uint32_t*,
                                   const uint8_t*) = {
        row2col_shortpair_impl<false, 1, 1>, row2col_shortpair_impl<false, 2, 1>,
        row2col_shortpair_impl<false, 4, 1>, row2col_shortpair_impl<false, 8, 1>,
        row2col_shortpair_impl<false, 1, 2>, row2col_shortpair_impl<false, 2, 2>,
        row2col_shortpair_impl<false, 4, 2>, row2col_shortpair_impl<false, 8, 2>,
        row2col_shortpair_impl<false, 1, 4>, row2col_shortpair_impl<false, 2, 4>,
        row2col_shortpair_impl<false, 4, 4>, row2col_shortpair_impl<false, 8, 4>,
        row2col_shortpair_impl<false, 1, 8>, row2col_shortpair_impl<false, 2, 8>,
        row2col_shortpair_impl<false, 4, 8>, row2col_shortpair_impl<false, 8, 8>,
        row2col_shortpair_impl<true, 1, 1>,  row2col_shortpair_impl<true, 2, 1>,
        row2col_shortpair_impl<true, 4, 1>,  row2col_shortpair_impl<true, 8, 1>,
        row2col_shortpair_impl<true, 1, 2>,  row2col_shortpair_impl<true, 2, 2>,
        row2col_shortpair_impl<true, 4, 2>,  row2col_shortpair_impl<true, 8, 2>,
        row2col_shortpair_impl<true, 1, 4>,  row2col_shortpair_impl<true, 2, 4>,
        row2col_shortpair_impl<true, 4, 4>,  row2col_shortpair_impl<true, 8, 4>,
        row2col_shortpair_impl<true, 1, 8>,  row2col_shortpair_impl<true, 2, 8>,
        row2col_shortpair_impl<true, 4, 8>,  row2col_shortpair_impl<true, 8, 8>};
    row2col_shortpair_fn[dispatch_const + (is_row_fixedlen ? 16 : 0)](
        num_rows, col_vals_A, col_vals_B, row_offsets, row_vals);
#if defined(ARROW_HAVE_AVX2)
  }
#endif
}

template <bool is_col_fixedlen, bool is_row_fixedlen, bool update_row_offsets>
void KeyTranspose::col2row_long(uint32_t num_rows, const uint8_t* col_non_nulls,
                                const uint32_t* col_offsets, const uint8_t* col_vals,
                                const uint32_t* row_offsets, uint8_t* row_vals,
                                uint32_t* row_offsets_updated) {
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
    const uint64_t* src = reinterpret_cast<const uint64_t*>(col_vals + col_offset);
    uint32_t row_offset;
    if (is_row_fixedlen) {
      row_offset = row_fixed_len * i;
    } else {
      row_offset = row_offsets[i];
    }
    uint64_t* dst = reinterpret_cast<uint64_t*>(row_vals + row_offset);
    uint32_t istripe;
    for (istripe = 0; istripe < length / 8; ++istripe) {
      dst[istripe] = src[istripe];
    }
    if ((length % 8) > 0) {
      uint64_t mask_last = ~0ULL >> (8 * (8 * (istripe + 1) - length));
      dst[istripe] = (dst[istripe] & ~mask_last) | (src[istripe] & mask_last);
    }
    if (update_row_offsets) {
      row_offsets_updated[i] = row_offset + length;
    }
  }
}

template <bool is_col_fixedlen, bool is_row_fixedlen, bool update_row_offsets>
void KeyTranspose::row2col_long(uint32_t num_rows, const uint32_t* col_offsets,
                                uint8_t* col_vals, const uint32_t* row_offsets,
                                const uint8_t* row_vals, uint32_t* row_offsets_updated) {
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
    uint64_t* dst = reinterpret_cast<uint64_t*>(col_vals + col_offset);
    uint32_t row_offset;
    if (is_row_fixedlen) {
      row_offset = row_fixed_len * i;
    } else {
      row_offset = row_offsets[i];
    }
    const uint64_t* src = reinterpret_cast<const uint64_t*>(row_vals + row_offset);
    uint32_t istripe;
    for (istripe = 0; istripe < (length + 7) / 8; ++istripe) {
      dst[istripe] = src[istripe];
    }
    if (update_row_offsets) {
      row_offsets_updated[i] = row_offset + length;
    }
  }
}

void KeyTranspose::offsets_to_lengths(util::CPUInstructionSet instruction_set,
                                      uint32_t num_rows, const uint32_t* offsets,
                                      uint32_t* lengths) {
  uint32_t num_processed = 0;
#if defined(ARROW_HAVE_AVX2)
  if (instruction_set == util::CPUInstructionSet::avx2) {
    // The function call below processes whole 8 rows together.
    num_processed = num_rows - (num_rows % 8);
    offsets_to_lengths_avx2(num_processed, offsets, lengths);
  }
#endif
  for (uint32_t i = num_processed; i < num_rows; ++i) {
    uint32_t length = offsets[i + 1] - offsets[i];
    lengths[i] = length;
  }
}

void KeyTranspose::lengths_to_offsets(uint32_t num_rows, uint32_t first_offset,
                                      const uint32_t* lengths, uint32_t* offsets) {
  uint32_t offset = first_offset;
  for (uint32_t i = 0; i < num_rows; ++i) {
    uint32_t length = lengths[i];
    offsets[i] = offset;
    offset += length;
  }
  offsets[num_rows] = offset;
}

template <bool is_row_fixedlen>
void KeyTranspose::memset_selection_of_values(int col_len, uint32_t num_rows,
                                              const uint16_t* row_ids,
                                              const uint32_t* row_offsets, uint8_t* rows,
                                              uint8_t byte_value) {
  if (col_len == 1) {
    for (uint32_t i = 0; i < num_rows; ++i) {
      uint8_t* row_base = rows + (is_row_fixedlen ? row_ids[i] * row_offsets[0]
                                                  : row_offsets[row_ids[i]]);
      *row_base = byte_value;
    }
  } else if (col_len == 2) {
    uint16_t value = (static_cast<uint16_t>(byte_value) * 0x0101);
    for (uint32_t i = 0; i < num_rows; ++i) {
      uint8_t* row_base = rows + (is_row_fixedlen ? row_ids[i] * row_offsets[0]
                                                  : row_offsets[row_ids[i]]);
      *reinterpret_cast<uint16_t*>(row_base) = value;
    }
  } else if (col_len == 4) {
    uint32_t value = (static_cast<uint32_t>(byte_value) * 0x01010101UL);
    for (uint32_t i = 0; i < num_rows; ++i) {
      uint8_t* row_base = rows + (is_row_fixedlen ? row_ids[i] * row_offsets[0]
                                                  : row_offsets[row_ids[i]]);
      *reinterpret_cast<uint32_t*>(row_base) = value;
    }
  } else if (col_len == 8) {
    uint64_t value = (static_cast<uint64_t>(byte_value) * 0x0101010101010101ULL);
    for (uint32_t i = 0; i < num_rows; ++i) {
      uint8_t* row_base = rows + (is_row_fixedlen ? row_ids[i] * row_offsets[0]
                                                  : row_offsets[row_ids[i]]);
      *reinterpret_cast<uint64_t*>(row_base) = value;
    }
  } else {
    uint64_t value = (static_cast<uint64_t>(byte_value) * 0x0101010101010101ULL);
    for (uint32_t i = 0; i < num_rows; ++i) {
      uint8_t* row_base = rows + (is_row_fixedlen ? row_ids[i] * row_offsets[0]
                                                  : row_offsets[row_ids[i]]);
      int j;
      for (j = 0; j < col_len / 8; ++j) {
        reinterpret_cast<uint64_t*>(row_base)[j] = value;
      }
      int tail = col_len % 8;
      if (tail) {
        uint64_t mask = ~0ULL >> (8 * (8 - tail));
        reinterpret_cast<uint64_t*>(row_base)[j] =
            (reinterpret_cast<const uint64_t*>(row_base)[j] & ~mask) | (value & mask);
      }
    }
  }
}

void KeyTranspose::col2row_null_bitvector(int col_id, int num_cols,
                                          uint32_t col_num_bits_set,
                                          const uint16_t* col_bits_set_ids,
                                          uint8_t* row_oriented_bits) {
  int log_row_bits = 0;
  while (num_cols > (1 << log_row_bits)) {
    ++log_row_bits;
  }
  for (uint32_t i = 0; i < col_num_bits_set; ++i) {
    uint16_t row_id = col_bits_set_ids[i];
    uint64_t output_bit_id = col_id + (row_id << log_row_bits);
    row_oriented_bits[output_bit_id / 8] |= (1 << (output_bit_id & 7));
  }
}

void KeyTranspose::row2col_null_bitvector(util::CPUInstructionSet instruction_set,
                                          uint64_t num_rows, int num_cols,
                                          uint8_t** col_non_null_bitvectors,
                                          const uint8_t* row_oriented_bits,
                                          int temp_vector_length,
                                          uint16_t* temp_vector_16) {
  ARROW_DCHECK(temp_vector_length >= 8);
  int log_row_bits = 0;
  while (num_cols > (1 << log_row_bits)) {
    ++log_row_bits;
  }
  for (int col = 0; col < num_cols; ++col) {
    memset(col_non_null_bitvectors[col], 0xff, num_rows / 8);
    if ((num_rows % 8) > 0) {
      col_non_null_bitvectors[col][num_rows / 8] = (1 << (num_rows % 8)) - 1;
    }
  }
  uint64_t local_batch_size_max = temp_vector_length - (temp_vector_length % 8);
  for (uint64_t base = 0; base < (num_rows << log_row_bits);) {
    uint64_t local_batch_size =
        std::min((num_rows << log_row_bits) - base, local_batch_size_max);
    int num_bits_set;
    util::BitUtil::bits_to_indexes(instruction_set, static_cast<int>(local_batch_size),
                                   row_oriented_bits + (base / 8), num_bits_set,
                                   temp_vector_16);
    for (int i = 0; i < num_bits_set; ++i) {
      uint64_t bit_id = temp_vector_16[i] + base;
      uint64_t row_id = (bit_id >> log_row_bits);
      uint64_t col_id = (bit_id & ((1 << log_row_bits) - 1));
      col_non_null_bitvectors[col_id][row_id / 8] &= ~(1 << (row_id & 7));
    }

    base += local_batch_size;
  }
}

void KeyTranspose::col2row(
    util::CPUInstructionSet instruction_set, uint32_t num_cols, uint32_t num_rows,
    // Column widths for all columns.
    // Undefined for varying length column (identified by offset vector not being null).
    // Column width of 0 means 1 bit per row.
    // Column widths are not restricted to powers of 2.
    const uint32_t* col_widths,
    // Bit vectors marking non-null values in columns cannot be null.
    // A constant bit vector with all bits set to 1 needs to be provided, even if the
    // input column does not specify one.
    const uint8_t** col_non_nulls, const uint32_t** col_offsets_maybe_missing,
    const uint8_t** col_vals,
    // In case of varying length rows - byte offsets for output rows with extra element
    // indicating total size of rows. In case of fixed length rows - there is only one
    // element that indicates row size in bytes. If any of the input columns has non null
    // offsets array pointer the row is varying length and otherwise it is fixed length.
    const uint32_t* row_offsets, uint8_t* row_vals,
    // Bit vector marking null values in any of the columns with 1.
    // Bits are organized in row oriented way (all null bits for the first row are
    // followed by all null bits for the second). It uses ceil(log2(num_cols)) bits per
    // row.
    uint8_t* row_nulls,
    // Temporary vectors with at least num_rows 32-bit elements with additional reserved
    // at least 32 bytes at the end, allocated and provided by the caller.
    uint32_t* temp_vector_32_A, uint32_t* temp_vector_32_B) {
  // 1. Find out if we have fixed length row or varying length row.
  //
  bool is_row_fixedlen = true;
  for (uint32_t icol = 0; icol < num_cols; ++icol) {
    if (col_offsets_maybe_missing[icol]) {
      is_row_fixedlen = false;
    }
  }

  // 2. Process all fixed width columns
  //
  uint32_t row_fixed_offset = 0;
  for (uint32_t icol = 0; icol < num_cols;) {
    const uint8_t* col_vals_modified_A = col_vals[icol];
    uint32_t col_width_modified_A = col_widths[icol];
    // Prepare bit vector column
    if (col_widths[icol] == 0) {
      util::BitUtil::bits_to_bytes(instruction_set, num_rows, col_vals[icol],
                                   reinterpret_cast<uint8_t*>(temp_vector_32_A));
      col_vals_modified_A = reinterpret_cast<const uint8_t*>(temp_vector_32_A);
      col_width_modified_A = 1;
    }
    // Prepare length column
    if (col_offsets_maybe_missing[icol]) {
      offsets_to_lengths(instruction_set, num_rows, col_offsets_maybe_missing[icol],
                         temp_vector_32_A);
      col_vals_modified_A = reinterpret_cast<const uint8_t*>(temp_vector_32_A);
    }
    bool is_short_A = col_widths[icol] == 0 || col_widths[icol] == 1 ||
                      col_widths[icol] == 2 || col_widths[icol] == 4 ||
                      col_widths[icol] == 8;
    bool is_short_B = (icol + 1 < num_cols) &&
                      (col_widths[icol + 1] == 0 || col_widths[icol + 1] == 1 ||
                       col_widths[icol + 1] == 2 || col_widths[icol + 1] == 4 ||
                       col_widths[icol + 1] == 8);
    if (is_short_A && is_short_B) {
      const uint8_t* col_vals_modified_B = col_vals[icol + 1];
      uint32_t col_width_modified_B = col_widths[icol + 1];
      // Prepare bit vector column
      if (col_widths[icol + 1] == 0) {
        util::BitUtil::bits_to_bytes(instruction_set, num_rows, col_vals[icol + 1],
                                     reinterpret_cast<uint8_t*>(temp_vector_32_B));
        col_vals_modified_B = reinterpret_cast<const uint8_t*>(temp_vector_32_B);
        col_width_modified_B = 1;
      }
      // Prepare length column
      if (col_offsets_maybe_missing[icol + 1]) {
        offsets_to_lengths(instruction_set, num_rows, col_offsets_maybe_missing[icol + 1],
                           temp_vector_32_B);
        col_vals_modified_B = reinterpret_cast<const uint8_t*>(temp_vector_32_B);
      }
      col2row_shortpair(instruction_set, is_row_fixedlen, col_width_modified_A,
                        col_width_modified_B, num_rows, col_vals_modified_A,
                        col_vals_modified_B, row_offsets, row_vals + row_fixed_offset);
      row_fixed_offset += col_width_modified_A + col_width_modified_B;
      icol += 2;
    } else {
      if (is_short_A) {
        col2row_short(num_rows, col_width_modified_A, col_vals_modified_A,
                      is_row_fixedlen, row_offsets, row_vals + row_fixed_offset);
      } else if (is_row_fixedlen) {
#if defined(ARROW_HAVE_AVX2)
        if (instruction_set == util::CPUInstructionSet::avx2) {
          col2row_long_avx2<true, true, false>(num_rows, nullptr, &col_width_modified_A,
                                               col_vals_modified_A, row_offsets,
                                               row_vals + row_fixed_offset, nullptr);
        } else {
#endif
          col2row_long<true, true, false>(num_rows, nullptr, &col_width_modified_A,
                                          col_vals_modified_A, row_offsets,
                                          row_vals + row_fixed_offset, nullptr);
#if defined(ARROW_HAVE_AVX2)
        }
#endif
      } else {
#if defined(ARROW_HAVE_AVX2)
        if (instruction_set == util::CPUInstructionSet::avx2) {
          col2row_long_avx2<true, false, false>(num_rows, nullptr, &col_width_modified_A,
                                                col_vals_modified_A, row_offsets,
                                                row_vals + row_fixed_offset, nullptr);
        } else {
#endif
          col2row_long<true, false, false>(num_rows, nullptr, &col_width_modified_A,
                                           col_vals_modified_A, row_offsets,
                                           row_vals + row_fixed_offset, nullptr);
#if defined(ARROW_HAVE_AVX2)
        }
#endif
      }
      row_fixed_offset += col_width_modified_A;
      ++icol;
    }
  }

  // 3. Process variable length columns.
  //
  // How many variable length columns do we have?
  //
  uint32_t num_col_varlen = 0;
  for (uint32_t icol = 0; icol < num_cols; ++icol) {
    if (col_offsets_maybe_missing[icol]) {
      ++num_col_varlen;
    }
  }
  uint32_t num_col_varlen_processed = 0;
  for (uint32_t icol = 0; icol < num_cols; ++icol) {
    if (col_offsets_maybe_missing[icol] == nullptr) {
      continue;
    }
#if defined(ARROW_HAVE_AVX2)
    if (instruction_set == util::CPUInstructionSet::avx2) {
      if (num_col_varlen_processed + 1 == num_col_varlen) {
        // This is the last variable length column
        // - no updating of row offsets
        //
        col2row_long_avx2<false, false, false>(
            num_rows, col_non_nulls[icol], col_offsets_maybe_missing[icol],
            col_vals[icol],
            num_col_varlen_processed == 0 ? row_offsets : temp_vector_32_A,
            row_vals + row_fixed_offset, nullptr);
      } else {
        col2row_long_avx2<false, false, true>(
            num_rows, col_non_nulls[icol], col_offsets_maybe_missing[icol],
            col_vals[icol],
            num_col_varlen_processed == 0 ? row_offsets : temp_vector_32_A,
            row_vals + row_fixed_offset, temp_vector_32_A);
      }
    } else {
#endif
      if (num_col_varlen_processed + 1 == num_col_varlen) {
        // This is the last variable length column
        // - no updating of row offsets
        //
        col2row_long<false, false, false>(
            num_rows, col_non_nulls[icol], col_offsets_maybe_missing[icol],
            col_vals[icol],
            num_col_varlen_processed == 0 ? row_offsets : temp_vector_32_A,
            row_vals + row_fixed_offset, nullptr);
      } else {
        col2row_long<false, false, true>(
            num_rows, col_non_nulls[icol], col_offsets_maybe_missing[icol],
            col_vals[icol],
            num_col_varlen_processed == 0 ? row_offsets : temp_vector_32_A,
            row_vals + row_fixed_offset, temp_vector_32_A);
      }
#if defined(ARROW_HAVE_AVX2)
    }
#endif
    ++num_col_varlen_processed;
  }

  // 4. Process non-null bit vectors.
  //
  int log_row_null_bits = 0;
  while (num_cols > (1UL << log_row_null_bits)) {
    ++log_row_null_bits;
  }
  memset(row_nulls, 0, ((num_rows << log_row_null_bits) + 7) / 8);
  row_fixed_offset = 0;
  for (uint32_t icol = 0; icol < num_cols; ++icol) {
    int num_ids;
    uint16_t* ids = reinterpret_cast<uint16_t*>(temp_vector_32_A);
    util::BitUtil::bits_to_indexes<0>(instruction_set, num_rows, col_non_nulls[icol],
                                      num_ids, ids);

    uint32_t col_width_modified = col_widths[icol] == 0 ? 1 : col_widths[icol];

    // 4a. Change columns values for nulls.
    //
    bool is_length_column = (col_offsets_maybe_missing[icol] != nullptr);
    if (is_row_fixedlen) {
      memset_selection_of_values<true>(col_width_modified, num_ids, ids, row_offsets,
                                       row_vals + row_fixed_offset,
                                       is_length_column ? 0 : 0xae);
    } else {
      memset_selection_of_values<false>(col_width_modified, num_ids, ids, row_offsets,
                                        row_vals + row_fixed_offset,
                                        is_length_column ? 0 : 0xae);
    }

    // 4b. Update row oriented nulls array.
    //
    col2row_null_bitvector(icol, num_cols, num_ids, ids, row_nulls);

    row_fixed_offset += col_width_modified;
  }
}

void KeyTranspose::row2col(
    util::CPUInstructionSet instruction_set, bool only_varlen_buffers, uint32_t num_cols,
    uint64_t num_rows,
    // Column widths for all columns.
    // Undefined for varying length column (identified by offset vector not being null).
    // Column width of 0 means 1 bit per row.
    // Column widths are not restricted to powers of 2.
    const uint32_t* col_widths,
    // Bit vectors marking non-null values in columns cannot be null.
    // A constant bit vector with all bits set to 1 needs to be provided, even if the
    // input column does not specify one.
    uint8_t** col_non_nulls,
    // If not null, the first element will be used as the base for newly added offsets.
    uint32_t** col_offsets_maybe_missing, uint8_t** col_vals,
    // In case of varying length rows - byte offsets for output rows with extra element
    // indicating total size of rows. In case of fixed length rows - there is only one
    // element that indicates row size in bytes. If any of the input columns has non null
    // offsets array pointer the row is varying length and otherwise it is fixed length.
    const uint32_t* row_offsets, const uint8_t* row_vals,
    // Bit vector marking null values in any of the columns with 1.
    // Bits are organized in row oriented way (all null bits for the first row are
    // followed by all null bits for the second). It uses ceil(log2(num_cols)) bits per
    // row.
    const uint8_t* row_nulls,
    // Temporary vectors with at least num_rows 32-bit elements with additional reserved
    // at least 32 bytes at the end, allocated and provided by the caller.
    uint32_t temp_vector_length, uint32_t* temp_vector_32_A, uint32_t* temp_vector_32_B) {
  // 1. Find out if we have fixed length row or varying length row.
  //
  bool is_row_fixedlen = true;
  for (uint32_t icol = 0; icol < num_cols; ++icol) {
    if (col_offsets_maybe_missing[icol]) {
      is_row_fixedlen = false;
    }
  }

  // 2. Process all fixed width columns
  //
  if (!only_varlen_buffers) {
    uint32_t row_fixed_offset = 0;
    for (uint32_t icol = 0; icol < num_cols;) {
      uint8_t* col_vals_A = col_vals[icol];
      uint32_t col_width_A = col_widths[icol];
      // Prepare bit vector column
      if (col_widths[icol] == 0) {
        col_vals_A = reinterpret_cast<uint8_t*>(temp_vector_32_A);
        col_width_A = 1;
      }
      // Prepare length column
      if (col_offsets_maybe_missing[icol]) {
        col_vals_A = reinterpret_cast<uint8_t*>(temp_vector_32_A);
      }
      bool is_short_A = col_widths[icol] == 0 || col_widths[icol] == 1 ||
                        col_widths[icol] == 2 || col_widths[icol] == 4 ||
                        col_widths[icol] == 8;
      bool is_short_B = (icol + 1 < num_cols) &&
                        (col_widths[icol + 1] == 0 || col_widths[icol + 1] == 1 ||
                         col_widths[icol + 1] == 2 || col_widths[icol + 1] == 4 ||
                         col_widths[icol + 1] == 8);
      uint32_t icol_next;
      if (is_short_A && is_short_B) {
        uint8_t* col_vals_B = col_vals[icol + 1];
        uint32_t col_width_B = col_widths[icol + 1];
        // Prepare bit vector column
        if (col_widths[icol + 1] == 0) {
          col_vals_B = reinterpret_cast<uint8_t*>(temp_vector_32_B);
          col_width_B = 1;
        }
        // Prepare length column
        if (col_offsets_maybe_missing[icol + 1]) {
          col_vals_B = reinterpret_cast<uint8_t*>(temp_vector_32_B);
        }
        row2col_shortpair(instruction_set, is_row_fixedlen, col_width_A, col_width_B,
                          static_cast<uint32_t>(num_rows), col_vals_A, col_vals_B,
                          row_offsets, row_vals + row_fixed_offset);
        if (col_offsets_maybe_missing[icol + 1]) {
          lengths_to_offsets(static_cast<uint32_t>(num_rows),
                             col_offsets_maybe_missing[icol + 1][0],
                             reinterpret_cast<const uint32_t*>(col_vals_B),
                             col_offsets_maybe_missing[icol + 1]);
        }
        if (col_widths[icol + 1] == 0) {
          util::BitUtil::bytes_to_bits(instruction_set, static_cast<int>(num_rows),
                                       col_vals_B, col_vals[icol + 1]);
        }
        row_fixed_offset += col_width_A + col_width_B;
        icol_next = icol + 2;
      } else {
        if (is_short_A) {
          row2col_short(static_cast<uint32_t>(num_rows), col_width_A, col_vals_A,
                        is_row_fixedlen, row_offsets, row_vals + row_fixed_offset);
        } else if (is_row_fixedlen) {
#if defined(ARROW_HAVE_AVX2)
          if (instruction_set == util::CPUInstructionSet::avx2) {
            row2col_long_avx2<true, true, false>(static_cast<uint32_t>(num_rows),
                                                 &col_width_A, col_vals_A, row_offsets,
                                                 row_vals + row_fixed_offset, nullptr);
          } else {
#endif
            row2col_long<true, true, false>(static_cast<uint32_t>(num_rows), &col_width_A,
                                            col_vals_A, row_offsets,
                                            row_vals + row_fixed_offset, nullptr);
#if defined(ARROW_HAVE_AVX2)
          }
#endif
        } else {
#if defined(ARROW_HAVE_AVX2)
          if (instruction_set == util::CPUInstructionSet::avx2) {
            row2col_long_avx2<true, false, false>(static_cast<uint32_t>(num_rows),
                                                  &col_width_A, col_vals_A, row_offsets,
                                                  row_vals + row_fixed_offset, nullptr);
          } else {
#endif
            row2col_long<true, false, false>(static_cast<uint32_t>(num_rows),
                                             &col_width_A, col_vals_A, row_offsets,
                                             row_vals + row_fixed_offset, nullptr);
#if defined(ARROW_HAVE_AVX2)
          }
#endif
        }
        row_fixed_offset += col_width_A;
        icol_next = icol + 1;
      }
      if (col_offsets_maybe_missing[icol]) {
        lengths_to_offsets(static_cast<uint32_t>(num_rows),
                           col_offsets_maybe_missing[icol][0],
                           reinterpret_cast<const uint32_t*>(col_vals_A),
                           col_offsets_maybe_missing[icol]);
      }
      if (col_widths[icol] == 0) {
        util::BitUtil::bytes_to_bits(instruction_set, static_cast<int>(num_rows),
                                     col_vals_A, col_vals[icol]);
      }
      icol = icol_next;
    }
  }

  // 3. Process variable length columns.
  //
  // How many variable length columns do we have?
  //
  if (only_varlen_buffers) {
    uint32_t num_col_varlen = 0;
    uint32_t row_fixed_offset = 0;
    for (uint32_t icol = 0; icol < num_cols; ++icol) {
      if (col_offsets_maybe_missing[icol]) {
        ++num_col_varlen;
        row_fixed_offset += sizeof(uint32_t);
      } else {
        row_fixed_offset += col_widths[icol] == 0 ? 1 : col_widths[icol];
      }
    }
    uint32_t num_col_varlen_processed = 0;
    for (uint32_t icol = 0; icol < num_cols; ++icol) {
      if (col_offsets_maybe_missing[icol] == nullptr) {
        continue;
      }
#if defined(ARROW_HAVE_AVX2)
      if (instruction_set == util::CPUInstructionSet::avx2) {
        if (num_col_varlen_processed + 1 == num_col_varlen) {
          // This is the last variable length column
          // - no updating of row offsets
          //
          row2col_long_avx2<false, false, false>(
              static_cast<uint32_t>(num_rows), col_offsets_maybe_missing[icol],
              col_vals[icol],
              num_col_varlen_processed == 0 ? row_offsets : temp_vector_32_A,
              row_vals + row_fixed_offset, nullptr);
        } else {
          row2col_long_avx2<false, false, true>(
              static_cast<uint32_t>(num_rows), col_offsets_maybe_missing[icol],
              col_vals[icol],
              num_col_varlen_processed == 0 ? row_offsets : temp_vector_32_A,
              row_vals + row_fixed_offset, temp_vector_32_A);
        }
      } else {
#endif
        if (num_col_varlen_processed + 1 == num_col_varlen) {
          // This is the last variable length column
          // - no updating of row offsets
          //
          row2col_long<false, false, false>(
              static_cast<uint32_t>(num_rows), col_offsets_maybe_missing[icol],
              col_vals[icol],
              num_col_varlen_processed == 0 ? row_offsets : temp_vector_32_A,
              row_vals + row_fixed_offset, nullptr);
        } else {
          row2col_long<false, false, true>(
              static_cast<uint32_t>(num_rows), col_offsets_maybe_missing[icol],
              col_vals[icol],
              num_col_varlen_processed == 0 ? row_offsets : temp_vector_32_A,
              row_vals + row_fixed_offset, temp_vector_32_A);
        }
#if defined(ARROW_HAVE_AVX2)
      }
#endif
      ++num_col_varlen_processed;
    }
  }

  // 4. Process non-null bit vectors.
  //
  if (!only_varlen_buffers) {
    row2col_null_bitvector(instruction_set, num_rows, num_cols, col_non_nulls, row_nulls,
                           temp_vector_length,
                           reinterpret_cast<uint16_t*>(temp_vector_32_A));
  }
}

}  // namespace exec
}  // namespace arrow
