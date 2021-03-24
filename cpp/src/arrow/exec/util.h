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
#include <vector>

#include "arrow/exec/common.h"
#include "arrow/memory_pool.h"
#include "arrow/result.h"
#include "arrow/status.h"
#include "arrow/util/logging.h"

#if defined(__clang__) || defined(__GNUC__)
#define BYTESWAP(x) __builtin_bswap64(x)
#define ROTL(x, n) (((x) << (n)) | ((x) >> (32 - (n))))
#elif defined(_MSC_VER)
#include <intrin.h>
#define BYTESWAP(x) _byteswap_uint64(x)
#define ROTL(x, n) _rotl((x), (n))
#endif

namespace arrow {
namespace exec {
namespace util {

enum class CPUInstructionSet {
  scalar,
  avx2,   // All of: AVX2, BMI2
  avx512  // In addition to avx2, all of: AVX512-F, AVX512-BW, AVX512-DQ, AVX512-CD
};

class TempBufferAlloc {
 public:
  TempBufferAlloc() {}
  ~TempBufferAlloc() {
    for (size_t i = 0; i < pages_.size(); ++i) {
      pool_->Free(pages_[i], elements_per_page_ * max_element_bytes_ + padding_);
    }
  }
  Status init(MemoryPool* pool, int32_t num_elements) {
    pool_ = pool;
    elements_per_page_ = num_elements;
    ARROW_DCHECK(pages_.empty());
    ARROW_RETURN_NOT_OK(alloc_page());
    page_to_use_ = 0;
    return Status::OK();
  }
  inline Status allocate_buffer(uint8_t*& buffer, uint32_t num_element_bytes) {
    ARROW_DCHECK(num_element_bytes <= elements_per_page_);
    if (max_element_bytes_ - page_bytes_allocated_[page_to_use_] <
        static_cast<int>(num_element_bytes)) {
      if (page_to_use_ == static_cast<int>(pages_.size()) - 1) {
        ARROW_RETURN_NOT_OK(alloc_page());
      }
      page_to_use_++;
    }
    buffer =
        pages_[page_to_use_] + page_bytes_allocated_[page_to_use_] * elements_per_page_;
    page_bytes_allocated_[page_to_use_] += num_element_bytes;
    allocated_sizes_.push_back(num_element_bytes);
    return Status::OK();
  }
  void release_buffer(int num_element_bytes) {
    ARROW_DCHECK(num_element_bytes == allocated_sizes_.back());
    allocated_sizes_.pop_back();
    page_bytes_allocated_[page_to_use_] -= num_element_bytes;
    if (page_bytes_allocated_[page_to_use_] == 0 && page_to_use_ > 0 &&
        page_bytes_allocated_[page_to_use_ - 1] < max_element_bytes_) {
      --page_to_use_;
    }
  }
  int32_t get_num_elements() const { return elements_per_page_; }
  static constexpr int max_element_bytes_ = 16;

 private:
  Status alloc_page() {
    uint8_t* new_page;
    ARROW_RETURN_NOT_OK(
        pool_->Allocate(elements_per_page_ * max_element_bytes_ + padding_, &new_page));
    pages_.push_back(new_page);
    page_bytes_allocated_.push_back(0);
    return Status::OK();
  }
  // Padding at the end of the buffer that may be needed for SIMD
  static constexpr uint32_t padding_ = 64;
  uint32_t elements_per_page_;
  MemoryPool* pool_;
  int page_to_use_;
  std::vector<uint8_t*> pages_;
  std::vector<int> page_bytes_allocated_;
  // TODO: This is only used in DEBUG
  std::vector<int> allocated_sizes_;
};

template <typename T>
class TempBuffer {
 public:
  explicit TempBuffer(TempBufferAlloc* alloc, int multiplicity = 1)
      : alloc_(alloc), multiplicity_(multiplicity), buffer_() {
    ARROW_DCHECK(multiplicity > 0);
    ARROW_DCHECK(sizeof(T) * multiplicity <= TempBufferAlloc::max_element_bytes_);
  }
  ~TempBuffer() {
    if (buffer_) {
      alloc_->release_buffer(sizeof(T) * multiplicity_);
    }
  }
  inline Status alloc() {
    return alloc_->allocate_buffer(buffer_, sizeof(T) * multiplicity_);
  }
  T* mutable_data() { return reinterpret_cast<T*>(buffer_); }

 private:
  TempBufferAlloc* alloc_;
  int multiplicity_;
  uint8_t* buffer_;
};

class BitUtil {
 public:
  template <int bit_to_search = 1>
  static void bits_to_indexes(CPUInstructionSet instruction_set, const int num_bits,
                              const uint8_t* bits, int& num_indexes, uint16_t* indexes);

  // Input and output indexes may be pointing to the same data (in-place filtering).
  template <int bit_to_search = 1>
  static void bits_filter_indexes(CPUInstructionSet instruction_set, const int num_bits,
                                  const uint8_t* bits, const uint16_t* input_indexes,
                                  int& num_indexes, uint16_t* indexes);

  // Input and output indexes may be pointing to the same data (in-place filtering).
  static void bits_split_indexes(CPUInstructionSet instruction_set, const int num_bits,
                                 const uint8_t* bits, int& num_indexes_bit0,
                                 uint16_t* indexes_bit0, uint16_t* indexes_bit1);

  static void bits_to_bytes(CPUInstructionSet instruction_set, const int num_bits,
                            const uint8_t* bits, uint8_t* bytes);
  static void bytes_to_bits(CPUInstructionSet instruction_set, const int num_bits,
                            const uint8_t* bytes, uint8_t* bits);

  static void bit_vector_lookup(CPUInstructionSet instruction_set, const int num_lookups,
                                const uint32_t* bit_ids, const uint8_t* bits,
                                uint8_t* result);

 private:
  inline static void bits_to_indexes_helper(uint64_t word, uint16_t base_index,
                                            int& num_indexes, uint16_t* indexes);
  inline static void bits_filter_indexes_helper(uint64_t word,
                                                const uint16_t* input_indexes,
                                                int& num_indexes, uint16_t* indexes);
  template <int bit_to_search, bool filter_input_indexes>
  static void bits_to_indexes_internal(CPUInstructionSet instruction_set,
                                       const int num_bits, const uint8_t* bits,
                                       const uint16_t* input_indexes, int& num_indexes,
                                       uint16_t* indexes);
  static void bits_to_bytes_internal(const int num_bits, const uint8_t* bits,
                                     uint8_t* bytes);
  static void bytes_to_bits_internal(const int num_bits, const uint8_t* bytes,
                                     uint8_t* bits);

#if defined(ARROW_HAVE_AVX2)
  template <int bit_to_search>
  static void bits_to_indexes_avx2(const int num_bits, const uint8_t* bits,
                                   int& num_indexes, uint16_t* indexes);
  template <int bit_to_search>
  static void bits_filter_indexes_avx2(const int num_bits, const uint8_t* bits,
                                       const uint16_t* input_indexes, int& num_indexes,
                                       uint16_t* indexes);
  static void bits_to_bytes_avx2(const int num_bits, const uint8_t* bits, uint8_t* bytes);
  static void bytes_to_bits_avx2(const int num_bits, const uint8_t* bytes, uint8_t* bits);
  static void bit_vector_lookup_avx2(const int num_lookups, const uint32_t* bit_ids,
                                     const uint8_t* bits, uint8_t* result);
#endif
};

}  // namespace util
}  // namespace exec
}  // namespace arrow
