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
#include <memory>

#include "arrow/memory_pool.h"
#include "arrow/result.h"
#include "arrow/status.h"
#include "arrow/buffer.h"
#include "common.h"
#include "util.h"

namespace arrow {
namespace exec {

/// Converts between key representation as a collection of arrays for
/// individual columns and another representation as a single array of rows
/// combining data from all columns into one value.
/// This conversion is reversible.
/// Row-oriented storage is beneficial when there is a need for random access
/// of individual rows and at the same time all included columns are likely to
/// be accessed together, as in the case of hash table key.
class KeyEncoder {
 public:
  struct KeyEncoderContext {
    bool has_avx2() const { return instr == util::CPUInstructionSet::avx2; }
    util::CPUInstructionSet instr;
    util::TempVectorStack* stack;
  };

  /// Description of a storage format for rows produced by encoder.
  struct KeyRowMetadata {
    /// Is row a varying-length binary, using offsets array to find a beginning of a row,
    /// or is it a fixed-length binary.
    bool is_fixed_length;
    /// For a fixed-length binary row, common size of rows in bytes.
    /// For a varying-length binary, size of all encoded fixed-length key columns.
    /// Encoded fixed-length key columns in that case prefix the information
    /// about all varying-length key columns.
    uint32_t fixed_length;
    /// Size in bytes of optional cumulative lengths of varying-length key columns,
    /// used when the row is not fixed length.
    /// Zero for fixed-length row.
    /// This number is equal to the number of varying-length key columns multiplied
    /// by sizeof(uint32_t), which is the size of a single cumulative length.
    uint32_t cumulative_lengths_length;
    /// Fixed number of bytes per row that are used to encode null masks.
    /// Null masks indicate for a single row which of its key columns are null.
    /// Nth bit in the sequence of bytes assigned to a row represents null
    /// information for Nth key column.
    int null_masks_bytes_per_row;
  };

  class KeyRowArray {
   public:
    KeyRowArray();
    Status Init(MemoryPool* pool, const KeyRowMetadata& metadata);
    void Clean();
    Status ResizeFixedLengthBuffers(int64_t num_extra_rows);
    Status ResizeOptionalVaryingLengthBuffer(int64_t num_extra_bytes);
    Status AppendSelectionFrom(const KeyRowArray& from, uint32_t num_rows_to_append,
                               uint16_t* source_row_ids);
    const KeyRowMetadata& get_metadata() const { return metadata_; }
    int64_t get_length() const { return num_rows_; }
    const uint8_t* data(int i) const {
      ARROW_DCHECK(i >= 0 && i <= max_buffers_);
      return buffers_[i];
    }
    uint8_t* mutable_data(int i) {
      ARROW_DCHECK(i >= 0 && i <= max_buffers_);
      return mutable_buffers_[i];
    }
    uint32_t* get_mutable_offsets() {
      return reinterpret_cast<uint32_t*>(mutable_data(1));
    }
    const uint32_t* get_offsets() const {
      return reinterpret_cast<const uint32_t*>(data(1));
    }
    const uint8_t* get_null_masks() const { return null_masks_->data(); }
    uint8_t* get_null_masks() { return null_masks_->mutable_data(); }

   private:
    int64_t size_null_masks(int64_t num_rows);
    int64_t size_offsets(int64_t num_rows);
    int64_t size_rows_fixed_length(int64_t num_rows);
    int64_t size_rows_varying_length(int64_t num_bytes);
    void update_buffer_pointers();
    static constexpr int64_t padding_for_vectors = 64;
    MemoryPool* pool_;
    KeyRowMetadata metadata_;
    /// Buffers can only expand during lifetime and never shrink.
    std::unique_ptr<ResizableBuffer> null_masks_;
    std::unique_ptr<ResizableBuffer> offsets_;
    std::unique_ptr<ResizableBuffer> rows_;
    static constexpr int max_buffers_ = 3;
    const uint8_t* buffers_[max_buffers_];
    uint8_t* mutable_buffers_[max_buffers_];
    int64_t num_rows_;
    int64_t rows_capacity_;
    int64_t bytes_capacity_;
  };

  /// Description of a storage format of a single key column as needed
  /// for the purpose of row encoding.
  struct KeyColumnMetadata {
    KeyColumnMetadata() {}
    KeyColumnMetadata(bool is_fixed_length_in, uint32_t fixed_length_in)
        : is_fixed_length(is_fixed_length_in), fixed_length(fixed_length_in) {}
    /// Is column storing a varying-length binary, using offsets array
    /// to find a beginning of a value, or is it a fixed-length binary.
    bool is_fixed_length;
    /// For a fixed-length binary column: number of bytes per value.
    /// Zero has a special meaning, indicating a bit vector with one bit per value.
    /// For a varying-length binary column: number of bytes per offset.
    uint32_t fixed_length;
  };

  /// A lightweight description of an array representing one of key columns.
  class KeyColumnArray {
   public:
    KeyColumnArray() {}
    /// Create as a mix of buffers according to the mask from two descriptions
    /// (Nth bit is set to 0 if Nth buffer from the first input
    /// should be used and is set to 1 otherwise).
    /// Metadata is inherited from the first input.
    KeyColumnArray(const KeyColumnMetadata& metadata, const KeyColumnArray& left,
                   const KeyColumnArray& right, int buffer_id_to_replace);
    /// Create for writing
    KeyColumnArray(const KeyColumnMetadata& metadata, int64_t length, uint8_t* buffer0,
                   uint8_t* buffer1, uint8_t* buffer2);
    /// Create as a window view of original description that is offset
    /// by a given number of rows.
    /// The number of rows used in offset must be divisible by 8
    /// in order to not split bit vectors within a single byte.
    KeyColumnArray(const KeyColumnArray& from, int64_t start, int64_t length);
    uint8_t* mutable_data(int i) {
      ARROW_DCHECK(i >= 0 && i <= max_buffers_);
      return mutable_buffers_[i];
    }
    const uint8_t* data(int i) const {
      ARROW_DCHECK(i >= 0 && i <= max_buffers_);
      return buffers_[i];
    }
    uint32_t* get_mutable_offsets() {
      return reinterpret_cast<uint32_t*>(mutable_data(1));
    }
    const uint32_t* get_offsets() const {
      return reinterpret_cast<const uint32_t*>(data(1));
    }
    const KeyColumnMetadata& get_metadata() const { return metadata_; }
    int64_t get_length() const { return length_; };

   private:
    static constexpr int max_buffers_ = 3;
    const uint8_t* buffers_[max_buffers_];
    uint8_t* mutable_buffers_[max_buffers_];
    KeyColumnMetadata metadata_;
    int64_t length_;
  };

  /*
      void PrepareMetadata(
          std::vector<KeyColumMetdata>& col_metadata,
          KeyRowMetadata* out_row_metadata);

      void PrepareKeyColumnArrays(
          const std::vector<KeyColumnArray>& cols_in,
          std::vector<KeyColumnArray>* out_cols,
          std::vector<KeyColumnArray>* out_varbinary_cols);

      void GetOutputBufferSizesForEncode();
          const KeyRowMetadata& row_metadata,
          const std::vector<KeyColumnArray>& varbinary_cols,
          int64_t* out_num_rows_required,
          int64_t* out_num_bytes_required);
  */

  void Encode();

  void Decode();

 private:
  class TransformBoolean {
   public:
    static KeyColumnArray ArrayReplace(const KeyColumnArray& column,
                                       const KeyColumnArray& temp);
    static void PreEncode(const KeyColumnArray& input, KeyColumnArray& output,
                          KeyEncoderContext* ctx);
    static void PostDecode(const KeyColumnArray& input, KeyColumnArray& output,
                           KeyEncoderContext* ctx);
  };

  class EncoderInteger {
   public:
    static void Encode(uint32_t* offset_within_row, KeyRowArray& rows,
                       const KeyColumnArray& col, KeyEncoderContext* ctx,
                       KeyColumnArray& temp);
    static void Decode(uint32_t start_row, uint32_t num_rows, uint32_t* offset_within_row,
                       const KeyRowArray& rows, KeyColumnArray& col,
                       KeyEncoderContext* ctx, KeyColumnArray& temp);
    static bool UsesTransform(const KeyColumnArray& column);
    static KeyColumnArray ArrayReplace(const KeyColumnArray& column,
                                       KeyColumnArray& temp);
    static void PreEncode(const KeyColumnArray& input, KeyColumnArray& output,
                          KeyEncoderContext* ctx);
    static void PostDecode(const KeyColumnArray& input, KeyColumnArray& output,
                           KeyEncoderContext* ctx);

   private:
    static bool IsBoolean(const KeyColumnMetadata& metadata);
  };

  class EncoderBinary {
   public:
    static void Encode(uint32_t* offset_within_row, KeyRowArray& rows,
                       const KeyColumnArray& col, KeyEncoderContext* ctx,
                       KeyColumnArray& temp);
    static void Decode(uint32_t start_row, uint32_t num_rows, uint32_t* offset_within_row,
                       const KeyRowArray& rows, KeyColumnArray& col,
                       KeyEncoderContext* ctx, KeyColumnArray& temp);
    static bool IsInteger(const KeyColumnMetadata& metadata);

   private:
    template <bool is_row_fixed_length, bool is_encoding, class COPY_FN>
    static inline void EncodeDecodeHelper(uint32_t start_row, uint32_t num_rows,
                                          uint32_t offset_within_row,
                                          const KeyRowArray* rows_const,
                                          KeyRowArray* rows_mutable_maybe_null,
                                          const KeyColumnArray* col_const,
                                          KeyColumnArray* col_mutable_maybe_null,
                                          COPY_FN copy_fn);
    template <bool is_row_fixed_length>
    static void EncodeImp(uint32_t offset_within_row, KeyRowArray& rows,
                          const KeyColumnArray& col);
    template <bool is_row_fixed_length>
    static void DecodeImp(uint32_t start_row, uint32_t num_rows,
                          uint32_t offset_within_row, const KeyRowArray& rows,
                          KeyColumnArray& col);
#if defined(ARROW_HAVE_AVX2)
    template <bool is_row_fixed_length>
    static void EncodeImp_avx2(uint32_t offset_within_row, KeyRowArray& rows,
                               const KeyColumnArray& col);
    template <bool is_row_fixed_length>
    static void DecodeImp_avx2(uint32_t start_row, uint32_t num_rows,
                               uint32_t offset_within_row, const KeyRowArray& rows,
                               KeyColumnArray& col);
#endif
    static void ColumnMemsetNulls(uint32_t offset_within_row, KeyRowArray& rows,
                                  const KeyColumnArray& col, KeyEncoderContext* ctx,
                                  KeyColumnArray& temp_vector_16bit, uint8_t byte_value);
    template <bool is_row_fixed_length, uint32_t col_width>
    static void ColumnMemsetNullsImp(uint32_t offset_within_row, KeyRowArray& rows,
                                     const KeyColumnArray& col, KeyEncoderContext* ctx,
                                     KeyColumnArray& temp_vector_16bit,
                                     uint8_t byte_value);
  };

  class EncoderBinaryPair {
   public:
    static bool CanProcessPair(const KeyColumnMetadata& col1,
                               const KeyColumnMetadata& col2) {
      return EncoderBinary::IsInteger(col1) && EncoderBinary::IsInteger(col2);
    }
    static void Encode(uint32_t* offset_within_row, KeyRowArray& rows,
                       const KeyColumnArray& col1, const KeyColumnArray& col2,
                       KeyEncoderContext* ctx, KeyColumnArray& temp1,
                       KeyColumnArray& temp2);
    static void Decode(uint32_t start_row, uint32_t num_rows, uint32_t* offset_within_row,
                       const KeyRowArray& rows, KeyColumnArray& col1,
                       KeyColumnArray& col2, KeyEncoderContext* ctx,
                       KeyColumnArray& temp1, KeyColumnArray& temp2);

   private:
    template <bool is_row_fixed_length, int col_width_1, int col_width_2>
    static void EncodeImp(uint32_t num_rows_to_skip, uint32_t offset_within_row,
                          KeyRowArray& rows, const KeyColumnArray& col1,
                          const KeyColumnArray& col2);
    template <bool is_row_fixed_length, int col_width_1, int col_width_2>
    static void DecodeImp(uint32_t num_rows_to_skip, uint32_t start_row,
                          uint32_t num_rows, uint32_t offset_within_row,
                          const KeyRowArray& rows, KeyColumnArray& col1,
                          KeyColumnArray& col2);
#if defined(ARROW_HAVE_AVX2)
    template <bool is_row_fixed_length, uint32_t col_width>
    static uint32_t EncodeImp_avx2(uint32_t offset_within_row, KeyRowArray& rows,
                                   const KeyColumnArray& col1,
                                   const KeyColumnArray& col2);
    template <bool is_row_fixed_length, uint32_t col_width>
    static uint32_t DecodeImp_avx2(uint32_t start_row, uint32_t num_rows,
                                   uint32_t offset_within_row, const KeyRowArray& rows,
                                   KeyColumnArray& col1, KeyColumnArray& col2);
#endif
  };

  class EncoderOffsets {
   public:
    // In order not to repeat work twice,
    // encoding combines in a single pass computing of:
    // a) row offsets for varying-length rows
    // b) within each new row, the cumulative length array
    // of varying-length values within a row.
    static void Encode(KeyRowArray& rows,
                       const std::vector<KeyColumnArray>& varbinary_cols,
                       KeyEncoderContext* ctx);
    static void Decode(uint32_t start_row, uint32_t num_rows, KeyRowArray& rows,
                       std::vector<KeyColumnArray>& varbinary_cols,
                       std::vector<uint32_t>& varbinary_cols_base_offset,
                       KeyEncoderContext* ctx);

   private:
    static void EncodeImp(uint32_t num_rows_to_skip, KeyRowArray& rows,
                          const std::vector<KeyColumnArray>& varbinary_cols);
#if defined(ARROW_HAVE_AVX2)
    static uint32_t EncodeImp_avx2(KeyRowArray& rows,
                                   const std::vector<KeyColumnArray>& varbinary_cols,
                                   KeyColumnArray& temp_buffer_32B_per_col);
#endif
  };

  class EncoderVarBinary {
   public:
    static void Encode(uint32_t varbinary_col_id, KeyRowArray& rows,
                       const KeyColumnArray& col, KeyEncoderContext* ctx);
    static void Decode(uint32_t start_row, uint32_t num_rows, uint32_t varbinary_col_id,
                       const KeyRowArray& rows, KeyColumnArray& col,
                       KeyEncoderContext* ctx);

   private:
    template <bool first_varbinary_col, bool is_encoding, class COPY_FN>
    static inline void EncodeDecodeHelper(uint32_t start_row, uint32_t num_rows,
                                          uint32_t varbinary_col_id,
                                          const KeyRowArray* rows_const,
                                          KeyRowArray* rows_mutable_maybe_null,
                                          const KeyColumnArray* col_const,
                                          KeyColumnArray* col_mutable_maybe_null,
                                          COPY_FN copy_fn);
    template <bool first_varbinary_col>
    static void EncodeImp(uint32_t varbinary_col_id, KeyRowArray& rows,
                          const KeyColumnArray& col);
    template <bool first_varbinary_col>
    static void DecodeImp(uint32_t start_row, uint32_t num_rows,
                          uint32_t varbinary_col_id, const KeyRowArray& rows,
                          KeyColumnArray& col);
#if defined(ARROW_HAVE_AVX2)
    template <bool first_varbinary_col>
    static void EncodeImp_avx2(uint32_t varbinary_col_id, KeyRowArray& rows,
                               const KeyColumnArray& col);
    template <bool first_varbinary_col>
    static void DecodeImp_avx2(uint32_t start_row, uint32_t num_rows,
                               uint32_t varbinary_col_id, const KeyRowArray& rows,
                               KeyColumnArray& col);
#endif
  };

  class EncoderNulls {
   public:
    static void Encode(KeyRowArray& rows, const std::vector<KeyColumnArray>& cols,
                       KeyEncoderContext* ctx, KeyColumnArray& temp_vector_16bit);
    static void Decode(uint32_t start_row, uint32_t num_rows, const KeyRowArray& rows,
                       std::vector<KeyColumnArray>& cols);
  };
};

template <bool is_row_fixed_length, bool is_encoding, class COPY_FN>
inline void KeyEncoder::EncoderBinary::EncodeDecodeHelper(
    uint32_t start_row, uint32_t num_rows, uint32_t offset_within_row,
    const KeyRowArray* rows_const, KeyRowArray* rows_mutable_maybe_null,
    const KeyColumnArray* col_const, KeyColumnArray* col_mutable_maybe_null,
    COPY_FN copy_fn) {
  ARROW_DCHECK(col_const && col_const->get_metadata().is_fixed_length);
  uint32_t col_width = col_const->get_metadata().fixed_length;

  if (is_row_fixed_length) {
    uint32_t row_width = rows_const->get_metadata().fixed_length;
    for (uint32_t i = 0; i < num_rows; ++i) {
      const uint8_t* src;
      uint8_t* dst;
      if (is_encoding) {
        src = col_const->data(1) + col_width * i;
        dst = rows_mutable_maybe_null->mutable_data(1) + row_width * (start_row + i) +
              offset_within_row;
      } else {
        src = rows_const->data(1) + row_width * (start_row + i) + offset_within_row;
        dst = col_mutable_maybe_null->mutable_data(1) + col_width * i;
      }
      copy_fn(dst, src, col_width);
    }
  } else {
    const uint32_t* row_offsets = rows_const->get_offsets();
    for (uint32_t i = 0; i < num_rows; ++i) {
      const uint8_t* src;
      uint8_t* dst;
      if (is_encoding) {
        src = col_const->data(1) + col_width * i;
        dst = rows_mutable_maybe_null->mutable_data(2) + row_offsets[start_row + i] +
              offset_within_row;
      } else {
        src = rows_const->data(2) + row_offsets[start_row + i] + offset_within_row;
        dst = col_mutable_maybe_null->mutable_data(1) + col_width * i;
      }
      copy_fn(dst, src, col_width);
    }
  }
}

template <bool first_varbinary_col, bool is_encoding, class COPY_FN>
inline void KeyEncoder::EncoderVarBinary::EncodeDecodeHelper(
    uint32_t start_row, uint32_t num_rows, uint32_t varbinary_col_id,
    const KeyRowArray* rows_const, KeyRowArray* rows_mutable_maybe_null,
    const KeyColumnArray* col_const, KeyColumnArray* col_mutable_maybe_null,
    COPY_FN copy_fn) {
  // Column and rows need to be varying length
  ARROW_DCHECK(!rows_const->get_metadata().is_fixed_length &&
               !col_const->get_metadata().is_fixed_length);

  const uint32_t* row_offsets_for_batch = rows_const->get_offsets() + start_row;

  const uint32_t* col_offsets = col_const->get_offsets();

  // Offset to the beginning of varbinary part of the row,
  // which comes after the fixed length column values and cummulative lengths of
  // varbinary values.
  const uint32_t first_varbinary_offset =
      rows_const->get_metadata().fixed_length +
      rows_const->get_metadata().cumulative_lengths_length;
  // Offset withing each row to the cummulative varbinary value length related to
  // the given varbinary column id.
  const uint32_t cumulative_length_offset =
      rows_const->get_metadata().fixed_length + varbinary_col_id * sizeof(uint32_t);

  uint32_t col_offset_next = col_offsets[0];
  for (uint32_t i = 0; i < num_rows; ++i) {
    uint32_t col_offset = col_offset_next;
    col_offset_next = col_offsets[i + 1];
    uint32_t length = col_offset_next - col_offset;

    // Find offset to the beginning of varbinary data of this column in encoded row.
    // The offset is the cumulative length of varbinary values for the previous varbinary
    // column or zero for the first varbinary column.
    uint32_t row_offset = row_offsets_for_batch[i];
    if (!first_varbinary_col) {
      row_offset += reinterpret_cast<const uint32_t*>(rows_const->data(2) + row_offset +
                                                      cumulative_length_offset)[-1];
    }
    row_offset += first_varbinary_offset;

    const uint8_t* src;
    uint8_t* dst;
    if (is_encoding) {
      src = col_const->data(2) + col_offset;
      dst = rows_mutable_maybe_null->mutable_data(2) + row_offset;
    } else {
      src = rows_const->data(2) + row_offset;
      dst = col_mutable_maybe_null->mutable_data(2) + col_offset;
    }
    copy_fn(dst, src, length);
  }
}

}  // namespace exec
}  // namespace arrow
