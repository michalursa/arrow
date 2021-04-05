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

#include "key_encode.h"

#include <memory.h>
#include <algorithm>

#include "arrow/util/bit_util.h"
#include "common.h"
#include "util.h"

namespace arrow {
namespace exec {

KeyEncoder::KeyRowArray::KeyRowArray()
    : pool_(nullptr), rows_capacity_(0), bytes_capacity_(0) {}

Status KeyEncoder::KeyRowArray::Init(MemoryPool* pool, const KeyRowMetadata& metadata) {
  pool_ = pool;
  metadata_ = metadata;

  ARROW_DCHECK(!null_masks_ && !offsets_ && !rows_);

  constexpr int64_t rows_capacity = 8;
  constexpr int64_t bytes_capacity = 1024;

  // Null masks
  ARROW_ASSIGN_OR_RAISE(null_masks_,
                        AllocateResizableBuffer(size_null_masks(rows_capacity), pool_));
  memset(null_masks_->mutable_data(), 0, size_null_masks(rows_capacity));

  // Offsets and rows
  if (!metadata.is_fixed_length) {
    ARROW_ASSIGN_OR_RAISE(offsets_,
                          AllocateResizableBuffer(size_offsets(rows_capacity), pool_));
    memset(offsets_->mutable_data(), 0, size_offsets(rows_capacity));
    reinterpret_cast<uint32_t*>(offsets_->mutable_data())[0] = 0;

    ARROW_ASSIGN_OR_RAISE(
        rows_, AllocateResizableBuffer(size_rows_varying_length(bytes_capacity), pool_));
    memset(rows_->mutable_data(), 0, size_rows_varying_length(bytes_capacity));
    bytes_capacity_ = size_rows_varying_length(bytes_capacity) - padding_for_vectors;
  } else {
    ARROW_ASSIGN_OR_RAISE(
        rows_, AllocateResizableBuffer(size_rows_fixed_length(rows_capacity), pool_));
    memset(rows_->mutable_data(), 0, size_rows_fixed_length(rows_capacity));
    bytes_capacity_ = size_rows_fixed_length(rows_capacity) - padding_for_vectors;
  }

  update_buffer_pointers();

  rows_capacity_ = rows_capacity;

  num_rows_ = 0;

  return Status::OK();
}

void KeyEncoder::KeyRowArray::Clean() {
  num_rows_ = 0;
  if (!metadata_.is_fixed_length) {
    reinterpret_cast<uint32_t*>(offsets_->mutable_data())[0] = 0;
  }
}

int64_t KeyEncoder::KeyRowArray::size_null_masks(int64_t num_rows) {
  return num_rows * metadata_.null_masks_bytes_per_row + padding_for_vectors;
}

int64_t KeyEncoder::KeyRowArray::size_offsets(int64_t num_rows) {
  return (num_rows + 1) * sizeof(uint32_t) + padding_for_vectors;
}

int64_t KeyEncoder::KeyRowArray::size_rows_fixed_length(int64_t num_rows) {
  return num_rows * metadata_.fixed_length + padding_for_vectors;
}

int64_t KeyEncoder::KeyRowArray::size_rows_varying_length(int64_t num_bytes) {
  return num_bytes + padding_for_vectors;
}

void KeyEncoder::KeyRowArray::update_buffer_pointers() {
  buffers_[0] = mutable_buffers_[0] = null_masks_->mutable_data();
  if (metadata_.is_fixed_length) {
    buffers_[1] = mutable_buffers_[1] = rows_->mutable_data();
    buffers_[2] = mutable_buffers_[2] = nullptr;
  } else {
    buffers_[1] = mutable_buffers_[1] = offsets_->mutable_data();
    buffers_[2] = mutable_buffers_[2] = rows_->mutable_data();
  }
}

Status KeyEncoder::KeyRowArray::ResizeFixedLengthBuffers(int64_t num_extra_rows) {
  if (rows_capacity_ >= num_rows_ + num_extra_rows) {
    return Status::OK();
  }

  int64_t rows_capacity_new = std::max(static_cast<int64_t>(1), 2 * rows_capacity_);
  while (rows_capacity_new < num_rows_ + num_extra_rows) {
    rows_capacity_new *= 2;
  }

  // Null masks
  RETURN_NOT_OK(null_masks_->Resize(size_null_masks(rows_capacity_new), false));
  memset(null_masks_->mutable_data() + size_null_masks(rows_capacity_), 0,
         size_null_masks(rows_capacity_new) - size_null_masks(rows_capacity_));

  // Either offsets or rows
  if (!metadata_.is_fixed_length) {
    RETURN_NOT_OK(offsets_->Resize(size_offsets(rows_capacity_new), false));
    memset(offsets_->mutable_data() + size_offsets(rows_capacity_), 0,
           size_offsets(rows_capacity_new) - size_offsets(rows_capacity_));
  } else {
    RETURN_NOT_OK(rows_->Resize(size_rows_fixed_length(rows_capacity_new), false));
    memset(rows_->mutable_data() + size_rows_fixed_length(rows_capacity_), 0,
           size_rows_fixed_length(rows_capacity_new) -
               size_rows_fixed_length(rows_capacity_));
    bytes_capacity_ = size_rows_fixed_length(rows_capacity_new) - padding_for_vectors;
  }

  update_buffer_pointers();

  rows_capacity_ = rows_capacity_new;

  return Status::OK();
}

Status KeyEncoder::KeyRowArray::ResizeOptionalVaryingLengthBuffer(
    int64_t num_extra_bytes) {
  int64_t num_bytes = get_offsets()[num_rows_];
  if (bytes_capacity_ >= num_bytes + num_extra_bytes || metadata_.is_fixed_length) {
    return Status::OK();
  }

  int64_t bytes_capacity_new = std::max(static_cast<int64_t>(1), 2 * bytes_capacity_);
  while (bytes_capacity_new < num_bytes + num_extra_bytes) {
    bytes_capacity_new *= 2;
  }

  RETURN_NOT_OK(rows_->Resize(size_rows_varying_length(bytes_capacity_new), false));
  memset(rows_->mutable_data() + size_rows_varying_length(bytes_capacity_), 0,
         size_rows_varying_length(bytes_capacity_new) -
             size_rows_varying_length(bytes_capacity_));

  update_buffer_pointers();

  bytes_capacity_ = bytes_capacity_new;

  return Status::OK();
}

Status KeyEncoder::KeyRowArray::AppendSelectionFrom(const KeyRowArray& from,
                                                    uint32_t num_rows_to_append,
                                                    uint16_t* source_row_ids) {
  ARROW_DCHECK(metadata_.is_fixed_length == from.get_metadata().is_fixed_length &&
               metadata_.fixed_length == from.get_metadata().fixed_length &&
               metadata_.cumulative_lengths_length ==
                   from.get_metadata().cumulative_lengths_length &&
               metadata_.null_masks_bytes_per_row ==
                   from.get_metadata().null_masks_bytes_per_row);

  RETURN_NOT_OK(ResizeFixedLengthBuffers(num_rows_to_append));

  if (!metadata_.is_fixed_length) {
    // Varying-length rows
    const uint32_t* from_offsets =
        reinterpret_cast<const uint32_t*>(from.offsets_->data());
    uint32_t* to_offsets = reinterpret_cast<uint32_t*>(offsets_->mutable_data());
    uint32_t total_length = to_offsets[num_rows_];
    uint32_t total_length_to_append = 0;
    for (uint32_t i = 0; i < num_rows_to_append; ++i) {
      uint16_t row_id = source_row_ids[i];
      uint32_t length = from_offsets[row_id + 1] - from_offsets[row_id];
      total_length_to_append += length;
      to_offsets[num_rows_ + i + 1] = total_length + total_length_to_append;
    }

    RETURN_NOT_OK(ResizeOptionalVaryingLengthBuffer(total_length_to_append));

    const uint8_t* src = from.rows_->data();
    uint8_t* dst = rows_->mutable_data() + total_length;
    for (uint32_t i = 0; i < num_rows_to_append; ++i) {
      uint16_t row_id = source_row_ids[i];
      uint32_t length = from_offsets[row_id + 1] - from_offsets[row_id];
      const uint64_t* src64 =
          reinterpret_cast<const uint64_t*>(src + from_offsets[row_id]);
      uint64_t* dst64 = reinterpret_cast<uint64_t*>(dst);
      for (uint32_t j = 0; j < (length + 7) / 8; ++j) {
        dst64[j] = src64[j];
      }
      dst += length;
    }
  } else {
    // Fixed-length rows
    const uint8_t* src = from.rows_->data();
    uint8_t* dst = rows_->mutable_data() + num_rows_ * metadata_.fixed_length;
    for (uint32_t i = 0; i < num_rows_to_append; ++i) {
      uint16_t row_id = source_row_ids[i];
      uint32_t length = metadata_.fixed_length;
      const uint64_t* src64 = reinterpret_cast<const uint64_t*>(src + length * row_id);
      uint64_t* dst64 = reinterpret_cast<uint64_t*>(dst);
      for (uint32_t j = 0; j < (length + 7) / 8; ++j) {
        dst64[j] = src64[j];
      }
      dst += length;
    }
  }

  // Null masks
  uint32_t byte_length = metadata_.null_masks_bytes_per_row;
  uint64_t dst_byte_offset = num_rows_ * byte_length;
  const uint8_t* src_base = from.null_masks_->data();
  uint8_t* dst_base = null_masks_->mutable_data();
  for (uint32_t i = 0; i < num_rows_to_append; ++i) {
    uint32_t row_id = source_row_ids[i];
    int64_t src_byte_offset = row_id * byte_length;
    const uint8_t* src = src_base + src_byte_offset;
    uint8_t* dst = dst_base + dst_byte_offset;
    for (uint32_t ibyte = 0; ibyte < byte_length; ++ibyte) {
      dst[ibyte] = src[ibyte];
    }
    dst_byte_offset += byte_length;
  }

  num_rows_ += num_rows_to_append;

  return Status::OK();
}

KeyEncoder::KeyColumnArray::KeyColumnArray(const KeyColumnMetadata& metadata,
                                           const KeyColumnArray& left,
                                           const KeyColumnArray& right,
                                           int buffer_id_to_replace) {
  metadata_ = metadata;
  length_ = left.get_length();
  for (int i = 0; i < max_buffers_; ++i) {
    buffers_[i] = left.buffers_[i];
    mutable_buffers_[i] = left.mutable_buffers_[i];
  }
  buffers_[buffer_id_to_replace] = right.buffers_[buffer_id_to_replace];
  mutable_buffers_[buffer_id_to_replace] = right.mutable_buffers_[buffer_id_to_replace];
}

KeyEncoder::KeyColumnArray::KeyColumnArray(const KeyColumnMetadata& metadata,
                                           int64_t length, uint8_t* buffer0,
                                           uint8_t* buffer1, uint8_t* buffer2) {
  metadata_ = metadata;
  length_ = length;
  buffers_[0] = mutable_buffers_[0] = buffer0;
  buffers_[1] = mutable_buffers_[1] = buffer1;
  buffers_[2] = mutable_buffers_[2] = buffer2;
}

KeyEncoder::KeyColumnArray::KeyColumnArray(const KeyColumnArray& from, int64_t start,
                                           int64_t length) {
  ARROW_DCHECK((start % 8) == 0);
  metadata_ = from.metadata_;
  length_ = length;
  uint32_t fixed_size =
      !metadata_.is_fixed_length ? sizeof(uint32_t) : metadata_.fixed_length;
  buffers_[0] = from.buffers_[0] ? from.buffers_[0] + start / 8 : nullptr;
  mutable_buffers_[0] =
      from.mutable_buffers_[0] ? from.mutable_buffers_[0] + start / 8 : nullptr;
  buffers_[1] = from.buffers_[1] ? from.buffers_[1] + start * fixed_size : nullptr;
  mutable_buffers_[1] =
      from.mutable_buffers_[1] ? from.mutable_buffers_[1] + start * fixed_size : nullptr;
  buffers_[2] = from.buffers_[2];
  mutable_buffers_[2] = from.mutable_buffers_[2];
}

KeyEncoder::KeyColumnArray KeyEncoder::TransformBoolean::ArrayReplace(
    const KeyColumnArray& column, const KeyColumnArray& temp) {
  // Make sure that the temp buffer is large enough
  ARROW_DCHECK(temp.get_length() >= column.get_length() &&
               temp.get_metadata().is_fixed_length &&
               temp.get_metadata().fixed_length >= sizeof(uint8_t));
  KeyColumnMetadata metadata;
  metadata.is_fixed_length = true;
  metadata.fixed_length = sizeof(uint8_t);
  constexpr int buffer_index = 1;
  KeyColumnArray result = KeyColumnArray(metadata, column, temp, buffer_index);
  return result;
}

void KeyEncoder::TransformBoolean::PreEncode(const KeyColumnArray& input,
                                             KeyColumnArray& output,
                                             KeyEncoderContext* ctx) {
  // Make sure that metadata and lengths are compatible.
  ARROW_DCHECK(output.get_metadata().is_fixed_length ==
               input.get_metadata().is_fixed_length);
  ARROW_DCHECK(output.get_metadata().fixed_length == 1 &&
               input.get_metadata().fixed_length == 0);
  ARROW_DCHECK(output.get_length() == input.get_length());
  constexpr int buffer_index = 1;
  ARROW_DCHECK(input.data(buffer_index) != nullptr);
  ARROW_DCHECK(output.mutable_data(buffer_index) != nullptr);
  util::BitUtil::bits_to_bytes(ctx->instr, static_cast<int>(input.get_length()),
                               input.data(buffer_index),
                               output.mutable_data(buffer_index));
}

void KeyEncoder::TransformBoolean::PostDecode(const KeyColumnArray& input,
                                              KeyColumnArray& output,
                                              KeyEncoderContext* ctx) {
  // Make sure that metadata and lengths are compatible.
  ARROW_DCHECK(output.get_metadata().is_fixed_length ==
               input.get_metadata().is_fixed_length);
  ARROW_DCHECK(output.get_metadata().fixed_length == 0 &&
               input.get_metadata().fixed_length == 1);
  ARROW_DCHECK(output.get_length() == input.get_length());
  constexpr int buffer_index = 1;
  ARROW_DCHECK(input.data(buffer_index) != nullptr);
  ARROW_DCHECK(output.mutable_data(buffer_index) != nullptr);

  util::BitUtil::bytes_to_bits(ctx->instr, static_cast<int>(input.get_length()),
                               input.data(buffer_index),
                               output.mutable_data(buffer_index));
}

bool KeyEncoder::EncoderInteger::IsBoolean(const KeyColumnMetadata& metadata) {
  return metadata.is_fixed_length && metadata.fixed_length == 0;
}

bool KeyEncoder::EncoderInteger::UsesTransform(const KeyColumnArray& column) {
  return IsBoolean(column.get_metadata());
}

KeyEncoder::KeyColumnArray KeyEncoder::EncoderInteger::ArrayReplace(
    const KeyColumnArray& column, KeyColumnArray& temp) {
  if (IsBoolean(column.get_metadata())) {
    return TransformBoolean::ArrayReplace(column, temp);
  }
  return column;
}

void KeyEncoder::EncoderInteger::PreEncode(const KeyColumnArray& input,
                                           KeyColumnArray& output,
                                           KeyEncoderContext* ctx) {
  if (IsBoolean(input.get_metadata())) {
    TransformBoolean::PreEncode(input, output, ctx);
  }
}

void KeyEncoder::EncoderInteger::PostDecode(const KeyColumnArray& input,
                                            KeyColumnArray& output,
                                            KeyEncoderContext* ctx) {
  if (IsBoolean(input.get_metadata())) {
    TransformBoolean::PostDecode(input, output, ctx);
  }
}

void KeyEncoder::EncoderInteger::Encode(uint32_t* offset_within_row, KeyRowArray& rows,
                                        const KeyColumnArray& col, KeyEncoderContext* ctx,
                                        KeyColumnArray& temp) {
  KeyColumnArray col_prep;
  if (UsesTransform(col)) {
    col_prep = ArrayReplace(col, temp);
    PreEncode(col, col_prep, ctx);
  } else {
    col_prep = col;
  }

  uint32_t num_rows = static_cast<uint32_t>(col.get_length());

  // When we have a single fixed length column we can just do memcpy
  if (rows.get_metadata().is_fixed_length &&
      rows.get_metadata().fixed_length == col.get_metadata().fixed_length) {
    ARROW_DCHECK(*offset_within_row == 0);
    uint32_t row_size = col.get_metadata().fixed_length;
    memcpy(rows.mutable_data(1), col.data(1), num_rows * row_size);
  } else if (rows.get_metadata().is_fixed_length) {
    uint32_t row_size = rows.get_metadata().fixed_length;
    uint8_t* row_base = rows.mutable_data(1) + *offset_within_row;
    const uint8_t* col_base = col_prep.data(1);
    switch (col_prep.get_metadata().fixed_length) {
      case 1:
        for (uint32_t i = 0; i < num_rows; ++i) {
          row_base[i * row_size] = col_base[i];
        }
        break;
      case 2:
        for (uint32_t i = 0; i < num_rows; ++i) {
          *reinterpret_cast<uint16_t*>(row_base + i * row_size) =
              reinterpret_cast<const uint16_t*>(col_base)[i];
        }
        break;
      case 4:
        for (uint32_t i = 0; i < num_rows; ++i) {
          *reinterpret_cast<uint32_t*>(row_base + i * row_size) =
              reinterpret_cast<const uint32_t*>(col_base)[i];
        }
        break;
      case 8:
        for (uint32_t i = 0; i < num_rows; ++i) {
          *reinterpret_cast<uint64_t*>(row_base + i * row_size) =
              reinterpret_cast<const uint64_t*>(col_base)[i];
        }
        break;
      default:
        ARROW_DCHECK(false);
    }
  } else {
    const uint32_t* row_offsets = rows.get_offsets();
    uint8_t* row_base = rows.mutable_data(2) + *offset_within_row;
    const uint8_t* col_base = col_prep.data(1);
    switch (col_prep.get_metadata().fixed_length) {
      case 1:
        for (uint32_t i = 0; i < num_rows; ++i) {
          row_base[row_offsets[i]] = col_base[i];
        }
        break;
      case 2:
        for (uint32_t i = 0; i < num_rows; ++i) {
          *reinterpret_cast<uint16_t*>(row_base + row_offsets[i]) =
              reinterpret_cast<const uint16_t*>(col_base)[i];
        }
        break;
      case 4:
        for (uint32_t i = 0; i < num_rows; ++i) {
          *reinterpret_cast<uint32_t*>(row_base + row_offsets[i]) =
              reinterpret_cast<const uint32_t*>(col_base)[i];
        }
        break;
      case 8:
        for (uint32_t i = 0; i < num_rows; ++i) {
          *reinterpret_cast<uint64_t*>(row_base + row_offsets[i]) =
              reinterpret_cast<const uint64_t*>(col_base)[i];
        }
        break;
      default:
        ARROW_DCHECK(false);
    }
  }

  *offset_within_row += col_prep.get_metadata().fixed_length;
}

void KeyEncoder::EncoderInteger::Decode(uint32_t start_row, uint32_t num_rows,
                                        uint32_t* offset_within_row,
                                        const KeyRowArray& rows, KeyColumnArray& col,
                                        KeyEncoderContext* ctx, KeyColumnArray& temp) {
  KeyColumnArray col_prep;
  if (UsesTransform(col)) {
    col_prep = ArrayReplace(col, temp);
  } else {
    col_prep = col;
  }

  // When we have a single fixed length column we can just do memcpy
  if (rows.get_metadata().is_fixed_length &&
      col_prep.get_metadata().fixed_length == rows.get_metadata().fixed_length) {
    ARROW_DCHECK(*offset_within_row == 0);
    uint32_t row_size = rows.get_metadata().fixed_length;
    memcpy(col_prep.mutable_data(1), rows.data(1) + start_row * row_size,
           num_rows * row_size);
  } else if (rows.get_metadata().is_fixed_length) {
    uint32_t row_size = rows.get_metadata().fixed_length;
    const uint8_t* row_base = rows.data(1) + start_row * row_size;
    row_base += *offset_within_row;
    uint8_t* col_base = col_prep.mutable_data(1);
    switch (col_prep.get_metadata().fixed_length) {
      case 1:
        for (uint32_t i = 0; i < num_rows; ++i) {
          col_base[i] = row_base[i * row_size];
        }
        break;
      case 2:
        for (uint32_t i = 0; i < num_rows; ++i) {
          reinterpret_cast<uint16_t*>(col_base)[i] =
              *reinterpret_cast<const uint16_t*>(row_base + i * row_size);
        }
        break;
      case 4:
        for (uint32_t i = 0; i < num_rows; ++i) {
          reinterpret_cast<uint32_t*>(col_base)[i] =
              *reinterpret_cast<const uint32_t*>(row_base + i * row_size);
        }
        break;
      case 8:
        for (uint32_t i = 0; i < num_rows; ++i) {
          reinterpret_cast<uint64_t*>(col_base)[i] =
              *reinterpret_cast<const uint64_t*>(row_base + i * row_size);
        }
        break;
      default:
        ARROW_DCHECK(false);
    }
  } else {
    const uint32_t* row_offsets = rows.get_offsets() + start_row;
    const uint8_t* row_base = rows.data(2) + row_offsets[0];
    row_base += *offset_within_row;
    uint8_t* col_base = col_prep.mutable_data(1);
    switch (col_prep.get_metadata().fixed_length) {
      case 1:
        for (uint32_t i = 0; i < num_rows; ++i) {
          col_base[i] = row_base[row_offsets[i]];
        }
        break;
      case 2:
        for (uint32_t i = 0; i < num_rows; ++i) {
          reinterpret_cast<uint16_t*>(col_base)[i] =
              *reinterpret_cast<const uint16_t*>(row_base + row_offsets[i]);
        }
        break;
      case 4:
        for (uint32_t i = 0; i < num_rows; ++i) {
          reinterpret_cast<uint32_t*>(col_base)[i] =
              *reinterpret_cast<const uint32_t*>(row_base + row_offsets[i]);
        }
        break;
      case 8:
        for (uint32_t i = 0; i < num_rows; ++i) {
          reinterpret_cast<uint64_t*>(col_base)[i] =
              *reinterpret_cast<const uint64_t*>(row_base + row_offsets[i]);
        }
        break;
      default:
        ARROW_DCHECK(false);
    }
  }

  if (UsesTransform(col)) {
    PostDecode(col_prep, col, ctx);
  }

  *offset_within_row += col_prep.get_metadata().fixed_length;
}

bool KeyEncoder::EncoderBinary::IsInteger(const KeyColumnMetadata& metadata) {
  bool is_fixed_length = metadata.is_fixed_length;
  auto size = metadata.fixed_length;
  return is_fixed_length &&
         (size == 0 || size == 1 || size == 2 || size == 4 || size == 8);
}

void KeyEncoder::EncoderBinary::Encode(uint32_t* offset_within_row, KeyRowArray& rows,
                                       const KeyColumnArray& col, KeyEncoderContext* ctx,
                                       KeyColumnArray& temp) {
  uint32_t offset_within_row_before = *offset_within_row;

  if (IsInteger(col.get_metadata())) {
    EncoderInteger::Encode(offset_within_row, rows, col, ctx, temp);
  } else {
    KeyColumnArray col_prep;
    if (EncoderInteger::UsesTransform(col)) {
      col_prep = EncoderInteger::ArrayReplace(col, temp);
      EncoderInteger::PreEncode(col, col_prep, ctx);
    } else {
      col_prep = col;
    }

    bool is_row_fixed_length = rows.get_metadata().is_fixed_length;

#if defined(ARROW_HAVE_AVX2)
    if (ctx->has_avx2()) {
      if (is_row_fixed_length) {
        EncodeImp_avx2<true>(*offset_within_row, rows, col);
      } else {
        EncodeImp_avx2<false>(*offset_within_row, rows, col);
      }
    } else {
#endif
      if (is_row_fixed_length) {
        EncodeImp<true>(*offset_within_row, rows, col);
      } else {
        EncodeImp<false>(*offset_within_row, rows, col);
      }
#if defined(ARROW_HAVE_AVX2)
    }
#endif

    *offset_within_row += col.get_metadata().fixed_length;
  }

  ARROW_DCHECK(temp.get_metadata().is_fixed_length);
  ARROW_DCHECK(temp.get_length() * temp.get_metadata().fixed_length >=
               col.get_length() * static_cast<int64_t>(sizeof(uint16_t)));

  KeyColumnArray temp16bit(KeyColumnMetadata(true, sizeof(uint16_t)), col.get_length(),
                           nullptr, temp.mutable_data(1), nullptr);
  ColumnMemsetNulls(offset_within_row_before, rows, col, ctx, temp16bit, 0xae);
}

void KeyEncoder::EncoderBinary::Decode(uint32_t start_row, uint32_t num_rows,
                                       uint32_t* offset_within_row,
                                       const KeyRowArray& rows, KeyColumnArray& col,
                                       KeyEncoderContext* ctx, KeyColumnArray& temp) {
  if (IsInteger(col.get_metadata())) {
    EncoderInteger::Decode(start_row, num_rows, offset_within_row, rows, col, ctx, temp);
  } else {
    KeyColumnArray col_prep;
    if (EncoderInteger::UsesTransform(col)) {
      col_prep = EncoderInteger::ArrayReplace(col, temp);
    } else {
      col_prep = col;
    }

    bool is_row_fixed_length = rows.get_metadata().is_fixed_length;

#if defined(ARROW_HAVE_AVX2)
    if (ctx->has_avx2()) {
      if (is_row_fixed_length) {
        DecodeImp_avx2<true>(start_row, num_rows, *offset_within_row, rows, col);
      } else {
        DecodeImp_avx2<false>(start_row, num_rows, *offset_within_row, rows, col);
      }
    } else {
#endif
      if (is_row_fixed_length) {
        DecodeImp<true>(start_row, num_rows, *offset_within_row, rows, col);
      } else {
        DecodeImp<false>(start_row, num_rows, *offset_within_row, rows, col);
      }
#if defined(ARROW_HAVE_AVX2)
    }
#endif

    if (EncoderInteger::UsesTransform(col)) {
      EncoderInteger::PostDecode(col_prep, col, ctx);
    }

    *offset_within_row += col.get_metadata().fixed_length;
  }
}

template <bool is_row_fixed_length>
void KeyEncoder::EncoderBinary::EncodeImp(uint32_t offset_within_row, KeyRowArray& rows,
                                          const KeyColumnArray& col) {
  EncodeDecodeHelper<is_row_fixed_length, true>(
      0, static_cast<uint32_t>(col.get_length()), offset_within_row, &rows, &rows, &col,
      nullptr, [](uint8_t* dst, const uint8_t* src, int64_t length) {
        for (uint32_t istripe = 0; istripe < (length + 7) / 8; ++istripe) {
          uint64_t* dst64 = reinterpret_cast<uint64_t*>(dst);
          const uint64_t* src64 = reinterpret_cast<const uint64_t*>(src);
          dst64[istripe] = src64[istripe];
        }
      });
}

template <bool is_row_fixed_length>
void KeyEncoder::EncoderBinary::DecodeImp(uint32_t start_row, uint32_t num_rows,
                                          uint32_t offset_within_row,
                                          const KeyRowArray& rows, KeyColumnArray& col) {
  EncodeDecodeHelper<is_row_fixed_length, false>(
      start_row, num_rows, offset_within_row, &rows, nullptr, &col, &col,
      [](uint8_t* dst, const uint8_t* src, int64_t length) {
        for (uint32_t istripe = 0; istripe < (length + 7) / 8; ++istripe) {
          uint64_t* dst64 = reinterpret_cast<uint64_t*>(dst);
          const uint64_t* src64 = reinterpret_cast<const uint64_t*>(src);
          dst64[istripe] = src64[istripe];
        }
      });
}

void KeyEncoder::EncoderBinary::ColumnMemsetNulls(
    uint32_t offset_within_row, KeyRowArray& rows, const KeyColumnArray& col,
    KeyEncoderContext* ctx, KeyColumnArray& temp_vector_16bit, uint8_t byte_value) {
  void (*ColumnMemsetNullsImp_fn[])(uint32_t, KeyRowArray&, const KeyColumnArray&,
                                    KeyEncoderContext*, KeyColumnArray&, uint8_t) = {
      ColumnMemsetNullsImp<false, 1>,  ColumnMemsetNullsImp<false, 2>,
      ColumnMemsetNullsImp<false, 4>,  ColumnMemsetNullsImp<false, 8>,
      ColumnMemsetNullsImp<false, 16>, ColumnMemsetNullsImp<true, 1>,
      ColumnMemsetNullsImp<true, 2>,   ColumnMemsetNullsImp<true, 4>,
      ColumnMemsetNullsImp<true, 8>,   ColumnMemsetNullsImp<true, 16>};
  uint32_t col_width = col.get_metadata().fixed_length;
  int dispatch_const =
      (rows.get_metadata().is_fixed_length ? 5 : 0) +
      (col_width == 1 ? 0
                      : col_width == 2 ? 1 : col_width == 4 ? 2 : col_width == 8 ? 3 : 4);
  ColumnMemsetNullsImp_fn[dispatch_const](offset_within_row, rows, col, ctx,
                                          temp_vector_16bit, byte_value);
}

template <bool is_row_fixed_length, uint32_t col_width>
void KeyEncoder::EncoderBinary::ColumnMemsetNullsImp(
    uint32_t offset_within_row, KeyRowArray& rows, const KeyColumnArray& col,
    KeyEncoderContext* ctx, KeyColumnArray& temp_vector_16bit, uint8_t byte_value) {
  // Nothing to do when there are no nulls
  if (!col.data(0)) {
    return;
  }

  uint32_t num_rows = static_cast<uint32_t>(col.get_length());

  // Temp vector needs space for the required number of rows
  ARROW_DCHECK(temp_vector_16bit.get_length() >= num_rows);
  ARROW_DCHECK(temp_vector_16bit.get_metadata().is_fixed_length &&
               temp_vector_16bit.get_metadata().fixed_length == sizeof(uint16_t));
  uint16_t* temp_vector = reinterpret_cast<uint16_t*>(temp_vector_16bit.mutable_data(1));

  // Bit vector to index vector of null positions
  int num_selected;
  util::BitUtil::bits_to_indexes<0>(ctx->instr, static_cast<int>(col.get_length()),
                                    col.data(0), num_selected, temp_vector);

  for (int i = 0; i < num_selected; ++i) {
    uint32_t row_id = temp_vector[i];

    // Target binary field pointer
    uint8_t* dst;
    if (is_row_fixed_length) {
      dst = rows.mutable_data(1) + rows.get_metadata().fixed_length * row_id;
    } else {
      dst = rows.mutable_data(2) + rows.get_offsets()[row_id];
    }
    dst += offset_within_row;

    if (col_width == 1) {
      *dst = byte_value;
    } else if (col_width == 2) {
      *reinterpret_cast<uint16_t*>(dst) =
          (static_cast<uint16_t>(byte_value) * static_cast<uint16_t>(0x0101));
    } else if (col_width == 4) {
      *reinterpret_cast<uint32_t*>(dst) =
          (static_cast<uint32_t>(byte_value) * static_cast<uint32_t>(0x01010101));
    } else if (col_width == 8) {
      *reinterpret_cast<uint64_t*>(dst) =
          (static_cast<uint64_t>(byte_value) * 0x0101010101010101ULL);
    } else {
      uint64_t value = (static_cast<uint64_t>(byte_value) * 0x0101010101010101ULL);
      uint32_t col_width_actual = col.get_metadata().fixed_length;
      uint32_t j;
      for (j = 0; j < col_width_actual / 8; ++j) {
        reinterpret_cast<uint64_t*>(dst)[j] = value;
      }
      int tail = col_width_actual % 8;
      if (tail) {
        uint64_t mask = ~0ULL >> (8 * (8 - tail));
        reinterpret_cast<uint64_t*>(dst)[j] =
            (reinterpret_cast<const uint64_t*>(dst)[j] & ~mask) | (value & mask);
      }
    }
  }
}

void KeyEncoder::EncoderBinaryPair::Encode(uint32_t* offset_within_row, KeyRowArray& rows,
                                           const KeyColumnArray& col1,
                                           const KeyColumnArray& col2,
                                           KeyEncoderContext* ctx, KeyColumnArray& temp1,
                                           KeyColumnArray& temp2) {
  ARROW_DCHECK(CanProcessPair(col1.get_metadata(), col2.get_metadata()));

  KeyColumnArray col_prep[2];
  if (EncoderInteger::UsesTransform(col1)) {
    col_prep[0] = EncoderInteger::ArrayReplace(col1, temp1);
    EncoderInteger::PreEncode(col1, col_prep[0], ctx);
  } else {
    col_prep[0] = col1;
  }
  if (EncoderInteger::UsesTransform(col2)) {
    col_prep[1] = EncoderInteger::ArrayReplace(col2, temp2);
    EncoderInteger::PreEncode(col2, col_prep[1], ctx);
  } else {
    col_prep[1] = col2;
  }

  uint32_t col_width1 = col_prep[0].get_metadata().fixed_length;
  uint32_t col_width2 = col_prep[1].get_metadata().fixed_length;
  int log_col_width1 =
      col_width1 == 8 ? 3 : col_width1 == 4 ? 2 : col_width1 == 2 ? 1 : 0;
  int log_col_width2 =
      col_width2 == 8 ? 3 : col_width2 == 4 ? 2 : col_width2 == 2 ? 1 : 0;

  bool is_row_fixed_length = rows.get_metadata().is_fixed_length;

  uint32_t num_rows = static_cast<uint32_t>(col1.get_length());
  uint32_t num_processed = 0;
#if defined(ARROW_HAVE_AVX2)
  if (ctx->has_avx2() && col_width1 == col_width2) {
    uint32_t (*EncodeImp_avx2_fn[])(uint32_t, KeyRowArray&, const KeyColumnArray&,
                                    const KeyColumnArray&) = {
        EncodeImp_avx2<false, 1>, EncodeImp_avx2<false, 2>, EncodeImp_avx2<false, 4>,
        EncodeImp_avx2<false, 8>, EncodeImp_avx2<true, 1>,  EncodeImp_avx2<true, 2>,
        EncodeImp_avx2<true, 4>,  EncodeImp_avx2<true, 8>};
    int dispatch_const = (is_row_fixed_length ? 4 : 0) + log_col_width1;
    num_processed = EncodeImp_avx2_fn[dispatch_const](*offset_within_row, rows,
                                                      col_prep[0], col_prep[1]);
  }
#endif
  if (num_processed < num_rows) {
    void (*EncodeImp_fn[])(uint32_t, uint32_t, KeyRowArray&, const KeyColumnArray&,
                           const KeyColumnArray&) = {
        EncodeImp<false, 1, 1>, EncodeImp<false, 2, 1>, EncodeImp<false, 4, 1>,
        EncodeImp<false, 8, 1>, EncodeImp<false, 1, 2>, EncodeImp<false, 2, 2>,
        EncodeImp<false, 4, 2>, EncodeImp<false, 8, 2>, EncodeImp<false, 1, 4>,
        EncodeImp<false, 2, 4>, EncodeImp<false, 4, 4>, EncodeImp<false, 8, 4>,
        EncodeImp<false, 1, 8>, EncodeImp<false, 2, 8>, EncodeImp<false, 4, 8>,
        EncodeImp<false, 8, 8>, EncodeImp<true, 1, 1>,  EncodeImp<true, 2, 1>,
        EncodeImp<true, 4, 1>,  EncodeImp<true, 8, 1>,  EncodeImp<true, 1, 2>,
        EncodeImp<true, 2, 2>,  EncodeImp<true, 4, 2>,  EncodeImp<true, 8, 2>,
        EncodeImp<true, 1, 4>,  EncodeImp<true, 2, 4>,  EncodeImp<true, 4, 4>,
        EncodeImp<true, 8, 4>,  EncodeImp<true, 1, 8>,  EncodeImp<true, 2, 8>,
        EncodeImp<true, 4, 8>,  EncodeImp<true, 8, 8>};
    int dispatch_const = (log_col_width2 << 2) | log_col_width1;
    dispatch_const += (is_row_fixed_length ? 16 : 0);
    EncodeImp_fn[dispatch_const](num_processed, *offset_within_row, rows, col_prep[0],
                                 col_prep[1]);
  }
  *offset_within_row += col_width1 + col_width2;
}

template <bool is_row_fixed_length, int col_width_1, int col_width_2>
void KeyEncoder::EncoderBinaryPair::EncodeImp(uint32_t num_rows_to_skip,
                                              uint32_t offset_within_row,
                                              KeyRowArray& rows,
                                              const KeyColumnArray& col1,
                                              const KeyColumnArray& col2) {
  const uint8_t* src_A = col1.data(1);
  const uint8_t* src_B = col2.data(1);
  uint32_t col_width_A = col1.get_metadata().fixed_length;
  uint32_t col_width_B = col2.get_metadata().fixed_length;

  uint32_t num_rows = static_cast<uint32_t>(col1.get_length());

  for (uint32_t i = num_rows_to_skip; i < num_rows; ++i) {
    uint8_t* dst;
    if (is_row_fixed_length) {
      dst =
          rows.mutable_data(1) + rows.get_metadata().fixed_length * i + offset_within_row;
    } else {
      dst = rows.mutable_data(2) + rows.get_offsets()[i] + offset_within_row;
    }
    if (col_width_A == 1) {
      *dst = src_A[i];
    } else if (col_width_A == 2) {
      *reinterpret_cast<uint16_t*>(dst) = reinterpret_cast<const uint16_t*>(src_A)[i];
    } else if (col_width_A == 4) {
      *reinterpret_cast<uint32_t*>(dst) = reinterpret_cast<const uint32_t*>(src_A)[i];
    } else if (col_width_A == 8) {
      *reinterpret_cast<uint64_t*>(dst) = reinterpret_cast<const uint64_t*>(src_A)[i];
    }
    dst += col_width_A;
    if (col_width_B == 1) {
      *dst = src_B[i];
    } else if (col_width_B == 2) {
      *reinterpret_cast<uint16_t*>(dst) = reinterpret_cast<const uint16_t*>(src_B)[i];
    } else if (col_width_B == 4) {
      *reinterpret_cast<uint32_t*>(dst) = reinterpret_cast<const uint32_t*>(src_B)[i];
    } else if (col_width_B == 8) {
      *reinterpret_cast<uint64_t*>(dst) = reinterpret_cast<const uint64_t*>(src_B)[i];
    }
  }
}

void KeyEncoder::EncoderBinaryPair::Decode(uint32_t start_row, uint32_t num_rows,
                                           uint32_t* offset_within_row,
                                           const KeyRowArray& rows, KeyColumnArray& col1,
                                           KeyColumnArray& col2, KeyEncoderContext* ctx,
                                           KeyColumnArray& temp1, KeyColumnArray& temp2) {
  ARROW_DCHECK(CanProcessPair(col1.get_metadata(), col2.get_metadata()));

  KeyColumnArray col_prep[2];
  if (EncoderInteger::UsesTransform(col1)) {
    col_prep[0] = EncoderInteger::ArrayReplace(col1, temp1);
  } else {
    col_prep[0] = col1;
  }
  if (EncoderInteger::UsesTransform(col2)) {
    col_prep[1] = EncoderInteger::ArrayReplace(col2, temp2);
  } else {
    col_prep[1] = col2;
  }

  uint32_t col_width1 = col_prep[0].get_metadata().fixed_length;
  uint32_t col_width2 = col_prep[1].get_metadata().fixed_length;
  int log_col_width1 =
      col_width1 == 8 ? 3 : col_width1 == 4 ? 2 : col_width1 == 2 ? 1 : 0;
  int log_col_width2 =
      col_width2 == 8 ? 3 : col_width2 == 4 ? 2 : col_width2 == 2 ? 1 : 0;

  bool is_row_fixed_length = rows.get_metadata().is_fixed_length;

  uint32_t num_processed = 0;
#if defined(ARROW_HAVE_AVX2)
  if (ctx->has_avx2() && col_width1 == col_width2) {
    uint32_t (*DecodeImp_avx2_fn[])(uint32_t, uint32_t, uint32_t, const KeyRowArray&,
                                    KeyColumnArray&, KeyColumnArray&) = {
        DecodeImp_avx2<false, 1>, DecodeImp_avx2<false, 2>, DecodeImp_avx2<false, 4>,
        DecodeImp_avx2<false, 8>, DecodeImp_avx2<true, 1>,  DecodeImp_avx2<true, 2>,
        DecodeImp_avx2<true, 4>,  DecodeImp_avx2<true, 8>};
    int dispatch_const = log_col_width1 | (is_row_fixed_length ? 4 : 0);
    num_processed = DecodeImp_avx2_fn[dispatch_const](
        start_row, num_processed, *offset_within_row, rows, col_prep[0], col_prep[1]);
  }
#endif
  if (num_processed < num_rows) {
    void (*DecodeImp_fn[])(uint32_t, uint32_t, uint32_t, uint32_t, const KeyRowArray&,
                           KeyColumnArray&, KeyColumnArray&) = {
        DecodeImp<false, 1, 1>, DecodeImp<false, 2, 1>, DecodeImp<false, 4, 1>,
        DecodeImp<false, 8, 1>, DecodeImp<false, 1, 2>, DecodeImp<false, 2, 2>,
        DecodeImp<false, 4, 2>, DecodeImp<false, 8, 2>, DecodeImp<false, 1, 4>,
        DecodeImp<false, 2, 4>, DecodeImp<false, 4, 4>, DecodeImp<false, 8, 4>,
        DecodeImp<false, 1, 8>, DecodeImp<false, 2, 8>, DecodeImp<false, 4, 8>,
        DecodeImp<false, 8, 8>, DecodeImp<true, 1, 1>,  DecodeImp<true, 2, 1>,
        DecodeImp<true, 4, 1>,  DecodeImp<true, 8, 1>,  DecodeImp<true, 1, 2>,
        DecodeImp<true, 2, 2>,  DecodeImp<true, 4, 2>,  DecodeImp<true, 8, 2>,
        DecodeImp<true, 1, 4>,  DecodeImp<true, 2, 4>,  DecodeImp<true, 4, 4>,
        DecodeImp<true, 8, 4>,  DecodeImp<true, 1, 8>,  DecodeImp<true, 2, 8>,
        DecodeImp<true, 4, 8>,  DecodeImp<true, 8, 8>};
    int dispatch_const =
        (log_col_width2 << 2) | log_col_width1 | (is_row_fixed_length ? 16 : 0);
    DecodeImp_fn[dispatch_const](num_processed, start_row, num_rows, *offset_within_row,
                                 rows, col_prep[0], col_prep[1]);
  }

  if (EncoderInteger::UsesTransform(col1)) {
    EncoderInteger::PostDecode(col_prep[0], col1, ctx);
  }
  if (EncoderInteger::UsesTransform(col2)) {
    EncoderInteger::PostDecode(col_prep[1], col2, ctx);
  }

  *offset_within_row += col_width1 + col_width2;
}

template <bool is_row_fixed_length, int col_width_1, int col_width_2>
void KeyEncoder::EncoderBinaryPair::DecodeImp(uint32_t num_rows_to_skip,
                                              uint32_t start_row, uint32_t num_rows,
                                              uint32_t offset_within_row,
                                              const KeyRowArray& rows,
                                              KeyColumnArray& col1,
                                              KeyColumnArray& col2) {
  ARROW_DCHECK(rows.get_length() >= start_row + num_rows);
  ARROW_DCHECK(col1.get_length() == num_rows && col2.get_length() == num_rows);

  uint8_t* dst_A = col1.mutable_data(1);
  uint8_t* dst_B = col2.mutable_data(1);
  uint32_t col_width_A = col1.get_metadata().fixed_length;
  uint32_t col_width_B = col2.get_metadata().fixed_length;

  for (uint32_t i = num_rows_to_skip; i < num_rows; ++i) {
    const uint8_t* src;
    if (is_row_fixed_length) {
      src = rows.data(1) + rows.get_metadata().fixed_length * i + offset_within_row;
    } else {
      src = rows.data(2) + rows.get_offsets()[i] + offset_within_row;
    }
    if (col_width_A == 1) {
      dst_A[i] = *src;
    } else if (col_width_A == 2) {
      reinterpret_cast<uint16_t*>(dst_A)[i] = *reinterpret_cast<const uint16_t*>(src);
    } else if (col_width_A == 4) {
      reinterpret_cast<uint32_t*>(dst_A)[i] = *reinterpret_cast<const uint32_t*>(src);
    } else if (col_width_A == 8) {
      reinterpret_cast<uint64_t*>(dst_A)[i] = *reinterpret_cast<const uint64_t*>(src);
    }
    src += col_width_A;
    if (col_width_B == 1) {
      dst_B[i] = *src;
    } else if (col_width_B == 2) {
      reinterpret_cast<uint16_t*>(dst_B)[i] = *reinterpret_cast<const uint16_t*>(src);
    } else if (col_width_B == 4) {
      reinterpret_cast<uint32_t*>(dst_B)[i] = *reinterpret_cast<const uint32_t*>(src);
    } else if (col_width_B == 8) {
      reinterpret_cast<uint64_t*>(dst_B)[i] = *reinterpret_cast<const uint64_t*>(src);
    }
  }
}

void KeyEncoder::EncoderOffsets::Encode(KeyRowArray& rows,
                                        const std::vector<KeyColumnArray>& varbinary_cols,
                                        KeyEncoderContext* ctx) {
  ARROW_DCHECK(!varbinary_cols.empty());

  // Rows and columns must all be varying-length
  ARROW_DCHECK(!rows.get_metadata().is_fixed_length);
  for (size_t col = 0; col < varbinary_cols.size(); ++col) {
    ARROW_DCHECK(!varbinary_cols[col].get_metadata().is_fixed_length);
  }

  uint32_t num_rows = static_cast<uint32_t>(varbinary_cols[0].get_length());

  // The space in columns must be exactly equal to a space for offsets in rows
  ARROW_DCHECK(rows.get_length() == num_rows);
  for (size_t col = 0; col < varbinary_cols.size(); ++col) {
    ARROW_DCHECK(varbinary_cols[col].get_length() == num_rows);
  }

  uint32_t num_processed = 0;
#if defined(ARROW_HAVE_AVX2)
  if (ctx->has_avx2()) {
    // Create a temp vector sized based on the number of columns
    auto temp_buffer_holder = util::TempVectorHolder<uint32_t>(
        ctx->stack, static_cast<uint32_t>(varbinary_cols.size()) * 8);
    auto temp_buffer_32B_per_col =
        KeyColumnArray(KeyColumnMetadata(true, sizeof(uint32_t)), varbinary_cols.size() * 8,
                        nullptr, temp_buffer_holder.mutable_data(), nullptr);

    num_processed = EncodeImp_avx2(rows, varbinary_cols, temp_buffer_32B_per_col);
  }
#endif
  if (num_processed < num_rows) {
    EncodeImp(num_processed, rows, varbinary_cols);
  }
}

void KeyEncoder::EncoderOffsets::EncodeImp(
    uint32_t num_rows_to_skip, KeyRowArray& rows,
    const std::vector<KeyColumnArray>& varbinary_cols) {
  ARROW_DCHECK(varbinary_cols.size() > 0);

  // Add together first offset in every column.
  // This value will be mapped to zero when
  // computing varying-length component of output row offsets.
  int64_t sum_offsets_first = 0;
  for (size_t col = 0; col < varbinary_cols.size(); ++col) {
    ARROW_DCHECK(!varbinary_cols[col].get_metadata().is_fixed_length);
    const uint32_t* col_offsets = varbinary_cols[col].get_offsets();
    sum_offsets_first += col_offsets[0];
  }

  // There is a fixed-length part in every row.
  // This needs to be included in calculation of row offsets.
  int64_t fixed_part =
      rows.get_metadata().fixed_length + rows.get_metadata().cumulative_lengths_length;

  // Difference between output row offset and direct sum of offsets for all
  // columns
  int64_t offset_adjustment = fixed_part - sum_offsets_first;
  int64_t offset_adjustment_incr = fixed_part;

  uint32_t* row_offsets = rows.get_mutable_offsets();
  uint8_t* row_values = rows.mutable_data(2);
  uint32_t num_rows = static_cast<uint32_t>(varbinary_cols[0].get_length());

  if (num_rows_to_skip == 0) {
    row_offsets[0] = 0;
  }
  for (uint32_t i = num_rows_to_skip; i < num_rows; ++i) {
    uint32_t* cumulative_lengths = reinterpret_cast<uint32_t*>(
        row_values + row_offsets[i] + rows.get_metadata().fixed_length);

    // Zero out lengths for nulls.
    // Add horizontally offsets, after adjustment for nulls, to
    // produce row offsets.
    // Add horizontally and store in temp buffer lengths,
    // after adjustment for nulls,
    // to produce cumulative lengths of individual column values in each row.
    int64_t col_null_length_sum = 0;
    int64_t col_offset_sum = 0;
    int64_t col_length_sum = 0;
    for (size_t col = 0; col < varbinary_cols.size(); ++col) {
      const uint32_t* col_offsets = varbinary_cols[col].get_offsets();
      uint32_t col_offset = col_offsets[i + 1];
      uint32_t col_length = col_offset - col_offsets[i];

      const uint8_t* non_nulls = varbinary_cols[col].data(0);
      if (non_nulls && BitUtil::GetBit(non_nulls, i) == 0) {
        col_null_length_sum += col_length;
        col_offset -= col_length;
        col_length = 0;
      }

      col_offset_sum += col_offset;
      col_length_sum += col_length;

      cumulative_lengths[col] = static_cast<uint32_t>(col_length_sum);
    }

    // Add fixed-length part to row offsets
    int64_t row_offset = col_offset_sum + offset_adjustment;
    offset_adjustment += offset_adjustment_incr;

    row_offsets[i + 1] = static_cast<uint32_t>(row_offset);
  }
}

void KeyEncoder::EncoderOffsets::Decode(uint32_t start_row, uint32_t num_rows,
                                        KeyRowArray& rows,
                                        std::vector<KeyColumnArray>& varbinary_cols,
                                        std::vector<uint32_t>& varbinary_cols_base_offset,
                                        KeyEncoderContext* ctx) {
  ARROW_DCHECK(!varbinary_cols.empty());
  ARROW_DCHECK(varbinary_cols.size() == varbinary_cols_base_offset.size());

  // Rows and columns must all be varying-length
  ARROW_DCHECK(!rows.get_metadata().is_fixed_length);
  for (size_t col = 0; col < varbinary_cols.size(); ++col) {
    ARROW_DCHECK(!varbinary_cols[col].get_metadata().is_fixed_length);
  }

  // The space in columns must be exactly equal to a subset of rows selected
  ARROW_DCHECK(rows.get_length() >= start_row + num_rows);
  for (size_t col = 0; col < varbinary_cols.size(); ++col) {
    ARROW_DCHECK(varbinary_cols[col].get_length() == num_rows);
  }

  // Offsets of varbinary columns data within each encoded row are stored
  // in the same encoded row as an array of 32-bit integers.
  // This array follows immediately the data of fixed-length columns.
  // There is one element for each varying-length column.
  // The Nth element is the sum of all the lengths of varbinary columns data in
  // that row, up to and including Nth varbinary column.

  const uint32_t* row_offsets = rows.get_offsets() + start_row;

  const uint32_t cumulative_length_offset = rows.get_metadata().fixed_length;

  // Set the base offset for each column
  for (size_t col = 0; col < varbinary_cols.size(); ++col) {
    uint32_t* col_offsets = varbinary_cols[col].get_mutable_offsets();
    col_offsets[0] = varbinary_cols_base_offset[col];
  }

  for (uint32_t i = 0; i < num_rows; ++i) {
    // Find the beginning of cumulative lengths array for next row
    const uint8_t* row = rows.data(2) + row_offsets[i];
    const uint32_t* cumulative_lengths =
        reinterpret_cast<const uint32_t*>(row + cumulative_length_offset);

    // Update the offset of each column
    uint32_t cumulative_length = 0;
    for (size_t col = 0; col < varbinary_cols.size(); ++col) {
      uint32_t length = cumulative_lengths[col] - cumulative_length;
      cumulative_length = cumulative_lengths[col];
      uint32_t* col_offsets = varbinary_cols[col].get_mutable_offsets();
      col_offsets[i + 1] = col_offsets[i] + length;
    }
  }
}

void KeyEncoder::EncoderVarBinary::Encode(uint32_t varbinary_col_id, KeyRowArray& rows,
                                          const KeyColumnArray& col,
                                          KeyEncoderContext* ctx) {
#if defined(ARROW_HAVE_AVX2)
  if (ctx->has_avx2()) {
    if (varbinary_col_id == 0) {
      EncodeImp_avx2<true>(varbinary_col_id, rows, col);
    } else {
      EncodeImp_avx2<false>(varbinary_col_id, rows, col);
    }
  } else {
#endif
    if (varbinary_col_id == 0) {
      EncodeImp<true>(varbinary_col_id, rows, col);
    } else {
      EncodeImp<false>(varbinary_col_id, rows, col);
    }
#if defined(ARROW_HAVE_AVX2)
  }
#endif
}

void KeyEncoder::EncoderVarBinary::Decode(uint32_t start_row, uint32_t num_rows,
                                          uint32_t varbinary_col_id,
                                          const KeyRowArray& rows, KeyColumnArray& col,
                                          KeyEncoderContext* ctx) {
  // Output column varbinary buffer needs an extra 32B
  // at the end in avx2 version and 8B otherwise.
#if defined(ARROW_HAVE_AVX2)
  if (ctx->has_avx2()) {
    if (varbinary_col_id == 0) {
      DecodeImp_avx2<true>(start_row, num_rows, varbinary_col_id, rows, col);
    } else {
      DecodeImp_avx2<false>(start_row, num_rows, varbinary_col_id, rows, col);
    }
  } else {
#endif
    if (varbinary_col_id == 0) {
      DecodeImp<true>(start_row, num_rows, varbinary_col_id, rows, col);
    } else {
      DecodeImp<false>(start_row, num_rows, varbinary_col_id, rows, col);
    }
#if defined(ARROW_HAVE_AVX2)
  }
#endif
}

template <bool first_varbinary_col>
void KeyEncoder::EncoderVarBinary::EncodeImp(uint32_t varbinary_col_id, KeyRowArray& rows,
                                             const KeyColumnArray& col) {
  EncodeDecodeHelper<first_varbinary_col, true>(
      0, static_cast<uint32_t>(col.get_length()), varbinary_col_id, &rows, &rows, &col,
      nullptr, [](uint8_t* dst, const uint8_t* src, int64_t length) {
        uint64_t* dst64 = reinterpret_cast<uint64_t*>(dst);
        const uint64_t* src64 = reinterpret_cast<const uint64_t*>(src);
        uint32_t istripe;
        for (istripe = 0; istripe < length / 8; ++istripe) {
          dst64[istripe] = src64[istripe];
        }
        if ((length % 8) > 0) {
          uint64_t mask_last = ~0ULL >> (8 * (8 * (istripe + 1) - length));
          dst64[istripe] = (dst64[istripe] & ~mask_last) | (src64[istripe] & mask_last);
        }
      });
}

template <bool first_varbinary_col>
void KeyEncoder::EncoderVarBinary::DecodeImp(uint32_t start_row, uint32_t num_rows,
                                             uint32_t varbinary_col_id,
                                             const KeyRowArray& rows,
                                             KeyColumnArray& col) {
  EncodeDecodeHelper<first_varbinary_col, false>(
      start_row, num_rows, varbinary_col_id, &rows, nullptr, &col, &col,
      [](uint8_t* dst, const uint8_t* src, int64_t length) {
        for (uint32_t istripe = 0; istripe < (length + 7) / 8; ++istripe) {
          uint64_t* dst64 = reinterpret_cast<uint64_t*>(dst);
          const uint64_t* src64 = reinterpret_cast<const uint64_t*>(src);
          dst64[istripe] = src64[istripe];
        }
      });
}

void KeyEncoder::EncoderNulls::Encode(KeyRowArray& rows,
                                      const std::vector<KeyColumnArray>& cols,
                                      KeyEncoderContext* ctx,
                                      KeyColumnArray& temp_vector_16bit) {
  ARROW_DCHECK(cols.size() > 0);
  uint32_t num_rows = static_cast<uint32_t>(rows.get_length());

  // All input columns should have the same number of rows.
  // They may or may not have non-nulls bit-vectors allocated.
  for (size_t col = 0; col < cols.size(); ++col) {
    ARROW_DCHECK(cols[col].get_length() == num_rows);
  }

  // Temp vector needs space for the required number of rows
  ARROW_DCHECK(temp_vector_16bit.get_length() >= num_rows);
  ARROW_DCHECK(temp_vector_16bit.get_metadata().is_fixed_length &&
               temp_vector_16bit.get_metadata().fixed_length == sizeof(uint16_t));

  uint8_t* null_masks = rows.get_null_masks();
  uint32_t null_masks_bytes_per_row = rows.get_metadata().null_masks_bytes_per_row;
  memset(null_masks, 0, null_masks_bytes_per_row * num_rows);
  for (size_t col = 0; col < cols.size(); ++col) {
    const uint8_t* non_nulls = cols[col].data(0);
    if (!non_nulls) {
      continue;
    }
    int num_selected;
    util::BitUtil::bits_to_indexes<0>(
        ctx->instr, num_rows, non_nulls, num_selected,
        reinterpret_cast<uint16_t*>(temp_vector_16bit.mutable_data(1)));
    for (int i = 0; i < num_selected; ++i) {
      uint16_t row_id = reinterpret_cast<const uint16_t*>(temp_vector_16bit.data(1))[i];
      int64_t null_masks_bit_id = row_id * null_masks_bytes_per_row * 8 + col;
      BitUtil::SetBit(null_masks, null_masks_bit_id);
    }
  }
}

void KeyEncoder::EncoderNulls::Decode(uint32_t start_row, uint32_t num_rows,
                                      const KeyRowArray& rows,
                                      std::vector<KeyColumnArray>& cols) {
  // Every output column needs to have a space for exactly the required number
  // of rows. It also needs to have non-nulls bit-vector allocated and mutable.
  ARROW_DCHECK(cols.size() > 0);
  for (size_t col = 0; col < cols.size(); ++col) {
    ARROW_DCHECK(cols[col].get_length() == num_rows);
    ARROW_DCHECK(cols[col].mutable_data(0));
  }

  const uint8_t* null_masks = rows.get_null_masks();
  uint32_t null_masks_bytes_per_row = rows.get_metadata().null_masks_bytes_per_row;
  for (size_t col = 0; col < cols.size(); ++col) {
    uint8_t* non_nulls = cols[col].mutable_data(0);
    memset(non_nulls, 0xff, BitUtil::BytesForBits(num_rows));
    for (uint32_t row = 0; row < num_rows; ++row) {
      uint32_t null_masks_bit_id =
          (start_row + row) * null_masks_bytes_per_row * 8 + static_cast<uint32_t>(col);
      bool is_set = BitUtil::GetBit(null_masks, null_masks_bit_id);
      if (is_set) {
        BitUtil::ClearBit(non_nulls, row);
      }
    }
  }
}

}  // namespace exec
}  // namespace arrow
