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

#include <algorithm>

#include "arrow/compute/exec/key_encode.h"
#include "arrow/compute/exec/util.h"
#include "arrow/util/bit_util.h"

namespace arrow {
namespace compute {

template <class COPY_FN, class SET_NULL_FN>
void KeyEncoder::EncoderBinary::EncodeSelectedImp(
    uint32_t offset_within_row, KeyRowArray* rows, const KeyColumnArray& col,
    uint32_t num_selected, const uint16_t* selection, COPY_FN copy_fn,
    SET_NULL_FN set_null_fn) {
  bool is_fixed_length = rows->metadata().is_fixed_length;
  if (is_fixed_length) {
    uint32_t row_width = rows->metadata().fixed_length;
    const uint8_t* src_base = col.data(1);
    uint8_t* dst = rows->mutable_data(1) + offset_within_row;
    for (uint32_t i = 0; i < num_selected; ++i) {
      copy_fn(dst, src_base, selection[i]);
      dst += row_width;
    }
    if (col.data(0)) {
      const uint8_t* non_null_bits = col.data(0);
      uint8_t* dst = rows->mutable_data(1) + offset_within_row;
      for (uint32_t i = 0; i < num_selected; ++i) {
        bool is_null = !BitUtil::GetBit(non_null_bits, selection[i]);
        if (is_null) {
          set_null_fn(dst);
        }
        dst += row_width;
      }
    }
  } else {
    const uint8_t* src_base = col.data(1);
    uint8_t* dst = rows->mutable_data(2) + offset_within_row;
    const uint32_t* offsets = rows->offsets();
    for (uint32_t i = 0; i < num_selected; ++i) {
      copy_fn(dst + offsets[i], src_base, selection[i]);
    }
    if (col.data(0)) {
      const uint8_t* non_null_bits = col.data(0);
      uint8_t* dst = rows->mutable_data(2) + offset_within_row;
      const uint32_t* offsets = rows->offsets();
      for (uint32_t i = 0; i < num_selected; ++i) {
        bool is_null = !BitUtil::GetBit(non_null_bits, selection[i]);
        if (is_null) {
          set_null_fn(dst + offsets[i]);
        }
      }
    }
  }
  uint32_t col_width = col.metadata().fixed_length;
  if (col_width == 0) {
    col_width = 1;
  }
}

void KeyEncoder::EncoderBinary::EncodeSelected(uint32_t offset_within_row,
                                               KeyRowArray* rows,
                                               const KeyColumnArray& col,
                                               uint32_t num_selected,
                                               const uint16_t* selection) {
  uint32_t col_width = col.metadata().fixed_length;
  if (col_width == 0) {
    EncodeSelectedImp(
        offset_within_row, rows, col, num_selected, selection,
        [](uint8_t* dst, const uint8_t* src_base, uint16_t irow) {
          *dst = BitUtil::GetBit(src_base, irow) ? 0xff : 0x00;
        },
        [](uint8_t* dst) { *dst = 0xae; });
  } else if (col_width == 1) {
    EncodeSelectedImp(
        offset_within_row, rows, col, num_selected, selection,
        [](uint8_t* dst, const uint8_t* src_base, uint16_t irow) {
          *dst = src_base[irow];
        },
        [](uint8_t* dst) { *dst = 0xae; });
  } else if (col_width == 2) {
    EncodeSelectedImp(
        offset_within_row, rows, col, num_selected, selection,
        [](uint8_t* dst, const uint8_t* src_base, uint16_t irow) {
          *reinterpret_cast<uint16_t*>(dst) =
              reinterpret_cast<const uint16_t*>(src_base)[irow];
        },
        [](uint8_t* dst) { *reinterpret_cast<uint16_t*>(dst) = 0xaeae; });
  } else if (col_width == 4) {
    EncodeSelectedImp(
        offset_within_row, rows, col, num_selected, selection,
        [](uint8_t* dst, const uint8_t* src_base, uint16_t irow) {
          *reinterpret_cast<uint32_t*>(dst) =
              reinterpret_cast<const uint32_t*>(src_base)[irow];
        },
        [](uint8_t* dst) {
          *reinterpret_cast<uint32_t*>(dst) = static_cast<uint32_t>(0xaeaeaeae);
        });
  } else if (col_width == 8) {
    EncodeSelectedImp(
        offset_within_row, rows, col, num_selected, selection,
        [](uint8_t* dst, const uint8_t* src_base, uint16_t irow) {
          *reinterpret_cast<uint64_t*>(dst) =
              reinterpret_cast<const uint64_t*>(src_base)[irow];
        },
        [](uint8_t* dst) { *reinterpret_cast<uint64_t*>(dst) = 0xaeaeaeaeaeaeaeaeULL; });
  } else {
    EncodeSelectedImp(
        offset_within_row, rows, col, num_selected, selection,
        [col_width](uint8_t* dst, const uint8_t* src_base, uint16_t irow) {
          memcpy(dst, src_base + col_width * irow, col_width);
        },
        [col_width](uint8_t* dst) { memset(dst, 0xae, col_width); });
  }
}

void KeyEncoder::EncoderOffsets::GetRowOffsetsSelected(
    KeyRowArray* rows, const std::vector<KeyColumnArray>& cols, uint32_t num_selected,
    const uint16_t* selection) {
  if (rows->metadata().is_fixed_length) {
    return;
  }

  uint32_t* row_offsets = rows->mutable_offsets();
  for (uint32_t i = 0; i < num_selected; ++i) {
    row_offsets[i] = rows->metadata().fixed_length;
  }

  for (size_t icol = 0; icol < cols.size(); ++icol) {
    bool is_fixed_length = (cols[icol].metadata().is_fixed_length);
    if (!is_fixed_length) {
      const uint32_t* col_offsets = cols[icol].offsets();
      for (uint32_t i = 0; i < num_selected; ++i) {
        uint32_t irow = selection[i];
        uint32_t length = col_offsets[irow + 1] - col_offsets[irow];
        row_offsets[i] += KeyRowMetadata::padding_for_alignment(
            row_offsets[i], rows->metadata().string_alignment);
        row_offsets[i] += length;
      }
      const uint8_t* non_null_bits = cols[icol].data(0);
      if (non_null_bits) {
        const uint32_t* col_offsets = cols[icol].offsets();
        for (uint32_t i = 0; i < num_selected; ++i) {
          uint32_t irow = selection[i];
          bool is_null = !BitUtil::GetBit(non_null_bits, irow);
          if (is_null) {
            uint32_t length = col_offsets[irow + 1] - col_offsets[irow];
            row_offsets[i] -= length;
          }
        }
      }
    }
  }

  uint32_t sum = 0;
  int row_alignment = rows->metadata().row_alignment;
  for (uint32_t i = 0; i < num_selected; ++i) {
    uint32_t length = row_offsets[i];
    length += KeyRowMetadata::padding_for_alignment(length, row_alignment);
    row_offsets[i] = sum;
    sum += length;
  }
  row_offsets[num_selected] = sum;
}

template <bool has_nulls, bool is_first_varbinary>
void KeyEncoder::EncoderOffsets::EncodeSelectedImp(
    uint32_t ivarbinary, KeyRowArray* rows, const std::vector<KeyColumnArray>& cols,
    uint32_t num_selected, const uint16_t* selection) {
  const uint32_t* row_offsets = rows->offsets();
  uint8_t* row_base = rows->mutable_data(2) +
                      rows->metadata().varbinary_end_array_offset +
                      ivarbinary * sizeof(uint32_t);
  const uint32_t* col_offsets = cols[ivarbinary].offsets();
  const uint8_t* col_non_null_bits = cols[ivarbinary].data(0);

  for (uint32_t i = 0; i < num_selected; ++i) {
    uint32_t irow = selection[i];
    uint32_t length = col_offsets[irow + 1] - col_offsets[irow];
    if (has_nulls) {
      uint32_t null_multiplier = BitUtil::GetBit(col_non_null_bits, irow) ? 1 : 0;
      length *= null_multiplier;
    }
    uint32_t* row = reinterpret_cast<uint32_t*>(row_base + row_offsets[i]);
    if (is_first_varbinary) {
      row[0] = rows->metadata().fixed_length + length;
    } else {
      row[0] = row[-1] +
               KeyRowMetadata::padding_for_alignment(row[-1],
                                                     rows->metadata().string_alignment) +
               length;
    }
  }
}

void KeyEncoder::EncoderOffsets::EncodeSelected(KeyRowArray* rows,
                                                const std::vector<KeyColumnArray>& cols,
                                                uint32_t num_selected,
                                                const uint16_t* selection) {
  if (rows->metadata().is_fixed_length) {
    return;
  }
  uint32_t ivarbinary = 0;
  for (size_t icol = 0; icol < cols.size(); ++icol) {
    if (!cols[icol].metadata().is_fixed_length) {
      const uint8_t* non_null_bits = cols[icol].data(0);
      if (non_null_bits && ivarbinary == 0) {
        EncodeSelectedImp<true, true>(ivarbinary, rows, cols, num_selected, selection);
      } else if (non_null_bits && ivarbinary > 0) {
        EncodeSelectedImp<true, false>(ivarbinary, rows, cols, num_selected, selection);
      } else if (!non_null_bits && ivarbinary == 0) {
        EncodeSelectedImp<false, true>(ivarbinary, rows, cols, num_selected, selection);
      } else {
        EncodeSelectedImp<false, false>(ivarbinary, rows, cols, num_selected, selection);
      }
      ivarbinary++;
    }
  }
}

void KeyEncoder::EncoderVarBinary::EncodeSelected(uint32_t ivarbinary, KeyRowArray* rows,
                                                  const KeyColumnArray& cols,
                                                  uint32_t num_selected,
                                                  const uint16_t* selection) {
  const uint32_t* row_offsets = rows->offsets();
  uint8_t* row_base = rows->mutable_data(2);
  const uint32_t* col_offsets = cols.offsets();
  const uint8_t* col_base = cols.data(2);

  if (ivarbinary == 0) {
    for (uint32_t i = 0; i < num_selected; ++i) {
      uint8_t* row = row_base + row_offsets[i];
      uint32_t row_offset;
      uint32_t length;
      rows->metadata().first_varbinary_offset_and_length(row, &row_offset, &length);
      uint32_t irow = selection[i];
      memcpy(row + row_offset, col_base + col_offsets[irow], length);
    }
  } else {
    for (uint32_t i = 0; i < num_selected; ++i) {
      uint8_t* row = row_base + row_offsets[i];
      uint32_t row_offset;
      uint32_t length;
      rows->metadata().nth_varbinary_offset_and_length(row, ivarbinary, &row_offset,
                                                       &length);
      uint32_t irow = selection[i];
      memcpy(row + row_offset, col_base + col_offsets[irow], length);
    }
  }
}

void KeyEncoder::EncoderNulls::EncodeSelected(KeyRowArray* rows,
                                              const std::vector<KeyColumnArray>& cols,
                                              uint32_t num_selected,
                                              const uint16_t* selection) {
  uint8_t* null_masks = rows->null_masks();
  uint32_t null_mask_num_bytes = rows->metadata().null_masks_bytes_per_row;
  memset(null_masks, 0, null_mask_num_bytes * num_selected);
  for (size_t icol = 0; icol < cols.size(); ++icol) {
    const uint8_t* non_null_bits = cols[icol].data(0);
    if (non_null_bits) {
      for (uint32_t i = 0; i < num_selected; ++i) {
        uint32_t irow = selection[i];
        bool is_null = !BitUtil::GetBit(non_null_bits, irow);
        if (is_null) {
          BitUtil::SetBit(null_masks, i * null_mask_num_bytes * 8 + icol);
        }
      }
    }
  }
}

void KeyEncoder::PrepareEncodeSelected(int64_t start_row, int64_t num_rows,
                                       const std::vector<KeyColumnArray>& cols) {
  // Prepare column array vectors
  PrepareKeyColumnArrays(start_row, num_rows, cols);
}

Status KeyEncoder::EncodeSelected(KeyRowArray* rows, uint32_t num_selected,
                                  const uint16_t* selection) {
  rows->Clean();
  RETURN_NOT_OK(
      rows->AppendEmpty(static_cast<uint32_t>(num_selected), static_cast<uint32_t>(0)));

  EncoderOffsets::GetRowOffsetsSelected(rows, batch_varbinary_cols_, num_selected,
                                        selection);

  RETURN_NOT_OK(rows->AppendEmpty(static_cast<uint32_t>(0),
                                  static_cast<uint32_t>(rows->offsets()[num_selected])));

  for (size_t icol = 0; icol < batch_all_cols_.size(); ++icol) {
    if (batch_all_cols_[icol].metadata().is_fixed_length) {
      uint32_t offset_within_row = rows->metadata().column_offsets[icol];
      EncoderBinary::EncodeSelected(offset_within_row, rows, batch_all_cols_[icol],
                                    num_selected, selection);
    }
  }

  EncoderOffsets::EncodeSelected(rows, batch_varbinary_cols_, num_selected, selection);

  for (size_t icol = 0; icol < batch_varbinary_cols_.size(); ++icol) {
    EncoderVarBinary::EncodeSelected(static_cast<uint32_t>(icol), rows,
                                     batch_varbinary_cols_[icol], num_selected,
                                     selection);
  }

  EncoderNulls::EncodeSelected(rows, batch_all_cols_, num_selected, selection);

  return Status::OK();
}

}  // namespace compute
}  // namespace arrow
