#include <memory.h>

#include <algorithm>
#include <cstdint>

#include "arrow/compute/exec/key_encode.h"
#include "arrow/compute/exec/key_hash.h"
#include "arrow/compute/exec/util.h"

namespace arrow {
namespace compute {

// From:
// https://www.boost.org/doc/libs/1_37_0/doc/html/hash/reference.html#boost.hash_combine
// template <class T>
// inline void hash_combine(std::size_t& seed, const T& v)
//{
//    std::hash<T> hasher;
//    seed ^= hasher(v) + 0x9e3779b9 + (seed<<6) + (seed>>2);
//}
void Hashing::HashCombine(uint32_t num_rows, uint32_t* accumulated_hash,
                          const uint32_t* next_column_hash) {
  constexpr uint32_t unroll = 8;
  for (uint32_t i = 0; i < num_rows / unroll; ++i) {
    __m256i acc =
        _mm256_loadu_si256(reinterpret_cast<const __m256i*>(accumulated_hash) + i);
    __m256i next =
        _mm256_loadu_si256(reinterpret_cast<const __m256i*>(next_column_hash) + i);
    next = _mm256_add_epi32(next, _mm256_set1_epi32(0x9e3779b9));
    next = _mm256_add_epi32(next, _mm256_slli_epi32(acc, 6));
    next = _mm256_add_epi32(next, _mm256_srli_epi32(acc, 2));
    acc = _mm256_xor_si256(acc, next);
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(accumulated_hash) + i, acc);
  }
  uint32_t num_processed = num_rows / unroll * unroll;
  for (uint32_t i = num_processed; i < num_rows; ++i) {
    uint32_t acc = accumulated_hash[i];
    uint32_t next = next_column_hash[i];
    next += 0x9e3779b9 + (acc << 6) + (acc >> 2);
    acc ^= next;
    accumulated_hash[i] = acc;
  }
}

void Hashing::HashMultiColumn(const std::vector<KeyEncoder::KeyColumnArray>& cols,
                              KeyEncoder::KeyEncoderContext* ctx, uint32_t* out_hash) {
  uint32_t num_rows = static_cast<uint32_t>(cols[0].length());
  auto hash_temp_buf = util::TempVectorHolder<uint32_t>(ctx->stack, num_rows);
  auto hash_null_index_buf = util::TempVectorHolder<uint16_t>(ctx->stack, num_rows);

  bool is_first = true;
  bool has_varbinary = false;

  {
    auto byte_temp_buf = util::TempVectorHolder<uint8_t>(ctx->stack, num_rows);

    for (size_t icol = 0; icol < cols.size(); ++icol) {
      if (cols[icol].metadata().is_fixed_length) {
        uint32_t col_width = cols[icol].metadata().fixed_length;
        if (col_width == 0) {
          util::BitUtil::bits_to_bytes(ctx->hardware_flags, num_rows, cols[icol].data(1),
                                       byte_temp_buf.mutable_data());
        }
        Hashing::hash_fixed(
            ctx->hardware_flags, num_rows, col_width == 0 ? 1 : col_width,
            col_width == 0 ? byte_temp_buf.mutable_data() : cols[icol].data(1),
            is_first ? out_hash : hash_temp_buf.mutable_data());

        // Zero hash for nulls
        if (cols[icol].data(0)) {
          uint32_t* dst_hash = is_first ? out_hash : hash_temp_buf.mutable_data();
          int num_nulls;
          util::BitUtil::bits_to_indexes(0, ctx->hardware_flags, num_rows, cols[icol].data(0),
                                         &num_nulls, hash_null_index_buf.mutable_data());
          for (int i = 0; i < num_nulls; ++i) {
            uint16_t row_id = hash_null_index_buf.mutable_data()[i];
            dst_hash[row_id] = 0;
          }
        }

        if (!is_first) {
          HashCombine(num_rows, out_hash, hash_temp_buf.mutable_data());
        }
        is_first = false;
      } else {
        has_varbinary = true;
      }
    }
  }
  if (has_varbinary) {
    auto varbin_temp_buf = util::TempVectorHolder<uint32_t>(
        ctx->stack, 4 * static_cast<uint32_t>(cols[0].length()));

    for (size_t icol = 0; icol < cols.size(); ++icol) {
      if (!cols[icol].metadata().is_fixed_length) {
        Hashing::hash_varlen(
            ctx->hardware_flags, num_rows, cols[icol].offsets(), cols[icol].data(2),
            varbin_temp_buf.mutable_data(),  // Needs to hold 4 x 32-bit per row
            is_first ? out_hash : hash_temp_buf.mutable_data());

        // Zero hash for nulls
        if (cols[icol].data(0)) {
          uint32_t* dst_hash = is_first ? out_hash : hash_temp_buf.mutable_data();
          int num_nulls;
          util::BitUtil::bits_to_indexes(0, ctx->hardware_flags, num_rows, cols[icol].data(0),
                                         &num_nulls, hash_null_index_buf.mutable_data());
          for (int i = 0; i < num_nulls; ++i) {
            uint16_t row_id = hash_null_index_buf.mutable_data()[i];
            dst_hash[row_id] = 0;
          }
        }

        if (!is_first) {
          HashCombine(num_rows, out_hash, hash_temp_buf.mutable_data());
        }
        is_first = false;
      }
    }
  }
}

}  // namespace compute
}  // namespace arrow
