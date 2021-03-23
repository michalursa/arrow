#include <immintrin.h>
#include "common.h"
#include "util.h"

namespace arrow {
namespace exec {
namespace util {

#if defined(ARROW_HAVE_AVX2)

template <int bit_to_search>
void BitUtil::bits_to_indexes_avx2(const int num_bits, const uint8_t* bits,
                                   int& num_indexes, uint16_t* indexes) {
  // 64 bits at a time
  constexpr int unroll = 64;

  // The caller takes care of processing the remaining bits at the end outside of the
  // multiples of 64
  ARROW_DCHECK(num_bits % unroll == 0);

  uint8_t byte_indexes[64];
  const uint64_t incr = 0x0808080808080808ULL;
  const uint64_t mask = 0x0706050403020100ULL;
  num_indexes = 0;
  for (int i = 0; i < num_bits / unroll; ++i) {
    uint64_t word = reinterpret_cast<const uint64_t*>(bits)[i];
    if (bit_to_search == 0) {
      word = ~word;
    }
    uint64_t base = 0;
    int num_indexes_loop = 0;
    while (word) {
      uint64_t byte_indexes_next =
          _pext_u64(mask, _pdep_u64(word, UINT64_C(0X0101010101010101)) * 0xff) + base;
      *reinterpret_cast<uint64_t*>(byte_indexes + num_indexes_loop) = byte_indexes_next;
      base += incr;
      num_indexes_loop += static_cast<int>(POPCNT64(word & 0xff));
      word >>= 8;
    }
    // Unpack indexes to 16-bits and either add the base of i * 64 or shuffle input
    // indexes
    for (int j = 0; j < (num_indexes_loop + 15) / 16; ++j) {
      __m256i output = _mm256_cvtepi8_epi16(
          _mm_loadu_si128(reinterpret_cast<const __m128i*>(byte_indexes) + j));
      output = _mm256_add_epi16(output, _mm256_set1_epi16(i * 64));
      _mm256_storeu_si256(((__m256i*)(indexes + num_indexes)) + j, output);
    }
    num_indexes += num_indexes_loop;
  }
}
template void BitUtil::bits_to_indexes_avx2<0>(const int num_bits, const uint8_t* bits,
                                               int& num_indexes, uint16_t* indexes);
template void BitUtil::bits_to_indexes_avx2<1>(const int num_bits, const uint8_t* bits,
                                               int& num_indexes, uint16_t* indexes);

template <int bit_to_search>
void BitUtil::bits_filter_indexes_avx2(const int num_bits, const uint8_t* bits,
                                       const uint16_t* input_indexes, int& num_indexes,
                                       uint16_t* indexes) {
  // 64 bits at a time
  constexpr int unroll = 64;

  // The caller takes care of processing the remaining bits at the end outside of the
  // multiples of 64
  ARROW_DCHECK(num_bits % unroll == 0);

  const uint64_t mask = 0xfedcba9876543210ULL;
  num_indexes = 0;
  for (int i = 0; i < num_bits / unroll; ++i) {
    uint64_t word = reinterpret_cast<const uint64_t*>(bits)[i];
    if (bit_to_search == 0) {
      word = ~word;
    }

    int loop_id = 0;
    while (word) {
      uint64_t indexes_4bit =
          _pext_u64(mask, _pdep_u64(word, UINT64_C(0x1111111111111111)) * 0xf);
      // Unpack 4 bit indexes to 8 bits
      __m256i indexes_8bit = _mm256_set1_epi64x(indexes_4bit);
      indexes_8bit = _mm256_shuffle_epi8(
          indexes_8bit, _mm256_setr_epi64x(0x0303020201010000ULL, 0x0707060605050404ULL,
                                           0x0303020201010000ULL, 0x0707060605050404ULL));
      indexes_8bit = _mm256_blendv_epi8(
          _mm256_and_si256(indexes_8bit, _mm256_set1_epi8(0x0f)),
          _mm256_and_si256(_mm256_srli_epi32(indexes_8bit, 4), _mm256_set1_epi8(0x0f)),
          _mm256_set1_epi16(static_cast<uint16_t>(0xff00)));
      __m256i input =
          _mm256_loadu_si256(((const __m256i*)input_indexes) + 4 * i + loop_id);
      // Shuffle bytes to get low bytes in the first 128-bit lane and high bytes in the
      // second
      input = _mm256_shuffle_epi8(
          input, _mm256_setr_epi64x(0x0e0c0a0806040200ULL, 0x0f0d0b0907050301ULL,
                                    0x0e0c0a0806040200ULL, 0x0f0d0b0907050301ULL));
      input = _mm256_permute4x64_epi64(input, 0xd8);  // 0b11011000
      // Apply permutation
      __m256i output = _mm256_shuffle_epi8(input, indexes_8bit);
      // Move low and high bytes across 128-bit lanes to assemble back 16-bit indexes.
      // (This is the reverse of the byte permutation we did on the input)
      output = _mm256_permute4x64_epi64(output,
                                        0xd8);  // The reverse of swapping 2nd and 3rd
                                                // 64-bit element is the same permutation
      output = _mm256_shuffle_epi8(
          output, _mm256_setr_epi64x(0x0b030a0209010800ULL, 0x0f070e060d050c04ULL,
                                     0x0b030a0209010800ULL, 0x0f070e060d050c04ULL));
      _mm256_storeu_si256((__m256i*)(indexes + num_indexes), output);
      num_indexes += static_cast<int>(POPCNT64(word & 0xffff));
      word >>= 16;
      ++loop_id;
    }
  }
}
template void BitUtil::bits_filter_indexes_avx2<0>(const int num_bits,
                                                   const uint8_t* bits,
                                                   const uint16_t* input_indexes,
                                                   int& num_indexes, uint16_t* indexes);
template void BitUtil::bits_filter_indexes_avx2<1>(const int num_bits,
                                                   const uint8_t* bits,
                                                   const uint16_t* input_indexes,
                                                   int& num_indexes, uint16_t* indexes);

void BitUtil::bits_to_bytes_avx2(const int num_bits, const uint8_t* bits,
                                 uint8_t* bytes) {
  constexpr int unroll = 32;
  // Processing 32 bits at a time
  for (int i = 0; i < num_bits / unroll; ++i) {
    __m256i unpacked = _mm256_set1_epi32(reinterpret_cast<const uint32_t*>(bits)[i]);
    unpacked = _mm256_shuffle_epi8(
        unpacked, _mm256_setr_epi64x(0x0000000000000000ULL, 0x0101010101010101ULL,
                                     0x0202020202020202ULL, 0x0303030303030303ULL));
    __m256i bits_in_bytes = _mm256_set1_epi64x(0x8040201008040201ULL);
    unpacked =
        _mm256_cmpeq_epi8(bits_in_bytes, _mm256_and_si256(unpacked, bits_in_bytes));
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(bytes) + i, unpacked);
  }
}

void BitUtil::bytes_to_bits_avx2(const int num_bits, const uint8_t* bytes,
                                 uint8_t* bits) {
  constexpr int unroll = 32;
  // Processing 32 bits at a time
  for (int i = 0; i < num_bits / unroll; ++i) {
    reinterpret_cast<uint32_t*>(bits)[i] = _mm256_movemask_epi8(
        _mm256_loadu_si256(reinterpret_cast<const __m256i*>(bytes) + i));
  }
}

void BitUtil::bit_vector_lookup_avx2(const int num_lookups, const uint32_t* bit_ids,
                                     const uint8_t* bits, uint8_t* result) {
  constexpr int unroll = 8;
  for (int i = 0; i < num_lookups / unroll; ++i) {
    __m256i id = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(bit_ids) + i);
    __m256i bit = _mm256_i32gather_epi32(reinterpret_cast<const int*>(bits),
                                         _mm256_srli_epi32(id, 5), 4);
    bit = _mm256_srlv_epi32(bit, _mm256_and_si256(id, _mm256_set1_epi32(31)));
    bit = _mm256_shuffle_epi8(bit, _mm256_setr_epi32(0x0c080400UL, ~0UL, ~0UL, ~0UL,
                                                     0x0c080400UL, ~0UL, ~0UL, ~0UL));
    bit = _mm256_permutevar8x32_epi32(bit, _mm256_setr_epi32(0, 4, 1, 1, 1, 1, 1, 1));
    bit = _mm256_slli_epi32(bit, 7);
    result[i] = _mm256_movemask_epi8(bit);
  }
}

#endif  // ARROW_HAVE_AVX2

}  // namespace util
}  // namespace exec
}  // namespace arrow