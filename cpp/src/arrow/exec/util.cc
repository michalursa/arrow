#include "util.h"

namespace arrow {
namespace exec {
namespace util {

int BitUtil::popcnt_bitvector(const int num_bits, const uint8_t* bits) {
  constexpr int unroll = 64;
  int count = 0;
  for (int i = 0; i < num_bits / unroll; ++i) {
    uint64_t word = reinterpret_cast<const uint64_t*>(bits)[i];
    count += static_cast<int>(POPCNT64(word));
  }
  int tail = num_bits % unroll;
  if (tail) {
    uint64_t word = reinterpret_cast<const uint64_t*>(bits)[num_bits / unroll];
    word &= ~0ULL >> (64 - tail);
    count += static_cast<int>(POPCNT64(word));
  }
  return count;
}

inline void BitUtil::bits_to_indexes_helper(uint64_t word, uint16_t base_index,
                                            int& num_indexes, uint16_t* indexes) {
  while (word) {
    indexes[num_indexes++] = base_index + static_cast<uint16_t>(TZCNT64(word));
    word &= word - 1;
  }
}
inline void BitUtil::bits_filter_indexes_helper(uint64_t word,
                                                const uint16_t* input_indexes,
                                                int& num_indexes, uint16_t* indexes) {
  while (word) {
    indexes[num_indexes++] = input_indexes[TZCNT64(word)];
    word &= word - 1;
  }
}

template <int bit_to_search, bool filter_input_indexes>
void BitUtil::bits_to_indexes_internal(CPUInstructionSet instruction_set,
                                       const int num_bits, const uint8_t* bits,
                                       const uint16_t* input_indexes, int& num_indexes,
                                       uint16_t* indexes) {
  // 64 bits at a time
  constexpr int unroll = 64;
  int tail = num_bits % unroll;
#if defined(ARROW_HAVE_AVX2)
  if (instruction_set == util::CPUInstructionSet::avx2) {
    if (filter_input_indexes) {
      bits_filter_indexes_avx2<bit_to_search>(num_bits - tail, bits, input_indexes,
                                              num_indexes, indexes);
    } else {
      bits_to_indexes_avx2<bit_to_search>(num_bits - tail, bits, num_indexes, indexes);
    }
  } else
#endif
  {
    num_indexes = 0;
    for (int i = 0; i < num_bits / unroll; ++i) {
      uint64_t word = reinterpret_cast<const uint64_t*>(bits)[i];
      if (bit_to_search == 0) {
        word = ~word;
      }
      if (filter_input_indexes) {
        bits_filter_indexes_helper(word, input_indexes + i * 64, num_indexes, indexes);
      } else {
        bits_to_indexes_helper(word, i * 64, num_indexes, indexes);
      }
    }
  }
  // Optionally process the last partial word with masking out bits outside range
  if (tail) {
    uint64_t word = reinterpret_cast<const uint64_t*>(bits)[num_bits / unroll];
    if (bit_to_search == 0) {
      word = ~word;
    }
    word &= ~0ULL >> (64 - tail);
    if (filter_input_indexes) {
      bits_filter_indexes_helper(word, input_indexes + num_bits - tail, num_indexes,
                                 indexes);
    } else {
      bits_to_indexes_helper(word, num_bits - tail, num_indexes, indexes);
    }
  }
}
template void BitUtil::bits_to_indexes_internal<0, false>(CPUInstructionSet, const int,
                                                          const uint8_t*, const uint16_t*,
                                                          int&, uint16_t*);
template void BitUtil::bits_to_indexes_internal<0, true>(CPUInstructionSet, const int,
                                                         const uint8_t*, const uint16_t*,
                                                         int&, uint16_t*);
template void BitUtil::bits_to_indexes_internal<1, false>(CPUInstructionSet, const int,
                                                          const uint8_t*, const uint16_t*,
                                                          int&, uint16_t*);
template void BitUtil::bits_to_indexes_internal<1, true>(CPUInstructionSet, const int,
                                                         const uint8_t*, const uint16_t*,
                                                         int&, uint16_t*);

template <int bit_to_search>
void BitUtil::bits_to_indexes(CPUInstructionSet instruction_set, const int num_bits,
                              const uint8_t* bits, int& num_indexes, uint16_t* indexes) {
  bits_to_indexes_internal<bit_to_search, false>(instruction_set, num_bits, bits, nullptr,
                                                 num_indexes, indexes);
}
template void BitUtil::bits_to_indexes<0>(CPUInstructionSet instruction_set,
                                          const int num_bits, const uint8_t* bits,
                                          int& num_indexes, uint16_t* indexes);
template void BitUtil::bits_to_indexes<1>(CPUInstructionSet instruction_set,
                                          const int num_bits, const uint8_t* bits,
                                          int& num_indexes, uint16_t* indexes);

template <int bit_to_search>
void BitUtil::bits_filter_indexes(CPUInstructionSet instruction_set, const int num_bits,
                                  const uint8_t* bits, const uint16_t* input_indexes,
                                  int& num_indexes, uint16_t* indexes) {
  bits_to_indexes_internal<bit_to_search, true>(instruction_set, num_bits, bits,
                                                input_indexes, num_indexes, indexes);
}
template void BitUtil::bits_filter_indexes<0>(CPUInstructionSet, const int,
                                              const uint8_t*, const uint16_t*, int&,
                                              uint16_t*);
template void BitUtil::bits_filter_indexes<1>(CPUInstructionSet, const int,
                                              const uint8_t*, const uint16_t*, int&,
                                              uint16_t*);

void BitUtil::bits_split_indexes(CPUInstructionSet instruction_set, const int num_bits,
                                 const uint8_t* bits, int& num_indexes_bit0,
                                 uint16_t* indexes_bit0, uint16_t* indexes_bit1) {
  bits_to_indexes<0>(instruction_set, num_bits, bits, num_indexes_bit0, indexes_bit0);
  int num_indexes_bit1;
  bits_to_indexes<1>(instruction_set, num_bits, bits, num_indexes_bit1, indexes_bit1);
}

void BitUtil::bits_to_bytes_internal(const int num_bits, const uint8_t* bits,
                                     uint8_t* bytes) {
  constexpr int unroll = 8;
  // Processing 8 bits at a time
  for (int i = 0; i < (num_bits + unroll - 1) / unroll; ++i) {
    uint8_t bits_next = bits[i];
    // Clear the lowest bit and then make 8 copies of remaining 7 bits, each 7 bits apart
    // from the previous.
    uint64_t unpacked = static_cast<uint64_t>(bits_next & 0xfe) *
                        ((1ULL << 7) | (1ULL << 14) | (1ULL << 21) | (1ULL << 28) |
                         (1ULL << 35) | (1ULL << 42) | (1ULL << 49));
    unpacked |= (bits_next & 1);
    unpacked &= 0x0101010101010101ULL;
    unpacked *= 255;
    reinterpret_cast<uint64_t*>(bytes)[i] = unpacked;
  }
}

void BitUtil::bytes_to_bits_internal(const int num_bits, const uint8_t* bytes,
                                     uint8_t* bits) {
  constexpr int unroll = 8;
  // Process 8 bits at a time
  for (int i = 0; i < (num_bits + unroll - 1) / unroll; ++i) {
    uint64_t bytes_next = reinterpret_cast<const uint64_t*>(bytes)[i];
    bytes_next &= 0x0101010101010101ULL;
    bytes_next |= (bytes_next >> 7);  // Pairs of adjacent output bits in individual bytes
    bytes_next |= (bytes_next >> 14);  // 4 adjacent output bits in individual bytes
    bytes_next |= (bytes_next >> 28);  // All 8 output bits in the lowest byte
    bits[i] = static_cast<uint8_t>(bytes_next & 0xff);
  }
}

void BitUtil::bits_to_bytes(CPUInstructionSet instruction_set, const int num_bits,
                            const uint8_t* bits, uint8_t* bytes) {
  int num_processed = 0;
#if defined(ARROW_HAVE_AVX2)
  if (instruction_set == util::CPUInstructionSet::avx2) {
    // The function call below processes whole 32 bit chunks together.
    num_processed = num_bits - (num_bits % 32);
    bits_to_bytes_avx2(num_processed, bits, bytes);
  }
#endif
  // Processing 8 bits at a time
  constexpr int unroll = 8;
  for (int i = num_processed / unroll; i < (num_bits + unroll - 1) / unroll; ++i) {
    uint8_t bits_next = bits[i];
    // Clear the lowest bit and then make 8 copies of remaining 7 bits, each 7 bits apart
    // from the previous.
    uint64_t unpacked = static_cast<uint64_t>(bits_next & 0xfe) *
                        ((1ULL << 7) | (1ULL << 14) | (1ULL << 21) | (1ULL << 28) |
                         (1ULL << 35) | (1ULL << 42) | (1ULL << 49));
    unpacked |= (bits_next & 1);
    unpacked &= 0x0101010101010101ULL;
    unpacked *= 255;
    reinterpret_cast<uint64_t*>(bytes)[i] = unpacked;
  }
}

void BitUtil::bytes_to_bits(CPUInstructionSet instruction_set, const int num_bits,
                            const uint8_t* bytes, uint8_t* bits) {
  int num_processed = 0;
#if defined(ARROW_HAVE_AVX2)
  if (instruction_set == util::CPUInstructionSet::avx2) {
    // The function call below processes whole 32 bit chunks together.
    num_processed = num_bits - (num_bits % 32);
    bytes_to_bits_avx2(num_processed, bytes, bits);
  }
#endif
  // Process 8 bits at a time
  constexpr int unroll = 8;
  for (int i = num_processed / unroll; i < (num_bits + unroll - 1) / unroll; ++i) {
    uint64_t bytes_next = reinterpret_cast<const uint64_t*>(bytes)[i];
    bytes_next &= 0x0101010101010101ULL;
    bytes_next |= (bytes_next >> 7);  // Pairs of adjacent output bits in individual bytes
    bytes_next |= (bytes_next >> 14);  // 4 adjacent output bits in individual bytes
    bytes_next |= (bytes_next >> 28);  // All 8 output bits in the lowest byte
    bits[i] = static_cast<uint8_t>(bytes_next & 0xff);
  }
}

void BitUtil::bit_vector_lookup(CPUInstructionSet instruction_set, const int num_lookups,
                                const uint32_t* bit_ids, const uint8_t* bits,
                                uint8_t* result) {
  int num_processed = 0;
#if defined(ARROW_HAVE_AVX2)
  if (instruction_set == util::CPUInstructionSet::avx2) {
    // The function call below processes whole 8 lookups together.
    num_processed = num_lookups - (num_lookups % 8);
    bit_vector_lookup_avx2(num_processed, bit_ids, bits, result);
  }
#endif
  uint8_t next_result = 0;
  for (int i = num_processed; i < num_lookups; ++i) {
    uint32_t id = bit_ids[i];
    int bit = ((bits[id / 8] >> (id & 7)) & 1);
    next_result |= bit << (i & 7);
    if ((i & 7) == 7) {
      result[i / 8] = next_result;
      next_result = 0;
    }
  }
  if ((num_lookups % 8) > 0) {
    result[num_lookups / 8] = next_result;
  }
}

}  // namespace util
}  // namespace exec
}  // namespace arrow