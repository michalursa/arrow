#pragma once
#include "arrow/memory_pool.h"
#include "arrow/result.h"
#include "arrow/status.h"
#include "common.h"
#include "groupby_hash.h"
#include "groupby_map.h"
#include "groupby_storage.h"
#include "util.h"

namespace arrow {
namespace exec {

class GroupMap {
 public:
  ~GroupMap();

  void init(util::CPUInstructionSet instruction_set_in, MemoryPool* pool,
            uint32_t num_columns, const std::vector<bool>& is_fixed_len_in,
            const uint32_t* col_widths_in);

  void push_input(uint32_t num_rows, const uint8_t** non_null_buffers_maybe_null,
                  const uint8_t** fixedlen_buffers,
                  const uint8_t** varlen_buffers_maybe_null, uint32_t* group_ids);

  void pull_output_prepare(uint64_t& out_num_rows, bool& is_row_fixedlen);

  void pull_output_fixedlen_and_nulls(uint8_t** non_null_buffers,
                                      uint8_t** fixedlen_buffers,
                                      uint64_t* out_varlen_buffer_sizes);

  void pull_output_varlen(uint8_t** non_null_buffers, uint8_t** fixedlen_buffers,
                          uint8_t** varlen_buffers_maybe_null);

  uint64_t get_num_keys() const { return key_store.get_num_keys(); }

 private:
  static constexpr int log_minibatch_max = 10;
  static constexpr int minibatch_size_max = 1 << log_minibatch_max;
  static constexpr int minibatch_size_min = 128;

  void equal_callback(int num_keys_to_compare, const uint16_t* selection_may_be_null,
                      const uint32_t* group_ids, uint8_t* match_bitvector);
  Status append_callback(int num_keys, const uint16_t* selection);

  int minibatch_size;

  MemoryPool* memory_pool;
  util::TempBufferAlloc temp_buffers;
  KeyStore key_store;
  util::CPUInstructionSet instruction_set;
  SwissTable group_id_map;

  // One element per input column
  std::vector<uint32_t> col_widths;
  std::vector<bool> is_col_fixed_len;
  std::vector<const uint8_t*> col_non_nulls;
  std::vector<const uint8_t*> varlen_col_non_nulls;
  std::vector<const uint32_t*> col_offsets;
  std::vector<const uint32_t*> varlen_col_offsets;
  std::vector<const uint8_t*> col_values;
  std::vector<uint8_t*> out_col_non_nulls;
  std::vector<uint32_t*> out_col_offsets;
  std::vector<uint8_t*> out_col_values;

  // Constant bit vectors used e.g. when null bit vector information for column is missing
  std::vector<uint8_t> bit_vector_all_0;
  std::vector<uint8_t> bit_vector_all_1;

  std::vector<uint32_t> minibatch_hashes;
  std::vector<uint32_t> minibatch_temp;

  std::vector<uint8_t> minibatch_rows;
  std::vector<uint32_t> minibatch_offsets;
  std::vector<uint8_t> minibatch_nulls;
  std::vector<uint8_t> minibatch_any_nulls_buffer;
  const uint8_t* minibatch_any_nulls;
};

}  // namespace exec
}  // namespace arrow