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

#include <memory>
#include <string>

#include "arrow/csv/options.h"
#include "arrow/dataset/dataset.h"
#include "arrow/dataset/file_base.h"
#include "arrow/dataset/type_fwd.h"
#include "arrow/dataset/visibility.h"
#include "arrow/status.h"
#include "arrow/util/compression.h"

namespace arrow {
namespace dataset {

constexpr char kCsvTypeName[] = "csv";

/// \addtogroup dataset-file-formats
///
/// @{

/// \brief A FileFormat implementation that reads from and writes to Csv files
class ARROW_DS_EXPORT CsvFileFormat : public FileFormat {
 public:
  /// Options affecting the parsing of CSV files
  csv::ParseOptions parse_options = csv::ParseOptions::Defaults();

  std::string type_name() const override { return kCsvTypeName; }

  bool Equals(const FileFormat& other) const override;

  Result<bool> IsSupported(const FileSource& source) const override;

  /// \brief Return the schema of the file if possible.
  Result<std::shared_ptr<Schema>> Inspect(const FileSource& source) const override;

  /// \brief Open a file for scanning
  Result<ScanTaskIterator> ScanFile(
      const std::shared_ptr<ScanOptions>& options,
      const std::shared_ptr<FileFragment>& fragment) const override;

  Result<std::shared_ptr<FileWriter>> MakeWriter(
      std::shared_ptr<io::OutputStream> destination, std::shared_ptr<Schema> schema,
      std::shared_ptr<FileWriteOptions> options) const override {
    return Status::NotImplemented("writing fragment of CsvFileFormat");
  }

  std::shared_ptr<FileWriteOptions> DefaultWriteOptions() override { return NULLPTR; }
};

/// \brief Per-scan options for CSV fragments
struct ARROW_DS_EXPORT CsvFragmentScanOptions : public FragmentScanOptions {
  std::string type_name() const override { return kCsvTypeName; }

  /// CSV conversion options
  csv::ConvertOptions convert_options = csv::ConvertOptions::Defaults();

  /// CSV reading options
  ///
  /// Note that use_threads is always ignored.
  csv::ReadOptions read_options = csv::ReadOptions::Defaults();
};

/// @}

}  // namespace dataset
}  // namespace arrow
