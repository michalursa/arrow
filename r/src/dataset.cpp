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

#include "./arrow_types.h"

#if defined(ARROW_R_WITH_DATASET)

#include <arrow/dataset/api.h>
#include <arrow/filesystem/filesystem.h>
#include <arrow/ipc/writer.h>
#include <arrow/table.h>
#include <arrow/util/checked_cast.h>
#include <arrow/util/iterator.h>

namespace ds = ::arrow::dataset;
namespace fs = ::arrow::fs;

namespace cpp11 {

const char* r6_class_name<ds::Dataset>::get(const std::shared_ptr<ds::Dataset>& dataset) {
  auto type_name = dataset->type_name();

  if (type_name == "union") {
    return "UnionDataset";
  } else if (type_name == "filesystem") {
    return "FileSystemDataset";
  } else if (type_name == "in-memory") {
    return "InMemoryDataset";
  } else {
    return "Dataset";
  }
}

const char* r6_class_name<ds::FileFormat>::get(
    const std::shared_ptr<ds::FileFormat>& file_format) {
  auto type_name = file_format->type_name();
  if (type_name == "parquet") {
    return "ParquetFileFormat";
  } else if (type_name == "ipc") {
    return "IpcFileFormat";
  } else if (type_name == "csv") {
    return "CsvFileFormat";
  } else {
    return "FileFormat";
  }
}

}  // namespace cpp11

// Dataset, UnionDataset, FileSystemDataset

// [[dataset::export]]
std::shared_ptr<ds::ScannerBuilder> dataset___Dataset__NewScan(
    const std::shared_ptr<ds::Dataset>& ds) {
  auto context = std::make_shared<ds::ScanContext>();
  context->pool = gc_memory_pool();
  return ValueOrStop(ds->NewScan(std::move(context)));
}

// [[dataset::export]]
std::shared_ptr<arrow::Schema> dataset___Dataset__schema(
    const std::shared_ptr<ds::Dataset>& dataset) {
  return dataset->schema();
}

// [[dataset::export]]
std::string dataset___Dataset__type_name(const std::shared_ptr<ds::Dataset>& dataset) {
  return dataset->type_name();
}

// [[dataset::export]]
std::shared_ptr<ds::Dataset> dataset___Dataset__ReplaceSchema(
    const std::shared_ptr<ds::Dataset>& dataset,
    const std::shared_ptr<arrow::Schema>& schm) {
  return ValueOrStop(dataset->ReplaceSchema(schm));
}

// [[dataset::export]]
std::shared_ptr<ds::Dataset> dataset___UnionDataset__create(
    const ds::DatasetVector& datasets, const std::shared_ptr<arrow::Schema>& schm) {
  return ValueOrStop(ds::UnionDataset::Make(schm, datasets));
}

// [[dataset::export]]
std::shared_ptr<ds::Dataset> dataset___InMemoryDataset__create(
    const std::shared_ptr<arrow::Table>& table) {
  return std::make_shared<ds::InMemoryDataset>(table);
}

// [[dataset::export]]
cpp11::list dataset___UnionDataset__children(
    const std::shared_ptr<ds::UnionDataset>& ds) {
  return arrow::r::to_r_list(ds->children());
}

// [[dataset::export]]
std::shared_ptr<ds::FileFormat> dataset___FileSystemDataset__format(
    const std::shared_ptr<ds::FileSystemDataset>& dataset) {
  return dataset->format();
}

// [[dataset::export]]
std::shared_ptr<fs::FileSystem> dataset___FileSystemDataset__filesystem(
    const std::shared_ptr<ds::FileSystemDataset>& dataset) {
  return dataset->filesystem();
}

// [[dataset::export]]
std::vector<std::string> dataset___FileSystemDataset__files(
    const std::shared_ptr<ds::FileSystemDataset>& dataset) {
  return dataset->files();
}

// DatasetFactory, UnionDatasetFactory, FileSystemDatasetFactory

// [[dataset::export]]
std::shared_ptr<ds::Dataset> dataset___DatasetFactory__Finish1(
    const std::shared_ptr<ds::DatasetFactory>& factory, bool unify_schemas) {
  ds::FinishOptions opts;
  if (unify_schemas) {
    opts.inspect_options.fragments = ds::InspectOptions::kInspectAllFragments;
  }
  return ValueOrStop(factory->Finish(opts));
}

// [[dataset::export]]
std::shared_ptr<ds::Dataset> dataset___DatasetFactory__Finish2(
    const std::shared_ptr<ds::DatasetFactory>& factory,
    const std::shared_ptr<arrow::Schema>& schema) {
  return ValueOrStop(factory->Finish(schema));
}

// [[dataset::export]]
std::shared_ptr<arrow::Schema> dataset___DatasetFactory__Inspect(
    const std::shared_ptr<ds::DatasetFactory>& factory, bool unify_schemas) {
  ds::InspectOptions opts;
  if (unify_schemas) {
    opts.fragments = ds::InspectOptions::kInspectAllFragments;
  }
  return ValueOrStop(factory->Inspect(opts));
}

// [[dataset::export]]
std::shared_ptr<ds::DatasetFactory> dataset___UnionDatasetFactory__Make(
    const std::vector<std::shared_ptr<ds::DatasetFactory>>& children) {
  return ValueOrStop(ds::UnionDatasetFactory::Make(children));
}

// [[dataset::export]]
std::shared_ptr<ds::FileSystemDatasetFactory> dataset___FileSystemDatasetFactory__Make2(
    const std::shared_ptr<fs::FileSystem>& fs,
    const std::shared_ptr<fs::FileSelector>& selector,
    const std::shared_ptr<ds::FileFormat>& format,
    const std::shared_ptr<ds::Partitioning>& partitioning) {
  // TODO(fsaintjacques): Make options configurable
  auto options = ds::FileSystemFactoryOptions{};
  if (partitioning != nullptr) {
    options.partitioning = partitioning;
  }

  return arrow::internal::checked_pointer_cast<ds::FileSystemDatasetFactory>(
      ValueOrStop(ds::FileSystemDatasetFactory::Make(fs, *selector, format, options)));
}

// [[dataset::export]]
std::shared_ptr<ds::FileSystemDatasetFactory> dataset___FileSystemDatasetFactory__Make1(
    const std::shared_ptr<fs::FileSystem>& fs,
    const std::shared_ptr<fs::FileSelector>& selector,
    const std::shared_ptr<ds::FileFormat>& format) {
  return dataset___FileSystemDatasetFactory__Make2(fs, selector, format, nullptr);
}

// [[dataset::export]]
std::shared_ptr<ds::FileSystemDatasetFactory> dataset___FileSystemDatasetFactory__Make3(
    const std::shared_ptr<fs::FileSystem>& fs,
    const std::shared_ptr<fs::FileSelector>& selector,
    const std::shared_ptr<ds::FileFormat>& format,
    const std::shared_ptr<ds::PartitioningFactory>& factory) {
  // TODO(fsaintjacques): Make options configurable
  auto options = ds::FileSystemFactoryOptions{};
  if (factory != nullptr) {
    options.partitioning = factory;
  }

  return arrow::internal::checked_pointer_cast<ds::FileSystemDatasetFactory>(
      ValueOrStop(ds::FileSystemDatasetFactory::Make(fs, *selector, format, options)));
}

// FileFormat, ParquetFileFormat, IpcFileFormat

// [[dataset::export]]
std::string dataset___FileFormat__type_name(
    const std::shared_ptr<ds::FileFormat>& format) {
  return format->type_name();
}

// [[dataset::export]]
std::shared_ptr<ds::FileWriteOptions> dataset___FileFormat__DefaultWriteOptions(
    const std::shared_ptr<ds::FileFormat>& fmt) {
  return fmt->DefaultWriteOptions();
}

// [[dataset::export]]
std::shared_ptr<ds::ParquetFileFormat> dataset___ParquetFileFormat__Make(
    bool use_buffered_stream, int64_t buffer_size, cpp11::strings dict_columns) {
  auto fmt = std::make_shared<ds::ParquetFileFormat>();

  fmt->reader_options.use_buffered_stream = use_buffered_stream;
  fmt->reader_options.buffer_size = buffer_size;

  auto dict_columns_vector = cpp11::as_cpp<std::vector<std::string>>(dict_columns);
  auto& d = fmt->reader_options.dict_columns;
  std::move(dict_columns_vector.begin(), dict_columns_vector.end(),
            std::inserter(d, d.end()));

  return fmt;
}

// [[dataset::export]]
std::string dataset___FileWriteOptions__type_name(
    const std::shared_ptr<ds::FileWriteOptions>& options) {
  return options->type_name();
}

#if defined(ARROW_R_WITH_PARQUET)
// [[dataset::export]]
void dataset___ParquetFileWriteOptions__update(
    const std::shared_ptr<ds::ParquetFileWriteOptions>& options,
    const std::shared_ptr<parquet::WriterProperties>& writer_props,
    const std::shared_ptr<parquet::ArrowWriterProperties>& arrow_writer_props) {
  options->writer_properties = writer_props;
  options->arrow_writer_properties = arrow_writer_props;
}
#endif

// [[dataset::export]]
void dataset___IpcFileWriteOptions__update2(
    const std::shared_ptr<ds::IpcFileWriteOptions>& ipc_options, bool use_legacy_format,
    const std::shared_ptr<arrow::util::Codec>& codec,
    arrow::ipc::MetadataVersion metadata_version) {
  ipc_options->options->write_legacy_ipc_format = use_legacy_format;
  ipc_options->options->codec = codec;
  ipc_options->options->metadata_version = metadata_version;
}

// [[dataset::export]]
void dataset___IpcFileWriteOptions__update1(
    const std::shared_ptr<ds::IpcFileWriteOptions>& ipc_options, bool use_legacy_format,
    arrow::ipc::MetadataVersion metadata_version) {
  ipc_options->options->write_legacy_ipc_format = use_legacy_format;
  ipc_options->options->metadata_version = metadata_version;
}

// [[dataset::export]]
std::shared_ptr<ds::IpcFileFormat> dataset___IpcFileFormat__Make() {
  return std::make_shared<ds::IpcFileFormat>();
}

// [[dataset::export]]
std::shared_ptr<ds::CsvFileFormat> dataset___CsvFileFormat__Make(
    const std::shared_ptr<arrow::csv::ParseOptions>& parse_options) {
  auto format = std::make_shared<ds::CsvFileFormat>();
  format->parse_options = *parse_options;
  return format;
}

// DirectoryPartitioning, HivePartitioning

// [[dataset::export]]
std::shared_ptr<ds::DirectoryPartitioning> dataset___DirectoryPartitioning(
    const std::shared_ptr<arrow::Schema>& schm) {
  return std::make_shared<ds::DirectoryPartitioning>(schm);
}

// [[dataset::export]]
std::shared_ptr<ds::PartitioningFactory> dataset___DirectoryPartitioning__MakeFactory(
    const std::vector<std::string>& field_names) {
  return ds::DirectoryPartitioning::MakeFactory(field_names);
}

// [[dataset::export]]
std::shared_ptr<ds::HivePartitioning> dataset___HivePartitioning(
    const std::shared_ptr<arrow::Schema>& schm, const std::string& null_fallback) {
  std::vector<std::shared_ptr<arrow::Array>> dictionaries;
  return std::make_shared<ds::HivePartitioning>(schm, dictionaries, null_fallback);
}

// [[dataset::export]]
std::shared_ptr<ds::PartitioningFactory> dataset___HivePartitioning__MakeFactory(
    const std::string& null_fallback) {
  ds::HivePartitioningFactoryOptions options;
  options.null_fallback = null_fallback;
  return ds::HivePartitioning::MakeFactory(options);
}

// ScannerBuilder, Scanner

// [[dataset::export]]
void dataset___ScannerBuilder__ProjectNames(const std::shared_ptr<ds::ScannerBuilder>& sb,
                                            const std::vector<std::string>& cols) {
  StopIfNotOk(sb->Project(cols));
}

// [[dataset::export]]
void dataset___ScannerBuilder__ProjectExprs(
    const std::shared_ptr<ds::ScannerBuilder>& sb,
    const std::vector<std::shared_ptr<ds::Expression>>& exprs,
    const std::vector<std::string>& names) {
  // We have shared_ptrs of expressions but need the Expressions
  std::vector<ds::Expression> expressions;
  for (auto expr : exprs) {
    expressions.push_back(*expr);
  }
  StopIfNotOk(sb->Project(expressions, names));
}

// [[dataset::export]]
void dataset___ScannerBuilder__Filter(const std::shared_ptr<ds::ScannerBuilder>& sb,
                                      const std::shared_ptr<ds::Expression>& expr) {
  StopIfNotOk(sb->Filter(*expr));
}

// [[dataset::export]]
void dataset___ScannerBuilder__UseThreads(const std::shared_ptr<ds::ScannerBuilder>& sb,
                                          bool threads) {
  StopIfNotOk(sb->UseThreads(threads));
}

// [[dataset::export]]
void dataset___ScannerBuilder__BatchSize(const std::shared_ptr<ds::ScannerBuilder>& sb,
                                         int64_t batch_size) {
  StopIfNotOk(sb->BatchSize(batch_size));
}

// [[dataset::export]]
std::shared_ptr<arrow::Schema> dataset___ScannerBuilder__schema(
    const std::shared_ptr<ds::ScannerBuilder>& sb) {
  return sb->schema();
}

// [[dataset::export]]
std::shared_ptr<ds::Scanner> dataset___ScannerBuilder__Finish(
    const std::shared_ptr<ds::ScannerBuilder>& sb) {
  return ValueOrStop(sb->Finish());
}

// [[dataset::export]]
std::shared_ptr<arrow::Table> dataset___Scanner__ToTable(
    const std::shared_ptr<ds::Scanner>& scanner) {
  return ValueOrStop(scanner->ToTable());
}

// [[dataset::export]]
std::shared_ptr<arrow::Table> dataset___Scanner__head(
    const std::shared_ptr<ds::Scanner>& scanner, int n) {
  // TODO: make this a full Slice with offset > 0
  std::vector<std::shared_ptr<arrow::RecordBatch>> batches;
  std::shared_ptr<arrow::RecordBatch> current_batch;

  for (auto st : ValueOrStop(scanner->Scan())) {
    for (auto b : ValueOrStop(ValueOrStop(st)->Execute())) {
      current_batch = ValueOrStop(b);
      batches.push_back(current_batch->Slice(0, n));
      n -= current_batch->num_rows();
      if (n < 0) break;
    }
    if (n < 0) break;
  }
  return ValueOrStop(arrow::Table::FromRecordBatches(std::move(batches)));
}

// [[dataset::export]]
cpp11::list dataset___Scanner__Scan(const std::shared_ptr<ds::Scanner>& scanner) {
  auto it = ValueOrStop(scanner->Scan());
  std::vector<std::shared_ptr<ds::ScanTask>> out;
  std::shared_ptr<ds::ScanTask> scan_task;
  // TODO(npr): can this iteration be parallelized?
  for (auto st : it) {
    scan_task = ValueOrStop(st);
    out.push_back(scan_task);
  }

  return arrow::r::to_r_list(out);
}

// [[dataset::export]]
std::shared_ptr<arrow::Schema> dataset___Scanner__schema(
    const std::shared_ptr<ds::Scanner>& sc) {
  return sc->schema();
}

// [[dataset::export]]
cpp11::list dataset___ScanTask__get_batches(
    const std::shared_ptr<ds::ScanTask>& scan_task) {
  arrow::RecordBatchIterator rbi;
  rbi = ValueOrStop(scan_task->Execute());
  std::vector<std::shared_ptr<arrow::RecordBatch>> out;
  std::shared_ptr<arrow::RecordBatch> batch;
  for (auto b : rbi) {
    batch = ValueOrStop(b);
    out.push_back(batch);
  }
  return arrow::r::to_r_list(out);
}

// [[dataset::export]]
void dataset___Dataset__Write(
    const std::shared_ptr<ds::FileWriteOptions>& file_write_options,
    const std::shared_ptr<fs::FileSystem>& filesystem, std::string base_dir,
    const std::shared_ptr<ds::Partitioning>& partitioning, std::string basename_template,
    const std::shared_ptr<ds::Scanner>& scanner) {
  ds::FileSystemDatasetWriteOptions opts;
  opts.file_write_options = file_write_options;
  opts.filesystem = filesystem;
  opts.base_dir = base_dir;
  opts.partitioning = partitioning;
  opts.basename_template = basename_template;
  StopIfNotOk(ds::FileSystemDataset::Write(opts, scanner));
}

#endif
