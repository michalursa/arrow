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

#include <gmock/gmock-matchers.h>

#include <functional>
#include <memory>

#include "arrow/compute/exec.h"
#include "arrow/compute/exec/exec_plan.h"
#include "arrow/compute/exec/expression.h"
#include "arrow/compute/exec/test_util.h"
#include "arrow/record_batch.h"
#include "arrow/testing/future_util.h"
#include "arrow/testing/gtest_util.h"
#include "arrow/testing/matchers.h"
#include "arrow/testing/random.h"
#include "arrow/util/async_generator.h"
#include "arrow/util/logging.h"
#include "arrow/util/thread_pool.h"
#include "arrow/util/vector.h"

using testing::ElementsAre;
using testing::HasSubstr;
using testing::UnorderedElementsAreArray;

namespace arrow {

namespace compute {

TEST(ExecPlanConstruction, Empty) {
  ASSERT_OK_AND_ASSIGN(auto plan, ExecPlan::Make());

  ASSERT_RAISES(Invalid, plan->Validate());
}

TEST(ExecPlanConstruction, SingleNode) {
  ASSERT_OK_AND_ASSIGN(auto plan, ExecPlan::Make());
  auto node = MakeDummyNode(plan.get(), "dummy", /*inputs=*/{}, /*num_outputs=*/0);
  ASSERT_OK(plan->Validate());
  ASSERT_THAT(plan->sources(), ElementsAre(node));
  ASSERT_THAT(plan->sinks(), ElementsAre(node));

  ASSERT_OK_AND_ASSIGN(plan, ExecPlan::Make());
  node = MakeDummyNode(plan.get(), "dummy", /*inputs=*/{}, /*num_outputs=*/1);
  // Output not bound
  ASSERT_RAISES(Invalid, plan->Validate());
}

TEST(ExecPlanConstruction, SourceSink) {
  ASSERT_OK_AND_ASSIGN(auto plan, ExecPlan::Make());
  auto source = MakeDummyNode(plan.get(), "source", /*inputs=*/{}, /*num_outputs=*/1);
  auto sink = MakeDummyNode(plan.get(), "sink", /*inputs=*/{source}, /*num_outputs=*/0);

  ASSERT_OK(plan->Validate());
  EXPECT_THAT(plan->sources(), ElementsAre(source));
  EXPECT_THAT(plan->sinks(), ElementsAre(sink));
}

TEST(ExecPlanConstruction, MultipleNode) {
  ASSERT_OK_AND_ASSIGN(auto plan, ExecPlan::Make());

  auto source1 = MakeDummyNode(plan.get(), "source1", /*inputs=*/{}, /*num_outputs=*/2);

  auto source2 = MakeDummyNode(plan.get(), "source2", /*inputs=*/{}, /*num_outputs=*/1);

  auto process1 =
      MakeDummyNode(plan.get(), "process1", /*inputs=*/{source1}, /*num_outputs=*/2);

  auto process2 = MakeDummyNode(plan.get(), "process1", /*inputs=*/{source1, source2},
                                /*num_outputs=*/1);

  auto process3 =
      MakeDummyNode(plan.get(), "process3", /*inputs=*/{process1, process2, process1},
                    /*num_outputs=*/1);

  auto sink = MakeDummyNode(plan.get(), "sink", /*inputs=*/{process3}, /*num_outputs=*/0);

  ASSERT_OK(plan->Validate());
  ASSERT_THAT(plan->sources(), ElementsAre(source1, source2));
  ASSERT_THAT(plan->sinks(), ElementsAre(sink));
}

struct StartStopTracker {
  std::vector<std::string> started, stopped;

  StartProducingFunc start_producing_func(Status st = Status::OK()) {
    return [this, st](ExecNode* node) {
      started.push_back(node->label());
      return st;
    };
  }

  StopProducingFunc stop_producing_func() {
    return [this](ExecNode* node) { stopped.push_back(node->label()); };
  }
};

TEST(ExecPlan, DummyStartProducing) {
  StartStopTracker t;

  ASSERT_OK_AND_ASSIGN(auto plan, ExecPlan::Make());

  auto source1 = MakeDummyNode(plan.get(), "source1", /*inputs=*/{}, /*num_outputs=*/2,
                               t.start_producing_func(), t.stop_producing_func());

  auto source2 = MakeDummyNode(plan.get(), "source2", /*inputs=*/{}, /*num_outputs=*/1,
                               t.start_producing_func(), t.stop_producing_func());

  auto process1 =
      MakeDummyNode(plan.get(), "process1", /*inputs=*/{source1}, /*num_outputs=*/2,
                    t.start_producing_func(), t.stop_producing_func());

  auto process2 =
      MakeDummyNode(plan.get(), "process2", /*inputs=*/{process1, source2},
                    /*num_outputs=*/1, t.start_producing_func(), t.stop_producing_func());

  auto process3 =
      MakeDummyNode(plan.get(), "process3", /*inputs=*/{process1, source1, process2},
                    /*num_outputs=*/1, t.start_producing_func(), t.stop_producing_func());

  MakeDummyNode(plan.get(), "sink", /*inputs=*/{process3}, /*num_outputs=*/0,
                t.start_producing_func(), t.stop_producing_func());

  ASSERT_OK(plan->Validate());
  ASSERT_EQ(t.started.size(), 0);
  ASSERT_EQ(t.stopped.size(), 0);

  ASSERT_OK(plan->StartProducing());
  // Note that any correct reverse topological order may do
  ASSERT_THAT(t.started, ElementsAre("sink", "process3", "process2", "process1",
                                     "source2", "source1"));
  ASSERT_EQ(t.stopped.size(), 0);
}

TEST(ExecPlan, DummyStartProducingError) {
  StartStopTracker t;

  ASSERT_OK_AND_ASSIGN(auto plan, ExecPlan::Make());
  auto source1 = MakeDummyNode(
      plan.get(), "source1", /*num_inputs=*/{}, /*num_outputs=*/2,
      t.start_producing_func(Status::NotImplemented("zzz")), t.stop_producing_func());

  auto source2 =
      MakeDummyNode(plan.get(), "source2", /*num_inputs=*/{}, /*num_outputs=*/1,
                    t.start_producing_func(), t.stop_producing_func());

  auto process1 = MakeDummyNode(
      plan.get(), "process1", /*num_inputs=*/{source1}, /*num_outputs=*/2,
      t.start_producing_func(Status::IOError("xxx")), t.stop_producing_func());

  auto process2 =
      MakeDummyNode(plan.get(), "process2", /*num_inputs=*/{process1, source2},
                    /*num_outputs=*/1, t.start_producing_func(), t.stop_producing_func());

  auto process3 =
      MakeDummyNode(plan.get(), "process3", /*num_inputs=*/{process1, source1, process2},
                    /*num_outputs=*/1, t.start_producing_func(), t.stop_producing_func());

  MakeDummyNode(plan.get(), "sink", /*num_inputs=*/{process3}, /*num_outputs=*/0,
                t.start_producing_func(), t.stop_producing_func());

  ASSERT_OK(plan->Validate());
  ASSERT_EQ(t.started.size(), 0);
  ASSERT_EQ(t.stopped.size(), 0);

  // `process1` raises IOError
  ASSERT_RAISES(IOError, plan->StartProducing());
  ASSERT_THAT(t.started, ElementsAre("sink", "process3", "process2", "process1"));
  // Nodes that started successfully were stopped in reverse order
  ASSERT_THAT(t.stopped, ElementsAre("process2", "process3", "sink"));
}

namespace {

struct BatchesWithSchema {
  std::vector<ExecBatch> batches;
  std::shared_ptr<Schema> schema;
};

Result<ExecNode*> MakeTestSourceNode(ExecPlan* plan, std::string label,
                                     BatchesWithSchema batches_with_schema, bool parallel,
                                     bool slow) {
  DCHECK_GT(batches_with_schema.batches.size(), 0);

  auto opt_batches = arrow::internal::MapVector(
      [](ExecBatch batch) { return util::make_optional(std::move(batch)); },
      std::move(batches_with_schema.batches));

  AsyncGenerator<util::optional<ExecBatch>> gen;

  if (parallel) {
    // emulate batches completing initial decode-after-scan on a cpu thread
    ARROW_ASSIGN_OR_RAISE(
        gen, MakeBackgroundGenerator(MakeVectorIterator(std::move(opt_batches)),
                                     arrow::internal::GetCpuThreadPool()));

    // ensure that callbacks are not executed immediately on a background thread
    gen = MakeTransferredGenerator(std::move(gen), arrow::internal::GetCpuThreadPool());
  } else {
    gen = MakeVectorGenerator(std::move(opt_batches));
  }

  if (slow) {
    gen = MakeMappedGenerator(std::move(gen), [](const util::optional<ExecBatch>& batch) {
      SleepABit();
      return batch;
    });
  }

  return MakeSourceNode(plan, label, std::move(batches_with_schema.schema),
                        std::move(gen));
}

Result<std::vector<ExecBatch>> StartAndCollect(
    ExecPlan* plan, AsyncGenerator<util::optional<ExecBatch>> gen) {
  RETURN_NOT_OK(plan->Validate());
  RETURN_NOT_OK(plan->StartProducing());

  auto maybe_collected = CollectAsyncGenerator(gen).result();
  ARROW_ASSIGN_OR_RAISE(auto collected, maybe_collected);

  plan->StopProducing();

  return arrow::internal::MapVector(
      [](util::optional<ExecBatch> batch) { return std::move(*batch); }, collected);
}

BatchesWithSchema MakeBasicBatches() {
  BatchesWithSchema out;
  out.batches = {
      ExecBatchFromJSON({int32(), boolean()}, "[[null, true], [4, false]]"),
      ExecBatchFromJSON({int32(), boolean()}, "[[5, null], [6, false], [7, false]]")};
  out.schema = schema({field("i32", int32()), field("bool", boolean())});
  return out;
}

BatchesWithSchema MakeRandomBatches(const std::shared_ptr<Schema>& schema,
                                    int num_batches = 10, int batch_size = 4) {
  BatchesWithSchema out;

  random::RandomArrayGenerator rng(42);
  out.batches.resize(num_batches);

  for (int i = 0; i < num_batches; ++i) {
    out.batches[i] = ExecBatch(*rng.BatchOf(schema->fields(), batch_size));
    // add a tag scalar to ensure the batches are unique
    out.batches[i].values.emplace_back(i);
  }
  return out;
}
}  // namespace

TEST(ExecPlanExecution, SourceSink) {
  for (bool slow : {false, true}) {
    SCOPED_TRACE(slow ? "slowed" : "unslowed");

    for (bool parallel : {false, true}) {
      SCOPED_TRACE(parallel ? "parallel" : "single threaded");

      ASSERT_OK_AND_ASSIGN(auto plan, ExecPlan::Make());

      auto basic_data = MakeBasicBatches();

      ASSERT_OK_AND_ASSIGN(auto source, MakeTestSourceNode(plan.get(), "source",
                                                           basic_data, parallel, slow));

      auto sink_gen = MakeSinkNode(source, "sink");

      ASSERT_THAT(StartAndCollect(plan.get(), sink_gen),
                  ResultWith(UnorderedElementsAreArray(basic_data.batches)));
    }
  }
}

TEST(ExecPlanExecution, SourceSinkError) {
  ASSERT_OK_AND_ASSIGN(auto plan, ExecPlan::Make());

  auto basic_data = MakeBasicBatches();
  auto it = basic_data.batches.begin();
  AsyncGenerator<util::optional<ExecBatch>> gen =
      [&]() -> Result<util::optional<ExecBatch>> {
    if (it == basic_data.batches.end()) {
      return Status::Invalid("Artificial error");
    }
    return util::make_optional(*it++);
  };

  auto source = MakeSourceNode(plan.get(), "source", {}, gen);
  auto sink_gen = MakeSinkNode(source, "sink");

  ASSERT_THAT(StartAndCollect(plan.get(), sink_gen),
              Raises(StatusCode::Invalid, HasSubstr("Artificial")));
}

TEST(ExecPlanExecution, StressSourceSink) {
  for (bool slow : {false, true}) {
    SCOPED_TRACE(slow ? "slowed" : "unslowed");

    for (bool parallel : {false, true}) {
      SCOPED_TRACE(parallel ? "parallel" : "single threaded");

      int num_batches = slow && !parallel ? 30 : 300;

      ASSERT_OK_AND_ASSIGN(auto plan, ExecPlan::Make());

      auto random_data = MakeRandomBatches(
          schema({field("a", int32()), field("b", boolean())}), num_batches);

      ASSERT_OK_AND_ASSIGN(auto source, MakeTestSourceNode(plan.get(), "source",
                                                           random_data, parallel, slow));

      auto sink_gen = MakeSinkNode(source, "sink");

      ASSERT_THAT(StartAndCollect(plan.get(), sink_gen),
                  ResultWith(UnorderedElementsAreArray(random_data.batches)));
    }
  }
}

TEST(ExecPlanExecution, SourceFilterSink) {
  ASSERT_OK_AND_ASSIGN(auto plan, ExecPlan::Make());

  auto basic_data = MakeBasicBatches();

  ASSERT_OK_AND_ASSIGN(auto source,
                       MakeTestSourceNode(plan.get(), "source", basic_data,
                                          /*parallel=*/false, /*slow=*/false));

  ASSERT_OK_AND_ASSIGN(auto predicate,
                       equal(field_ref("i32"), literal(6)).Bind(*basic_data.schema));

  ASSERT_OK_AND_ASSIGN(auto filter, MakeFilterNode(source, "filter", predicate));

  auto sink_gen = MakeSinkNode(filter, "sink");

  ASSERT_THAT(StartAndCollect(plan.get(), sink_gen),
              ResultWith(UnorderedElementsAreArray(
                  {ExecBatchFromJSON({int32(), boolean()}, "[]"),
                   ExecBatchFromJSON({int32(), boolean()}, "[[6, false]]")})));
}

TEST(ExecPlanExecution, SourceProjectSink) {
  ASSERT_OK_AND_ASSIGN(auto plan, ExecPlan::Make());

  auto basic_data = MakeBasicBatches();

  ASSERT_OK_AND_ASSIGN(auto source,
                       MakeTestSourceNode(plan.get(), "source", basic_data,
                                          /*parallel=*/false, /*slow=*/false));

  std::vector<Expression> exprs{
      not_(field_ref("bool")),
      call("add", {field_ref("i32"), literal(1)}),
  };
  for (auto& expr : exprs) {
    ASSERT_OK_AND_ASSIGN(expr, expr.Bind(*basic_data.schema));
  }

  ASSERT_OK_AND_ASSIGN(auto projection, MakeProjectNode(source, "project", exprs));

  auto sink_gen = MakeSinkNode(projection, "sink");

  ASSERT_THAT(StartAndCollect(plan.get(), sink_gen),
              ResultWith(UnorderedElementsAreArray(
                  {ExecBatchFromJSON({boolean(), int32()}, "[[false, null], [true, 5]]"),
                   ExecBatchFromJSON({boolean(), int32()},
                                     "[[null, 6], [true, 7], [true, 8]]")})));
}

TEST(ExecPlanExecution, GroupByOnly) {
  BatchesWithSchema input;
  input.batches = {
      ExecBatchFromJSON({int32(), int32()}, "[[1, 12], [2, 7], [1, 3]]"),
      ExecBatchFromJSON({int32(), int32()}, "[[5, 5], [2, 3], [1, 8]]")};
  input.schema = schema({field("a", int32()), field("b", int32())});

  ASSERT_OK_AND_ASSIGN(auto plan, ExecPlan::Make());
  ASSERT_OK_AND_ASSIGN(auto scan, MakeTestSourceNode(plan.get(), "source", input, /*parallel=*/false, /*slow=*/false));
  ASSERT_OK_AND_ASSIGN(auto gby, MakeGroupByNode(scan,
                                                 "gby",
                                                 {"a"},
                                                 {"b"},
                                                 {{"hash_sum", nullptr}}));
  auto sink_gen = MakeSinkNode(gby, "sink");

  ASSERT_THAT(StartAndCollect(plan.get(), sink_gen),
              ResultWith(UnorderedElementsAreArray(
                  {ExecBatchFromJSON({int64(), int32()}, "[[23, 1], [10, 2], [5, 5]]")}
              )));
}

TEST(ExecPlanExecution, FilterProjectGroupByFilter) {
  // TODO: implement
}

}  // namespace compute
}  // namespace arrow
