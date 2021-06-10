#!/bin/bash
#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

set -e

arrow_dir=${1}
cpp_build_dir=${2}
java_dir=${arrow_dir}/java

export ARROW_TEST_DATA=${arrow_dir}/testing/data

pushd $java_dir

# build the entire project
mvn clean install -P arrow-jni -Darrow.cpp.build.dir=$cpp_build_dir

MODULES=(
  adapter/avro
  adapter/jdbc
  adapter/orc
  algorithm
  compression
  dataset
  flight/flight-core
  flight/flight-grpc
  format
  gandiva
  memory/memory-core
  memory/memory-netty
  memory/memory-unsafe
  performance
  plasma
  tools
  vector
)

# copy all jars to distribution folder, excluding the unit tests
for module in "${MODULES[@]}"; do
  find $module/target/ -name "*.jar" -not -name "*tests*" -not -name "*benchmarks*" -exec cp  {} $cpp_build_dir \;
done

popd
