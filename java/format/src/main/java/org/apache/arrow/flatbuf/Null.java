// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
// 
// http://www.apache.org/licenses/LICENSE-2.0
// 
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// 
// automatically generated by the FlatBuffers compiler, do not modify

package org.apache.arrow.flatbuf;

import java.nio.*;
import java.lang.*;
import java.util.*;
import com.google.flatbuffers.*;

@SuppressWarnings("unused")
/**
 * These are stored in the flatbuffer in the Type union below
 */
public final class Null extends Table {
  public static void ValidateVersion() { Constants.FLATBUFFERS_1_12_0(); }
  public static Null getRootAsNull(ByteBuffer _bb) { return getRootAsNull(_bb, new Null()); }
  public static Null getRootAsNull(ByteBuffer _bb, Null obj) { _bb.order(ByteOrder.LITTLE_ENDIAN); return (obj.__assign(_bb.getInt(_bb.position()) + _bb.position(), _bb)); }
  public void __init(int _i, ByteBuffer _bb) { __reset(_i, _bb); }
  public Null __assign(int _i, ByteBuffer _bb) { __init(_i, _bb); return this; }


  public static void startNull(FlatBufferBuilder builder) { builder.startTable(0); }
  public static int endNull(FlatBufferBuilder builder) {
    int o = builder.endTable();
    return o;
  }

  public static final class Vector extends BaseVector {
    public Vector __assign(int _vector, int _element_size, ByteBuffer _bb) { __reset(_vector, _element_size, _bb); return this; }

    public Null get(int j) { return get(new Null(), j); }
    public Null get(Null obj, int j) {  return obj.__assign(__indirect(__element(j), bb), bb); }
  }
}

