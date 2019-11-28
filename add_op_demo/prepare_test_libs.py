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
"""Script to prepare test_addone.so"""
import tvm
import os

from tvm import relay
import numpy as np

def test():
    N = 5
    x = tvm.nd.array(np.arange(N, dtype=np.float32))
    y = tvm.nd.array(np.zeros(N, dtype=np.float32))
    x = relay.var('a', shape=(N, 1))
    y = relay.var('b', shape=(N, 1))
    z = relay.op.custom_bcast_geomean(x, y)
    # z = relay.op.add(x, y)
    mod = relay.module.Module.from_expr(z)
    graph, lib, params = relay.build(mod, target='llvm')
    print('Exporting library...')
    lib.export_library('lib/lib.so')
    with open('lib/graph.json', "w") as fo: fo.write(graph)
    with open('lib/params.bin', "wb") as fo: fo.write(relay.save_param_dict(params))

    ctx = tvm.cpu()
    e = relay.create_executor(mod=mod, ctx=ctx)
    m = tvm.contrib.graph_runtime.create(graph, lib, ctx)
    # m.load_params(**params)
    a = tvm.nd.array(np.ones([N, 1]).astype(np.float32))
    b = tvm.nd.array(2*np.ones([N, 1]).astype(np.float32))
    c = tvm.nd.array(3*np.ones([N, 1]).astype(np.float32))
    m.run(**{'a': a, 'b': b})
    out = m.get_output(0)
    print(out)
    m.run(**{'a': a, 'b': c})
    out = m.get_output(0)
    print(out)

def prepare_test_libs(base_path):
    n = tvm.var("n")
    A = tvm.placeholder((n,), name='A')
    B = tvm.compute(A.shape, lambda *i: A(*i) + 1.0, name='B')
    s = tvm.create_schedule(B.op)
    print(B)
    # Compile library as dynamic library
    fadd_dylib = tvm.build(s, [A, B], "llvm", name="addone")
    dylib_path = os.path.join(base_path, "test_addone.so")
    # fadd_dylib.export_library(dylib_path)

if __name__ == "__main__":
    lib_dir = 'lib'
    if not os.path.exists(lib_dir): os.mkdir(lib_dir)
    # prepare_test_libs(lib_dir)
    test()
