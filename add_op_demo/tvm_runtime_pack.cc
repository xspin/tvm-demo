/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \brief This is an all in one TVM runtime file.
 *
 *   You only have to use this file to compile libtvm_runtime to
 *   include in your project.
 *
 *  - Copy this file into your project which depends on tvm runtime.
 *  - Compile with -std=c++11
 *  - Add the following include path
 *     - /path/to/tvm/include/
 *     - /path/to/tvm/3rdparty/dmlc-core/include/
 *     - /path/to/tvm/3rdparty/dlpack/include/
 *   - Add -lpthread -ldl to the linked library.
 *   - You are good to go.
 *   - See the Makefile in the same folder for example.
 *
 *  The include files here are presented with relative path
 *  You need to remember to change it to point to the right file.
 *
 */
#include "/home/chengx/tvm/src/runtime/c_runtime_api.cc"
#include "/home/chengx/tvm/src/runtime/cpu_device_api.cc"
#include "/home/chengx/tvm/src/runtime/workspace_pool.cc"
#include "/home/chengx/tvm/src/runtime/module_util.cc"
#include "/home/chengx/tvm/src/runtime/module.cc"
#include "/home/chengx/tvm/src/runtime/registry.cc"
#include "/home/chengx/tvm/src/runtime/file_util.cc"
#include "/home/chengx/tvm/src/runtime/threading_backend.cc"
#include "/home/chengx/tvm/src/runtime/thread_pool.cc"
#include "/home/chengx/tvm/src/runtime/ndarray.cc"

// NOTE: all the files after this are optional modules
// that you can include remove, depending on how much feature you use.

// Likely we only need to enable one of the following
// If you use Module::Load, use dso_module
// For system packed library, use system_lib_module
#include "/home/chengx/tvm/src/runtime/dso_module.cc"
#include "/home/chengx/tvm/src/runtime/system_lib_module.cc"

// Graph runtime
#include "/home/chengx/tvm/src/runtime/graph/graph_runtime.cc"

// Uncomment the following lines to enable RPC
// #include "/home/chengx/tvm/src/runtime/rpc/rpc_session.cc"
// #include "/home/chengx/tvm/src/runtime/rpc/rpc_event_impl.cc"
// #include "/home/chengx/tvm/src/runtime/rpc/rpc_server_env.cc"

// These macros enables the device API when uncommented.
#define TVM_CUDA_RUNTIME 1
#define TVM_METAL_RUNTIME 1
#define TVM_OPENCL_RUNTIME 1

// Uncomment the following lines to enable Metal
// #include "/home/chengx/tvm/src/runtime/metal/metal_device_api.mm"
// #include "/home/chengx/tvm/src/runtime/metal/metal_module.mm"

// Uncomment the following lines to enable CUDA
// #include "/home/chengx/tvm/src/runtime/cuda/cuda_device_api.cc"
// #include "/home/chengx/tvm/src/runtime/cuda/cuda_module.cc"

// Uncomment the following lines to enable OpenCL
// #include "/home/chengx/tvm/src/runtime/opencl/opencl_device_api.cc"
// #include "/home/chengx/tvm/src/runtime/opencl/opencl_module.cc"
