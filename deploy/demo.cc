#include <dlpack/dlpack.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/packed_func.h>
#include <cuda.h>
#include <algorithm>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <ctime>
#include <cstdlib>

#define DEVICE_TYPE kDLCPU // kDLCPU or kDLGPU

void test()
{
    std::string lib_path, graph_path, params_path;
    std::string prefix = (DEVICE_TYPE==kDLCPU)? "model/llvm_" : "model/cuda_";
    lib_path = prefix + "lib.so";
    graph_path = prefix + "graph.json";
    params_path = prefix + "params.bin";
	std::string input_name = "input_1";
    
    // tvm module for compiled functions
    tvm::runtime::Module mod_syslib = tvm::runtime::Module::LoadFromFile(lib_path);
    // json graph
    std::ifstream json_in(graph_path, std::ios::in);
    std::string json_data((std::istreambuf_iterator<char>(json_in)), std::istreambuf_iterator<char>());
    json_in.close();
    // parameters in binary
    std::ifstream params_in(params_path, std::ios::binary);
    std::string params_data((std::istreambuf_iterator<char>(params_in)), std::istreambuf_iterator<char>());
    params_in.close();
    // parameters need to be TVMByteArray type to indicate the binary data
    TVMByteArray params_arr;
    params_arr.data = params_data.c_str();
    params_arr.size = params_data.length();

    int dtype_code = kDLFloat;
    int dtype_bits = 32;
    int dtype_lanes = 1;
    int device_id = 0;

    // get global function module for graph runtime
    tvm::runtime::Module mod = (*tvm::runtime::Registry::Get("tvm.graph_runtime.create"))(json_data, mod_syslib, (int)DEVICE_TYPE, device_id);

    int device_type = kDLCPU;
    DLTensor* x;
    int in_ndim = 4;
    int64_t in_shape[4] = {1, 3, 224, 224};
    TVMArrayAlloc(in_shape, in_ndim, dtype_code, dtype_bits, dtype_lanes, device_type, device_id, &x);

    // load image data saved in binary
    // const std::string data_filename = "cat.bin";
    // std::ifstream data_fin(data_filename, std::ios::binary);
    // if(!data_fin) throw std::runtime_error("Could not open: " + data_filename);
    // data_fin.read(static_cast<char*>(x->data), 3 * 224 * 224 * 4);
    std::srand(std::time(nullptr)); 
    for (auto i=0; i<1*3*224*224; i++) {
        static_cast<float*>(x->data)[i] = (float) std::rand()/RAND_MAX; 
    }
    x->strides = nullptr;
    x->byte_offset = 0;

    // get the function from the module(set input data)
    tvm::runtime::PackedFunc set_input = mod.GetFunction("set_input");
    // get the function from the module(load patameters)
    tvm::runtime::PackedFunc load_params = mod.GetFunction("load_params");
    load_params(params_arr);
    // get the function from the module(run it)
    tvm::runtime::PackedFunc run = mod.GetFunction("run");
    
    set_input(input_name, x);
    run();

    DLTensor* y;
    int out_ndim = 2;
    int64_t out_shape[2] = {1, 1000, };
    TVMArrayAlloc(out_shape, out_ndim, dtype_code, dtype_bits, dtype_lanes, device_type, device_id, &y);

    // get the function from the module(get output data)
    tvm::runtime::PackedFunc get_output = mod.GetFunction("get_output");
    get_output(0, y);

    // get the maximum position in output vector
    auto y_iter = static_cast<float*>(y->data);
    auto max_iter = std::max_element(y_iter, y_iter + 1000);
    auto max_index = std::distance(y_iter, max_iter);
    std::cout << "The maximum position in output vector is: " << max_index << std::endl;

    TVMArrayFree(x);
    TVMArrayFree(y);
}


int main(int argc, char **argv) {
    if (DEVICE_TYPE==kDLCPU) {
        std::cout << "Device type: CPU" << std::endl;
    } else {
        std::cout << "Device type: GPU" << std::endl;
    }
    test();
    return 0;
}