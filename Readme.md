
## TVM 编译模型和使用

修改 TVM 路径
- 修改 makefile 中 TVM 路径：`TVM_ROOT`
- 修改 deploy/tvm_runtime_pack.cc 中 #include TVM 路径

使用 TVM 编译和导出模型
```shell
$ python tune_compile.py
```

编译 C++ 代码

```shell
$ cd deploy
$ make
```

运行demo

```shell
$ make run
```

编译和使用 GPU 模型
- 修改 tune_compile.py 中 option['target'] 为 'cuda'
- 修改 deploy/demo.cc 中 DEVICE_TYPE 为 kDLGPU

-----

## TVM 安装

> Relay 是 TVM 中用来替代 NNVM 的模块，其本身被认为是 NNVM 第二代。 
>  https://zhuanlan.zhihu.com/p/91283238 

### 安装 LLVM

#### Debian/Ubuntu

方法1：

```shell
$ #(optinal) apt-add-repository deb ... # https://apt.llvm.org/
$ apt update
$ apt install llvm-6.0 llvm-6.0-dev llvm-6.0-runtime
```

方法2：

```shell
$ bash -c "$(wget -O - https://apt.llvm.org/llvm.sh)"
```

Fix `lsb_release: command not found`

```shell
$ apt update && apt install lsb-core
```

#### 源码编译

 http://releases.llvm.org/download.html 

 http://llvm.org/docs/CMake.html 

```shell
$ cd path/to/llvm-src
$ mkdir build && cd build
$ cmake ../
$ cmake --build .
$ cmake --build . --target install # if install to system
```

 No CMAKE_CXX_COMPILER could be found 

```shell
$ apt install build-essential
```

### 编译 TVM

```shell
$ git clone --recursive https://github.com/apache/incubator-tvm
$ cd tvm && mkdir build
$ cp cmake/config.cmake build
$ cd build
$ vi config.cmake # edit options: LLVM, CUDA, etc. 
$ cmake ..
$ make -j4
```

 https://docs.tvm.ai/install/from_source.html 

### 安装 TVM 到 Python

方法1：

```shell
$ export TVM_HOME=/path/to/tvm
$ export PYTHONPATH=$TVM_HOME/python:$TVM_HOME/topi/python:$TVM_HOME/nnvm/python:${PYTHONPATH}
```

方法2：

```shell
$ cd /path/to/tvm
$ cd python; python setup.py install --user; cd ..
$ cd topi/python; python setup.py install --user; cd ../..
$ cd nnvm/python; python setup.py install --user; cd ../..
```

Dependencies

- Necessary `pip3 install --user numpy decorator attrs`
-  RPC Tracker `pip3 install --user tornado`
-  auto-tuning module `pip3 install --user tornado psutil xgboost`
-  build tvm to compile a model `sudo apt install antlr4; pip3 install --user mypy orderedset antlr4-python3-runtime`

TVMError: Check failed: bf != nullptr: Target llvm is not enabled

>  cmake automatically identify LLVM path. In your case you can set the llvm-config path manually in `config.cmake` as below: `set(USE_LLVM /path/to/llvm/bin/llvm-config)` 

nvcc not found

```shell
$ export PATH=/path/to/cuda/bin:$PATH    # e.g. /usr/local/cuda/bin
```


## 添加自定义算子
[add_op_demo](add_op_demo/Readme.md)