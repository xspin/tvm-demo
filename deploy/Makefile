
TVM_ROOT=/home/chengx/tvm

DMLC_CORE=${TVM_ROOT}/3rdparty/dmlc-core

PKG_CFLAGS = -std=c++11 -O2 -fPIC\
	-I${TVM_ROOT}/include\
	-I${DMLC_CORE}/include\
	-I${TVM_ROOT}/3rdparty/dlpack/include\
	-I/usr/local/cuda/include 

PKG_LDFLAGS = -ldl -lpthread -L/usr/local/cuda/lib64 -lcudart -lcuda

all: bin/demo

run: bin/demo
	bin/demo

lib/libtvm_runtime_pack.o: tvm_runtime_pack.cc
	@mkdir -p $(@D)
	$(CXX) -c $(PKG_CFLAGS) -o $@  $^ 
	
lib/demo.o: demo.cc
	@mkdir -p $(@D)
	$(CXX) -c $(PKG_CFLAGS) -o $@  $^	
	
bin/demo: lib/demo.o lib/libtvm_runtime_pack.o
	@mkdir -p $(@D)
	$(CXX) $(PKG_CFLAGS) $(PKG_LDFLAGS) -o $@  $^	

clean:
	@rm -rf lib
	@rm -rf bin