import tvm
import numpy as np

def verify(mod, fname):
  # Get the function from the module
  f = mod.get_function(fname)
  # Use tvm.nd.array to convert numpy ndarray to tvm
  # NDArray type, so that function can be invoked normally
  N = 10 
  x = tvm.nd.array(np.arange(N, dtype=np.float32))
  y = tvm.nd.array(np.zeros(N, dtype=np.float32))
  # Invoke the function
  f(x, y)
  np_x = x.asnumpy() 
  np_y = y.asnumpy() 
  # Verify correctness of function
  assert(np.all([xi+1 == yi for xi, yi in zip(np_x, np_y)]))
  print("Finish verification...")
  

if __name__ == "__main__":
  # The normal dynamic loading method for deployment
  mod_dylib = tvm.module.load("lib/test_addone.so")
  print("Verify dynamic loading from test_addone.so")
  verify(mod_dylib, "addone")
  # There might be methods to use the system lib way in
  # python, but dynamic loading is good enough for now.
