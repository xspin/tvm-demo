# TVM 添加算子

https://docs.tvm.ai/dev/relay_add_op.html

## 添加 expr_op
`tvm/include/tvm/expr_operator.h`

## Broadcast 类型算子

`tvm/topi/include/topi/braodcast.h` 添加 OP 定义
```c++
TOPI_DEFINE_BCAST_OP(custom_bcast_geomean, { return tvm::sqrt(a*b); });
```

`tvm/topi/src/topi.cc` topi 注册 OP
```c++
TOPI_REGISTER_BCAST_OP("topi.custom_bcast_geomean", topi::custom_bcast_geomean);
```

`tvm/src/relay/op/tensor/binary.cc` relay 注册 OP
```C++
RELAY_REGISTER_BINARY_OP("custom_bcast_geomean")
.describe("Elementwise geometric mean with with broadcasting")
.set_support_level(1)
.set_attr<FTVMCompute>("FTVMCompute", RELAY_BINARY_COMPUTE(topi::custom_bcast_geomean));
```

`tvm/python/tvm/relay/op/tensor.py` 实现 Python 相应接口
```python
def custom_bcast_geomean(lhs, rhs):
    """Geomean with numpy-style broadcasting.
    """
    return _make.custom_bcast_geomean(lhs, rhs)
```

`tvm/python/tvm/relay/op/_tensor.py` 注册相应 schedule 和 shape 类型
```python
register_schedule("custom_bcast_geomean", schedule_broadcast)
register_shape_func("custom_bcast_geomean", False, broadcast_shape_func)
```


### Python 中调用自定义 OP

```python
import tvm
from tvm import relay

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
m.run(**{'a': a, 'b': b})
out = m.get_output(0)
print(out)
```


## Elemwise 类型算子

`tvm/topi/include/topi/elemwise.h` 
```c++
inline Tensor custom_elemwise_plus(const Tensor& x,
                       const Expr& a,
                       std::string name = "T_custom_plus",
                       std::string tag = kElementWise) {
  return compute(x->shape, [&](const Array<Var>& i) {
    return x(i) + a;
  }, name, tag);
}
```

`tvm/topi/src/topi.cc` 
```c++
TVM_REGISTER_GLOBAL("topi.custom_elemwise_plus")
.set_body([](TVMArgs args, TVMRetValue *rv) {
  *rv = cast(args[0], args[1]);
  })
```

`tvm/src/relay/op/tensor/unary.cc`

