# 添加自定义 Keras Layer


## Keras 自定义层

```python
from keras import backend as K
from keras.layers import Layer

class CustomLayer(Layer):

    def __init__(self, output_dim, activation=None, **kwargs):
        self.output_dim = output_dim
        self.activation = keras.activations.get(activation)
        super(CustomLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel', 
                                      shape=(input_shape[1], self.output_dim),
                                      initializer='uniform',
                                      trainable=True)
        super(CustomLayer, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        out = K.dot(x, self.kernel)
        if self.activation is not None:
            out = self.activation(out)
        return out

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)
```

## tvm 中添加相应转化函数

`tvm/python/tvm/relay/frontend/keras.py` 

```python
def _convert_custom_layer(inexpr, keras_layer, etab):
    weightList = keras_layer.get_weights()
    kernel = etab.new_const(weightList[0].transpose([1, 0]))
    params = {'weight':kernel, 'units':weightList[0].shape[1]}
    input_shape = keras_layer.input_shape
    input_dim = len(input_shape)
    if input_dim > 2:
        raise tvm.error.OpAttributeInvalid(
                'Input shape {} is not valid for CustomLayer.'.format(input_shape))
        inexpr = _op.squeeze(inexpr, axis=0)
    out = _op.nn.dense(data=inexpr, **params)
    # defuse activation
    if sys.version_info.major < 3:
        act_type = keras_layer.activation.func_name
    else:
        act_type = keras_layer.activation.__name__
    if act_type != 'linear':
        out = _convert_activation(out, act_type, etab)
    if input_dim > 2:
        out = _op.expand_dims(out, axis=0)
    return out

_convert_map['CustomLayer'] = _convert_custom_layer
```

