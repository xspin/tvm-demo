import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"   
import keras
from keras import backend as K
from keras.layers import Layer
import tvm
from tvm import relay
import numpy as np

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

model = keras.models.Sequential()
model.add(keras.layers.InputLayer(input_shape=(10,), name='Input'))
model.add(keras.layers.Dense(10, use_bias=False, name='Dense'))
model.add(CustomLayer(5, activation='linear', name='MyLayer'))
# model.compile(optimizer='sgd', loss='mean_squared_error')

print(model.summary())

data_shape = (1, model.input_shape[1])
mod, params = relay.frontend.from_keras(model, {model.input_names[0]: data_shape})

## build
graph, lib, params = relay.build(mod, params=params, target='llvm')

ctx = tvm.cpu()
e = relay.create_executor(mod=mod, ctx=ctx)
m = tvm.contrib.graph_runtime.create(graph, lib, ctx)

x = tvm.nd.array(np.ones(data_shape).astype(np.float32))
m.run(**{model.input_names[0]: x}, **params)
out = m.get_output(0)
print(out)