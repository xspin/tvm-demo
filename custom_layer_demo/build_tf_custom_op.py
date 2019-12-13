import numpy as np
import tvm
from tvm import relay
import tvm.relay.testing.tf as tf_testing
import tensorflow as tf
import tensorflow.keras as keras
import os
import logging
tf.get_logger().setLevel(logging.ERROR)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"   

ph_input = tf.placeholder(tf.float32, shape=[None, 10], name='Input_ph')
ph_ytrue = tf.placeholder(tf.float32, shape=[None, 5], name='ytrue_ph')

# inputs = keras.layers.Input(shape=(10,), tensor=ph_input, name='Inputs')
x = keras.layers.Dense(10, use_bias=False, name='Dense')(ph_input)

# custom operations
w = tf.Variable(tf.zeros([10, 8]))
b = tf.Variable(tf.zeros([8]))
x = tf.nn.sigmoid(tf.matmul(x, w)+b, name='custom_op')

outputs = tf.layers.dense(x, 5, activation='softmax', name='Outputs')

losses = tf.keras.losses.categorical_crossentropy(ph_ytrue, outputs)

# model = keras.models.Model(inputs, outputs)
# model.compile('sgd', loss=tf.keras.losses.CategoricalCrossentropy())
# model.summary()

sess = tf.Session()
sess.run(tf.global_variables_initializer())

graph_def = tf_testing.AddShapesToGraphDef(sess, 'Outputs/Softmax')

# shape_dict = {'DecodeJpeg/contents': x.shape}
mod, params = relay.frontend.from_tensorflow(graph_def, layout=None)
with relay.build_config(opt_level=3):
    graph, lib, params = relay.build(mod, target='llvm', params=params)

ctx = tvm.cpu()
e = relay.create_executor(mod=mod, ctx=ctx)
m = tvm.contrib.graph_runtime.create(graph, lib, ctx)

x = tvm.nd.array(np.ones([1,10]).astype(np.float32))
m.set_input(**params)
m.run(**{'Input_ph': x})
out = m.get_output(0)
print(out)