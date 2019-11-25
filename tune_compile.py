import tvm
from tvm import relay
from tvm import autotvm
from tvm.autotvm.tuner import XGBTuner, GATuner, RandomTuner, GridSearchTuner
from tvm.autotvm.graph_tuner import DPTuner, PBQPTuner
import tvm.contrib.graph_runtime as runtime
import os
import numpy as np
import tensorflow as tf
tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))


def tune_tasks(mod, params, option):
    print('Creating tasks ...')
    tasks = autotvm.task.extract_from_program(mod["main"], 
                                                target=option['target'],
                                                params=params, 
                                                ops=(relay.op.nn.conv2d, 
                                                     relay.op.nn.dense))
    if option['try_winograd']:
        for i in range(len(tasks)):
            try:  # try winograd template
                tsk = autotvm.task.create(tasks[i].name, 
                                          tasks[i].args,
                                          tasks[i].target, 
                                          tasks[i].target_host, 'winograd')
                input_channel = tsk.workload[1][1]
                if input_channel >= 64:
                    tasks[i] = tsk
            except Exception as err:
                print(err)

    # create tmp log file
    tmp_log_file = option['log_file']
    if os.path.exists(tmp_log_file): os.remove(tmp_log_file)

    for i, tsk in enumerate(reversed(tasks)):

        # converting conv2d tasks to conv2d_NCHWc tasks
        # op_name = tsk.workload[0]
        # if op_name == 'conv2d':
        #     func_create = 'topi_x86_conv2d_NCHWc'
        # elif op_name == 'depthwise_conv2d_nchw':
        #     func_create = 'topi_x86_depthwise_conv2d_NCHWc_from_nchw'
        # else:
        #     func_create = tasks[i].name

        # task = autotvm.task.create(func_create, args=tsk.args,
        #                            target=target, template_key='direct')
        # task.workload = tsk.workload
        # tsk = task

        prefix = "[Task %2d/%2d] (%s)" %(i+1, len(tasks), tsk.name)

        # create tuner
        tuner = option['tuner']
        if tuner == 'xgb' or tuner == 'xgb-rank':
            tuner_obj = XGBTuner(tsk, loss_type='rank')
        elif tuner == 'ga':
            tuner_obj = GATuner(tsk, pop_size=100)
        elif tuner == 'random':
            tuner_obj = RandomTuner(tsk)
        elif tuner == 'gridsearch':
            tuner_obj = GridSearchTuner(tsk)
        else:
            raise ValueError("Invalid tuner: " + tuner)

        if option['use_transfer_learning']:
            if os.path.isfile(tmp_log_file):
                tuner_obj.load_history(autotvm.record.load_from_file(tmp_log_file))

        # do tuning
        n_trial = min(option['n_trial'], len(tsk.config_space))
        tuner_obj.tune(n_trial=n_trial,
                       early_stopping=option['early_stopping'],
                       measure_option=option['measure_option'],
                       callbacks=[
                           autotvm.callback.progress_bar(n_trial, prefix=prefix),
                           autotvm.callback.log_to_file(tmp_log_file)])

    if os.path.exists(option['log_best_file']):
        os.remove(option['log_best_file'])
    autotvm.record.pick_best(option['log_file'], option['log_best_file'])

# Use graph tuner to achieve graph level optimal schedules
# Set use_DP=False if it takes too long to finish.
def tune_graph(graph, dshape, records, opt_sch_file, use_DP=True):
    target_op = [relay.nn.conv2d]
    Tuner = DPTuner if use_DP else PBQPTuner
    executor = Tuner(graph, {input_name: dshape}, records, target_op, target)
    executor.benchmark_layout_transform(min_exec_num=2000)
    executor.run()
    executor.write_opt_sch2record_file(opt_sch_file)

def get_keras_model(model_path=None, batch_size=1):
    import keras
    if model_path is None:
        model = keras.applications.resnet50.ResNet50(include_top=True, 
                                                    weights=None,
                                                    input_shape=(224, 224, 3), 
                                                    classes=1000)
    else:
        print('Loading keras model from', model_path)
        model = keras.models.load_model(model_path)
    input_name = model.layers[0].name
    input_shape = model.layers[0].batch_input_shape
    data_shape = (batch_size, input_shape[-1]) + input_shape[1:-1]
    output_shape = (batch_size, model.layers[-1].units)
    mod, params = relay.frontend.from_keras(model, {input_name: data_shape})
    print(f'data_shape: {data_shape}')
    print(f'output_shape: {output_shape}')
    sample_shape = input_shape[1:]
    return model, mod, params, sample_shape, data_shape, output_shape, input_name

target = 'llvm' # llvm or cuda or ...

option = {
    'target': target,
    'log_file': 'log/%s.log'%target,
    'log_best_file': 'log/%s_best.log'%target,
    'graph_best_file': 'log/%s_graph_best.log'%target,
    'path_lib': 'deploy/model/%s_lib.so'%target, 
    'path_graph': 'deploy/model/%s_graph.json'%target,
    'path_params': 'deploy/model/%s_params.bin'%target,
    'tuner': 'xgb', # random, ga, gridsearch, xgb
    'n_trial': 10,
    'early_stopping': 600,
    'try_winograd': False,
    'use_transfer_learning': False,
    'measure_option': autotvm.measure_option(
        builder=autotvm.LocalBuilder(timeout=100),
        runner=autotvm.LocalRunner(number=4, repeat=3, timeout=4, min_repeat_ms=150),
    ),
}

if __name__ == "__main__":
    if not os.path.exists('log'): os.mkdir('log')

    print('Loading model...')
    model, mod, params_, sample_shape, data_shape, output_shape, input_name = get_keras_model()

    print("Tuning kernels...")
    # tune_tasks(mod, params_, option)

    print("Tuning graph...")
    # tune_graph(mod["main"], data_shape, option['log_file'], option['graph_best_file'])

    print("Compile...")
    # if use tune_tasks
    # with autotvm.apply_history_best(option['log_best_file']): 
    # if use tune_graph
    # with autotvm.apply_graph_best(option['graph_best_file']):
    with relay.build_config(opt_level=3):
            graph, lib, params = relay.build_module.build(mod, target=option['target'], params=params_)

    print('Exporting library...')
    lib.export_library(option['path_lib'])
    with open(option['path_graph'], "w") as fo: fo.write(graph)
    with open(option['path_params'], "wb") as fo: fo.write(relay.save_param_dict(params))

    print('Loading library...')
    loaded_lib = tvm.module.load(option['path_lib'])
    loaded_graph = open(option['path_graph']).read()
    loaded_params = bytearray(open(option['path_params'], 'rb').read())

    print('Runing...')
    ctx = tvm.context(option['target'], 0)
    data_tvm = tvm.nd.array((np.random.uniform(size=data_shape)).astype('float32'))
    m = tvm.contrib.graph_runtime.create(loaded_graph, loaded_lib, ctx)
    m.load_params(loaded_params)
    m.run(**{input_name:data_tvm}) #or m.set_input(input_name, data_tvm); m.run()
    out = m.get_output(0)
    print(out.asnumpy().argmax())

    print('Done')