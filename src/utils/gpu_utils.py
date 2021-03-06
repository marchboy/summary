import os
import tensorflow as tf


def config_gpu(use_cpu=False, gpu_memory=6):
    if use_cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        # gpus = tf.config.experimental.list_physical_devices('GPU')
        gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
        print(gpus)
        # tf.config.experimental.set_memory_growth(gpus[0], enable=True)   # 需要时申请GPU显存
        if gpus:
            try:
                tf.config.experimental.set_virtual_device_configuration(
                    gpus[0],
                    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=100)])
                logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
            except RuntimeError as e:
                print(e)
