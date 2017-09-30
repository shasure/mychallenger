#!/usr/bin/env python
# encoding: utf-8


"""
@version: 0.0
@author: hailang
@Email: seahailang@gmail.com
@software: PyCharm Community Edition
@file: config.py
@time: 2017/9/20 16:06
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

server_or_local_prefix = '/home/zsy'
# server_or_local_prefix = ''

''' model detail'''

tf.app.flags.DEFINE_integer('en_vocab_size', 32589,
                            '''english vocabulary size''')
tf.app.flags.DEFINE_integer('en_embedded_size', 200,
                            '''english embedded size''')
tf.app.flags.DEFINE_integer('en_max_length', 20,
                            '''''')
tf.app.flags.DEFINE_integer('zh_vocab_size', 31936,
                            '''english vocabulary size''')
tf.app.flags.DEFINE_integer('zh_embedded_size', 200,
                            '''english embedded size''')
tf.app.flags.DEFINE_integer('zh_max_length', 20,
                            '''''')
tf.app.flags.DEFINE_integer('batch_size', 64,
                            '''batch size''')
tf.app.flags.DEFINE_integer('attention_size', 100,
                            '''''')

'''train detail'''
tf.app.flags.DEFINE_float('learning_rate', 0.01,
                          '''initial learning rate''')
tf.app.flags.DEFINE_integer('decay_step', 1000,
                            '''decay step''')
tf.app.flags.DEFINE_float('decay_rate', 0.99,
                          '''decay weight''')
tf.app.flags.DEFINE_float('max_gradient', 1.00,
                          '''clipped max gradient''')
tf.app.flags.DEFINE_integer('num_batches', 100,
                            'the number of batches to run')

'''train flags'''
tf.app.flags.DEFINE_boolean('is_inference', False,
                            '''inference flag''')
tf.app.flags.DEFINE_boolean('is_train', True,
                            '''train flag''')

'''dir'''
# tf.app.flags.DEFINE_string("train_dir", 'E:\\AI_Challenger\\my_challenge\\zh_en',
#                            '''train_dir''')
# tf.app.flags.DEFINE_integer('en_vocab_dir', 'E:\\AI_Challenger\\my_challenge\\vocab.en',
#                             '''en vocabulary dir''')
# tf.app.flags.DEFINE_integer('zh_vocab_dir', 'E:\\AI_Challenger\\my_challenge\\vocab.zh',
#                             '''zh vocabulary dir''')

tf.app.flags.DEFINE_string('data_dir',
                           server_or_local_prefix + '/datasets/AI_Challenger.data/ai_challenger_translation_train_20170904/translation_train_data_20170904',
                           'corpus data dir')

tf.app.flags.DEFINE_string('ckpt_dir', server_or_local_prefix + '/train_dir/mychallenger',
                           'checkpoint dir')

'''device'''
tf.app.flags.DEFINE_string('compute_device', 'gpu',
                           'device used for computing')
tf.app.flags.DEFINE_integer('num_gpus', 3,
                            'the number of gpu devices')

''''''

if __name__ == '__main__':
    pass
