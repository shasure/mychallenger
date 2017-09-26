#!/usr/bin/env python
# encoding: utf-8


"""
@version: 0.0
@author: hailang
@Email: seahailang@gmail.com
@software: PyCharm Community Edition
@file: record_input.py
@time: 2017/9/26 12:35
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from datetime import datetime

FLAGS = tf.app.flags.FLAGS



def inputs(filenames):
    reader = tf.TFRecordReader()
    filenames = tf.train.string_input_producer(filenames)
    example = reader.read(filenames)[1]
    feature_map ={
        'en':tf.VarLenFeature(dtype=tf.int64),
        'zh':tf.VarLenFeature(dtype=tf.int64)
    }
    features = tf.parse_single_example(example,feature_map)

    return features

def test_input(argv=None):
    filename = ['./en_zh.record%d-8'%(i) for i in range(8)]
    features = inputs(filename)
    with tf.Session() as sess:
        tf.train.queue_runner.start_queue_runners()
        print(sess.run(features))

if __name__ == '__main__':
    tf.app.run(test_input,argv=None)