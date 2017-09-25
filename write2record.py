#!/usr/bin/env python
# encoding: utf-8


"""
@version: 0.0
@author: hailang
@Email: seahailang@gmail.com
@software: PyCharm Community Edition
@file: write2record.py
@time: 2017/9/7 19:53
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from datetime import datetime

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('train_dir',
                           '../ai_challenger_translation_train_20170904/translation_train_data_20170904',
                           '''train data dir''')
tf.app.flags.DEFINE_string('output_dir',
                           './output',
                           'output data dir')



if __name__ == '__main__':
    pass