#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__author__ = 'zsy'
__mtime__ = '2017/09/29'"""

from tfrecordwrapper import TFRecordWrapper, IntList
import tensorflow as tf
from config import FLAGS


class Corpus2TFRecord(TFRecordWrapper):
    def __init__(self, output_dir, shards_prefix, shuffle=False):
        super(Corpus2TFRecord, self).__init__(output_dir, shards_prefix, shuffle)
        self.keys = None
        self.types = None

    def get_keys(self):
        self.keys = ['zh', 'en', 'zh_length', 'en_length']
        # self.keys = ['zh', 'en']
        return self.keys

    def get_types(self):
        self.types = [IntList(isfix=False), IntList(isfix=False), IntList(), IntList()]
        # self.types = [IntList(isfix=False), IntList(isfix=False)]
        return self.types


if __name__ == '__main__':
    wp = Corpus2TFRecord(output_dir=FLAGS.data_dir, shards_prefix='en_zh.record')
    example = wp.dataset_batch(shuffle=False)
    with tf.Session() as sess:
        print(sess.run(example))
