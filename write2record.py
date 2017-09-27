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
import threading
import multiprocessing
from multiprocessing import Process, Manager
import pickle
import numpy as np
import sys

from build_vocab import chinese_parse, english_parse,sentence2int

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('train_dir',
                           '../ai_challenger_translation_train_20170904/translation_train_data_20170904',
                           '''train data dir''')
tf.app.flags.DEFINE_string('output_dir',
                           './output',
                           'output data dir')

def _int64_feature(value):
    if not isinstance(value,list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _convert_to_example(zh_int_list,en_int_list):
    feature_dict={
        'zh': _int64_feature(zh_int_list),
        'en': _int64_feature(en_int_list),
        'zh_length':_int64_feature(len(zh_int_list)),
        'en_length':_int64_feature(len(en_int_list))
    }
    return tf.train.Example(
        features=tf.train.Features(feature=feature_dict)
    )

# def process(zh_en_file,output_file):
#     writer = tf.python_io.TFRecordWriter(output_file)
#     with open(zh_en_file,'r') as file:
#         lines = file.readlines()
#         for i in range(int(len(lines)/2)):
#             en = lines[2*i]
#             print(en)
#             zh = lines[2*i+1]
#             print(zh)
#             zh = zh.strip().split(' ')
#             en = en.strip().split(' ')
#             zh = [int(i) for i in zh]
#             en = [int(i) for i in en]
#             example = _convert_to_example(zh,en)
#             writer.write(example.SerializeToString())

def zh_process(zh,num_threads,thread_id,i,zh_vocab):
    return sentence2int(chinese_parse(zh[i*num_threads+thread_id].strip()), zh_vocab)

def en_process(en,num_threads,thread_id,i,en_vocab):
    return sentence2int(english_parse(en[i*num_threads+thread_id].strip()), en_vocab)


def process(zh,en,zh_vocab,en_vocab,writer_name):
    assert len(zh) == len(en), 'bad data set'
    writer = tf.python_io.TFRecordWriter(writer_name)
    for i in range(len(zh)):
        # print(i)
        zh_line = sentence2int(chinese_parse(zh[i].strip()), zh_vocab)
        en_line = sentence2int(english_parse(en[i].strip()), en_vocab)
        example = _convert_to_example(zh_line,en_line)
        writer.write(example.SerializeToString())

def main(argv=None):
    start = datetime.now()
    with open(FLAGS.train_dir+'/train.zh.m','r',encoding='utf-8') as file:
        zh = file.readlines()
    with open(FLAGS.train_dir+'/train.en.m','r',encoding='utf-8') as file:
        en = file.readlines()

    num_threads = 8
    zh_list = [[] for _ in range(num_threads)]
    en_list = [[] for _ in range(num_threads)]
    output_file = 'en_zh.record'
    writers = [output_file + '%d-%d' % (thread_id, num_threads) for thread_id in range(num_threads)]
    for i in range(int(len(zh)/num_threads)):
        for j in range(num_threads):
            zh_list[j].append(zh[i*num_threads+j])
            en_list[j].append(en[i * num_threads + j])
    del zh
    del en



    with open('vocab.zh','rb') as file:
        # zh_vocab = manager.dict(pickle.load(file))
        zh_vocab = (pickle.load(file))
    with open('vocab.en','rb') as file:
        # en_vocab = manager.dict(pickle.load(file))
        en_vocab = (pickle.load(file))
        Pool = multiprocessing.Pool(processes=8)
        # coord = tf.train.Coordinator()

        # threads = [multiprocessing.Process(target = process,args=(zh_list[thread_id],en_list[thread_id],
        #                                                   zh_vocab,en_vocab,thread_id,writers[thread_id]))\
        #           for thread_id in range(num_threads)]
        # for t in threads:
        #     t.start()
        # for t in threads:
        #     t.join()
        # process(zh_list[1],en_list[1],zh_vocab,en_vocab,writers[1])

        # '''zh, en, zh_vocab, en_vocab, thread_id, writer'''
        targets = [Pool.apply_async(process, args=(zh_list[thread_id], en_list[thread_id],
                                                                 zh_vocab, en_vocab,writers[thread_id]))\
                   for thread_id in range(num_threads)]
        for t in targets:
            t.get()
    print(datetime.now()-start)



if __name__=='__main__':
    main()


