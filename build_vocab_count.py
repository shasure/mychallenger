#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__author__ = 'zsy'
__mtime__ = '2017/09/30'"""

import multiprocessing as mul
import pickle
import sys
from collections import Counter

from build_vocab import chinese_parse, english_parse
from config import FLAGS

num_processes = 16


def build_partial_vocab_dict(partial_lines, filename):
    vocab_count = {}
    maxLen = 0
    if filename.split('.')[-2] == 'zh':
        parse = lambda x: chinese_parse(x)
    else:
        parse = lambda x: english_parse(x)
    # parse  = lambda x : pynlpir.segment(x,pos_tagging=False)
    for i, line in enumerate(partial_lines):
        try:
            line = parse(line)
        except:
            print('unicode error line ' + str(i + 1))
        if len(line) > maxLen:
            maxLen = len(line)
        for word in line:
            if word not in vocab_count:
                # print(word)
                vocab_count[word] = 1
            else:
                vocab_count[word] += 1
    return vocab_count, maxLen


if __name__ == '__main__':
    pool = mul.Pool(num_processes)
    if sys.argv[1] == 'zh':
        relative_name = '/train.zh.m'
        filename = FLAGS.data_dir + '/train.zh.m'
    else:
        relative_name = '/train.en.m'
        filename = FLAGS.data_dir + '/train.en.m'
    filename_mul = [filename for _ in range(num_processes)]
    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    lines_splits = [[] for _ in range(num_processes)]
    for i in range(len(lines)):
        lines_splits[i % num_processes].append(lines[i])

    del lines

    results = pool.starmap(build_partial_vocab_dict, zip(lines_splits, filename_mul))
    dict_list = []
    max_len_list = []
    for res in results:
        dict_list.append(res[0])
        max_len_list.append(res[1])

    print('max_len: %d' % max(max_len_list))
    counter = Counter()
    map_res = map(Counter, dict_list)
    for res in map_res:
        counter = counter + res
    if 'zh' in relative_name:
        with open('full_vocab_count_zh', 'wb') as file:
            pickle.dump(counter, file)
    else:
        with open('full_vocab_count_en', 'wb') as file:
            pickle.dump(counter, file)
