#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__author__ = 'zsy'
__mtime__ = '2017/09/30'"""
import pickle

import sys

import IPython

if __name__ == '__main__':
    if sys.argv[1] == 'zh':
        with open('full_vocab_count_zh', 'rb') as f:
            vocab_count = pickle.load(f)
    else:
        with open('full_vocab_count_en', 'rb') as f:
            vocab_count = pickle.load(f)

    threshold = range(100, 0, -10)
    for i in threshold:
        d = {}
        for key, val in vocab_count.items():
            if val >= i:
                d[key] = val
        print('%d : %d' % (i, len(d)))

    IPython.embed()