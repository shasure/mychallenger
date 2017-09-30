#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__author__ = 'zsy'
__mtime__ = '2017/09/30'"""
import os
import pickle

from config import FLAGS

df = 80

if __name__ == '__main__':
    for filename in ['full_vocab_count_zh', 'full_vocab_count_en']:
        with open(filename, 'rb') as f:
            vocab_count = pickle.load(f)

        vocab = {}
        idx = 0
        vocab['<eos>'] = idx
        idx += 1
        vocab['<pad>'] = idx
        idx += 1
        vocab['<unk>'] = idx
        idx += 1
        for key, val in vocab_count.items():
            if val >= df:
                vocab[key] = idx
                idx += 1
        print(filename + str(len(vocab)))
        print('idx: %d' % idx)

        if 'zh' in filename:
            with open(os.path.join(FLAGS.data_dir, 'vocab.zh'), 'wb') as f:
                pickle.dump(vocab, f)
        else:
            with open(os.path.join(FLAGS.data_dir, 'vocab.en'), 'wb') as f:
                pickle.dump(vocab, f)

