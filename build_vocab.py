#!/usr/bin/env python
# -*- coding:utf-8 -*-


"""
@version: 0.0
@author: hailang
@Email: seahailang@gmail.com
@software: PyCharm Community Edition
@file: preprocess.py
@time: 2017/9/7 17:01
"""
import jieba
import pickle
import IPython
import os
import pynlpir
import re
import sys
import time
pynlpir.open()


def chinese_parse(s):
    return pynlpir.segment(s,pos_tagging=False)

def english_parse(s):
    pattern = re.compile('[a-zA-Z]+|\d+\.?\d+[a-zA-Z]*|[\-,\.\'\?\!"]')
    return re.findall(pattern,s)


def build_vocab(filename):
    with open(filename,'r',encoding='utf-8') as file:
        vocab = {}
        maxLen=0
        if filename.split('.')[-2] == 'zh':
            parse = lambda x: chinese_parse(x)
        else:
            parse = lambda x: english_parse(x)
        # parse  = lambda x : pynlpir.segment(x,pos_tagging=False)
        idx = 0
        for line in file.readlines():
            # print(line)
            line = parse(line)
            if len(line)>maxLen:
                maxLen = len(line)
            for word in line:
                if word not in vocab:
                    print(word)
                    vocab[word] = idx
                    idx +=1
            # time.sleep(3)
        vocab['<pad>'] = idx
        idx+=1
        vocab['<unk>'] = idx
        idx+=1
        vocab['<eos>'] = idx
    return vocab,maxLen

class CorpusInfo(object):
    def __init__(self,filename):
        self.vocab,self.max_len = build_vocab(filename)
    def write(self,output):
        with open(output,'wb') as file:
            pickle.dump(self.vocab,file)

def sentence2int(sentence,vocab):
    return [vocab[word] if word in vocab else vocab['<unk>'] for word in sentence]

class Config(object):
    def __init__(self):
        pass

def convert():
    zh_file = open(config.zh_dir, 'r', encoding='utf-8')
    en_file = open(config.en_dir, 'r', encoding='utf-8')
    out_file = open('./zh_en', 'w')
    vocab_zh = pickle.load(open('vocab.zh','rb'))
    vocab_en = pickle.load(open('vocab.en','rb'))
    for zh in zh_file.readlines():


        print(zh.strip())
        zh = sentence2int(chinese_parse(zh.strip()), vocab_zh)
        print(zh)
        en = en_file.readline()
        print(en.strip())
        en = sentence2int(english_parse(en.strip()), vocab_en)
        print(en)
        for i in en:
            out_file.write(str(i) + ' ')
        out_file.write('\n')
        for i in zh:
            out_file.write(str(i) + ' ')
        out_file.write('\n')
        print('\n')
    out_file.close()


if __name__ == '__main__':
    config = Config()
    config.zh_dir=os.path.abspath('../ai_challenger_translation_train_20170904/translation_train_data_20170904/train.zh.m')
    config.en_dir=os.path.abspath('../ai_challenger_translation_train_20170904/translation_train_data_20170904/train.en.m')
    # if sys.argv[1] == 'zh':
    #     ZH= CorpusInfo(config.zh_dir)
    #     config.zh_max_len=ZH.max_len
    #     ZH.write('vocab.zh')
    # else:
    #     EN = CorpusInfo(config.en_dir)
    #     config.en_max_len = EN.max_len
    #     EN.write('vocab.en')

    convert()

    IPython.embed()


    '''Five on one. Five on one. Yeah, not the greatest odds.'''

