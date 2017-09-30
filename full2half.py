#!/usr/bin/env python
# encoding: utf-8


"""
@version: 0.0
@author: hailang
@Email: seahailang@gmail.com
@software: PyCharm Community Edition
@file: full2half.py
@time: 2017/9/14 14:00
"""

import sys

import os


def strQ2B(ustring):
    rstring = ''
    for uchar in ustring:
        inside_code = ord(uchar)
        if inside_code == 0x3000:  # 处理空格
            inside_code = 0x0020
        elif inside_code == 8226 or inside_code == 183:  # 着重符号和间隔符改为空格
            inside_code = 0x0020
        else:
            inside_code -= 0xfee0
        if inside_code < 0x0020 or inside_code > 0x7e:
            rstring += uchar
        else:
            rstring += chr(inside_code)
    return rstring


if __name__ == '__main__':
    data_dir = '/home/zsy/datasets/AI_Challenger.data/ai_challenger_translation_train_20170904/translation_train_data_20170904'
    file = open(os.path.join(data_dir, sys.argv[2]), 'w', encoding='utf-8')
    for line in open(os.path.join(data_dir, sys.argv[1]), 'r', encoding='utf-8'):
        s = strQ2B(line.strip())
        # print(s)
        file.write(s + '\n')
