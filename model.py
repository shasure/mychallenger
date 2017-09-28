#!/usr/bin/env python
# encoding: utf-8


"""
@version: 0.0
@author: hailang
@Email: seahailang@gmail.com
@software: PyCharm Community Edition
@file: model.py
@time: 2017/9/15 10:21
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from datetime import datetime
from tensorflow.python.layers.core import Dense
import os
import time
FLAGS = tf.app.flags.FLAGS


class Config(object):
    def __init__(self):
        pass


config = Config()
config.encoder_fw_units=[100,80,60]
config.encoder_bw_units=[100,80,60]
num = sum(config.encoder_fw_units)+sum(config.encoder_bw_units)
config.out_cell_units=[num,num]


class Translator(object):
    def __init__(self):
        self.config = tf.ConfigProto(log_device_placement=True)
        self.config.gpu_options.allow_growth = True
        self.config.allow_soft_placement  = True
        self.train_dataset = Dataset()
        self.eval_dataset = Dataset()
        self.test_dataset = Dataset()



    def model(self):

        # 将来替换为record input
        inputs = tf.placeholder(dtype=tf.int32, shape=(FLAGS.batch_size, FLAGS.en_max_length))
        targets = tf.placeholder(dtype=tf.int32, shape=(FLAGS.batch_size, FLAGS.zh_max_length))
        start_tokens = tf.placeholder(tf.int32, shape=[], name='start_tokens')
        end_token = tf.placeholder(tf.int32, shape=[], name='end_token')
        en_len_sequence = tf.placeholder(dtype=tf.int32, shape=FLAGS.batch_size)
        zh_len_sequence = tf.placeholder(dtype=tf.int32, shape=FLAGS.batch_size, name='batch_seq_length')



        en_embedding_matrix = tf.get_variable(name='embedding_matrix',
                                           shape=(FLAGS.en_vocab_size, FLAGS.en_embedded_size),
                                           dtype=tf.float32,
                                           # regularizer=tf.nn.l2_loss,
                                           initializer=tf.truncated_normal_initializer(mean=0, stddev=0.01)
                                           )
        zh_embedding_matrix = tf.get_variable(name='zh_embedding_matrix',
                                              shape=(FLAGS.zh_vocab_size, FLAGS.zh_embedded_size),
                                              dtype=tf.float32,
                                              # regularizer=tf.nn.l2_loss,
                                              initializer=tf.truncated_normal_initializer(mean=0, stddev=0.01))

        tf.add_to_collection(tf.GraphKeys.LOSSES, tf.nn.l2_loss(en_embedding_matrix))
        tf.add_to_collection(tf.GraphKeys.LOSSES, tf.nn.l2_loss(zh_embedding_matrix))

        tf.summary.histogram('zh_embedding_matrix',zh_embedding_matrix)
        tf.summary.histogram('en_embedding_matrix',en_embedding_matrix)
        with tf.device('/cpu:0'):
            embedded = tf.nn.embedding_lookup(en_embedding_matrix, inputs)
            target_embedded = tf.nn.embedding_lookup(zh_embedding_matrix, targets)

        with tf.name_scope("encoder"):
            cells_fw = [tf.contrib.rnn.GRUCell(num) for num in config.encoder_fw_units]
            cells_bw = [tf.contrib.rnn.GRUCell(num) for num in config.encoder_bw_units]
            outputs, states_fw, states_bw = \
                tf.contrib.rnn.stack_bidirectional_dynamic_rnn(cells_fw,
                                                               cells_bw,
                                                               embedded,
                                                               dtype=tf.float32,
                                                               sequence_length=en_len_sequence)

            dense_fw = tf.concat(states_fw, axis=1)
            dense_bw = tf.concat(states_bw, axis=1)
            states = tf.concat([dense_bw, dense_fw], axis=1)

            tf.summary.histogram('encoder_state',states)


        with tf.name_scope("decoder"):
            attention_m = \
                tf.contrib.seq2seq.BahdanauAttention(
                    FLAGS.attention_size,
                    outputs,
                    en_len_sequence)
            cell_out = [tf.contrib.rnn.GRUCell(num) for num in config.out_cell_units]
            cell_attention = \
                [tf.contrib.seq2seq.AttentionWrapper(
                    cell_out[i], attention_m) for i in range(len(config.out_cell_units))]
            cells = tf.contrib.rnn.MultiRNNCell(cell_attention)

            initial_state = cells.zero_state(dtype=tf.float32, batch_size=FLAGS.batch_size)
            initial_state = list(initial_state)
            initial_state[0] = initial_state[0].clone(cell_state=states)
            initial_state = tuple(initial_state)

            if FLAGS.is_inference:
                helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(zh_embedding_matrix, start_tokens, end_token)
            else:
                helper = tf.contrib.seq2seq.TrainingHelper(target_embedded, zh_len_sequence)

            dense = Dense(FLAGS.zh_vocab_size)
            decoder = tf.contrib.seq2seq.BasicDecoder(cells, helper, initial_state, dense)
            logits, final_states, final_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(decoder)
            weights = tf.constant(1.0, shape=[FLAGS.batch_size, FLAGS.zh_max_length])
            inference_losses = tf.contrib.seq2seq.sequence_loss(logits.rnn_output, targets, weights)
            tf.summary.scalar('inference_loss',inference_losses)
            tf.add_to_collection(tf.GraphKeys.LOSSES, inference_losses)
            losses = tf.add_n(tf.get_collection(tf.GraphKeys.LOSSES))
            tf.summary.scalar('losses',losses)
            eval = sequence_equal(logits.sample_id,targets)
            tf.summary.scalar('eval',eval)

            global_step = tf.train.get_or_create_global_step()

            learning_rate = tf.train.exponential_decay(FLAGS.learning_rate,
                                                       global_step,
                                                       FLAGS.decay_step,
                                                       FLAGS.decay_rate)
            tf.summary.scalar('learning_rate',learning_rate)

            opt = tf.train.GradientDescentOptimizer(learning_rate)

            grads_and_vars = opt.compute_gradients(losses)
            clipped_grads_and_vars = tf.contrib.training.clip_gradient_norms(grads_and_vars,FLAGS.max_gradient)
            apply_grads_op = opt.apply_gradients(clipped_grads_and_vars, global_step)

            summary_op = tf.summary.merge_all()

            if FLAGS.is_inference:
                return logits.sample_id,[inputs,en_len_sequence,start_tokens,end_token]
            elif FLAGS.is_train:
                return [global_step,eval,losses,apply_grads_op,summary_op],[inputs,en_len_sequence,targets,zh_len_sequence]
            else:
                return [global_step,eval,losses]



    def train(self):
        # dataset will be changed when the input data is prepared
        start = datetime.now()
        dataset = self.train_dataset
        train_op,feed_list = self.model()
        init_op = tf.global_variables_initializer()
        saver = tf.train.Saver(tf.global_variables())
        summary = tf.summary.FileWriter(FLAGS.ckpt_dir)
        with tf.Session(config=self.config) as sess:
            if not os.path.exists(FLAGS.ckpt_dir):
                sess.run(init_op)
            else:
                saver.restore(sess,FLAGS.ckpt_dir+'/model.ckpt')
            for step in range(FLAGS.max_step):
                en_batch, zh_batch = dataset.nextbatch(is_train=True)

                train_info = sess.run(train_op,feed_dict={feed_list[0]:en_batch.data,
                                             feed_list[1]:en_batch.len_sequence,
                                             feed_list[2]:zh_batch.data,
                                             feed_list[3]:zh_batch.len_sequence})
                if train_info[0]%100==0:
                    print(datetime.now()-start)
                    print('\t')
                    for info in train_info[:-1]:
                        print(info+'\t')
                    summary.add_summary(train_info[-1],train_info[0])
                    if not os.path.exists(FLAGS.ckpt_dir):
                        os.mkdir(FLAGS.ckpt_dir)
                    saver.save(sess,FLAGS.ckpt_dir+'/model.ckpt')



    def eval(self):
        # dataset will be changed when the input data is prepared
        dataset = self.eval_dataset
        max_examples = dataset.max_examples()
        assert max_examples%FLAGS.batch_size == 0
        eval_op = self.model()
        eval = 0
        losses = 0
        summary = tf.summary.FileWriter(FLAGS.ckpt_dir)
        summary_op = tf.summary.scalar('eval_score',eval)
        with tf.Session(config=self.config) as sess:
            while True:
                for i in range(int(max_examples/FLAGS.batch_size)):
                    eval_info,summary_str = sess.run(eval_op,summary_op)
                    summary.add_summary(summary_str,eval_info[0])
                    eval += eval_info[1]
                    losses += eval_info[2]
                eval = eval/max_examples*FLAGS.batch_size
                print(eval,losses)
                time.sleep(100)







    def test(self):
        dataset = self.test_dataset
        max_examples = dataset.max_examples()
        assert max_examples % FLAGS.batch_size == 0
        output = self.model()
        result =[]
        with tf.Session(config=self.config) as sess:
            for i in range(int(max_examples/FLAGS.batch_size)):
                output_id = sess.run(output)
                result.extend(output_id)
        with open(FLAGS.test_dir+'result_raw.csv','w',encoding='utf-8') as file:
            for line in result:
                for i in line:
                    file.write(str(i)+',')
                file.write('\n')
        return result

    def run(self):
        if FLAGS.is_inference:
            self.test()
        elif FLAGS.is_train:
            self.train()
        else:
            self.eval()





def sequence_equal(x_batch,y_batch,sequence_length):
    equal_info = [0 for _ in range(x_batch.shape[0])]
    for i in range(x_batch.shape[0]):
        equal_info[i] = tf.reduce_sum(tf.cast(tf.equal(x_batch[i,:sequence_length[i]],y_batch[i,:sequence_length[i]]),tf.float32))
    sum = tf.add_n(equal_info)
    return sum/tf.reduce_sum(sequence_length)



class Dataset():
    def __init__(self,data_dir):
        self.data_dir = data_dir
        self.index = 0

    def next_batch(self,is_train):
        pass
        if is_train:
            return Batch(...),Batch(...)
        else:
            return Batch(...)
    def max_examples(self):
        pass


class Batch():
    def __init__(self):
        pass

    def padding(self):
        pass
        self.data = ...
        self.len_sequence = ...

if __name__ == '__main__':
    pass