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

import os
import time

import tensorflow as tf
from tensorflow.python.layers.core import Dense

from corpus2tfrecord import Corpus2TFRecord

FLAGS = tf.app.flags.FLAGS


class Config(object):
    def __init__(self):
        pass


config = Config()
config.encoder_fw_units = [1000, 1000]
config.encoder_bw_units = [1000, 1000]
config.decoder_units = [1000, 1000]
num = sum(config.encoder_bw_units)  # decoder的初始化只用后向
config.encoder_output_status_size = num


class Translator(object):
    def __init__(self):
        self.config = tf.ConfigProto(log_device_placement=False)
        self.config.gpu_options.allow_growth = True
        self.config.allow_soft_placement = True
        self.cpu_device = '/cpu:0'
        self.devices = ['/%s:%i' % (FLAGS.compute_device, i) for i in range(FLAGS.num_gpus)]
        self.batch_size = FLAGS.batch_size  # batch per device

    def model(self):

        input_data_type = tf.int32
        # global step
        with tf.device(self.cpu_device):
            global_step = tf.contrib.framework.get_or_create_global_step()

        # input
        with tf.device(self.cpu_device):
            inputs, targets, en_len_sequence, zh_len_sequence = get_batch_and_preprocess(
                self.batch_size, input_data_type)

            start_tokens = tf.tile([0], [self.batch_size])
            end_token = tf.convert_to_tensor(0)
            # input,将来替换为record input
            # inputs = tf.placeholder(dtype=tf.int32, shape=(FLAGS.batch_size, FLAGS.en_max_length))
            # targets = tf.placeholder(dtype=tf.int32, shape=(FLAGS.batch_size, FLAGS.zh_max_length))
            # en_len_sequence = tf.placeholder(dtype=tf.int32, shape=FLAGS.batch_size)
            # zh_len_sequence = tf.placeholder(dtype=tf.int32, shape=FLAGS.batch_size, name='batch_seq_length')

            # start_tokens = tf.placeholder(tf.int32, shape=[], name='start_tokens')
            # end_token = tf.placeholder(tf.int32, shape=[], name='end_token')

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

            tf.summary.histogram('zh_embedding_matrix', zh_embedding_matrix)  # 是否应该使用
            tf.summary.histogram('en_embedding_matrix', en_embedding_matrix)

            embedded = tf.nn.embedding_lookup(en_embedding_matrix, inputs)
            target_embedded = tf.nn.embedding_lookup(zh_embedding_matrix, targets)

        # inference
        with tf.name_scope("encoder"):
            cells_fw = [tf.contrib.rnn.DeviceWrapper(tf.contrib.rnn.GRUCell(num), self.devices[i]) for i, num in enumerate(config.encoder_fw_units)]
            cells_bw = [tf.contrib.rnn.DeviceWrapper(tf.contrib.rnn.GRUCell(num), self.devices[i]) for i, num in enumerate(config.encoder_bw_units)]
            outputs, states_fw, states_bw = \
                tf.contrib.rnn.stack_bidirectional_dynamic_rnn(cells_fw,
                                                               cells_bw,
                                                               embedded,
                                                               dtype=tf.float32,
                                                               sequence_length=en_len_sequence)

            # dense_fw = tf.concat(states_fw, axis=1)
            # dense_bw = tf.concat(states_bw, axis=1)
            # states = tf.concat([dense_bw, dense_fw], axis=1)

            tf.summary.histogram('encoder_state', states_bw)

        with tf.name_scope("decoder"):
            attention_m = tf.contrib.seq2seq.BahdanauAttention(
                FLAGS.attention_size,
                outputs,
                en_len_sequence)
            cell_out = [tf.contrib.rnn.DeviceWrapper(tf.contrib.rnn.GRUCell(num), self.devices[-1]) for num in config.decoder_units]  # decoder两层
            cell_attention = [tf.contrib.seq2seq.AttentionWrapper(
                cell_out[i], attention_m) for i in range(len(config.decoder_units))]  # 所以有两个AttentionWrapper
            cell_attention = [tf.contrib.rnn.DeviceWrapper(cell, self.devices[i]) for i, cell in enumerate(cell_attention)]
            cells = tf.contrib.rnn.MultiRNNCell(cell_attention)

            # initial_state = cells.zero_state(dtype=tf.float32, batch_size=FLAGS.batch_size)
            # initial_state = list(initial_state)
            # initial_state[0] = initial_state[0].clone(cell_state=states)  # 只将encoder的status复制到第一层
            # initial_state = tuple(initial_state)

            if FLAGS.is_inference:
                helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(zh_embedding_matrix, start_tokens, end_token)
            else:
                helper = tf.contrib.seq2seq.TrainingHelper(target_embedded, zh_len_sequence)

            dense = Dense(FLAGS.zh_vocab_size, use_bias=False)
            initial_states = cells.zero_state(batch_size=self.batch_size, dtype=tf.float32)
            for initial_state, state_bw in zip(initial_states, states_bw):
                initial_state = initial_state.clone(cell_state=state_bw)
            decoder = tf.contrib.seq2seq.BasicDecoder(cells, helper, initial_states, dense)  # 在这里初始化cell的state
            logits, final_states, final_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(decoder)

            # loss
            max_zh_len = tf.reduce_max(zh_len_sequence)
            weights = tf.sequence_mask(zh_len_sequence, max_zh_len, dtype=tf.float32)
            inference_losses = tf.contrib.seq2seq.sequence_loss(logits.rnn_output, targets, weights)
            tf.summary.scalar('inference_loss', inference_losses)
            tf.add_to_collection(tf.GraphKeys.LOSSES, inference_losses)
            losses = tf.add_n(tf.get_collection(tf.GraphKeys.LOSSES))
            tf.summary.scalar('losses', losses)
            # eval = sequence_equal(logits.sample_id, targets)
            # tf.summary.scalar('eval', eval)

            learning_rate = tf.train.exponential_decay(FLAGS.learning_rate,
                                                       global_step,
                                                       FLAGS.decay_step,
                                                       FLAGS.decay_rate)
            tf.summary.scalar('learning_rate', learning_rate)

            opt = tf.train.GradientDescentOptimizer(learning_rate)

            grads_and_vars = opt.compute_gradients(losses)
            clipped_grads_and_vars = tf.contrib.training.clip_gradient_norms(grads_and_vars, FLAGS.max_gradient)
            apply_grads_op = opt.apply_gradients(clipped_grads_and_vars, global_step)

            if FLAGS.is_inference:
                return logits.sample_id, [inputs, en_len_sequence, start_tokens, end_token]
            elif FLAGS.is_train:
                return {'loss': losses, 'train_op': apply_grads_op}
            else:
                return [global_step, losses]

    def train(self):
        # dataset will be changed when the input data is prepared
        train_op = self.model()  # build model

        saver = tf.train.Saver(tf.global_variables())
        summary_writer = tf.summary.FileWriter(FLAGS.ckpt_dir)

        global_step_op = tf.train.get_global_step()

        with tf.Session(config=self.config) as sess:
            # try to restore model and global_step or init variables
            load_checkpoint(saver, sess, FLAGS.ckpt_dir, global_step_op)

            start = time.time()
            for step in range(FLAGS.num_batches):

                results, global_step = sess.run([train_op, global_step_op])

                if global_step % 10 == 0:
                    summary_op = tf.summary.merge_all()
                    summary_str = sess.run(summary_op)
                    end = time.time()
                    # 输出global_step and losses
                    print('%d\t%.3f\t' % (global_step, results['loss']), end='\t')
                    print('%.3f sentences/sec' % ((10 * self.batch_size) / (end - start),))
                    start = end
                    summary_writer.add_summary(summary_str, global_step)
                    if not tf.gfile.Exists(FLAGS.ckpt_dir):
                        tf.gfile.MakeDirs(FLAGS.ckpt_dir)
                    saver.save(sess, os.path.join(FLAGS.ckpt_dir, 'model.ckpt'), global_step)

    def eval(self):
        # dataset will be changed when the input data is prepared
        dataset = self.eval_dataset
        max_examples = dataset.max_examples()
        assert max_examples % FLAGS.batch_size == 0
        eval_op = self.model()
        eval = 0
        losses = 0
        summary = tf.summary.FileWriter(FLAGS.ckpt_dir)
        summary_op = tf.summary.scalar('eval_score', eval)
        with tf.Session(config=self.config) as sess:
            while True:
                for i in range(int(max_examples / FLAGS.batch_size)):
                    eval_info, summary_str = sess.run(eval_op, summary_op)
                    summary.add_summary(summary_str, eval_info[0])
                    eval += eval_info[1]
                    losses += eval_info[2]
                eval = eval / max_examples * FLAGS.batch_size
                print(eval, losses)
                time.sleep(100)

    def test(self):
        dataset = self.test_dataset
        max_examples = dataset.max_examples()
        assert max_examples % FLAGS.batch_size == 0
        output = self.model()
        result = []
        with tf.Session(config=self.config) as sess:
            for i in range(int(max_examples / FLAGS.batch_size)):
                output_id = sess.run(output)
                result.extend(output_id)
        with open(FLAGS.test_dir + 'result_raw.csv', 'w', encoding='utf-8') as file:
            for line in result:
                for i in line:
                    file.write(str(i) + ',')
                file.write('\n')
        return result


def sequence_equal(x_batch, y_batch, sequence_length):
    equal_info = [0 for _ in range(x_batch.shape[0])]
    for i in range(x_batch.shape[0]):
        equal_info[i] = tf.reduce_sum(
            tf.cast(tf.equal(x_batch[i, :sequence_length[i]], y_batch[i, :sequence_length[i]]), tf.float32))
    sum = tf.add_n(equal_info)
    return sum / tf.reduce_sum(sequence_length)


def get_batch_and_preprocess(batch_size, input_data_type):
    wp = Corpus2TFRecord(output_dir=FLAGS.data_dir, shards_prefix='en_zh.record')
    zh, en, zh_length, en_length = wp.dataset_batch(batch_size=batch_size, shuffle=False)
    zh = tf.sparse_tensor_to_dense(zh, default_value=1)
    en = tf.sparse_tensor_to_dense(en, default_value=1)
    zh_length = tf.cast(zh_length, input_data_type)
    en_length = tf.cast(en_length, input_data_type)
    return en, zh, en_length, zh_length


def load_checkpoint(saver, sess, ckpt_dir, global_step_tensor):
    ckpt = tf.train.get_checkpoint_state(ckpt_dir)
    if ckpt and ckpt.model_checkpoint_path:
        if os.path.isabs(ckpt.model_checkpoint_path):
            # Restores from checkpoint with absolute path.
            model_checkpoint_path = ckpt.model_checkpoint_path
        else:
            # Restores from checkpoint with relative path.
            model_checkpoint_path = os.path.join(ckpt_dir, ckpt.model_checkpoint_path)
        # Assuming model_checkpoint_path looks something like:
        #   /my-favorite-path/imagenet_train/model.ckpt-0,
        # extract global_step from it.
        global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        if not global_step.isdigit():
            global_step = 0
        else:
            global_step = int(global_step)
        saver.restore(sess, model_checkpoint_path)
        print('Successfully loaded model from %s.' % ckpt.model_checkpoint_path)
        # assign to global step
        sess.run(global_step_tensor.assign(global_step))
    else:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)


def main(_):
    with tf.Graph().as_default():  # Graph
        trans = Translator()
        if FLAGS.is_train:
            trans.train()
        else:
            trans.eval()


if __name__ == '__main__':
    tf.app.run()
