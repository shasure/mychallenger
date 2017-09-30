# coding:utf-8
import os

import pandas as pd
import tfrecordwrapper as wp
import tensorflow as tf
import random


class CsvTFRecordWrapper(wp.TFRecordWrapper):
    def __init__(self, output_dir=None, shards_prefix=None, shuffle=False):
        super(CsvTFRecordWrapper, self).__init__(output_dir, shards_prefix, shuffle)
        self.keys = None
        self.types = None

    def process_one_file(self, filename):
        data = pd.read_csv(filename)
        # age的null用均值替换
        age = data['Age']
        mean_age = age.mean()
        data.loc[age.isnull(), 'Age'] = mean_age
        # cabin
        cabin = data['Cabin']
        data.loc[cabin.isnull(), 'Cabin'] = ''
        # Embarked
        embarked = data['Embarked']
        data.loc[embarked.isnull(), 'Embarked'] = ''

        rows = data.shape[0]
        for row in range(rows):
            feature_list = []
            for i, key in enumerate(self.keys):
                # test store fixlen list, read with FixedLenFeature, should use shape
                if key == 'Embarked':
                    feature_list.append([str(i) for i in range(random.randint(1, 5))])
                    feature_list.append(['a', 'b'])
                    continue
                if key == 'Age':
                    feature_list.append([1, 2])
                    continue
                feature_list.append(data[key][row])

            yield feature_list

    def get_types(self):
        self.types = [wp.IntList()] * 3 + [wp.StrList()] * 2 + [wp.FloatList(shape=[2])] + [wp.IntList()] * 2 + [
            wp.StrList()] + [wp.FloatList()] + [wp.StrList()] + [wp.StrList(isfix=False)]
        return self.types

    def get_keys(self):
        self.keys = ['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare',
                     'Cabin', 'Embarked']
        return self.keys


if __name__ == '__main__':
    data_dir = os.path.join('/datasets', 'csv')
    filename_p = os.path.join(data_dir, 'train*.csv')

    cw = CsvTFRecordWrapper(output_dir='/datasets/csv', shards_prefix='csv')

    # write TFRecord
    # cw.process_dataset(filename_p, num_threads=2, num_shards=2)

    # read TFRecord
    cw_r = CsvTFRecordWrapper(output_dir='/datasets/csv', shards_prefix='csv')

    '''RecordInput'''
    # def pre_process_func(*args):  # 预处理直接返回原来的值
    #     return args
    #
    #
    # batch_examples = cw_r.record_input_batch(pre_process_func, buffer_size=1)
    #
    # with tf.Session() as sess:
    #     print(sess.run([batch_examples[11][31]]))
    #
    #     name, embarked, age, pid = sess.run(
    #         [batch_examples[3][0], batch_examples[11][0], batch_examples[5][0],
    #          batch_examples[0][0]])
    #     print(name, type(name))
    #     # print(embarked, type(embarked), embarked.values, type(embarked.values))
    #     print(age, type(age))
    #     print(pid, type(pid))
    #     # print(embarked.dense_shape)


    '''shuffle batch'''
    # def pre_process_func(name, embarked):
    #     return name, embarked
    #
    #
    # batch_examples = cw_r.shuffle_batch(num_threads=12, pre_process_func=pre_process_func, cols=['Name', 'Embarked'])
    # with tf.Session() as sess:
    #     coord = tf.train.Coordinator()
    #     threads = tf.train.start_queue_runners(sess, coord=coord)
    #     print(sess.run(batch_examples))
    #     coord.request_stop()
    #     coord.join(threads)

    '''batch'''
    # def pre_process_func(id, name):
    #     return name + '--'
    #
    #
    # batch_examples = cw_r.batch(pre_process_func, num_threads=12, cols=['PassengerId', 'Name'])
    # with tf.Session() as sess:
    #     coord = tf.train.Coordinator()
    #     threads = tf.train.start_queue_runners(sess, coord=coord)
    #     for i in range(5):
    #         print(sess.run([batch_examples[i] for i in range(32)]))
    #     coord.request_stop()
    #     coord.join(threads)

    '''shuffle batch join'''
    # def pre_process_func(name):
    #     return name
    #
    #
    # batch_examples = cw_r.shuffle_batch_join(pre_process_func=pre_process_func, num_examples_per_epoch=891,
    #                                          num_threads=2, cols=['Name'])
    # with tf.Session() as sess:
    #     coord = tf.train.Coordinator()
    #     threads = tf.train.start_queue_runners(sess, coord=coord)
    #     for i in range(5):
    #         print(sess.run([batch_examples[i] for i in range(32)]))
    #     coord.request_stop()
    #     coord.join()

    '''dataset batch'''
    def pre_process_func(passengerid, embarked):
        return passengerid, embarked

    batch_examples = cw_r.dataset_batch(pre_process_func, num_epochs=1, batch_size=2, shuffle=False, buffer_size=1000,
                                        cols=['PassengerId', 'Embarked'])
    with tf.Session() as sess:
        while True:
            try:
                print(sess.run(batch_examples))
            except tf.errors.OutOfRangeError:
                break

    '''dataset_read_batch_features'''
    # features = cw_r.dataset_read_batch_features(num_epochs=1, batch_size=3, shuffle=True,
    #                                             cols=['PassengerId', 'Embarked'])
    # with tf.Session() as sess:
    #     while True:
    #         try:
    #             print(sess.run(features))
    #         except tf.errors.OutOfRangeError:
    #             break
