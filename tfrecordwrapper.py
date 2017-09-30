# coding: utf-8
import collections
import os
import sys
import threading
from datetime import datetime
from random import random

import numpy as np
import tensorflow as tf
from tensorflow.contrib.data import TFRecordDataset
from tensorflow.contrib.data.python.ops import dataset_ops
from tensorflow.python.ops import data_flow_ops, parsing_ops
from tensorflow.python.framework import sparse_tensor

"""
关于用户输入数据的假设：
1.用户输入数据存储在1个或多个文件中
2.从每个文件中可以提取出一个example，或者一个example的大部分
3.每个文件可以提取多个example
4.每个文件对应一个TFRecord shard，一个TFRecord shard可以对应多个文件
"""


class TFRecordWrapper(object):
    """对TFRecord的读和写进行封装

    注意读取TFRecord时，如果有变长字段(VarLenFeature)，会返回SparseTensor
    """

    def __init__(self, output_dir, shards_prefix, shuffle=False):
        """
        :param output_dir: TFRecord输出路径
        :param shards_prefix: TFRecord文件的命名前缀
        :param shuffle: 是否在写入TFRecord时打乱文件名
        """
        self.output_dir = output_dir
        self.shards_prefix = shards_prefix
        self.shuffle = shuffle
        filename_p = '%s-*-of-*' % (self.shards_prefix,)
        self.pattern = os.path.join(self.output_dir, filename_p)
        if shards_prefix and output_dir:
            self.pattern = self._tfrecord_filename_pattern()

    def _tfrecord_filename_pattern(self):
        filename_p = '%s-*-of-*' % (self.shards_prefix,)
        pattern = os.path.join(self.output_dir, filename_p)
        return pattern

    def _gen_filenames(self):
        """
        返回要处理的文件的文件名

        :return: filenames list
        """
        filenames = tf.gfile.Glob(self.glob)

        # Shuffle the ordering of all image files in order to guarantee
        # random ordering of the images with respect to label in the
        # saved TFRecord files. Make the randomization repeatable.
        if self.shuffle:
            shuffled_index = list(range(len(filenames)))
            random.seed(12345)
            random.shuffle(shuffled_index)

            filenames = [filenames[i] for i in shuffled_index]

        print('Found %d  files ' % len(filenames))
        return filenames

    def process_dataset(self, glob, num_threads, num_shards):
        """
        多线程处理文件，转换成TFRecord。这里要求num_shards % num_threads == 0

        :param glob: 要处理的文件的匹配字符串
        :param num_threads: 写TFRecord文件的线程数
        :param num_shards: TFRecord文件分片数
        :return: None
        """
        self.glob = glob
        self.num_threads = num_threads
        self.num_shards = num_shards
        self.num_shards_len = len(str(num_threads))
        # Break all images into batches with a [ranges[i][0], ranges[i][1]].
        # 将file分成num_threads份

        filenames = self._gen_filenames()
        spacing = np.linspace(0, len(filenames), self.num_threads + 1).astype(np.int)
        ranges = []
        for i in range(len(spacing) - 1):
            ranges.append([spacing[i], spacing[i + 1]])

        # Launch a thread for each batch.
        print('Launching %d threads for spacings: %s' % (self.num_threads, ranges))
        sys.stdout.flush()

        # Create a mechanism for monitoring when all threads are finished.
        coord = tf.train.Coordinator()

        threads = []
        for thread_index in range(len(ranges)):
            args = (self, thread_index, ranges, self.shards_prefix, filenames, self.num_shards, self.output_dir)
            t = threading.Thread(target=TFRecordWrapper._process_files_batch, args=args)
            t.start()
            threads.append(t)

        # Wait for all the threads to terminate.
        coord.join(threads)
        print('%s: Finished writing all %d files in data set.' % (datetime.now(), len(filenames)))
        sys.stdout.flush()

    def get_types(self):
        """
        按顺序返回各个字段的类型
        :return: list of types
        """
        return []

    def get_keys(self):
        """
        按顺序返回各个字段的存储在TFRecord中的名称
        :return: list of str
        """
        return []

    def process_one_file(self, filename):
        """
        如何处理一个文件，每次返回一个example的所有feature（按照get_keys和get_types的顺序）
        :param filename: 要处理的文件名
        :return: list of feature
        """
        return []

    def _convert_to_example(self, keys, feature_list):
        """将features转换成example"""
        types = self.get_types()
        feature_dict = {keys[i]: types[i].write_feature(feature_list[i]) for i in range(len(keys))}

        example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
        return example

    def _process_files_batch(self, thread_index, ranges, shards_prefix, filenames, num_shards, output_dir):
        """Processes and saves data as TFRecord in 1 thread.

        """
        # Each thread produces N shards where N = int(num_shards / num_threads).
        # For instance, if num_shards = 128, and the num_threads = 2, then the first
        # thread would produce shards [0, 64).
        num_threads = len(ranges)
        assert not num_shards % num_threads
        num_shards_per_batch = int(num_shards / num_threads)

        shard_ranges = np.linspace(ranges[thread_index][0],
                                   ranges[thread_index][1],
                                   num_shards_per_batch + 1).astype(int)
        num_files_in_thread = ranges[thread_index][1] - ranges[thread_index][0]

        counter = 0
        keys = self.get_keys()  # 获取字段名
        for s in range(num_shards_per_batch):
            # Generate a sharded version of the file name, e.g. 'train-00002-of-00010'
            shard = thread_index * num_shards_per_batch + s
            output_filename = '%s-%s-of-%s' % (
                shards_prefix, str(shard).zfill(self.num_shards_len), str(num_shards).zfill(self.num_shards_len))
            output_file = os.path.join(output_dir, output_filename)
            writer = tf.python_io.TFRecordWriter(output_file)

            shard_counter = 0
            files_in_shard = np.arange(shard_ranges[s], shard_ranges[s + 1], dtype=int)  # 类似range

            for i in files_in_shard:
                filename = filenames[i]  # filename是绝对路径

                for feature_list in self.process_one_file(filename):
                    # features转example
                    example = self._convert_to_example(keys, feature_list)

                    writer.write(example.SerializeToString())
                    shard_counter += 1
                    counter += 1

                    if not counter % 1000:
                        print('%s [thread %d]: Processed %d of %d examples in thread batch.' %
                              (datetime.now(), thread_index, counter, num_files_in_thread))
                        sys.stdout.flush()

            writer.close()
            print('%s [thread %d]: Wrote %d examples to %s' %
                  (datetime.now(), thread_index, shard_counter, output_file))
            sys.stdout.flush()
        print('%s [thread %d]: Wrote %d examples to %d shards.' %
              (datetime.now(), thread_index, counter, num_files_in_thread))
        sys.stdout.flush()

    def record_input_batch(self, pre_process_func=None, seed=301, parallelism=64, buffer_size=10000, batch_size=32,
                           cols=None):
        """使用RecordInput从TFRecord随机读取一个batch的数据

        :param pre_process_func: 预处理函数。为None时不进行预处理。预处理函数接收的参数数目需要和len(cols)相同，返回参数数目不限制
        :param seed: 随机种子
        :param parallelism: 并发数
        :param buffer_size: The maximum number of records the buffer will contain.
        :param batch_size: 一次返回多少records
        :param cols: 要返回TFRecord中的哪些feature。get_keys()函数返回值得子集。
        :return: 一个batch的数据
        """
        if cols is None:
            cols = self.get_keys()
        record_input = data_flow_ops.RecordInput(  # return : A tensor of shape [batch_size].
            file_pattern=self.pattern,
            seed=seed,
            parallelism=parallelism,
            buffer_size=buffer_size,
            batch_size=batch_size,
            name='record_input')
        records = record_input.get_yield_op()
        records = tf.split(records, batch_size, 0)
        records = [tf.reshape(record, []) for record in records]
        batch_examples = [[] for _ in range(len(cols))]
        keys = self.get_keys()
        types = self.get_types()
        for i in range(batch_size):
            value = records[i]
            cols_types = [types[keys.index(col)] for col in cols]  # 返回cols对应的类型
            features = _parse_single_example_proto_cols_closure(cols_types, cols)(value)

            if pre_process_func is not None:  # 是否进行预处理
                features = pre_process_func(*features)
                # 调用者可能使用pre_process_func返回单个Tensor，需要转换成Sequence
                if not isinstance(features, collections.Sequence):
                    features = (features,)
            for j, feature in enumerate(features):
                batch_examples[j].append(feature)
        return batch_examples

    def _filenames_queue(self):
        """创建filenames queue"""
        filenames = tf.gfile.Glob(self.pattern)
        filename_queue = tf.train.string_input_producer(filenames)
        return filename_queue

    def _tfrecord_read(self, cols):
        """返回一个example"""
        filename_queue = self._filenames_queue()
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
        cols_types = [self.get_types()[self.get_keys().index(col)] for col in cols]  # 返回cols对应的类型
        features = _parse_single_example_proto_cols_closure(cols_types, cols)(serialized_example)

        return features

    def shuffle_batch(self, pre_process_func=None, batch_size=32, num_threads=1, cols=None,
                      num_examples_per_epoch=10000,
                      min_fraction_of_examples_in_queue=0.4):
        """需要使用tf.train.start_queue_runners

        :param pre_process_func: 预处理函数
        :param batch_size:
        :param num_threads: 读取线程数
        :param cols: 要返回TFRecord中的哪些feature。get_keys()函数返回值得子集。
        :param num_examples_per_epoch: 每个epoch有多少example
        :param min_fraction_of_examples_in_queue: 队列中元素占所有example元素的比重，用来保证shuffle的效果
        :return: 一个batch数据
        """
        if cols is None:
            cols = self.get_keys()
        features = self._tfrecord_read(cols)
        if pre_process_func is not None:  # 是否进行预处理
            features = pre_process_func(*features)
            # 调用者可能使用pre_process_func返回单个Tensor，需要转换成Sequence
            if not isinstance(features, collections.Sequence):
                features = (features,)
        min_queue_examples = int(num_examples_per_epoch * min_fraction_of_examples_in_queue)
        capacity = min_queue_examples + (num_threads + 3) * batch_size
        batch = tf.train.shuffle_batch(features, batch_size=batch_size, num_threads=num_threads,
                                       capacity=capacity, min_after_dequeue=min_queue_examples)
        # 不对SparseTensor进行转换，SparseTensor -> Tensor由调用者进行
        # output_batch = []
        # for batch_elem_tensor, col_type in zip(batch, cols_types):
        #     if isinstance(batch_elem_tensor, sparse_tensor.SparseTensor):
        #         output_batch.append(tf.sparse_tensor_to_dense(batch_elem_tensor, col_type.get_type_default_value()))
        #     else:
        #         output_batch.append(batch_elem_tensor)
        # return output_batch
        return batch

    def batch(self, pre_process_func=None, batch_size=32, num_threads=1, cols=None, capacity=10000):
        """需要使用tf.train.start_queue_runners

        发现有局部的乱序存在"""
        if cols is None:
            cols = self.get_keys()
        features = self._tfrecord_read(cols)
        if pre_process_func is not None:  # 是否进行预处理
            features = pre_process_func(*features)
            # 调用者可能使用pre_process_func返回单个Tensor，需要转换成Sequence
            if not isinstance(features, collections.Sequence):
                features = (features,)
        return tf.train.batch(features, batch_size=batch_size, num_threads=num_threads,
                              capacity=capacity)

    def shuffle_batch_join(self, pre_process_func=None, batch_size=32, num_threads=1, cols=None,
                           num_examples_per_epoch=10000,
                           min_fraction_of_examples_in_queue=0.4):
        """需要使用tf.train.start_queue_runners"""
        if cols is None:
            cols = self.get_keys()
        features = self._tfrecord_read(cols)
        if pre_process_func is not None:  # 是否进行预处理
            features = pre_process_func(*features)
            # 调用者可能使用pre_process_func返回单个Tensor，需要转换成Sequence
            if not isinstance(features, collections.Sequence):
                features = (features,)
        features_list = [features for _ in range(num_threads)]
        min_queue_examples = int(num_examples_per_epoch * min_fraction_of_examples_in_queue)
        capacity = min_queue_examples + (num_threads + 3) * batch_size
        return tf.train.shuffle_batch_join(features_list, batch_size=batch_size,
                                           capacity=capacity, min_after_dequeue=min_queue_examples)

    def dataset_batch(self, pre_process_func=None, batch_size=32, shuffle=True, buffer_size=10000, num_threads_map=1,
                      num_epochs=None,
                      cols=None):
        """使用dataset返回一个batch数据（tf.__version__ >=1.3）。写dataset.map函数时需要注意要返回tuple而不是list，并且由于不能
        返回SparseTensor，所以将SpareTensor拆分成3个Tensor。这部分通过参考read_batch_features函数来实现。

        :param pre_process_func: 预处理函数
        :param batch_size:
        :param shuffle: 读取时是否打乱顺序
        :param buffer_size: map和shuffle的buffer大小
        :param num_threads_map: map转换使用的线程数
        :param num_epochs: 获取多少epoch数据，None表示无限
        :param cols:  要返回TFRecord中的哪些feature。get_keys()函数返回值得子集。
        :return: 一个batch数据
        """
        filenames = tf.gfile.Glob(self.pattern)
        dataset = TFRecordDataset(filenames)
        if cols is None:
            cols = self.get_keys()
        # ---两个map耗时更长---
        # map函数不能返回list，必须返回tuple
        # dataset = dataset.map(self._parse_example_proto, num_threads=num_threads_map, output_buffer_size=buffer_size)
        # 原因：源码nest.flatten(ret)函数中的is_sequence(nest)对list类型的nest返回false
        # dataset = dataset.map(lambda feature_dict: tuple(feature_dict[col] for col in cols),
        #                       num_threads=num_threads_map,
        #                       output_buffer_size=buffer_size)

        # 采用closure，两个map合成一个
        keys = self.get_keys()
        types = self.get_types()
        cols_types = [types[keys.index(col)] for col in cols]  # 返回cols对应的类型
        if shuffle:
            dataset = dataset.shuffle(buffer_size=buffer_size)
        dataset = dataset.repeat(num_epochs)
        dataset = dataset.batch(batch_size)
        dataset = dataset.map(_parse_example_proto_cols_closure(cols_types, cols),
                              num_threads=num_threads_map, output_buffer_size=buffer_size)
        sparse_bool_list = []
        if pre_process_func is not None:  # 是否进行预处理
            dataset = dataset.map(_preprocess_sparsetensor(pre_process_func, cols_types, sparse_bool_list))
        iterator = dataset.make_one_shot_iterator()
        output = iterator.get_next()
        # 3 tensor -> sparsetensor
        output_sparse = []
        index = 0
        if pre_process_func and sparse_bool_list:  # 用户进行preprocess
            for sparse_bool in sparse_bool_list:
                if sparse_bool:
                    output_sparse.append(sparse_tensor.SparseTensor(indices=output[index], values=output[index + 1],
                                                                    dense_shape=output[index + 2]))
                    index += 3
                else:
                    output_sparse.append(output[index])
                    index += 1
        else:  # 用户没有preprocess
            for col_type in cols_types:
                if col_type.get_isfix():
                    output_sparse.append(output[index])
                    index += 1
                else:
                    output_sparse.append(sparse_tensor.SparseTensor(
                        indices=output[index],
                        values=output[index + 1],
                        dense_shape=output[index + 2]))
                    index += 3
        return output_sparse

    def dataset_read_batch_features(self, batch_size=32, cols=None, shuffle=True, num_epochs=None, capacity=10000):
        """直接使用dataset_ops.read_batch_features没法自定义pre_precess_func"""
        filenames = tf.gfile.Glob(self.pattern)
        keys = self.get_keys()
        types = self.get_types()
        cols_types = [types[keys.index(col)] for col in cols]  # 返回cols对应的类型
        feature_map = {cols[i]: cols_types[i].read_feature() for i in range(len(cols))}
        features = dataset_ops.read_batch_features(filenames, batch_size=batch_size, features=feature_map,
                                                   reader=TFRecordDataset, randomize_input=shuffle,
                                                   num_epochs=num_epochs,
                                                   capacity=capacity)
        # for t, k in zip(cols_types, cols):  # VarLenFeature返回SparseTensor，在这里转换成dense Tensor
        #     if not t.get_isfix():
        #         features[k] = tf.sparse_tensor_to_dense(features[k], default_value=t.get_type_default_value())
        return tuple(features[col] for col in cols)


def _parse_example_proto_cols_closure(cols_types, cols):
    """解析examples，并对SparseTensor进行处理"""
    features = {cols[i]: cols_types[i].read_feature() for i in range(len(cols))}

    def _parse_example_proto_cols(serialized):
        parsed = parsing_ops.parse_example(serialized, features)
        result = []
        for key in cols:
            val = parsed[key]
            if isinstance(val, sparse_tensor.SparseTensor):
                result.extend([val.indices, val.values, val.dense_shape])  # SparseTensor -> 3 Tensor
            else:
                result.append(val)
        return tuple(result)

    return _parse_example_proto_cols


def _preprocess_sparsetensor(pre_process_func, cols_types, sparse_bool_list):
    def real_preprocess_func(*map_input):
        map_index = 0
        preprocess_input = []
        for col_type in cols_types:
            if col_type.get_isfix():
                preprocess_input.append(map_input[map_index])
                map_index += 1
            else:
                preprocess_input.append(sparse_tensor.SparseTensor(
                    indices=map_input[map_index],
                    values=map_input[map_index + 1],
                    dense_shape=map_input[map_index + 2]))
                map_index += 3
        preprocess_output = pre_process_func(*preprocess_input)
        # 调用者可能使用pre_process_func返回单个Tensor，需要转换成Sequence
        if not isinstance(preprocess_output, collections.Sequence):
            preprocess_output = (preprocess_output,)
        map_output = []
        # sparsetensor -> tensor,并记录sparsetensor的位置
        for feature in preprocess_output:
            if isinstance(feature, sparse_tensor.SparseTensor):
                sparse_bool_list.append(True)
                map_output.extend([feature.indices, feature.values, feature.dense_shape])  # SparseTensor -> 3 Tensor
            else:
                sparse_bool_list.append(False)
                map_output.append(feature)

        return tuple(map_output)

    return real_preprocess_func


def _parse_single_example_proto_cols_closure(cols_types, cols):
    def _parse_single_example_proto_cols(example_serialized):
        """解析single example proto"""
        feature_map = {cols[i]: cols_types[i].read_feature() for i in range(len(cols))}
        features = tf.parse_single_example(example_serialized, feature_map)
        # 不对SparseTensor进行转换
        # for t, k in zip(cols_types, cols):  # VarLenFeature返回SparseTensor，在这里转换成dense Tensor
        #     if not t.get_isfix():
        #         features[k] = tf.sparse_tensor_to_dense(features[k], default_value=t.get_type_default_value())
        return tuple(features[col] for col in cols)

    return _parse_single_example_proto_cols


'''基本类型'''


class FeatureListBase(object):
    def __init__(self, shape=None, isfix=True, dtype=None):
        self.isfix = isfix
        self.dtype = dtype
        self.shape = shape
        if self.shape is None:
            self.shape = []

    def write_feature(self, value):
        assert False

    def read_feature(self):
        if self.isfix:
            return tf.FixedLenFeature(shape=self.shape, dtype=self.dtype)
        return tf.VarLenFeature(dtype=self.dtype)

    def get_isfix(self):
        return self.isfix

    def get_type_default_value(self):
        assert False


class IntList(FeatureListBase):
    def __init__(self, shape=None, isfix=True, dtype=tf.int64):
        super(IntList, self).__init__(shape, isfix, dtype)

    def write_feature(self, value):
        if not isinstance(value, list):
            value = [value]
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

    def get_type_default_value(self):
        return 0


class FloatList(FeatureListBase):
    def __init__(self, shape=None, isfix=True, dtype=tf.float32):
        super(FloatList, self).__init__(shape, isfix, dtype)

    def write_feature(self, value):
        if not isinstance(value, list):
            value = [value]
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))

    def get_type_default_value(self):
        return 0.0


class BytesList(FeatureListBase):
    def __init__(self, shape=None, isfix=True, dtype=tf.string):
        super(BytesList, self).__init__(shape, isfix, dtype)

    def write_feature(self, value):
        if not isinstance(value, list):
            value = [value]
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

    def get_type_default_value(self):
        return b''


class StrList(FeatureListBase):
    def __init__(self, shape=None, isfix=True, dtype=tf.string):
        super(StrList, self).__init__(shape, isfix, dtype)

    def write_feature(self, value):
        if not isinstance(value, list):
            value = [value.encode()]
        else:
            value = [v.encode() for v in value]
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

    def get_type_default_value(self):
        return b''
