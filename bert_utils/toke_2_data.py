# -*- coding:utf-8 -*-
# @Time      : 2021-08-02 15:03
# @Author    : 年少无为呀！
# @FileName  : toke_2_data.py
# @Software  : PyCharm
import tensorflow as tf
import csv
import os
import tokenization
from parameter import args





class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for prediction."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, delimiter="\t", quotechar=None):
        """Reads a tab separated value file."""
        with tf.io.gfile.GFile(input_file, "r") as f:
            reader = csv.reader(f, delimiter=delimiter, quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
          guid: Unique id for the example.
          text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
          text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
          label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class SentimentCorpProcessor(DataProcessor):
    """
    定义当前数据集的数据处理方式
    """

    def get_train_examples(self, data_dir):
        """
        基于给定的文件夹路径data_dir加载文件夹内部的训练数据并返回
        :param data_dir:
        :return:
        """
        return self._create_examples(
            lines=self._read_tsv(os.path.join(data_dir, "train.csv"), delimiter=','),
            set_type="train")

    def get_dev_examples(self, data_dir):
        """
        基于给定的文件夹路径data_dir加载文件夹内部的验证数据并返回
        :param data_dir:
        :return:
        """
        return self._create_examples(
            lines=self._read_tsv(os.path.join(data_dir, "dev.csv"), delimiter=','),
            set_type="dev")

    def get_test_examples(self, data_dir):
        """
        基于给定的文件夹路径data_dir加载文件夹内部的测试数据并返回
        :param data_dir:
        :return:
        """
        return self._create_examples(
            lines=self._read_tsv(os.path.join(data_dir, "test.csv"), delimiter=','),
            set_type="test")

    def get_labels(self):
        # return ["negative", "positive"]
        # 实际上这里返回的列表就是训练数据中的实际的label标签值
        return args.label_list

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            if len(line) != 2:
                continue
            if len(line[1]) == 0:
                continue

            guid = "%s-%s" % (set_type, i)
            text = tokenization.convert_to_unicode(line[1])  # 对原始文本进行转换操作(转换为UTF-8的数据格式)
            if set_type == "test":
                label = "0"
            else:
                label = tokenization.convert_to_unicode(line[0])
            examples.append(InputExample(guid=guid, text_a=text, label=label))
        return examples
