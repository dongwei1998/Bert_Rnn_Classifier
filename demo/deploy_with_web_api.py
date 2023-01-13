# -*- coding:utf-8 -*-
# @Time      : 2021-08-04 15:03
# @Author    : 年少无为呀！
# @FileName  : deploy_with_web_api.py
# @Software  : PyCharm

from bert_utils.parameter import args
import tensorflow as tf
from bert_utils import tokenization,modeling
from bert_utils.model_build import model_fn_builder
import collections
from flask import Flask, jsonify, request
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

class PaddingInputExample(object):
    """Fake example so the num input examples is a multiple of the batch size.

    When running eval/predict on the TPU, we need to pad the number of examples
    to be a multiple of the batch size, because the TPU requires a fixed batch
    size. The alternative is to drop the last batch, which is bad because it means
    the entire output data won't be generated.

    We use this class instead of `None` because treating `None` as padding
    battches could cause silent errors.
    """


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 input_ids,
                 input_mask,
                 segment_ids,
                 label_id,
                 is_real_example=True):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.is_real_example = is_real_example


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""
    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

def convert_single_example(ex_index, example, label_list, max_seq_length, tokenizer):
    """Converts a single `InputExample` into a single `InputFeatures`."""
    # 1. 如果是填充的样本，直接返回特殊值
    if isinstance(example, PaddingInputExample):
        return InputFeatures(
            input_ids=[0] * max_seq_length,
            input_mask=[0] * max_seq_length,
            segment_ids=[0] * max_seq_length,
            label_id=0,
            is_real_example=False)

    # 2. 构建标签名称和标签索引id之间的映射关系
    label_map = {}
    for (idx, label) in enumerate(label_list):
        label_map[label] = idx

    # 3. 对文本进行单词转换的划分(中文是以字为单词)
    tokens_a = tokenizer.tokenize(example.text_a)
    tokens_b = None
    if example.text_b:
        tokens_b = tokenizer.tokenize(example.text_b)

    # 字符串长度限制，超过做截取
    # max_length >= [CLS] + text_a + [SEP] + text_b + [SEP]
    # max_length >= [CLS] + text_a + [SEP]
    if tokens_b:
        _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
    else:
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[0:(max_seq_length - 2)]

    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    for token in tokens_a:
        tokens.append(token)
        segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)

    if tokens_b:
        for token in tokens_b:
            tokens.append(token)
            segment_ids.append(1)
        tokens.append("[SEP]")
        segment_ids.append(1)

    # token转换为id
    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # 构建input mask: 实际值位置为1，填充值位置为0
    input_mask = [1] * len(input_ids)

    # 数据填充 Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    # 样本标签id
    label_map_ = {str(v): k for k, v in label_map.items()}
    label_ = label_map_[example.label]
    label_id = label_map[label_]

    # 构建对象并返回
    feature = InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        label_id=label_id,
        is_real_example=True)
    return feature

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
def file_based_convert_examples_to_features(
        examples,  # List列表，内部为InputExample对象
        label_list,  # 标签数据，List列表，内部为具体的类别字符串
        max_seq_length,  # 当前模型训练允许的最长序列长度，当序列实际长度超过该值的时候，进行截断，小于该值，进行填充
        tokenizer,  # 主要应用：文本转单词以及单词转id
        output_file  # TFRecord数据保存的磁盘路径
):
    """Convert a set of `InputExample`s to a TFRecord file."""

    def create_int_feature(values):
        f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
        return f
    writer = tf.python_io.TFRecordWriter(output_file)
    for (ex_index, example) in enumerate(examples):
        # 处理单个Example对象
        feature = convert_single_example(ex_index, example, label_list, max_seq_length, tokenizer)

        # 构建TensorFlow的Example对象
        features = collections.OrderedDict()
        features["input_ids"] = create_int_feature(feature.input_ids)
        features["input_mask"] = create_int_feature(feature.input_mask)
        features["segment_ids"] = create_int_feature(feature.segment_ids)
        features["label_ids"] = create_int_feature([feature.label_id])
        features["is_real_example"] = create_int_feature([int(feature.is_real_example)])
        tf_example = tf.train.Example(features=tf.train.Features(feature=features))

        # Example对象序列化输出
        writer.write(tf_example.SerializeToString())
    writer.close()
def file_based_input_fn_builder(
        input_file,  # TFRecord数据所在的磁盘路径
        seq_length,  # 序列长度
        is_training,  # 是否是训练操作
        drop_remainder  #
):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""
    # 1. 构建TFRecord中字符串和id之间的映射关系
    name_to_features = {
        "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
        "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "label_ids": tf.FixedLenFeature([], tf.int64),
        "is_real_example": tf.FixedLenFeature([], tf.int64),
    }

    def _decode_record(record, name_to_features):
        """Decodes a record to a TensorFlow example."""
        example = tf.parse_single_example(record, name_to_features)

        # tf2.0.Example only supports tf2.0.int64, but the TPU only supports tf2.0.int32. So cast all int64 to int32.
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            example[name] = t

        return example

    def input_fn(params, config, mode):
        """The actual input function."""
        batch_size = params["batch_size"]

        # For training, we want a lot of parallel reading and shuffling.
        # For eval, we want no shuffling and parallel reading doesn't matter.
        d = tf.data.TFRecordDataset(input_file)
        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)

        d = d.apply(
            tf.contrib.data.map_and_batch(
                lambda record: _decode_record(record, name_to_features),
                batch_size=batch_size,
                drop_remainder=drop_remainder))

        return d

    return input_fn

class Predictor(object):
    def __init__(self):
        # 加载bert模型的配置信息
        bert_config = modeling.BertConfig.from_json_file(args.bert_config_file)

        # 获取映射关系
        label_list = args.label_list

        # 基于字典文件构建数据token处理的对象(分词、词转id、id转词等功能，中文中以字作为词)
        self.tokenizer = tokenization.FullTokenizer(
            vocab_file=args.vocab_file, do_lower_case=args.do_lower_case)

        is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
        run_config = tf.contrib.tpu.RunConfig(
            cluster=None,
            master=args.master,
            model_dir=args.output_dir,
            save_checkpoints_steps=args.save_checkpoints_steps,
            tpu_config=tf.contrib.tpu.TPUConfig(
                iterations_per_loop=args.iterations_per_loop,
                num_shards=args.num_tpu_cores,
                per_host_input_for_training=is_per_host))

        model_fn = model_fn_builder(
            bert_config=bert_config,
            num_labels=len(label_list),
            init_checkpoint=args.init_checkpoint,
            learning_rate=args.learning_rate,
            num_train_steps=None,
            num_warmup_steps=None,
            use_tpu=args.use_tpu,
            use_one_hot_embeddings=args.use_tpu)

        # 如果没有TPU，会自动转为CPU/GPU的Estimator
        self.estimator = tf.contrib.tpu.TPUEstimator(
            use_tpu=args.use_tpu,
            model_fn=model_fn,
            config=run_config,
            train_batch_size=args.train_batch_size,
            eval_batch_size=args.eval_batch_size,
            predict_batch_size=args.predict_batch_size)


    def predict(self, text,label='0'):
        predict_examples = [InputExample('predict', text, '', label)]
        # 2. 得到实际的样本数目
        num_actual_predict_examples = len(predict_examples)

        # 3. 如果是TPU运行，进行必要的数据样本填充(填充PaddingInputExample对象)
        if args.use_tpu:
            while len(predict_examples) % args.predict_batch_size != 0:
                predict_examples.append(PaddingInputExample())

        # 4. 将预测数据转换为TFRecord格式的数据保存到磁盘(InputExample -> InputFeatures -> TFRecord)
        predict_file = os.path.join(args.prob_cache_file, "predict.tf_record")
        file_based_convert_examples_to_features(
            predict_examples,  # 预测数据，List列表，内部为InputExample对象
            args.label_list,  # 标签数据，List列表，内部为具体的类别字符串
            args.max_seq_length,  # 当前模型训练允许的最长序列长度，当序列实际长度超过该值的时候，进行截断，小于该值，进行填充
            self.tokenizer,  # 主要应用：文本转单词以及单词转id
            predict_file  # TFRecord数据保存的磁盘路径
        )
        tf.logging.info("***** Running prediction*****")
        tf.logging.info("  Num examples = %d (%d actual, %d padding)",
                        len(predict_examples), num_actual_predict_examples,
                        len(predict_examples) - num_actual_predict_examples)
        tf.logging.info("  Batch size = %d", args.predict_batch_size)

        # 6. 数据加载函数的构建
        predict_drop_remainder = True if args.use_tpu else False
        predict_input_fn = file_based_input_fn_builder(
            input_file=predict_file,
            seq_length=args.max_seq_length,
            is_training=False,
            drop_remainder=predict_drop_remainder)

        # 7. 预测，得到预测结果
        result = self.estimator.predict(input_fn=predict_input_fn)
        for (i, prediction) in enumerate(result):
            # print(prediction)
            probabilities = prediction["probabilities"]  # 获取预测结果
            probabilities_list = probabilities.tolist()
            # print(list(probabilities))                                # [0.9877944  0.00433629 0.00271852 0.00515076]
            # print(probabilities.argmax())                       # 0
            # print(args.label_list[probabilities.argmax()])      # 胜诉
            return args.label_list,probabilities_list,max(probabilities_list),args.label_list[probabilities.argmax()]

if __name__ == '__main__':

    # 一、应用构建
    detector = Predictor()

    # text = '1.案情简介：承保范围方程x000d详细内容：2014年9月，申诉人与被申诉人签订了既有买卖合同，双方达成一致，申请人应以713916元的价格购买被申请人开发的清华花园13号商住楼为现房，申请人于2014年9月19日付清全部房款，绿洲公司交付房屋，由于被申请人已承诺，合同中未约定该房产的登记日期，但房屋送达后，被申请人仍未要求出具详细的x000D房产的法律证明u2。潍坊市仲裁委员会经审理认为，该房屋合同是在自愿平等待遇的基础上签订的，其内容不违反法律的强制性规定，合法有效，绿洲地产明知涉案房屋被典当给王艳，便与移动公司签订了购房协议，故意隐瞒抵押事实，导致了两人死亡移动公司未能办理房产登记，无法达成合同目的，基于上述事实和理由，仲裁庭于2020年8月作出判决：“x000D已详细说明u1在房屋清洁领域；确认申请人与被申请人x000d之间现有房屋买卖合同的有效性，详细说明u2，122899空白版合同终止x000D详细说明u3.12289被申请人必须在收到本决定后三十天内将人民币713916元的购买价款退还申请人，并赔偿人民币499741.20元（713916*70%）。x000D详细说明u4公共服务区；驳回申请人潍坊市在移动安全领域的其他申请，本案仲裁费16491元，由被申请人绿洲地产承担。'
    # label_list, prob_value_list, prob_value, label_value = detector.predict(text)
    # print(type(label_list), type(prob_value_list), type(prob_value), type(label_value))
    # print(label_list, prob_value_list, str(prob_value), label_value)

    app = Flask(__name__)
    app.config['JSON_AS_ASCII'] = False


    @app.route('/')
    @app.route('/index')
    def _index():
        return "你好，欢迎使用Flask Web API，进入文本分类任务!!!"


    @app.route('/predict', methods=['POST'])
    def _predict():
        tf.logging.info("法律案件结果预测.....")
        try:
            # 参数获取
            text = request.form.get("text")

            # 参数检查
            if text is None:
                return jsonify({
                    'code': 503,
                    'msg': '请给定参数text！！！'
                })

            # 直接调用预测的API
            label_list, prob_value_list, prob_value,label_value = detector.predict(text)
            return jsonify({
                'code': 200,
                'msg': '成功',
                'data': [
                    {
                        'text': text,
                        'label_list':label_list,            # 标签类别列表
                        'prob_value_list':prob_value_list,  # 预测概率列表
                        'prob_value': prob_value,           # 最大概率
                        'label_value':label_value     # 预测类别
                    }
                ]
            })
        except Exception as e:
            tf.logging.error("异常信息为:{}".format(e))
            return jsonify({
                'code': 504,
                'msg': '预测数据失败!!!'
            })


    # 启动
    app.run(host='0.0.0.0', port=8885)

