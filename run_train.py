# -*- coding:utf-8 -*-
# @Time      : 2021-08-01 12:54
# @Author    : 年少无为呀！
# @FileName  : run_train.py
# @Software  : PyCharm


from parameter import args
import tensorflow as tf
import tokenization,modeling,toke_2_data
from model_build import model_fn_builder
from data_help import excel2csv_data_create
import os
import collections
import logging.config
logging.config.fileConfig(args.logini)
logger=logging.getLogger('applog')




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
        # Modifies `tokens_a` and `tokens_b` in place so that the total
        # length is less than the specified length.
        # Account for [CLS], [SEP], [SEP] with "- 3"
        _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
    else:
        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[0:(max_seq_length - 2)]

    # 添加分隔符以及获取segment ids
    # The convention in BERT is:
    # (a) For sequence pairs:
    #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
    #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
    # (b) For single sequences:
    #  tokens:   [CLS] the dog is hairy . [SEP]
    #  type_ids: 0     0   0   0  0     0 0
    #
    # Where "type_ids" are used to indicate whether this is the first
    # sequence or the second sequence. The embedding vectors for `type=0` and
    # `type=1` were learned during pre-training and are added to the wordpiece
    # embedding vector (and position vector). This is not *strictly* necessary
    # since the [SEP] token unambiguously separates the sequences, but it makes
    # it easier for the model to learn the concept of sequences.
    #
    # For classification tasks, the first vector (corresponding to [CLS]) is
    # used as the "sentence vector". Note that this only makes sense because
    # the entire model is fine-tuned.
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
    # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
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
    # if args.do_predict:
    #     label_map_ = {str(v): k for k, v in label_map.items()}
    #     label_ = label_map_[example.label]
    #     label_id = label_map[label_]
    # else:
    label_id = label_map[example.label]

    # 控制台输出可视化以下
    if ex_index < 5:
        tf.logging.info("*** Example ***")
        tf.logging.info("guid: %s" % (example.guid))
        tf.logging.info("tokens: %s" % " ".join(
            [tokenization.printable_text(x) for x in tokens]))
        tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        tf.logging.info("label: %s (id = %s)" % (example.label, label_id))

    # 构建对象并返回
    feature = InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        label_id=label_id,
        is_real_example=True)
    return feature


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
        if ex_index % 10000 == 0:
            tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

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




def main(_):

    # 查看tensorflow版本
    logger.info(f'tensorflow version {tf.__version__}')


    # # 数据生成
    # logger.info('数据生成')
    # excel2csv_data_create(args)

    # 对bert模型的初始化的磁盘路径进行校验
    tokenization.validate_case_matches_checkpoint(args.do_lower_case,
                                                  args.init_checkpoint)

    # 参数校验，至少是train、eval或者predict的其中一个操作
    if not args.do_train and not args.do_eval and not args.do_predict:
        raise ValueError(
            "至少有一个 `do_train`, `do_eval` or `do_predict' 为真.")

    # 加载bert模型的配置信息
    bert_config = modeling.BertConfig.from_json_file(args.bert_config_file)

    # 参数校验，当前模型训练的序列最长长度不能超过原始的bert模型预训练的模型长度信息
    if args.max_seq_length > bert_config.max_position_embeddings:
        raise ValueError(
            f"不能使用序列长度为 {args.max_seq_length},应为bert模型最大支持序列长度为{bert_config.max_position_embeddings}")

    # 模型最终数据输出文件夹的构建
    if not os.path.isdir(args.output_dir):
        logger.info(f'创建模型以及可视化文件保存路径--->{args.output_dir}')
        os.makedirs(args.output_dir)

    # 定义take和数据加载之间的映射关系
    processors = {
        "zhfw": toke_2_data.SentimentCorpProcessor
    }

    # 获取任务名称
    task_name = args.task_name


    # 数据校验：查看任务是否有对应的数据加载对象
    if task_name not in processors:
        logger.debug(f'当前任务没有数据映射模块:{task_name}')
        raise ValueError("当前任务没有数据映射模块: %s" % (task_name))

    # 获取具体的数据加载对象
    processor = processors[task_name]()

    # 获取标签列表，一个list集合，内部为标签的字符串值
    label_list = processor.get_labels()

    # 基于字典文件构建数据token处理的对象(分词、词转id、id转词等功能，中文中以字作为词)
    tokenizer = tokenization.FullTokenizer(
        vocab_file=args.vocab_file, do_lower_case=args.do_lower_case)

    # TPU运行的相关参数的设置
    tpu_cluster_resolver = None
    if args.use_tpu and args.tpu_name:
        tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
            args.tpu_name, zone=args.tpu_zone, project=args.gcp_project)

    is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
    run_config = tf.contrib.tpu.RunConfig(
        cluster=tpu_cluster_resolver,
        master=args.master,
        model_dir=args.output_dir,
        save_checkpoints_steps=args.save_checkpoints_steps,
        tpu_config=tf.contrib.tpu.TPUConfig(
            iterations_per_loop=args.iterations_per_loop,
            num_shards=args.num_tpu_cores,
            per_host_input_for_training=is_per_host))

    train_examples = None
    num_train_steps = None
    num_warmup_steps = None
    # 模型训练 数据加载
    if args.do_train:
        # 加载训练数据
        train_examples = processor.get_train_examples(args.data_dir)
        num_train_steps = int(
            len(train_examples) / args.train_batch_size * args.num_train_epochs)
        num_warmup_steps = int(num_train_steps * args.warmup_proportion)
    # 自定义模型用于estimator训练
    model_fn = model_fn_builder(
        bert_config=bert_config,
        num_labels=len(label_list),
        init_checkpoint=args.init_checkpoint,
        learning_rate=args.learning_rate,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        use_tpu=args.use_tpu,
        use_one_hot_embeddings=args.use_tpu)

    # 如果没有TPU，会自动转为CPU/GPU的Estimator
    estimator = tf.contrib.tpu.TPUEstimator(
        use_tpu=args.use_tpu,
        model_fn=model_fn,
        config=run_config,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        predict_batch_size=args.predict_batch_size)

    # 训练操作
    if args.do_train:
        # 1. 将训练数据转换为TFRecord格式的数据保存到磁盘(InputExample -> InputFeatures -> TFRecord)
        train_file = os.path.join(args.output_dir, "train.tf_record")
        file_based_convert_examples_to_features(
            train_examples,  # 训练数据，List列表，内部为InputExample对象
            label_list,  # 标签数据，List列表，内部为具体的类别字符串
            args.max_seq_length,  # 当前模型训练允许的最长序列长度，当序列实际长度超过该值的时候，进行截断，小于该值，进行填充
            tokenizer,  # 主要应用：文本转单词以及单词转id
            train_file  # TFRecord数据保存的磁盘路径
        )
        tf.logging.info("***** Running training *****")
        tf.logging.info("  Num examples = %d", len(train_examples))
        tf.logging.info("  Batch size = %d", args.train_batch_size)
        tf.logging.info("  Num steps = %d", num_train_steps)

        # 2. 基于TFRecord的数据文件构建一个数据读入的解析函数
        train_input_fn = file_based_input_fn_builder(
            input_file=train_file,
            seq_length=args.max_seq_length,
            is_training=True,
            drop_remainder=True)

        # 3. 直接训练
        estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)

    # 验证操作
    if args.do_eval:
        # 1. 获取验证数据集， List列表，内部为InputExample对象
        eval_examples = processor.get_dev_examples(args.data_dir)

        # 2. 得到实际的样本数目
        num_actual_eval_examples = len(eval_examples)

        # 3. 如果是TPU运行，进行必要的数据样本填充(填充PaddingInputExample对象)
        if args.use_tpu:
            # TPU requires a fixed batch size for all batches, therefore the number
            # of examples must be a multiple of the batch size, or else examples
            # will get dropped. So we pad with fake examples which are ignored
            # later on. These do NOT count towards the metric (all tf2.0.metrics
            # support a per-instance weight, and these get a weight of 0.0).
            while len(eval_examples) % args.eval_batch_size != 0:
                eval_examples.append(PaddingInputExample())

        # 4. 将验证数据转换为TFRecord格式的数据保存到磁盘(InputExample -> InputFeatures -> TFRecord)
        eval_file = os.path.join(args.output_dir, "eval.tf_record")
        file_based_convert_examples_to_features(
            eval_examples,  # 验证数据，List列表，内部为InputExample对象
            label_list,  # 标签数据，List列表，内部为具体的类别字符串
            args.max_seq_length,  # 当前模型训练允许的最长序列长度，当序列实际长度超过该值的时候，进行截断，小于该值，进行填充
            tokenizer,  # 主要应用：文本转单词以及单词转id
            eval_file  # TFRecord数据保存的磁盘路径
        )
        tf.logging.info("***** Running evaluation *****")
        tf.logging.info("  Num examples = %d (%d actual, %d padding)",
                        len(eval_examples), num_actual_eval_examples,
                        len(eval_examples) - num_actual_eval_examples)
        tf.logging.info("  Batch size = %d", args.eval_batch_size)

        # 5. 计算验证的步长数目(批次的数目)
        # This tells the estimator to run through the entire set.
        eval_steps = None
        # However, if running eval on the TPU, you will need to specify the number of steps.
        if args.use_tpu:
            assert len(eval_examples) % args.eval_batch_size == 0
            eval_steps = int(len(eval_examples) // args.eval_batch_size)

        # 6. 数据加载函数的构建
        eval_drop_remainder = True if args.use_tpu else False
        eval_input_fn = file_based_input_fn_builder(
            input_file=eval_file,
            seq_length=args.max_seq_length,
            is_training=False,
            drop_remainder=eval_drop_remainder)

        # 7. 验证数据的获取
        result = estimator.evaluate(input_fn=eval_input_fn, steps=eval_steps)

        # 8. 验证结果数据的保存
        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        with tf.gfile.GFile(output_eval_file, "w") as writer:
            tf.logging.info("***** Eval results *****")
            for key in sorted(result.keys()):
                tf.logging.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

if __name__ == '__main__':
    tf.compat.v1.app.run()