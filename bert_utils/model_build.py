# -*- coding:utf-8 -*-
# @Time      : 2021-08-02 17:27
# @Author    : 年少无为呀！
# @FileName  : model_build.py
# @Software  : PyCharm
import tensorflow as tf
from bert_utils import modeling
from bert_utils import text_rnn
from bert_utils import optimization
def create_model(bert_config, is_training, input_ids, input_mask, segment_ids,
                 labels, num_labels, use_one_hot_embeddings):
    # 1. 构建bert模型
    model = modeling.BertModel(
        config=bert_config,  # bert模型的配置信息
        is_training=is_training,  # 是否是训练操作, 训练: True, 验证/预测: False
        input_ids=input_ids,  # 模型输入的id
        input_mask=input_mask,  # 模型输入的掩码
        token_type_ids=segment_ids,  # 模型输入的segment id
        use_one_hot_embeddings=use_one_hot_embeddings  # embedding计算方式
    )

    # # 最后一层第一个时刻的输出，[N,E]
    # model.get_pooled_output()
    # # 最后一层所有时刻的输出，Tensor, 形状为: [N,T,E]
    # model.get_sequence_output()
    # # 所有层的所有时刻的输出 list列表，内部为每一层的输出Tensor, 形状为: [N,T,E]
    # model.get_all_encoder_layers()
    # # # bert中embedding table的输出
    # # model.get_embedding_output()
    # # # bert中的embedding table的值
    # # model.get_embedding_table()

    # 2. 获取Bert模型的输出（获取最后一层对应的输出）
    output_layer = model.get_sequence_output()  # [batch_size, max_seq_length, hidden_size]

    # 3. 基于Bert的输出以及模型的希望输出构建后面的分类模型
    classification_model = text_rnn.TextRNN(network_name="TextRNN",
                                            initializer=None,
                                            num_labels=num_labels,
                                            num_units=128,  # 给定内部RNN的神经元数目
                                            sequence_input_embedding=output_layer,
                                            sequence_target_label=labels
                                            )
    # 返回总的损失函数、每个样本的损失、logits的值(置信度值)、样本所属类别的概率值
    return classification_model.total_loss, classification_model.per_example_loss, \
           classification_model.logits, classification_model.probabilities

def model_fn_builder(bert_config, num_labels, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu=False,
                     use_one_hot_embeddings=False):
    """Returns `model_fn` closure for TPUEstimator."""

    def model_fn(features, labels, mode, config, params):  # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""
        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

        # 1. 从加载的数据中获取对应的Tensor对象(原始信息)
        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        label_ids = features["label_ids"]
        is_real_example = None
        if "is_real_example" in features:
            is_real_example = tf.cast(features["is_real_example"], dtype=tf.float32)
        else:
            is_real_example = tf.ones(tf.shape(label_ids), dtype=tf.float32)
        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        # 2. 构建模型(前向过程+损失函数的构建)
        # 返回值: 总的损失函数、每个样本的损失函数、logits<没有经过概率转换的置信度>、样本的预测概率值
        (total_loss, per_example_loss, logits, probabilities) = create_model(
            bert_config,  # Bert模型的配置参数
            is_training,  # 是否是训练操作
            input_ids,  # 模型的输入input ids
            input_mask,  # 模型的输入input mask
            segment_ids,  # 模型的输入input segment ids
            label_ids,  # 模型的输入label ids(实际标签id)
            num_labels,  # 类别数目
            use_one_hot_embeddings  # embedding的实现方式: FC or embedding_lookup
        )

        # 3. 对于bert模型的参数，使用bert模型给定的初始化路径进行初始化操作(定义)
        tvars = tf.trainable_variables()  # 获取所有的训练变量
        initialized_variable_names = {}
        scaffold_fn = None
        if init_checkpoint:
            # 加载具体哪些参数需要从bert模型中恢复(根据名称获取具体哪些需要加载恢复)
            (assignment_map, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(
                tvars,
                init_checkpoint
            )
            if use_tpu:
                def tpu_scaffold():
                    tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
                    return tf.train.Scaffold()

                scaffold_fn = tpu_scaffold
            else:
                # 初始化操作
                tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        tf.logging.info("**** Trainable Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape, init_string)

        output_spec = None
        if mode == tf.estimator.ModeKeys.TRAIN:
            # 1. 构建优化对象
            train_op = optimization.create_optimizer(
                total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)

            # 2. 构建Estimeator算法封装对象的输出Spec对象
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                train_op=train_op,
                scaffold_fn=scaffold_fn)
        elif mode == tf.estimator.ModeKeys.EVAL:
            def metric_fn(per_example_loss, label_ids, logits, is_real_example):
                # 预测的索引下标(获取最大索引)
                predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
                # 和实际值一起构建出准确率
                accuracy = tf.metrics.accuracy(labels=label_ids, predictions=predictions, weights=is_real_example)
                loss = tf.metrics.mean(values=per_example_loss, weights=is_real_example)
                precision = tf.metrics.precision(labels=label_ids, predictions=predictions, weights=is_real_example)
                recall = tf.metrics.recall(labels=label_ids, predictions=predictions, weights=is_real_example)
                return {
                    "eval_accuracy": accuracy,
                    "eval_loss": loss,
                    "eval_recall": recall,
                    "eval_precision": precision
                }

            eval_metrics = (metric_fn, [per_example_loss, label_ids, logits, is_real_example])
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,  # 给定操作类型为验证操作
                loss=total_loss,  # 损失
                eval_metrics=eval_metrics,  # metrics的信息，验证操作计算的metrics具体有哪些
                scaffold_fn=scaffold_fn)
        else:
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,  # 给定预测操作
                predictions={"probabilities": probabilities},  # 预测返回的结果具体有哪些，属于一个字典
                scaffold_fn=scaffold_fn)
        return output_spec

    return model_fn

