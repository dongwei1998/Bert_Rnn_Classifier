# -- encoding:utf-8 --


import tensorflow as tf



class TextRNN(object):
    """
    A RNN for text classification
    """

    def __init__(self, network_name, initializer, num_labels, num_units,
                 sequence_input_embedding, sequence_target_label):
        self.network_name = network_name  # 名称
        self.initializer = initializer  # 变量初始化器
        self.num_labels = num_labels  # 类别数目
        self.num_units = num_units  # RNN中神经元的数目

        with tf.variable_scope(self.network_name, initializer=self.initializer):
            # 2. Build RNN output
            with tf.variable_scope("rnn"):
                # a. 构建RNN结构
                cell_fw = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.num_units)
                cell_bw = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.num_units)

                # b. 数据输入，并处理
                (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell_fw, cell_bw=cell_bw,
                                                                            inputs=sequence_input_embedding,
                                                                            dtype=tf.float32)

                # c. 获取最后一个时刻的输出
                output = tf.concat((output_fw[:, -1, :], output_bw[:, -1, :]), -1)

            # 3. Build FC output
            with tf.variable_scope("fc"):
                in_units = output.get_shape()[-1]
                w = tf.get_variable(name='w', shape=[in_units, self.num_labels])
                b = tf.get_variable(name='b', shape=[self.num_labels])
                self.logits = tf.nn.xw_plus_b(output, weights=w, biases=b, name='scores')
                self.probabilities = tf.nn.softmax(self.logits, axis=-1)

            # 4. Build Loss
            with tf.variable_scope("loss"):
                self.per_example_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=sequence_target_label,
                                                                                       logits=self.logits)
                self.loss = tf.reduce_mean(self.per_example_loss)
                tf.losses.add_loss(self.loss)
                self.total_loss = tf.losses.get_total_loss(name='total_loss')
