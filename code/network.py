# encoding: utf-8
"""
@author: bqw
@time: 2020/5/11 16:18
@file: network.py
@desc: 模型BERT-CRF的网络结构
"""
import tensorflow as tf
from bert import modeling
from tensorflow.contrib.layers.python.layers import initializers
from tensorflow.contrib import crf


class BertCrf(object):
    def __init__(self, bert_config, is_training, input_ids, input_mask, segment_ids,
                 labels, num_labels, max_seq_length, use_one_hot_embeddings):
        # load bert
        bert = modeling.BertModel(config=bert_config,
                                  is_training=is_training,
                                  input_ids=input_ids,
                                  input_mask=input_mask,
                                  token_type_ids=segment_ids,
                                  use_one_hot_embeddings=use_one_hot_embeddings)
        # 获取bert的输出
        output_layer = bert.get_sequence_output()
        # self.all_encoder_layers = bert.get_all_encoder_layers()
        if is_training:
            output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)
        hidden_size = output_layer.shape[-1].value
        output_layer = tf.reshape(output_layer, [-1, hidden_size])
        tf.logging.info(" The dimension of bert output:%s" % output_layer.shape)

        # 全连接层
        output_weight = tf.get_variable("output_weights", [num_labels, hidden_size],
                                        initializer=tf.truncated_normal_initializer(stddev=0.02))
        output_bias = tf.get_variable("output_bias", [num_labels], initializer=tf.zeros_initializer())
        logits = tf.matmul(output_layer, output_weight, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)
        self.logits = tf.reshape(logits, [-1, max_seq_length, num_labels])

        # 使用全连接层的输出计算MNLP分数
        self.probs = tf.nn.softmax(self.logits, axis=-1)
        self.best_probs = tf.reduce_max(self.probs, axis=-1)
        self.mnlp_score = tf.reduce_mean(tf.log(self.best_probs), axis=-1)

        # 计算输入样本的长度
        used = tf.sign(tf.abs(input_ids))
        lengths = tf.reduce_sum(used, reduction_indices=1)

        # crf层
        with tf.variable_scope("crf"):
            trans = tf.get_variable("transitions",
                                    shape=[num_labels, num_labels],
                                    initializer=initializers.xavier_initializer())
            if labels is None:
                self.loss = None
            else:
                log_likelihood, trans = tf.contrib.crf.crf_log_likelihood(
                    inputs=self.logits,
                    tag_indices=labels,
                    transition_params=trans,
                    sequence_lengths=lengths)
                self.loss = tf.reduce_mean(-log_likelihood)

            self.predicts, self.score = crf.crf_decode(potentials=self.logits, transition_params=trans,
                                                       sequence_length=lengths)