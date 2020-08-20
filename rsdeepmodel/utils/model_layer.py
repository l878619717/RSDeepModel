# -*- coding:utf-8 -*
'''
功能：通用的模型内部模块，如参数初始化、embedding-pooling、target-attention、multi-heads self-attention等
'''
import sys

import numpy as np
import tensorflow as tf


def emb_init(name, feat_num, embedding_size, zero_first_row=True, pre_trained=False, trained_emb_path=None):
    if not pre_trained:
        with tf.variable_scope("weight_matrix"):
            embeddings = tf.get_variable(name=name,
                                         dtype=tf.float32,
                                         shape=(feat_num, embedding_size),
                                         initializer=tf.contrib.layers.xavier_initializer())

        if zero_first_row:  # The first row of initialization is zero
            embeddings = tf.concat((tf.zeros(shape=[1, embedding_size]), embeddings[1:]), 0)
    else:
        pass
        with tf.variable_scope("pre-trained_weight_matrix"):
            load_emb = np.load(tf.gfile.GFile(trained_emb_path, "rb"))
            embeddings = tf.constant(load_emb, dtype=tf.float32, name=name)
            sys.stdout.flush()

    return embeddings


def nonzero_reduce_mean(emb):  # nonzero-mean-pooling
    axis_2_sum = tf.reduce_sum(emb, axis=2)
    multi_cate_nonzero = tf.count_nonzero(axis_2_sum, 1, keepdims=True, dtype=float)
    multi_cate_sum = tf.reduce_sum(emb, axis=1)
    reduce_mean_emb = tf.div_no_nan(multi_cate_sum, multi_cate_nonzero)
    return reduce_mean_emb


class InteractingLayer:  # multi-heads self-attention
    def __init__(self, num_layer, att_emb_size=32, seed=2020, head_num=3, use_res=1):
        if head_num <= 0:
            raise ValueError('head_num must be a int > 0')
        self.num_layer = num_layer
        self.att_emb_size = att_emb_size
        self.seed = seed
        self.head_num = head_num
        self.use_res = use_res

    def __call__(self, inputs):
        input_shape = inputs.get_shape().as_list()
        if len(input_shape) != 3:
            raise ValueError("Unexpected inputs dimensions %d, expect to be 3 dimensions" % len(input_shape))

        embedding_size = int(input_shape[-1])
        self.w_query = tf.get_variable(name=str(self.num_layer) + '_query',
                                       dtype=tf.float32,
                                       shape=(embedding_size, self.att_emb_size * self.head_num),
                                       initializer=tf.contrib.layers.xavier_initializer(seed=self.seed))
        self.w_key = tf.get_variable(name=str(self.num_layer) + '_key',
                                     dtype=tf.float32,
                                     shape=(embedding_size, self.att_emb_size * self.head_num),
                                     initializer=tf.contrib.layers.xavier_initializer(seed=self.seed + 1))
        self.w_value = tf.get_variable(name=str(self.num_layer) + '_value',
                                       dtype=tf.float32,
                                       shape=(embedding_size, self.att_emb_size * self.head_num),
                                       initializer=tf.contrib.layers.xavier_initializer(seed=self.seed + 2))
        if self.use_res:
            self.w_res = tf.get_variable(name=str(self.num_layer) + '_res',
                                         dtype=tf.float32,
                                         shape=(embedding_size, self.att_emb_size * self.head_num),
                                         initializer=tf.contrib.layers.xavier_initializer(seed=self.seed))

        querys = tf.tensordot(inputs, self.w_query, axes=1)  # None F D*head_num
        keys = tf.tensordot(inputs, self.w_key, axes=1)
        values = tf.tensordot(inputs, self.w_value, axes=1)

        # head_num None F D
        querys = tf.stack(tf.split(querys, self.head_num, axis=2))
        keys = tf.stack(tf.split(keys, self.head_num, axis=2))
        values = tf.stack(tf.split(values, self.head_num, axis=2))

        inner_product = tf.matmul(querys, keys, transpose_b=True)  # head_num None F F
        # Scale
        inner_product = inner_product / (keys.get_shape().as_list()[-1] ** 0.5)
        # Activation
        self.normalized_att_scores = tf.nn.softmax(inner_product)

        result = tf.matmul(self.normalized_att_scores, values)  # head_num None F D
        result = tf.concat(tf.split(result, self.head_num, axis=0), axis=-1)  # 1 None F D*head_num
        result = tf.squeeze(result, axis=0)  # None F D*head_num

        if self.use_res:
            result += tf.tensordot(inputs, self.w_res, axes=1)
        result = tf.nn.relu(result)

        return result


def attention(queries, keys, keys_length):  # target-attention
    """
      queries:     [B, H] 前面的B代表的是batch_size，H代表向量维度。
      keys:        [B, T, H] T是一个batch中，当前特征最大的长度，每个样本代表一个样本的特征
      keys_length: [B]
    """
    # H 每个query词的隐藏层神经元是多少，也就是H
    queries_hidden_units = queries.get_shape().as_list()[-1]
    # tf.tile为复制函数，1代表在B上保持一致，tf.shape(keys)[1] 代表在H上复制这么多次, 那么queries最终shape为(B, H*T)
    queries = tf.tile(queries, [1, tf.shape(keys)[1]])
    # queries.shape(B, T, H) 其中每个元素(T,H)代表T行H列，其中每个样本中，每一行的数据都是一样的
    queries = tf.reshape(queries, [-1, tf.shape(keys)[1], queries_hidden_units])
    # 下面4个变量的shape都是(B, T, H)，按照最后一个维度concat，所以shape是(B, T, H*4), 在这块就将特征中的每个item和目标item连接在了一起
    din_all = tf.concat([queries, keys, queries - keys, queries * keys], axis=-1)
    # (B, T, 80)
    d_layer_1_all = tf.layers.dense(din_all, 80, activation=tf.nn.sigmoid, name='f1_att', reuse=tf.AUTO_REUSE)
    # (B, T, 40)
    d_layer_2_all = tf.layers.dense(d_layer_1_all, 40, activation=tf.nn.sigmoid, name='f2_att', reuse=tf.AUTO_REUSE)
    # (B, T, 1)
    d_layer_3_all = tf.layers.dense(d_layer_2_all, 1, activation=None, name='f3_att', reuse=tf.AUTO_REUSE)
    # (B, 1, T)
    # 每一个样本都是 [1,T] 的维度，和原始特征的维度一样，但是这时候每个item已经是特征中的一个item和目标item混在一起的数值了
    d_layer_3_all = tf.reshape(d_layer_3_all, [-1, 1, tf.shape(keys)[1]])
    outputs = d_layer_3_all
    # Mask，每一行都有T个数字，keys_length长度为B，假设第1 2个数字是5,6，那么key_masks第1 2行的前5 6个数字为True
    key_masks = tf.sequence_mask(keys_length, tf.shape(keys)[1])  # [B, T]
    key_masks = tf.expand_dims(key_masks, 1)  # [B, 1, T]
    # 创建一个和outputs的shape保持一致的变量，值全为1，再乘以(-2 ** 32 + 1)，所以每个值都是(-2 ** 32 + 1)
    paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
    outputs = tf.where(key_masks, outputs, paddings)  # [B, 1, T]

    # Scale
    outputs = outputs / (keys.get_shape().as_list()[-1] ** 0.5)  # T，根据特征数目来做拉伸
    # Activation
    outputs = tf.nn.softmax(outputs)  # [B, 1, T]
    # Weighted sum
    outputs = tf.matmul(outputs, keys)  # [B, 1, H]
    return outputs


def attention_multi(queries, keys, keys_length):  # target-attention for multi-feats-queries
    """
      queries:     [B, N, H] (e.g. N is the number of ads)
      keys:        [B, T, H]
      keys_length: [B]
    """
    # H 每个query词的隐藏层神经元是多少，也就是H
    queries_hidden_units = queries.get_shape().as_list()[-1]
    # N
    queries_nums = queries.get_shape().as_list()[1]
    queries = tf.tile(queries, [1, 1, tf.shape(keys)[1]])
    queries = tf.reshape(queries, [-1, queries_nums, tf.shape(keys)[1], queries_hidden_units])  # shape : [B, N, T, H]
    max_len = tf.shape(keys)[1]
    keys = tf.tile(keys, [1, queries_nums, 1])
    keys = tf.reshape(keys, [-1, queries_nums, max_len, queries_hidden_units])  # shape : [B, N, T, H]
    din_all = tf.concat([queries, keys, queries - keys, queries * keys], axis=-1)
    d_layer_1_all = tf.layers.dense(din_all, 80, activation=tf.nn.sigmoid, name='f1_att', reuse=tf.AUTO_REUSE)
    d_layer_2_all = tf.layers.dense(d_layer_1_all, 40, activation=tf.nn.sigmoid, name='f2_att', reuse=tf.AUTO_REUSE)
    d_layer_3_all = tf.layers.dense(d_layer_2_all, 1, activation=None, name='f3_att', reuse=tf.AUTO_REUSE)  # [B, N, T, 1]
    d_layer_3_all = tf.reshape(d_layer_3_all, [-1, queries_nums, 1, max_len])  # [B, N, 1, T]
    outputs = d_layer_3_all
    # Mask
    key_masks = tf.sequence_mask(keys_length, max_len)  # [B, T]
    key_masks = tf.tile(key_masks, [1, queries_nums])  # [B, N, T]
    key_masks = tf.reshape(key_masks, [-1, queries_nums, 1, max_len])  # shape : [B, N, 1, T]
    paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
    outputs = tf.where(key_masks, outputs, paddings)  # [B, N, 1, T]

    # Scale
    outputs = outputs / (keys.get_shape().as_list()[-1] ** 0.5)

    # Activation
    outputs = tf.nn.softmax(outputs)  # [B, N, 1, T]
    # Weighted sum
    outputs = tf.matmul(outputs, keys)  # [B, N, 1, H]

    # Pooling
    # outputs = tf.reduce_mean(outputs, axis=1)  # [B, 1, H] mean-pooling
    outputs = tf.reduce_sum(outputs, axis=1)  # [B, 1, H] sum-pooling
    return outputs
