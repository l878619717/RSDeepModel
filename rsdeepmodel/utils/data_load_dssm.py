# -*- coding:utf-8 -*-
'''
功能：加载tfrecord
专门用于dssm读取数据，将user侧和item侧特征分开
"user_cont_feats": user侧连续型特征
"user_cate_feats": user侧单值离散型特征
"item_cont_feats": item侧连续型特征
"item_cate_feats": item侧单值离散型特征
多值特征还是按照默认单独为列，在模型内部区分多值特征是user侧或item侧
-------
'''
import tensorflow as tf


def parse_example(example, params):
    features = {
        'label': tf.FixedLenFeature([1], tf.float32),
        # 'user_cont_feats': tf.FixedLenFeature([params.user_cont_field_count], tf.float32), 暂时没有
        'user_cate_feats': tf.FixedLenFeature([params.user_cate_field_count], tf.int64),
        'item_cont_feats': tf.FixedLenFeature([params.item_cont_field_count], tf.float32),
        'item_cate_feats': tf.FixedLenFeature([params.item_cate_field_count], tf.int64),
    }
    for name_topN in params.multi_cate_field_list:  # 这里默认params.multi_feats_type=sparse
        features[name_topN[0]] = tf.VarLenFeature(tf.int64)

    parsed_features = tf.parse_single_example(example, features)
    label = parsed_features['label']
    parsed_features.pop('label')

    return parsed_features, label


def input_fn(data_path, params):
    data_set = tf.data.TFRecordDataset(data_path)

    data_set = data_set.map(lambda x: parse_example(x, params), num_parallel_calls=params.num_threads) \
        .batch(params.batch_size) \
        .repeat(params.epochs) \
        .prefetch(params.batch_size)

    iterator = data_set.make_one_shot_iterator()
    feature_dict, label = iterator.get_next()
    return feature_dict, label
