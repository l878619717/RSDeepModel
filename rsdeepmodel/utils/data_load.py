# -*- coding:utf-8 -*-
'''
功能：加载tfrecord

-------
所有连续特征合并为features["cont_feats"], 所有单值离散特征合并为features["cate_feats"], 此时由于位置和特征一一对应, 故必须补齐缺失值, 即数据类型为dense类型->FixedLenFeature
多值离散特征不合并, 各自为一维特征(features["name1"], features["name2"], ..., features["nameN"])
-------
由于多值离散特征fetures["name*"]最终往往会进行sum_pool或mean_pool处理, 故可以不补齐缺失值, 即sparse类型->VarLenFeature, 配合embedding_lookup_sparse使用即可
当然多值离散特征也可以像连续特征/单值离散特征一样, 补齐为dense类型->FixedLenFeature, 配合embedding_lookup和reduce_sum/reduce_mean即可
但多值离散特征往往维度较大且稀疏, 建议使用sparse方式以节省空间与时间, 除了一些特殊情况(如din)
-------
关于使用了target-attention的din与dinfm模型, 多值离散特征用target-attention机制代替了简单的sum/mean-pooling, 所以不能使用embedding_lookup_sparse, 即不支持sparse数据
这里使用pdded_batch实现了'sparse2dense', 但还是建议在生成tfrecord数据之前直接利用spark补齐缺失值生成dense数据, spark效率更高
-------
'''
import tensorflow as tf


def parse_example(example, params):
    features = {
        'label': tf.FixedLenFeature([1], tf.float32),
        'cont_feats': tf.FixedLenFeature([params.cont_field_count], tf.float32),
        'cate_feats': tf.FixedLenFeature([params.cate_field_count], tf.int64),
    }
    if params.multi_feats_type == 'dense':
        for name_topN in params.multi_cate_field_list:
            features[name_topN[0]] = tf.FixedLenFeature([name_topN[1]], tf.int64)
    elif params.multi_feats_type in ['sparse', 'sparse2dense']:
        for name_topN in params.multi_cate_field_list:
            features[name_topN[0]] = tf.VarLenFeature(tf.int64)

    parsed_features = tf.parse_single_example(example, features)
    label = parsed_features['label']
    parsed_features.pop('label')

    if params.multi_feats_type == 'sparse2dense':  # 配合padded_batch使用
        for name_topN in params.multi_cate_field_list:
            parsed_features[name_topN[0]] = tf.sparse_tensor_to_dense(parsed_features[name_topN[0]])

    return parsed_features, label


def input_fn(data_path, params):
    data_set = tf.data.TFRecordDataset(data_path)

    if params.multi_feats_type in ['dense', 'sparse']:
        data_set = data_set.map(lambda x: parse_example(x, params), num_parallel_calls=params.num_threads) \
            .batch(params.batch_size) \
            .repeat(params.epochs) \
            .prefetch(params.batch_size)

    elif params.multi_feats_type == 'sparse2dense':  # 利用padded_batch实现sparse数据转为dense
        pad_shapes = dict()
        pad_shapes_label = tf.TensorShape([1])
        pad_shapes["cont_feats"] = tf.TensorShape([params.cont_field_count])
        pad_shapes["cate_feats"] = tf.TensorShape([params.cate_field_count])
        for name_topN in params.multi_cate_field_list:
            pad_shapes[name_topN[0]] = tf.TensorShape([name_topN[1]])
        pad_shapes = (pad_shapes, pad_shapes_label)
        data_set = data_set.map(lambda x: parse_example(x, params), num_parallel_calls=params.num_threads) \
            .padded_batch(params.batch_size, padded_shapes=pad_shapes) \
            .repeat(params.epochs) \
            .prefetch(params.batch_size)

    else:
        print("multi_feats_type error!!!")
        exit(-1)

    iterator = data_set.make_one_shot_iterator()
    feature_dict, label = iterator.get_next()
    return feature_dict, label
