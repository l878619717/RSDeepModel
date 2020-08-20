# -*- coding:utf-8 -*
'''
功能: 模型共用模块，如训练、预测、保存等
'''

import tensorflow as tf


def model_optimizer(params, mode, labels, out):
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=out)

    else:
        labels = tf.identity(labels, name='labels')
        auc = tf.metrics.auc(labels=labels, predictions=out, name='auc')
        metrics = {
            'auc': auc
        }
        loss = tf.reduce_mean(tf.losses.log_loss(labels=labels, predictions=out))

        # ------bulid optimizer------
        if params.optimizer == 'Adam':
            optimizer = tf.train.AdamOptimizer(learning_rate=params.learning_rate)
        elif params.optimizer == 'Adagrad':
            optimizer = tf.train.AdagradOptimizer(learning_rate=params.learning_rate)
        elif params.optimizer == 'Momentum':
            optimizer = tf.train.MomentumOptimizer(learning_rate=params.learning_rate, momentum=0.95)
        elif params.optimizer == 'ftrl':
            optimizer = tf.train.FtrlOptimizer(learning_rate=params.learning_rate)
        elif params.optimizer == 'Adadelta':
            optimizer = tf.train.AdadeltaOptimizer(learning_rate=params.learning_rate)

        if mode == tf.estimator.ModeKeys.TRAIN:
            train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
        else:
            train_op = None

        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss,
            eval_metric_ops=metrics,
            train_op=train_op)


def model_save_pb(params, model):
    features = {
        'cont_feats': tf.FixedLenFeature(dtype=tf.float32, shape=[params.cont_field_count]),
        'cate_feats': tf.FixedLenFeature(dtype=tf.int64, shape=[params.cate_field_count]),
    }
    if params.multi_feats_type in ['dense', 'sparse2dense']:
        for field_name in params.multi_cate_field_list:
            features[field_name[0]] = tf.FixedLenFeature(field_name[1], tf.int64)
    elif params.multi_feats_type == 'sparse':
        for field_name in params.multi_cate_field_list:
            features[field_name[0]] = tf.VarLenFeature(tf.int64)

    if params.model_pb_type == 'parsing':
        serving_input_receiver_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(features)
    elif params.model_pb_type == 'raw':
        serving_input_receiver_fn = tf.estimator.export.build_raw_serving_input_receiver_fn(features)

    return model.export_savedmodel(params.model_pb, serving_input_receiver_fn)
