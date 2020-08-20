# -*- coding:utf-8 -*-

# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function

# import argparse
import shutil
# import sys
import os
import glob
from datetime import date, timedelta
from time import time

import random
import tensorflow as tf
from tensorflow.python.layers import normalization
from tensorflow.python.framework import sparse_tensor as sparse_tensor_lib
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops

# os.environ["CUDA_VISIBLE_DEVICES"] = '1'

#################### CMD Arguments ####################
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer("num_threads", 1, "Number of threads")
tf.app.flags.DEFINE_integer("user_field_size", 9, "Number of fields")
tf.app.flags.DEFINE_integer("item_field_size", 5, "Number of fields")
tf.app.flags.DEFINE_integer("embedding_size", 512, "Embedding size")
tf.app.flags.DEFINE_integer("num_epochs", 1, "Number of epochs")
tf.app.flags.DEFINE_integer("batch_size", 64, "Number of batch size")
tf.app.flags.DEFINE_integer("log_steps", 1000, "save summary every steps")
tf.app.flags.DEFINE_float("learning_rate", 0.001, "learning rate")
tf.app.flags.DEFINE_string("optimizer", 'Adam', "optimizer type {Adam, Adagrad, GD, Momentum}")
tf.app.flags.DEFINE_string("user_deep_layers", '256,128,64', "deep layers")
tf.app.flags.DEFINE_string("item_deep_layers", '256,128,64', "deep layers")
tf.app.flags.DEFINE_string("dropout", '0.5,0.5,0.5', "dropout rate")
tf.app.flags.DEFINE_string("data_dir", '', "data dir")
tf.app.flags.DEFINE_string("train_data", 'data/textline/demo/demo-textline', "")
tf.app.flags.DEFINE_string("val_data", 'data/textline/demo/demo-textline', "")
tf.app.flags.DEFINE_string("test_data", '', "")
tf.app.flags.DEFINE_string("infer_result", '', "")
tf.app.flags.DEFINE_string("model_dir", 'data/model_save_dir/dssm_textline', "model check point dir")
tf.app.flags.DEFINE_string("servable_model_dir", 'data/model_save_pb/dssm_textline', "export servable model for TensorFlow Serving")
tf.app.flags.DEFINE_string("task_type", 'train', "task type {train, infer, eval, export}")
tf.app.flags.DEFINE_boolean("clear_existing_model", False, "clear existing model or not")
# autoint
tf.app.flags.DEFINE_integer("autoint_layer_count", 2, "")
tf.app.flags.DEFINE_integer("autoint_emb_size", 16, "")
tf.app.flags.DEFINE_integer("autoint_head_count", 2, "")
tf.app.flags.DEFINE_integer("autoint_use_res", 1, "")

CSV_COLUMNS = ["label", "uid", "ucat1long", "ucat2long", "umedialong", "utaglong", "ucat1short", "ucat2short", "umediashort", "utagshort", "aid", "cat1", "cat2", "media", "tags"]
PREDICT_COLUMNS = ["itemid", "cat1", "cat2", "media", "tags"]
VID_HASH_SIZE = 6000000  # 1000000
CATE1_HASH_SIZE = 1000
CATE2_HASH_SIZE = 10000
CP_HASH_SIZE = 500000
TAG_HASH_SIZE = 500000
FIELD_DEFAULTS = [["0"]] * 15


def parse_csv(text):
    fields = tf.decode_csv(text, FIELD_DEFAULTS, field_delim=" ")
    print("====================:", fields)
    features = dict(zip(CSV_COLUMNS, fields))
    features["label"] = tf.string_to_number(tf.cast(features["label"], tf.string), out_type=tf.int64)

    sparse_cols = ["ucat1long", "ucat2long", "umedialong", "utaglong", "ucat1short", "ucat2short", "umediashort", "utagshort", "tags"]
    for col in sparse_cols:
        sparse_tensor = tf.string_split([features[col]], delimiter=",")
        features[col] = sparse_tensor
    labels = features.pop("label")

    return features, labels


def input_fn(filenames, batch_size=32, num_epochs=1, perform_shuffle=False):
    print('Parsing', filenames)
    dataset = tf.data.TextLineDataset(filenames).map(parse_csv, num_parallel_calls=FLAGS.num_threads)    # multi-thread pre-process then prefetch
    if perform_shuffle:
        dataset = dataset.shuffle(buffer_size=256)

    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size)  # Batch size to use
    dataset = dataset.prefetch(batch_size)
    return dataset.make_one_shot_iterator().get_next()


def build_model_columns(embedding_size):
    uid = tf.feature_column.categorical_column_with_hash_bucket(key="uid", hash_bucket_size=6000000)
    uid_emb = tf.feature_column.embedding_column(uid, dimension=embedding_size)

    ucat1long = tf.feature_column.categorical_column_with_hash_bucket(key="ucat1long", hash_bucket_size=10000)
    ucat1long_emb = tf.feature_column.embedding_column(ucat1long, dimension=embedding_size)

    ucat2long = tf.feature_column.categorical_column_with_hash_bucket(key="ucat2long", hash_bucket_size=100000)
    ucat2long_emb = tf.feature_column.embedding_column(ucat2long, dimension=embedding_size)

    umedialong = tf.feature_column.categorical_column_with_hash_bucket(key="umedialong", hash_bucket_size=1000000)
    umedialong_emb = tf.feature_column.embedding_column(umedialong, dimension=embedding_size)

    utaglong = tf.feature_column.categorical_column_with_hash_bucket(key="utaglong", hash_bucket_size=1000000)
    utaglong_emb = tf.feature_column.embedding_column(utaglong, dimension=embedding_size)

    ucat1short = tf.feature_column.categorical_column_with_hash_bucket(key="ucat1short", hash_bucket_size=10000)
    ucat1short_emb = tf.feature_column.embedding_column(ucat1short, dimension=embedding_size)

    ucat2short = tf.feature_column.categorical_column_with_hash_bucket(key="ucat2short", hash_bucket_size=100000)
    ucat2short_emb = tf.feature_column.embedding_column(ucat2short, dimension=embedding_size)

    umediashort = tf.feature_column.categorical_column_with_hash_bucket(key="umediashort", hash_bucket_size=1000000)
    umediashort_emb = tf.feature_column.embedding_column(umediashort, dimension=embedding_size)

    utagshort = tf.feature_column.categorical_column_with_hash_bucket(key="utagshort", hash_bucket_size=1000000)
    utagshort_emb = tf.feature_column.embedding_column(utagshort, dimension=embedding_size)

    aid = tf.feature_column.categorical_column_with_hash_bucket(key="aid", hash_bucket_size=1000000)
    aid_emb = tf.feature_column.embedding_column(aid, dimension=embedding_size)

    cat1 = tf.feature_column.categorical_column_with_hash_bucket(key="cat1", hash_bucket_size=10000)
    cat1_emb = tf.feature_column.embedding_column(cat1, dimension=embedding_size)

    cat2 = tf.feature_column.categorical_column_with_hash_bucket(key="cat2", hash_bucket_size=100000)
    cat2_emb = tf.feature_column.embedding_column(cat2, dimension=embedding_size)

    media = tf.feature_column.categorical_column_with_hash_bucket(key="media", hash_bucket_size=1000000)
    media_emb = tf.feature_column.embedding_column(media, dimension=embedding_size)

    tags = tf.feature_column.categorical_column_with_hash_bucket(key="tags", hash_bucket_size=1000000)
    tags_emb = tf.feature_column.embedding_column(tags, dimension=embedding_size)

    return [uid_emb, ucat1long_emb, ucat2long_emb, umedialong_emb, utaglong_emb, ucat1short_emb, ucat2short_emb, umediashort_emb, utagshort_emb, aid_emb, cat1_emb,  cat2_emb, media_emb, tags_emb]


class InteractingLayer:
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


def model_fn(features, labels, mode, params):
    """Bulid Model function f(x) for Estimator."""
    user_field_size = params["user_field_size"]
    item_field_size = params["item_field_size"]
    embedding_size = params["embedding_size"]
    learning_rate = params["learning_rate"]
    user_layers = list(map(int, params["user_deep_layers"].split(',')))
    item_layers = list(map(int, params["item_deep_layers"].split(',')))
    dropout = list(map(float, params["dropout"].split(',')))

    net = tf.feature_column.input_layer(features, params["feature_columns"])
    print("net shape:", net.shape)

    user_embeddings = tf.slice(net, [0, 0], [-1, user_field_size*embedding_size])
    # print("user shape:", user_embeddings.shape)
    item_embeddings = tf.slice(net, [0, user_field_size*embedding_size], [-1, -1])
    # print("item shape:", item_embeddings.shape)

    # autoint multi-head self-attention
    autoint_emb = tf.reshape(user_embeddings, shape=[-1, user_field_size, embedding_size])
    with tf.name_scope("AutoInt"):
        for i in range(params["autoint_layer_count"]):
            autoint_emb = InteractingLayer(num_layer=i, att_emb_size=params["autoint_emb_size"], seed=2020,
                                           head_num=params["autoint_head_count"], use_res=params["autoint_use_res"])(autoint_emb)
        autoint_emb = tf.layers.Flatten()(autoint_emb)

    user_embeddings = tf.concat([user_embeddings, autoint_emb], axis=1)

    for i in range(len(user_layers)):
        user_embeddings = tf.layers.dense(user_embeddings, user_layers[i], activation=tf.nn.relu)
        item_embeddings = tf.layers.dense(item_embeddings, item_layers[i], activation=tf.nn.relu)
        if mode == tf.estimator.ModeKeys.TRAIN:
            user_embeddings = tf.nn.dropout(user_embeddings, dropout[i])
            item_embeddings = tf.nn.dropout(item_embeddings, dropout[i])

    normalize_ub = tf.nn.l2_normalize(user_embeddings, axis=1)
    normalize_ib = tf.nn.l2_normalize(item_embeddings, axis=1)
    cos_similarity = tf.reduce_sum(tf.multiply(normalize_ub, normalize_ib), axis=1)
    # print("cos shape:", cos_similarity.shape)
    # print("label shape:", labels.shape)

    predictions = {"user_embedding": normalize_ub, "item_embedding": normalize_ib}

    export_outputs = {tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: tf.estimator.export.PredictOutput(predictions)}
    # Provide an estimator spec for `ModeKeys.PREDICT`
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            export_outputs=export_outputs)

    loss = tf.losses.log_loss(labels=labels, predictions=cos_similarity)

    # Provide an estimator spec for `ModeKeys.EVAL`
    eval_metric_ops = {
        "auc": tf.metrics.auc(labels, cos_similarity)
    }
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss,
            eval_metric_ops=eval_metric_ops)

    # ------bulid optimizer------
    if FLAGS.optimizer == 'Adam':
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    elif FLAGS.optimizer == 'Adagrad':
        optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate)
    elif FLAGS.optimizer == 'Momentum':
        optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.95)
    elif FLAGS.optimizer == 'ftrl':
        optimizer = tf.train.FtrlOptimizer(learning_rate)
    elif FLAGS.optimizer == 'Adadelta':
        optimizer = tf.train.AdadeltaOptimizer(learning_rate)

    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

    # Provide an estimator spec for `ModeKeys.TRAIN` modes
    if mode == tf.estimator.ModeKeys.TRAIN:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss,
            train_op=train_op)


def main(_):
    # ------check Arguments------
    print('task_type ', FLAGS.task_type)
    print('model_dir ', FLAGS.model_dir)
    print('data_dir ', FLAGS.data_dir)
    print('train_data', FLAGS.train_data)
    print('val_data', FLAGS.val_data)
    print('test_data', FLAGS.test_data)
    print('num_epochs ', FLAGS.num_epochs)
    print('embedding_size ', FLAGS.embedding_size)
    print('batch_size ', FLAGS.batch_size)
    print('user_layers ', FLAGS.user_deep_layers)
    print('item_layers ', FLAGS.item_deep_layers)
    print('dropout ', FLAGS.dropout)
    print('optimizer ', FLAGS.optimizer)
    print('learning_rate ', FLAGS.learning_rate)

    # ------init Envs------
    tr_files = glob.glob("%s*" % FLAGS.train_data)
    random.shuffle(tr_files)
    print("tr_files:", tr_files)
    va_files = glob.glob("%s*" % FLAGS.val_data)
    print("va_files:", va_files)
    te_files = glob.glob("%s*" % FLAGS.test_data)
    print("te_files:", te_files)

    if FLAGS.clear_existing_model:
        try:
            shutil.rmtree(FLAGS.model_dir)
        except Exception as e:
            print(e, "at clear_existing_model")
        else:
            print("existing model cleaned at %s" % FLAGS.model_dir)

    columns = build_model_columns(FLAGS.embedding_size)

    # ------bulid Tasks------
    model_params = {
        "user_field_size": FLAGS.user_field_size,
        "item_field_size": FLAGS.item_field_size,
        "embedding_size": FLAGS.embedding_size,
        "learning_rate": FLAGS.learning_rate,
        "user_deep_layers": FLAGS.user_deep_layers,
        "item_deep_layers": FLAGS.item_deep_layers,
        "dropout": FLAGS.dropout,
        "feature_columns": columns,
        "autoint_layer_count": FLAGS.autoint_layer_count,
        "autoint_emb_size": FLAGS.autoint_emb_size,
        "autoint_head_count": FLAGS.autoint_head_count,
        "autoint_use_res": FLAGS.autoint_use_res
    }
    config = tf.estimator.RunConfig().replace(session_config=tf.ConfigProto(device_count={'GPU': 1, 'CPU': FLAGS.num_threads}, gpu_options=tf.GPUOptions(allow_growth=True), inter_op_parallelism_threads=FLAGS.num_threads, intra_op_parallelism_threads=FLAGS.num_threads),
                                              log_step_count_steps=FLAGS.log_steps, save_summary_steps=FLAGS.log_steps, save_checkpoints_steps=1000, keep_checkpoint_max=3)
    # dataset = predict_input_fn(FLAGS.test_data, batch_size=1)
    # with tf.Session() as sess:
    #    for i in range(3):
    #        print(sess.run(dataset))
    model = tf.estimator.Estimator(model_fn=model_fn, model_dir=FLAGS.model_dir, params=model_params, config=config)

    if FLAGS.task_type == 'train':
        model.evaluate(input_fn=lambda: input_fn(va_files, num_epochs=1, batch_size=FLAGS.batch_size))
        model.train(input_fn=lambda: input_fn(tr_files, num_epochs=FLAGS.num_epochs, batch_size=FLAGS.batch_size), steps=1000000)  # , hooks=[hook])
        feature_spec = tf.feature_column.make_parse_example_spec(columns)
        serving_input_receiver_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec)
        model.export_savedmodel(FLAGS.servable_model_dir, serving_input_receiver_fn)

    elif FLAGS.task_type == 'eval':
        model.evaluate(input_fn=lambda: input_fn(va_files, num_epochs=1, batch_size=FLAGS.batch_size))

    elif FLAGS.task_type == 'infer':
        print("=====================")
        preds = model.predict(input_fn=lambda: input_fn(FLAGS.test_data, batch_size=FLAGS.batch_size), predict_keys=["item_embedding"])
        f = open(FLAGS.test_data, "rb")
        with open(FLAGS.infer_result, "w") as fo:
            for line, p in zip(f, preds):
                id = line.decode("utf-8").split(" ")[10]
                # print(p)
                # fo.write("%f\n" % (prob['embedding']))
                emb = ','.join(["%.6f" % f for f in list(p["item_embedding"])])
                fo.write(id + "|" + id + "|" + emb + "\n")


if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
