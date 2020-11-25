# -*- coding:utf-8 -*-

import tensorflow as tf

import utils.model_layer as my_layer
import utils.model_op as op


def model_fn(features, labels, mode, params):
    tf.set_random_seed(2020)

    cont_feats = features["cont_feats"]
    cate_feats = features["cate_feats"]

    cont_feats_index = tf.Variable([[i for i in range(params.cont_field_count)]], trainable=False, dtype=tf.int64, name="cont_feats_index")
    cont_feats_index = tf.add(cont_feats_index, params.cate_emb_space_size)

    feats_size = params.cont_field_count + params.cate_emb_space_size
    feats_emb = my_layer.emb_init(name='feats_emb', feat_num=feats_size, embedding_size=params.embedding_size)

    cont_emb = tf.nn.embedding_lookup(feats_emb, ids=cont_feats_index)  # [None, F, embedding_size]
    cont_value = tf.reshape(cont_feats, shape=[-1, params.cont_field_count, 1])
    embeddings = tf.multiply(cont_emb, cont_value)
    cate_emb = tf.nn.embedding_lookup(feats_emb, ids=cate_feats)
    embeddings = tf.concat([embeddings, cate_emb], axis=1)
    if params.multi_feats_type == 'dense':
        for name_topN in params.multi_cate_field_list:
            dense_embedding = tf.nn.embedding_lookup(feats_emb, ids=features[name_topN[0]])  # [None, topN, embedding_size]
            dense_embedding = tf.reduce_sum(dense_embedding, axis=1)  # [None, embedding_size]
            dense_embedding = tf.reshape(dense_embedding, shape=[-1, 1, params.embedding_size])
            embeddings = tf.concat([embeddings, dense_embedding], axis=1)
    else:  # sparse
        for name_topN in params.multi_cate_field_list:
            sparse_embedding = tf.nn.embedding_lookup_sparse(feats_emb, sp_ids=features[name_topN[0]],
                                                             sp_weights=None, combiner='sum')  # [None, embedding_size]
            sparse_embedding = tf.reshape(sparse_embedding, shape=[-1, 1, params.embedding_size])
            embeddings = tf.concat([embeddings, sparse_embedding], axis=1)

    deep_emb = tf.reshape(embeddings, shape=[-1, params.total_field_count * params.embedding_size])
    wide_emb = deep_emb

    # AutoInteracting
    with tf.name_scope("AutoInt"):
        for i in range(params.autoint_layer_count):
            embeddings = my_layer.InteractingLayer(num_layer=i, att_emb_size=params.autoint_emb_size, seed=2020,
                                                   head_num=params.autoint_head_count, use_res=params.autoint_use_res)(embeddings)
        autoint_emb = tf.layers.Flatten()(embeddings)

    # deep
    len_layers = len(params.hidden_units)
    for i in range(0, len_layers):
        deep_emb = tf.layers.dense(inputs=deep_emb, units=params.hidden_units[i], activation=tf.nn.relu)

    # autoint & deep & wide
    output_embeddings = tf.concat([autoint_emb, deep_emb, wide_emb], axis=1)
    out = tf.layers.dense(inputs=output_embeddings, units=1)
    score = tf.identity(tf.nn.sigmoid(out), name='score')
    model_estimator_spec = op.model_optimizer(params, mode, labels, score)
    return model_estimator_spec


def model_estimator(params):
    tf.reset_default_graph()
    config = tf.estimator.RunConfig() \
        .replace(
        session_config=tf.ConfigProto(device_count={'GPU': params.is_GPU, 'CPU': params.num_threads}, gpu_options=tf.GPUOptions(allow_growth=True),
                                      inter_op_parallelism_threads=params.num_threads,
                                      intra_op_parallelism_threads=params.num_threads),
        log_step_count_steps=params.log_step_count_steps,
        save_checkpoints_steps=params.save_checkpoints_steps,
        keep_checkpoint_max=params.keep_checkpoint_max,
        save_summary_steps=params.save_summary_steps)

    model = tf.estimator.Estimator(
        model_fn=model_fn,
        config=config,
        model_dir=params.model_dir,
        params=params,
    )
    return model
