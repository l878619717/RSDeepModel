# -*- coding:utf-8 -*-

import tensorflow as tf

import utils.model_layer as my_layer
import utils.model_op as op


def cosine_similarity(user_vec, item_vec):
    # user·item
    user_dot_item = tf.reduce_sum(tf.multiply(user_vec, item_vec), axis=1)
    # |user|×|item|
    user_norm = tf.sqrt(tf.reduce_sum(tf.square(user_vec), axis=1))
    item_norm = tf.sqrt(tf.reduce_sum(tf.square(item_vec), axis=1))
    user_mul_item = tf.multiply(user_norm, item_norm)
    # cosine similarity
    cos_sim = tf.truediv(user_dot_item, user_mul_item)
    score = tf.reshape(cos_sim, shape=[-1, 1])
    return score


def tower(input_emb, params, w_name_input):
    flatten_input_emb = tf.layers.flatten(input_emb)
    deep_emb = flatten_input_emb
    wide_emb = flatten_input_emb
    # AutoInteracting
    with tf.name_scope("AutoInt"):
        for i in range(params.autoint_layer_count):
            input_emb = my_layer.InteractingLayer(num_layer=i, w_name=w_name_input, att_emb_size=params.autoint_emb_size, seed=2020,
                                                  head_num=params.autoint_head_count, use_res=params.autoint_use_res)(input_emb)
        autoint_emb = tf.layers.Flatten()(input_emb)

    # deep
    len_layers = len(params.hidden_units)
    for i in range(0, len_layers):
        deep_emb = tf.layers.dense(inputs=deep_emb, units=params.hidden_units[i], activation=tf.nn.relu)

    # autoint & deep & wide
    all_emb = tf.concat([autoint_emb, deep_emb, wide_emb], axis=1)
    out = tf.layers.dense(inputs=all_emb, units=params.hidden_units[-1], activation=tf.nn.relu)
    return out


def model_fn(features, labels, mode, params):
    tf.set_random_seed(2020)

    item_cont_feats = features["item_cont_feats"]
    item_cate_feats = features["item_cate_feats"]
    # user_cont_feats = features["user_cont_feats"] # 暂时没有
    user_cate_feats = features["user_cate_feats"]

    # init emb
    item_cont_feats_index = tf.Variable([[i for i in range(params.item_cont_field_count)]], trainable=False, dtype=tf.int64, name="item_cont_feats_index")
    item_cont_feats_index = tf.add(item_cont_feats_index, params.cate_emb_space_size)
    feats_size = params.item_cont_field_count + params.cate_emb_space_size
    feats_emb = my_layer.emb_init(name='feats_emb', feat_num=feats_size, embedding_size=params.embedding_size)

    # item_emb
    item_cont_emb = tf.nn.embedding_lookup(feats_emb, ids=item_cont_feats_index)  # [None, F, embedding_size]
    item_cont_value = tf.reshape(item_cont_feats, shape=[-1, params.item_cont_field_count, 1])
    item_embeddings = tf.multiply(item_cont_emb, item_cont_value)
    item_cate_emb = tf.nn.embedding_lookup(feats_emb, ids=item_cate_feats)
    item_embeddings = tf.concat([item_embeddings, item_cate_emb], axis=1)
    for name_topN in params.multi_cate_field_list:
        if name_topN[0] in ["item_topic_keys", "item_tags_keys", "item_tags_ext_keys"]:  # item
            sparse_embedding = tf.nn.embedding_lookup_sparse(feats_emb, sp_ids=features[name_topN[0]],
                                                             sp_weights=None, combiner='sum')  # [None, embedding_size]
            sparse_embedding = tf.reshape(sparse_embedding, shape=[-1, 1, params.embedding_size])
            item_embeddings = tf.concat([item_embeddings, sparse_embedding], axis=1)
    # user_emb
    user_embeddings = tf.nn.embedding_lookup(feats_emb, ids=user_cate_feats)
    for name_topN in params.multi_cate_field_list:
        if name_topN[0] in ["user_month_cat1_keys", "user_month_cat2_keys", "user_month_media_id_keys", "user_month_tags_keys",
                            "ruCat1ScoreKeys", "ruCat2ScoreKeys", "ruMediaIdScoreKeys", "ruTagsScoreKeys"]:  # user
            sparse_embedding = tf.nn.embedding_lookup_sparse(feats_emb, sp_ids=features[name_topN[0]],
                                                             sp_weights=None, combiner='sum')  # [None, embedding_size]
            sparse_embedding = tf.reshape(sparse_embedding, shape=[-1, 1, params.embedding_size])
            user_embeddings = tf.concat([user_embeddings, sparse_embedding], axis=1)

    # item vec
    item_vec = tf.identity(tower(item_embeddings, params, "item"), name='item_vec')
    # user vec
    user_vec = tf.identity(tower(user_embeddings, params, "user"), name='user_vec')
    # cosine_similarity
    score = tf.identity(cosine_similarity(user_vec, item_vec), name='score')

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
