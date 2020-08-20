# -*- coding:utf-8 -*-
import tensorflow as tf

import utils.model_layer as my_layer
import utils.model_op as op


def model_fn(labels, features, mode, params):
    tf.set_random_seed(2020)
    cont_feats = features["cont_feats"]
    cate_feats = features["cate_feats"]

    cont_feats_index = tf.Variable([[i for i in range(params.cont_field_count)]], trainable=False, dtype=tf.int64, name="cont_feats_index")
    cont_feats_index = tf.add(cont_feats_index, params.cate_emb_space_size)

    feats_size = params.cont_field_count + params.cate_emb_space_size
    feats_emb = my_layer.emb_init(name='feats_emb', feat_num=feats_size, embedding_size=params.embedding_size)

    # cont_feats
    cont_emb = tf.nn.embedding_lookup(feats_emb, ids=cont_feats_index)  # None * F * embedding_size
    cont_value = tf.reshape(cont_feats, shape=[-1, params.cont_field_count, 1])
    embeddings = tf.multiply(cont_emb, cont_value)
    # cate_feats
    cate_emb = tf.nn.embedding_lookup(feats_emb, ids=cate_feats)
    embeddings = tf.concat([embeddings, cate_emb], axis=1)
    # multi_cate_feats
    for name_topN in params.multi_cate_field_list:
        multi_cate_emb = tf.nn.embedding_lookup(feats_emb, ids=features[name_topN[0]])  # None, topN, embedding_size
        multi_cate_emb = tf.reduce_sum(multi_cate_emb, axis=1)  # None, embedding_size
        multi_cate_emb = tf.reshape(multi_cate_emb, shape=[-1, 1, params.embedding_size])
        embeddings = tf.concat([embeddings, multi_cate_emb], axis=1)
    # target-attention 1vN (e.g. item_cat1 & user_click_cat1)
    for k_v in params.target_att_1vN_list:
        item_feat = tf.split(cate_feats, params.cate_field_count, axis=1)[k_v[1]]
        user_feat = features[k_v[0]]
        nonzero_len = tf.count_nonzero(user_feat, axis=1)
        item_emb = tf.nn.embedding_lookup(feats_emb, ids=item_feat)  # [B, 1, H]
        item_emb = tf.reshape(item_emb, shape=[-1, params.embedding_size])  # [B, H]
        user_emb = tf.nn.embedding_lookup(feats_emb, ids=user_feat)  # [B, T, H])
        att_1vN_emb = my_layer.attention(item_emb, user_emb, nonzero_len)  # [B, 1, H]
        embeddings = tf.concat([embeddings, att_1vN_emb], axis=1)
    # target-attention NvN (e.g. item_tags & user_click_tags)
    for k_v in params.target_att_NvN_list:
        item_feat = features[k_v[1]]
        user_feat = features[k_v[0]]
        nonzero_len = tf.count_nonzero(user_feat, axis=1)
        item_emb = tf.nn.embedding_lookup(feats_emb, ids=item_feat)  # [B, N, H]
        user_emb = tf.nn.embedding_lookup(feats_emb, ids=user_feat)  # [B, T, H])
        att_NvN_emb = my_layer.attention_multi(item_emb, user_emb, nonzero_len)  # [B, 1, H]
        embeddings = tf.concat([embeddings, att_NvN_emb], axis=1)

    # deep
    embeddings = tf.layers.flatten(embeddings)
    len_layers = len(params.hidden_units)
    for i in range(0, len_layers):
        embeddings = tf.layers.dense(inputs=embeddings, units=params.hidden_units[i], activation=tf.nn.relu)
    out = tf.layers.dense(inputs=embeddings, units=1)
    score = tf.identity(tf.nn.sigmoid(out), name='score')
    model_estimator_spec = op.model_optimizer(params, mode, labels, score)
    return model_estimator_spec


def model_estimator(params):
    # shutil.rmtree(conf.model_dir, ignore_errors=True)
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
