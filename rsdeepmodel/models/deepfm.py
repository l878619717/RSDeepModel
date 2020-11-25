# -*- coding:utf-8 -*-

import tensorflow as tf
import utils.model_op as op


def model_fn(features, labels, mode, params):
    tf.set_random_seed(2020)

    cont_feats = features["cont_feats"]
    cate_feats = features["cate_feats"]
    cont_feats_index = tf.Variable([[i for i in range(params.cont_field_count)]], trainable=False, dtype=tf.int64, name="cont_feats_index")
    cont_feats_index = tf.add(cont_feats_index, params.cate_emb_space_size)

    feats_size = params.cont_field_count + params.cate_emb_space_size
    weights = dict()
    weights["feats_emb"] = tf.get_variable(name='feats_emb', shape=[feats_size, params.embedding_size], initializer=tf.glorot_normal_initializer())
    weights["feats_bias"] = tf.get_variable(name='feats_bias', shape=[feats_size, 1], initializer=tf.glorot_normal_initializer())

    with tf.name_scope('fm_part'):
        # FM_first_order [?, total_field_count]
        first_cont_emb = tf.nn.embedding_lookup(weights["feats_bias"], ids=cont_feats_index)  # [None, F, 1]
        first_cont_emb = tf.reshape(first_cont_emb, shape=[-1, params.cont_field_count])
        first_order = tf.multiply(first_cont_emb, cont_feats)
        first_cate_emb = tf.nn.embedding_lookup(weights["feats_bias"], ids=cate_feats)
        first_cate_emb = tf.reshape(first_cate_emb, shape=[-1, params.cate_field_count])
        first_order = tf.concat([first_order, first_cate_emb], axis=1)
        if params.multi_feats_type == 'dense':
            for name_topN in params.multi_cate_field_list:
                dense_embedding = tf.nn.embedding_lookup(weights["feats_bias"], ids=features[name_topN[0]])  # [None, topN, 1]
                dense_embedding = tf.reduce_sum(dense_embedding, axis=1)  # [None, 1]
                first_order = tf.concat([first_order, dense_embedding], axis=1)
        else:  # sparse
            for name_topN in params.multi_cate_field_list:
                sparse_embedding = tf.nn.embedding_lookup_sparse(weights["feats_bias"], sp_ids=features[name_topN[0]], sp_weights=None, combiner='sum')  # [None, 1]
                first_order = tf.concat([first_order, sparse_embedding], axis=1)

        # FM_second_order [?, embedding_size]
        second_cont_emb = tf.nn.embedding_lookup(weights["feats_emb"], ids=cont_feats_index)  # [None, F, embedding_size]
        second_cont_value = tf.reshape(cont_feats, shape=[-1, params.cont_field_count, 1])
        embeddings = tf.multiply(second_cont_emb, second_cont_value)
        second_cate_emb = tf.nn.embedding_lookup(weights["feats_emb"], ids=cate_feats)
        embeddings = tf.concat([embeddings, second_cate_emb], axis=1)  # [None, F, embedding_size]
        if params.multi_feats_type == 'dense':
            for name_topN in params.multi_cate_field_list:
                dense_embedding = tf.nn.embedding_lookup(weights["feats_emb"], ids=features[name_topN[0]])  # [None, topN, embedding_size]
                dense_embedding = tf.reduce_sum(dense_embedding, axis=1)  # [None, embedding_size]
                dense_embedding = tf.reshape(dense_embedding, shape=[-1, 1, params.embedding_size])
                embeddings = tf.concat([embeddings, dense_embedding], axis=1)
        else:  # sparse
            for name_topN in params.multi_cate_field_list:
                sparse_embedding = tf.nn.embedding_lookup_sparse(weights["feats_emb"], sp_ids=features[name_topN[0]],
                                                                 sp_weights=None, combiner='sum')  # [None, embedding_size]
                sparse_embedding = tf.reshape(sparse_embedding, shape=[-1, 1, params.embedding_size])
                embeddings = tf.concat([embeddings, sparse_embedding], axis=1)

        sum_emb = tf.reduce_sum(embeddings, 1)
        sum_square_emb = tf.square(sum_emb)
        square_emb = tf.square(embeddings)
        square_sum_emb = tf.reduce_sum(square_emb, 1)
        second_order = 0.5 * tf.subtract(sum_square_emb, square_sum_emb)
        # print(second_order)

        # FM_res [?, total_field_count + embedding_size]
        fm_res = tf.concat([first_order, second_order], axis=1)

    with tf.name_scope('deep_part'):
        deep_res = tf.reshape(embeddings, shape=[-1, params.total_field_count * params.embedding_size])
        len_layers = len(params.hidden_units)
        for i in range(0, len_layers):
            deep_res = tf.layers.dense(inputs=deep_res, units=params.hidden_units[i], activation=tf.nn.relu)

    with tf.name_scope('deep_fm'):
        input_size = params.total_field_count + params.embedding_size + params.hidden_units[-1]  # FM_res->total_field_count+embedding_size, Deep_res->hidden_units[-1]
        weights["concat_emb"] = tf.get_variable(name='concat_emb', shape=[input_size, 1], initializer=tf.glorot_normal_initializer())
        weights["concat_bias"] = tf.get_variable(name='concat_bias', shape=[1], initializer=tf.constant_initializer(0.01))

        feats_input = tf.concat([fm_res, deep_res], axis=1)
        out = tf.add(tf.matmul(feats_input, weights["concat_emb"]), weights["concat_bias"])
        y = tf.nn.sigmoid(out)

    model_estimator_spec = op.model_optimizer(params, mode, labels, y)

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
