# -*- coding:utf-8 -*-

import tensorflow as tf
import shutil
from utils import my_utils, data_load, model_op
from models import dnn, deepfm, din, dinfm, autoint

#################### CMD Arguments ####################
FLAGS = tf.flags.FLAGS
# ----------------common------------------
tf.flags.DEFINE_string("alg_name", "deepfm", "algorithm name")
tf.flags.DEFINE_string("task_mode", 'train', "task mode type {train, eval, infer, debug}")
tf.flags.DEFINE_integer("epochs", 1, "epochs of training")
tf.flags.DEFINE_integer("batch_size", 64, "Number of batch size")
tf.flags.DEFINE_integer("embedding_size", 16, "Embedding size")
tf.flags.DEFINE_list("hidden_units", [512, 256, 128], "the layers of dnn")
tf.flags.DEFINE_string("optimizer", 'Adam', "optimizer type {Adam, Adagrad, GD, Momentum}")
tf.flags.DEFINE_float("learning_rate", 0.001, "learning rate")
tf.flags.DEFINE_integer("learning_rate_decay_steps", 10000000, "")
tf.flags.DEFINE_float("learning_rate_decay_rate", 0.9, "")
tf.flags.DEFINE_list("dropout", [1, 1, 1, 1, 1], "dropout rate")
tf.flags.DEFINE_float("l2_reg", 0.00001, "")
tf.flags.DEFINE_boolean("clear_existing_model", True, "clear existing model or not")
# ----------------data(tfrecord)--------------------
tf.flags.DEFINE_string("multi_feats_type", 'dense', "multi_cate_feats type {dense, sparse}")  # for notes, see utils/data_load.py
tf.flags.DEFINE_string("train_data", 'data/tfrecord/demo/demo-' + FLAGS.multi_feats_type, "the path of train data")
tf.flags.DEFINE_string("eval_data", FLAGS.train_data, "the path of eval data") 
if FLAGS.alg_name in ['din', 'dinfm'] and FLAGS.multi_feats_type == 'sparse':  # target-attention函数的输入是dense_tensor, 所以多值离散特征需补齐缺失值, 不支持sparse
    FLAGS.multi_feats_type = 'sparse2dense'
# ----------------model_save--------------------
tf.flags.DEFINE_string("model_pb", "data/model_save_pb/" + FLAGS.alg_name, "the path for exporting model pb after training")
tf.flags.DEFINE_string("model_pb_type", "parsing", "model_pb export type {raw, parsing}")
tf.flags.DEFINE_string("model_dir", "data/model_save_dir/" + FLAGS.alg_name, "the path for saving model checkpoint between training")
tf.flags.DEFINE_integer("keep_checkpoint_max", 3, "")
tf.flags.DEFINE_integer("save_checkpoints_steps", 100, "")
tf.flags.DEFINE_integer("log_step_count_steps", 100, "")
tf.flags.DEFINE_integer("save_summary_steps", 500, "save summary every steps")
# ----------------device-----------------
tf.flags.DEFINE_integer("is_GPU", 1, "use GPU or not, 1->yes, 0->no")
tf.flags.DEFINE_integer("num_cpu", 20, "Number of CPU")
tf.flags.DEFINE_integer("num_threads", 1, "Number of threads")
# ---------------features----------------
tf.flags.DEFINE_integer("cate_emb_space_size", 9000000, "")
if FLAGS.task_mode == 'debug':  # make some fake data for debugging !!!开发中，暂时不可用
    tf.flags.DEFINE_string("debug_data", 'data/tfrecord/debug/', "the path of debug data")
    tf.flags.DEFINE_integer("cont_field_count", 3, "")
    tf.flags.DEFINE_integer("cate_field_count", 5, "")
    tf.flags.DEFINE_integer("multi_cate_field_count", 2, "")
    tf.flags.DEFINE_list("multi_cate_field_list", [(m1, 3), (m2, 5)], "")
    tf.flags.DEFINE_integer("total_field_count", 10, "")
    tf.flags.DEFINE_list("target_att_1vN_list", [], "")
    tf.flags.DEFINE_list("target_att_NvN_list", [], "")
else:  # parse feature-config files
    tf.flags.DEFINE_string("feats_conf", 'data/tfrecord/demo/feats_conf', "the path of feature config files")
    parse_feats_dict = my_utils.parse_feats_conf(FLAGS.feats_conf, FLAGS.alg_name)
    tf.flags.DEFINE_integer("cont_field_count", parse_feats_dict['cont_field_count'], "")
    tf.flags.DEFINE_integer("cate_field_count", parse_feats_dict['cate_field_count'], "")
    tf.flags.DEFINE_integer("multi_cate_field_count", parse_feats_dict['multi_cate_field_count'], "")
    tf.flags.DEFINE_list("multi_cate_field_list", parse_feats_dict['multi_cate_field_list'], "")
    tf.flags.DEFINE_integer("total_field_count", parse_feats_dict['total_field_count'], "")
    tf.flags.DEFINE_list("target_att_1vN_list", parse_feats_dict['target_att_1vN_list'], "")
    tf.flags.DEFINE_list("target_att_NvN_list", parse_feats_dict['target_att_NvN_list'], "")

# ------------model special parameters-----------
# dssm
tf.flags.DEFINE_integer("user_field_size", 0, "Number of fields")
tf.flags.DEFINE_integer("item_field_size", 0, "Number of fields")
# autoint
tf.flags.DEFINE_integer("autoint_layer_count", 2, "")
tf.flags.DEFINE_integer("autoint_emb_size", 16, "")
tf.flags.DEFINE_integer("autoint_head_count", 2, "")
tf.flags.DEFINE_integer("autoint_use_res", 1, "")


def check_arguments():
    print("-----------check Arguments------------START")
    print('alg_name: ', FLAGS.alg_name)
    print('task_mode: ', FLAGS.task_mode)
    print('model_dir: ', FLAGS.model_dir)
    print('model_pb: ', FLAGS.model_pb)
    print('clear_existing_model: ', FLAGS.clear_existing_model)
    print('train_data: ', FLAGS.train_data)
    print('eval_data: ', FLAGS.eval_data)
    print('epochs: ', FLAGS.epochs)
    print('embedding_size: ', FLAGS.embedding_size)
    print('batch_size: ', FLAGS.batch_size)
    print('dropout: ', FLAGS.dropout)
    print('optimizer: ', FLAGS.optimizer)
    print('learning_rate: ', FLAGS.learning_rate)
    print("total_field_count: ", FLAGS.total_field_count)
    print("cont_field_count: ", FLAGS.cont_field_count)
    print("cate_field_count: ", FLAGS.cate_field_count)
    print("multi_cate_field_count: ", FLAGS.multi_cate_field_count)
    print("multi_cate_field_list('feats_name', topN): ", FLAGS.multi_cate_field_list)
    print("multi_feats_type: ", FLAGS.multi_feats_type)
    print("target_att_1vN_list('user_feats_name', single_cate_index): ", FLAGS.target_att_1vN_list)
    print("target_att_NvN_list('user_feats_name', item_feats_name): ", FLAGS.target_att_NvN_list)
    print("-----------check Arguments-------------END")


def main(_):
    check_arguments()

    if FLAGS.clear_existing_model:
        try:
            shutil.rmtree(FLAGS.model_dir)
        except Exception as e:
            print(e, "at clear_existing_model")
        else:
            print("existing model cleaned at %s" % FLAGS.model_dir)

    if FLAGS.alg_name == 'dnn':
        model = dnn.model_estimator(FLAGS)
    elif FLAGS.alg_name == 'deepfm':
        model = deepfm.model_estimator(FLAGS)
    elif FLAGS.alg_name == 'din':
        model = din.model_estimator(FLAGS)
    elif FLAGS.alg_name == 'dinfm':
        model = dinfm.model_estimator(FLAGS)
    elif FLAGS.alg_name == 'autoint':
        model = autoint.model_estimator(FLAGS)
    else:
        print("ERROR!!! alg_name = %s is not exit!" % FLAGS.alg_name)
        exit(-1)

    if FLAGS.task_mode == 'train':
        # model.evaluate(input_fn=lambda: data_load.input_fn(FLAGS.eval_data, FLAGS))
        model.train(input_fn=lambda: data_load.input_fn(FLAGS.train_data, FLAGS))
        model_op.model_save_pb(FLAGS, model)

    elif FLAGS.task_mode == 'eval':
        model.evaluate(input_fn=lambda: data_load.input_fn(FALGS.eval_data, FLAGS))

    elif FLAGS.task_mode == 'infer':
        pass
        # preds = model.predict(input_fn=lambda: data_load.input_fn(FLAGS.eval_data, FLAGS), predict_keys=["item_embedding"])
        # f = open(FLAGS.test_data, "rb")
        # with open(FLAGS.infer_result, "w") as fo:
        #     for line, p in zip(f, preds):
        #         id = line.decode("utf-8").split(" ")[10]
        #         # print(p)
        #         # fo.write("%f\n" % (prob['embedding']))
        #         emb = ','.join(["%.6f" % f for f in list(p["item_embedding"])])
        #         fo.write(id + "|" + id + "|" + emb + "\n")

    elif FLAGS.task_mode == 'debug':
        pass

    else:
        print("Task_mode Error!")


if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
