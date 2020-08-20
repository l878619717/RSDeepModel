# -*- coding:utf-8 -*
'''
功能: 1.通用工具类函数 2.解析配置文件lr.conf&dnn.conf
'''
import datetime
import os
import tensorflow as tf


def arg_parse(argv):
    parse_dict = dict()
    for i in range(1, len(argv)):
        line_parse = argv[i].split("=")
        key = line_parse[0].strip()
        value = line_parse[1].strip()
        parse_dict[key] = value
    return parse_dict


def venus_set_environ():
    # venus 参数设置
    os.environ['AWS_ACCESS_KEY_ID'] = "J2SU8BKYDQAKTKBHY1DV"
    os.environ['AWS_SECRET_ACCESS_KEY'] = "4icFh3queHjR2jPk8U2j7qM1ekw7HpGQqkVPDgZ4"
    os.environ['S3_ENDPOINT'] = "s3szoffline.sumeru.mig"
    os.environ['S3_USE_HTTPS'] = '0'
    os.environ['AWS_DEFAULT_REGION'] = 'default'


def get_file_list(input_path):
    file_list = tf.gfile.ListDirectory(input_path)
    print("file_list_len:", len(file_list))
    file_dir_list = []
    for file in file_list:
        if file[:4] == "part":
            file_path = input_path + file
            file_dir_list.append(file_path)

    return file_dir_list


def shift_date_time(dt_time, offset_day, time_structure='%Y%m%d'):
    dt = datetime.datetime(int(dt_time[0:4]), int(dt_time[4:6]),
                           int(dt_time[6:8]))
    delta = datetime.timedelta(days=offset_day)
    del_day_date = dt + delta
    del_day_time = del_day_date.strftime(time_structure)
    return del_day_time


def shift_hour_time(dt_time, offset_hour, time_structure='%Y%m%d%H'):
    dt = datetime.datetime(int(dt_time[0:4]), int(dt_time[4:6]),
                           int(dt_time[6:8]), int(dt_time[8:10]))
    delta = datetime.timedelta(hours=offset_hour)
    del_date = dt + delta
    del_time = del_date.strftime(time_structure)
    return del_time


def parse_feats_conf(conf_path, alg_name):  # 解析配置文件
    cont_field_count = 0
    cate_field_count = 0
    multi_cate_field_count = 0
    multi_cate_field_list = list()  # [(multi_cate_feat_name, topN)]
    total_field_count = 0
    # target-attention
    target_att_alg_name = ['din', 'dinfm']
    target_att_item_single = dict()  # {attention-id: item-index}  single_cate_feats(e.g. item_cat1) was merged into one column named 'cate_feats', so we need record the index
    target_att_item = dict()  # {attention-id: item-name} multi_cate_feats(e.g. item_tags) are independent columns, so it is enough to record column names
    target_att_user = dict()  # {attention-id: user-name} multi_cate_feats(e.g. user_click_cat1) are independent columns, so it is enough to record column names
    target_att_1vN_list = list()  # target-attention: 1-n [[user-name1, item-index1], [user-name2, item-index2], ...]
    target_att_NvN_list = list()  # target-attention: n-n [[user-name1, item-name1], [user-name2, item-name2], ...]

    files = os.listdir(conf_path)
    for file in files:
        if file != "dnn.conf" and file != "lr.conf":
            continue
        file_path = conf_path + "/" + file
        with open(file_path, 'r') as f:
            for line in f.readlines():
                line_data = line.strip()
                if line_data == '':
                    continue
                try:
                    config_arr = line_data.split("\t")
                    col_name = config_arr[0]
                    new_col_name = config_arr[1]
                    if new_col_name == "none":
                        new_col_name = col_name
                    result_type = config_arr[2]

                    is_drop = int(config_arr[8])
                    att_id = config_arr[10]  # attention_id

                    if is_drop == 1:
                        continue

                    total_field_count += 1

                    if result_type == 'arr':  # multi_cate
                        func_parse = config_arr[5]  # pre_parse_func = config_arr[4]
                        result_parse = config_arr[7]  # result_parse_func = config_arr[6]
                        # target-attention
                        if alg_name in target_att_alg_name and att_id != '0':
                            feat_type = col_name.split('_')[0]
                            if feat_type == "item":
                                target_att_item[att_id] = new_col_name
                            else:  # "user"
                                target_att_user[att_id] = new_col_name

                        # multi_cate_feats
                        top_n = result_parse if func_parse == 'none' else func_parse
                        multi_cate_field_list.append((new_col_name, int(top_n.split("=")[1])))
                        multi_cate_field_count += 1

                    elif result_type == 'string':  # single_cate
                        # target-attention
                        if alg_name in target_att_alg_name and att_id != '0':
                            feat_type = col_name.split('_')[0]
                            if feat_type == "item":
                                target_att_item_single[att_id] = cate_field_count

                        # single_cate_feats
                        cate_field_count += 1

                    elif result_type == 'float':  # cont
                        cont_field_count += 1
                    else:
                        print("%s is error!!!" % line_data)
                except Exception as e:
                    print("-----------feat_conf is Error!!!!-----------")
                    print(e)
                    print(line_data)
                    exit(-1)

    parse_feats_dict = dict()
    parse_feats_dict["cont_field_count"] = cont_field_count
    parse_feats_dict["cate_field_count"] = cate_field_count
    parse_feats_dict["multi_cate_field_count"] = multi_cate_field_count
    parse_feats_dict["multi_cate_field_list"] = multi_cate_field_list
    parse_feats_dict["total_field_count"] = total_field_count
    # target-attention
    for k, v in target_att_user.items():
        if k in target_att_item_single.keys():
            target_att_1vN_list.append((v, target_att_item_single[k]))  # e.g. [(user_click_cat1, 3), ...], '3' means 'item_cat1' is the third feat of 'cate_feats'
    parse_feats_dict["target_att_1vN_list"] = target_att_1vN_list
    for k, v in target_att_user.items():
        if k in target_att_item.keys():
            target_att_NvN_list.append((v, target_att_item[k]))  # e.g. [(user_click_tags: item_tags), ...], both are multi_feats
    parse_feats_dict["target_att_NvN_list"] = target_att_NvN_list

    return parse_feats_dict
