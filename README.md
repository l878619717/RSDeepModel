# RSDeepModel

深度学习模型（推荐场景CTR预估为背景）, 使用estimator进行封装, 包括dnn、deepfm、din、dinfm、autoint等模型.

| -          | version |
| :--------- | :-----: |
| Python     |   3.6   |
| TensorFlow | 1.12.0  |

***

## 一、快速开始

打开程序入口文件rsdeepmodel/main.py, 自定义相关参数, 直接运行即可, **本项目训练数据默认tfrecord格式**.

| -              | -                                           |
| :------------- | :------------------------------------------ |
| 程序入口       | rsdeepmodel/main.py                         | - |
| 数据(tfrecord) | rsdeepmodel/data/tfrecord/demo/demo-*       |
| 数据配置文件   | rsdeepmodel/data/tfrecord/demo/feats_conf/* |
| 模型           | rsdeepmodel/models/*                        |
| 工具函数       | rsdeepmodel/utils/*                         |

ps: 本项目也提供了一个textline数据格式的demo(textline-model-demo/dssm_textline.py), 这里没有将数据处理、模型训练等拆分开, 所有内容都在一个脚本中, 直接运行即可.

**如果只关注模型侧工程, 可以不往下看了, 如果关心配置文件的用途、训练数据如何生成, 下面有较为详细的解释**.
***

## 二、数据与配置文件

本项目采取配置文件的形式, 提供的demo数据与配置文件示例是对应的, 因为数据是按照配置文件生成的.

配置文件同时用于线上、线下特征构造. 线上线下都会读取同一份配置文件, 使用逻辑相同的函数. 当需要'增/删/查/改'特征时, 线上线下无需再更改代码, 只需更新配置文件即可达到目的. 这样一来大大提高了迭代效率, 并有效保证线上线下特征一致性.

### 1. 配置文件

与**离线spark特征构造生成训练数据**和**线上特征构造用于预测**不同, 本项目解析配置文件是为了获取训练模型所需的特征field计数以及attention相关参数的, 详情见下：

```
# 脚本路径: rsdeepmodel/utils/my_utils.py
def parse_feats_conf(conf_path, alg_name):  # 解析配置文件
    ...
    return
```
配置文件示例见 rsdeepmodel/data/tfrecord/demo/feats_conf/*, 其中lr.conf记录单值离散和多值离散特征, dnn.conf记录连续型特征. 目前共12列, 根据自身需要进行增删, 对应更改解析配置文件脚本即可.

| 序号 | 名称               | 释义                          | 示例                        | 备注                                                                                                                                                            |
| ---- | ------------------ | ----------------------------- | --------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1    | name               | 列名                          | user_id、item_id、ctx_qudao | user特征前缀user_;item特征前缀item_;ctx特征前缀ctx_;                                                                                                            |
| 2    | new_name           | 新列名                        | none、item_id_new           | 'none'表示不更改, 如果需要修改、新增则写新列名                                                                                                                  |
| 3    | result_type        | 新生成特征的数据类型          | string, arr, float          | 只有string、arr、float三种数据类型                                                                                                                              |
| 4    | feature_type       | 特征类型                      | wide, deep                  | 连续型特征-deep, 离散特征-wide                                                                                                                                  |
| 5    | handle_func        | 特征交叉方法                  | match_pos_arr_arr、none     | 如果不需要特征交叉方法则none                                                                                                                                    |
| 6    | handle_parse       | 特征交叉方法参数              | top=10、none                | 对应上面的handle_func, 不需要则none                                                                                                                             |
| 7    | result_func        | 特征处理方法                  | top_key_arr、normal、none   | 如果不需要特征处理方法则none                                                                                                                                    |
| 8    | result_parse       | 特征处理参数                  | top=10、none                | 不需要则none                                                                                                                                                    |
| 9    | is_drop            | 是否丢弃该特征                | 0、1                        | 中间结果特征可丢弃                                                                                                                                              |
| 10   | set_id             | 特征id                        | df                          | 特征hash, 区分不同'域'的特征                                                                                                                                    |
| 11   | taget_attention_id | 该特征训练时是否使用attention | 0、1、2...                  | 使用target-attention的两个特征使用相同非0值, 不需要attention则置0                                                                                               |
| 12   | mask_type          | 特征hash的类型                | low、mid、high              | 将mmh3的hash结果进行mask截断, 缩减embedding特征空间；e.g. 假设mid=20, set_id最大值为5(二进制101), 此时emb空间应该为2^23=8388608, 那么emb空间初始化要大于这个值, 比如本工程用的是900万(tf.flags.DEFINE_integer("cate_emb_space_size", 9000000, "")). |


### 2. 训练数据制作

离线pyspark(相比rdd, 推荐使用dataframe)生成训练数据不在本工程范围内, 这里简单举例, 解释下过程.

首先dataframe格式数据如下, 通过配置文件进行特征构造: 

| label | item_id | user_id | item_cat1 | user_click_cat1 | user_click_cat2     | item_day_ctr | item_3hour_ctr | ... |
| ----- | ------- | ------- | --------- | --------------- | ------------------- | ------------ | -------------- | --- |
| 0     | 1001001 | user1   | 0         | 0, 1, 101       | 10101, 12321, 12343 | 0.21322      | 0.23212        | ... |
| 1     | 1001002 | user1   | -         | 1, 3            | 12122, 12312        | 0.62122      | 0.71231        | ... |
| 0     | 1001003 | user2   | 101       | 1               | 12341               | 0.13533      | -              | ... |

通过读取配置文件进行特征处理后, 我们都会将所有归一化后的连续型特征合并到'cont_feats'列中, 所有的单值离散特征合并到'cate_feats'中, 由于此时特征和位置一一对应, 所以要补齐缺失值(可以自定义, 比如连续特征补0, 离散特征用字符串'none'代替), 换句话说, 'cont_feats'和'cate_feats'都是dense类型. 而所有多值离散特征都单独为列, 所以可以不补缺失值直接保存(即sparse类型), 当然如果需要也可以补齐为dense, 相比来说, sparse能节省大量空间时间, 推荐sparse. 

| label | cate_feats           | cont_feats       | user_click_cat1 | user_click_cat2     | ... |
| ----- | -------------------- | ---------------- | --------------- | ------------------- | --- |
| 0     | 1001001, user1, 0    | 0.21322, 0.23212 | 0, 1, 101       | 10101, 12321, 12343 | ... |
| 1     | 1001002, user1, none | 0.62122, 0.71231 | 1, 3            | 12122, 12312        | ... |
| 0     | 1001003, user2, 101  | 0.13533, 0.0     | 1               | 12341               | ... |

对所有的单值、多值离散特征进行'域'hash处理, '域'即setID, 比如item_cat1和user_click_cat1应该用同一个setID, hash方法使用mmh3方法:

```
# python  线下使用, 与线上保持参数设置一致即可, 如low=16, seed=100

import mmh3

low = 16
mid = 20
high = 24

def mmh3_hash(set_id, value, mask_type):
    def mask_high(x):
        mask = (1 << high) - 1
        return x & mask

    def mask_mid(x):
        mask = (1 << mid) - 1
        return x & mask

    def mask_low(x):
        mask = (1 << low) - 1
        return x & mask

    if mask_type == 'low':
        return set_id << low | mask_low(mmh3.hash(key=value, seed=100) & 0xffffffff)
    elif mask_type == 'mid':
        return set_id << mid | mask_mid(mmh3.hash(key=value, seed=100) & 0xffffffff)
    elif mask_type == 'high':
        return set_id << high | mask_high(mmh3.hash(key=value, seed=100) & 0xffffffff)
```
```
// golang  线上使用

import murmur3 //go语言的mmh3包,[下载链接:](https://github.com/spaolacci/murmur3)
func murmur3_hash(featureID string, mask string, featureValues string) int64{
    setID, err := strconv.Atoi(featureID)
    if err != nil {
        fmt.Println("setID:", setID, " strconv.Atoi() error!!")
    }
    var maskID uint32
    switch mask {
    case "high":
        maskID = 24
    case "mid":
        maskID = 20
    case "low":
        maskID = 16
    default:
        maskID = 16
    }
    value, _ := StrMmh3Int32(featureValues, 100)
    maskFeat := (uint32(setID) << maskID) | MaskHash(value, maskID)
    v = int64(maskFeat)
    return v
}

func MaskHash(x uint32, maskId uint32) uint32 {
	mask := (1 << maskId) - 1
	return x & uint32(mask)
}

func StrMmh3Int32(x string, seed uint32) (y uint32, err error) {
	value := murmur3.Sum32WithSeed([]byte(x), seed)
	return value, nil
}
```

最后, 保存为tfrecord数据即可:```feat_df.write.format("tfrecords").option("recordType", "Example").save(train_path)```