import json
import csv
import random
import pandas as pd

# train-claims.json 有1228条数据 最长的332 chars - 49 words
# 拆成claim和evidence数据对后有4122条数据
# evidence.json 有 1208827条数据 最长的是3148 chars - 479 words
# dev-claims.json 有 154条数据
# test-claims.json 有 153条数据
#

HAS_RELATION = 1
NO_RELATION = 0
NO_RELATION_NUM = 5

def prepare_pairs_data(train_or_dev_claims_path, new_file_name):
    # 打开JSON文件
    with open(train_or_dev_claims_path, 'r') as f:
        # 读取JSON数据 - 字典
        train_claims = json.load(f)

    # 打开evidence.json文件
    with open("data/evidence.json", 'r') as f:
        # 读取JSON数据 - 字典
        evidences = json.load(f)
        num_of_evidences = len(evidences)-1

    result = {}
    for key, value in train_claims.items():
        evidences_nos = value['evidences']
        for evidences_no in evidences_nos:
            evi_text = evidences[evidences_no]
            # claim 编号+evidence 编号 (用，连接) = key
            result[key + ',' + evidences_no] = [value['claim_text'], evi_text, HAS_RELATION]
        # 从evidence.json中找出没有被train_claims使用的evidence
        # 用于生成没有关系的数据对, 指定生成个数NO_RELATION_NUM
        counter = 0
        while counter < NO_RELATION_NUM:
            index = random.randint(0, num_of_evidences)
            text_index = list(evidences.keys())[index]
            if text_index in evidences_nos:
                continue
            evi_text = evidences[text_index]
            result[key + ',' + text_index] = [value['claim_text'], evi_text, NO_RELATION]
            counter += 1

    # 均匀打乱数据
    random.seed(69)
    print(f"From {train_or_dev_claims_path} get {len(result)} pairs data after adding no relation pairs.")
    result = dict(random.sample(result.items(), len(result)))

    data = [["id", "sentence", "label"]]
    for key, value in result.items():
        data.append([key, value[0] + '[SEP]' + value[1], value[2]])
    # 将result 写入csv文件
    with open(f"data/{new_file_name}", mode='w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerows(data)

    return result  # 返回字典 - 例子: value = [claim, evidence, label]

def data_analise(path):
    # 用pandas从csv加载数据
    print(f"====================={path}=====================")
    data = pd.read_csv(path)

    # 打印数据的形状
    print("Data Shape: ", data.shape)
    # 打印数据的列名
    print("Data Columns: ", data.columns)
    # 打印数据的统计信息
    print("Data Describe:")
    print(data.describe())



if __name__=="__main__":
    data_analise("data/train.csv")
    data_analise("data/dev.csv")





