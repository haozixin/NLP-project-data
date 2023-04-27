import csv
import json
import os
import random

import pandas as pd
import torch
from torch.utils.data import Dataset

from retrieve import SentimentClassifier
from retrieve_predictor import predict

HAS_RELATION = 1
NO_RELATION = 0
ERROR_PRED = 4

def generate_new_train_or_dev_data(train_claims_path, output_path):
    temp_output_dataset_path = "./data/temp_error_pred.csv"
    # 打开JSON文件
    with open(train_claims_path, 'r') as f:
        # 读取JSON数据 - 字典
        train_claims = json.load(f)
    evidences, num_of_evidences = get_evidence()
    initialize_output_file(output_path)

    for claim_key, claim_value in train_claims.items():
        batch_result = [["id", "sentence"]]
        duplicate = []
        temp2_path = "./data/temp_output.csv"
        if os.path.exists(temp2_path):
            os.remove(temp2_path)
        print("Finding wrong pred for claim_key: ", claim_key)
        while ERROR_PRED:
            random_index = random.randint(0, num_of_evidences)
            if random_index in duplicate:
                continue
            duplicate.append(random_index)
            evidence_id = list(evidences.keys())[random_index]
            if evidence_id not in claim_value['evidences']:
                input = [claim_key+','+evidence_id, claim_value["claim_text"]+'[SEP]'+evidences[evidence_id]]
                batch_result.append(input)
            if len(batch_result) >= 4000:
                temp_path = "./data/temp_new_train.csv"
                if os.path.exists(temp_path):
                    # delete the file
                    os.remove(temp_path)
                # 写入csv文件
                # 将result 写入csv文件
                with open(temp_path, mode='w', newline='', encoding='utf-8-sig') as f:
                    writer = csv.writer(f)
                    writer.writerows(batch_result)
                # ================以上，暂存输入数据 ===========

                predict(temp_path, temp_output_dataset_path)
                # delete the temp file
                os.remove(temp_path)
                # temp_output_dataset_path 文件会在下次调用predict时被覆盖
                df = pd.read_csv(temp_output_dataset_path)
                # 只要指定列
                df = df[["id", "sentence", "label"]]
                # 把label都变成0
                df["label"] = 0
                # 清除batch_result
                batch_result = [["id", "sentence"]]

                #===============暂存输出数据================
                # 存储到csv文件
                df.to_csv(temp2_path, index=False, mode='a', header=False)
                # 重新读数据
                df = pd.read_csv(temp2_path, header=None, names=["id", "sentence", "label"])
                # 如果数据超过N行
                if df.shape[0] < ERROR_PRED:
                    continue
                # ===========达到数量之后，就把数据写入output_path，然后删除temp_path， 然后break======
                # 写入数据到output_path
                df.to_csv(output_path, index=False, mode='a', header=False)

                # 删除temp_path
                os.remove(temp2_path)

                break


def initialize_output_file(output_path):
    if os.path.exists(output_path):
        # delete the file
        os.remove(output_path)
    df = pd.DataFrame(columns=["id", "sentence", "label"])
    df.to_csv(output_path, index=False, mode='w')


def get_evidence():
    # 打开evidence.json文件
    with open("data/evidence.json", 'r') as f:
        # 读取JSON数据 - 字典
        evidences = json.load(f)
        num_of_evidences = len(evidences) - 1
    return evidences, num_of_evidences


if __name__ == '__main__':
    generate_new_train_or_dev_data("./data/dev-claims.json", "./data/new_dev_data.csv")



