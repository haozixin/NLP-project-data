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
ERROR_PRED = 5

def generate_new_train_data(train_claims_path, output_path):
    temp_output_dataset_path = "./data/temp_error_pred.csv"
    # 打开JSON文件
    with open(train_claims_path, 'r') as f:
        # 读取JSON数据 - 字典
        train_claims = json.load(f)

    # 打开evidence.json文件
    with open("data/evidence.json", 'r') as f:
        # 读取JSON数据 - 字典
        evidences = json.load(f)
        num_of_evidences = len(evidences) - 1

    if os.path.exists(output_path):
        # delete the file
        os.remove(output_path)
    df = pd.DataFrame(columns=["id", "sentence", "label"])
    df.to_csv(output_path, index=False, mode='w')


    for claim_key, claim_value in train_claims.items():
        batch_result = [["id", "sentence"]]
        duplicate = []
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
            if len(batch_result) >= 1000:
                temp_path = "./data/temp_new_train.csv"
                if os.path.exists(temp_path):
                    # delete the file
                    os.remove(temp_path)
                # 写入csv文件
                # 将result 写入csv文件
                with open(temp_path, mode='w', newline='', encoding='utf-8-sig') as f:
                    writer = csv.writer(f)
                    writer.writerows(batch_result)
                predict(temp_path, temp_output_dataset_path)
                # delete the temp file
                os.remove(temp_path)

                df = pd.read_csv(temp_output_dataset_path)

                # 如果有N行数据，不包括表头
                if df.shape[0] < ERROR_PRED:
                    continue
                # 存储id和sentence，label列
                df = df[["id", "sentence", "label"]]
                # 存储到csv文件
                df.to_csv(output_path, index=False, mode='a')
                break
        os.remove(temp_output_dataset_path)




if __name__ == '__main__':
    generate_new_train_data("./data/train-claims.json", "./data/new_train_data.csv")





        # evidences_nos = value['evidences']
        # for evidences_no in evidences_nos:
        #     evi_text = evidences[evidences_no]
        #     # claim 编号+evidence 编号 (用，连接) = key
        #     result[key + ',' + evidences_no] = [value['claim_text'], evi_text, HAS_RELATION]
        # # 从evidence.json中找出没有被train_claims使用的evidence
        # # 用于生成没有关系的数据对, 指定生成个数NO_RELATION_NUM
        # counter = 0
        # while counter < NO_RELATION_NUM:
        #     index = random.randint(0, num_of_evidences)
        #     text_index = list(evidences.keys())[index]
        #     if text_index in evidences_nos:
        #         continue
        #     evi_text = evidences[text_index]
        #     result[key + ',' + text_index] = [value['claim_text'], evi_text, NO_RELATION]
        #     counter += 1

