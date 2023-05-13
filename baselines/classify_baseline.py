import csv
import json
import math
import random

from torch.utils.data import Dataset
from transformers import BertTokenizer, BertConfig
import pandas as pd
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from transformers import BertModel
import torch.optim as optim
import time
from preprocess import prepare_train_pairs_data
from sklearn.metrics import f1_score
import torch.nn.functional as F


def preprocess_data(train_or_dev_claims_path, new_file_name):
    with open(train_or_dev_claims_path, 'r') as f:
        train_claims = json.load(f)

    with open("data/evidence.json", 'r') as f:
        evidences = json.load(f)

    result = {}
    for key, value in train_claims.items():
        evidences_nos = value['evidences']
        for evidences_no in evidences_nos:
            evi_text = evidences[evidences_no]
            label = value['claim_label']
            # 对label {SUPPORTS, REFUTES, NOT_ENOUGH_INFO, DISPUTED} 进行编码
            label = label_mapping(label)
            # claim 编号+evidence 编号 (用，连接) = key
            result[key + ',' + evidences_no] = [value['claim_text'], evi_text, label]

    # 均匀打乱
    result = dict(random.sample(result.items(), len(result)))

    data = [["id", "sentence", "label"]]
    for key, value in result.items():
        data.append([key, value[0] + "[SEP]" + value[1], value[2]])

    with open(f"{new_file_name}", mode='w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerows(data)


def label_mapping(label):
    if label == 'SUPPORTS':
        label = 1
    else:
        label = 0
    return label

if __name__ == "__main__":
    preprocess_data("../data/train_claims.json", "train_for_classify_baseline.csv")