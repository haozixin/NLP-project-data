import csv
import json
import os
import time

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer

from classify import step2Classifier

# autodl-tmp/classify_data/classify280_100_large_sstcls_1.dat 0.554
# MODEL_FILE = "classify_models/410max_18batch_0.5305.dat"
# MODEL_FILE = "classify_models/410max_25batch_0.5337.dat"
MODEL_FILE = "classify_models/classify280_120_sstcls_0.538.dat"

def preprocess_data(retrieve_output_formated_file, output_path):
    """ take the format
    "claim-375": {
    "claim_text": "when 3 per cent of total annual global emissions of carbon dioxide are from humans and Australia prod\u00aduces 1.3 per cent of this 3 per cent, then no amount of emissions reductio\u00adn here will have any effect on global climate.",
    "claim_label": "NAN",
    "evidences": [
      "evidence-647121",
      "evidence-949910",
      "evidence-944007"
    ]
  }
   into the format:
    id,sentence
    for next prediction
    """
    # 打开JSON文件
    with open(retrieve_output_formated_file, 'r') as f:
        # 读取JSON数据 - 字典
        ready_to_classify_data = json.load(f)

    # 打开evidence.json文件
    with open("data/evidence.json", 'r') as f:
        # 读取JSON数据 - 字典
        evidences = json.load(f)

    result = {}
    for key, value in ready_to_classify_data.items():
        evidences_nos = value['evidences']
        for evidences_no in evidences_nos:
            evi_text = evidences[evidences_no]
            result[key + ',' + evidences_no] = [value['claim_text'], evi_text]

    data = [["id", "sentence"]]
    for key, value in result.items():
        data.append([key, value[0] + "[SEP]" + value[1]])

    with open(output_path, mode='w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerows(data)


class CDataset(Dataset):

    def __init__(self, filename, maxlen):

        # Store the contents of the file in a pandas dataframe
        self.df = pd.read_csv(filename, delimiter=',')

        # Initialize the BERT tokenizer
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        self.maxlen = maxlen

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):

        # Selecting the sentence and label at the specified index in the data frame
        sentence = self.df.loc[index, 'sentence']
        # label = self.df.loc[index, 'label']
        # label_tensor = torch.tensor(int(label))

        # 对sentence进行预处理，先根据【SEP】分句
        claim = sentence.split("[SEP]")[0]
        evidence = sentence.split("[SEP]")[1]

        encoded_input = self.tokenizer.encode_plus(
            claim,  # 要编码的句子
            evidence,
            add_special_tokens=True,  # 添加特殊令牌
            max_length=self.maxlen,  # 最大长度
            padding='max_length',  # 填充到最大长度
            truncation=True,  # 截断输入序列以适合最大长度
            return_tensors='pt',  # 返回 PyTorch 张量
        )

        # 获取句子的token ids
        tokens_ids_tensor = encoded_input['input_ids'].squeeze()
        attn_mask = encoded_input['attention_mask'].squeeze()
        segment_ids = encoded_input['token_type_ids'].squeeze()

        # position tokens
        position_ids = [i for i in range(self.maxlen)]
        position_ids = torch.tensor(position_ids)
        return tokens_ids_tensor, attn_mask, segment_ids, position_ids

class ClassifyPredictor:
    def __init__(self, maxlen, gpu):
        self.classifier = step2Classifier()
        self.classifier.load_state_dict(torch.load(MODEL_FILE))
        self.classifier.eval()
        self.classifier.cuda(gpu)
        self.gpu = gpu
        self.maxlen = maxlen

    def classify(self, predict_dataset_path, output_dataset_path):
        self.output_file_path = output_dataset_path
        self.predict_dataset_path = predict_dataset_path
        self.predict_dataset = CDataset(predict_dataset_path, self.maxlen)
        self.output_file_path = output_dataset_path
        self.dataloader = DataLoader(self.predict_dataset, batch_size=200, shuffle=False,
                                     num_workers=5)

        with torch.no_grad():
            all_preds = []
            all_probs = []
            for seq, attn_masks, segment_ids, position_ids in self.dataloader:
                seq, attn_masks, segment_ids, position_ids = seq.cuda(self.gpu), attn_masks.cuda(self.gpu), \
                                                       segment_ids.cuda(self.gpu), position_ids.cuda(self.gpu)
                logits = self.classifier(seq, attn_masks, segment_ids, position_ids)
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(probs, dim=1)
                all_probs.extend(probs[:, 1].cpu().numpy())
                all_preds.extend(preds.cpu().numpy())

            # 将预测结果写入文件
            df = self.predict_dataset.df
            # 2. 将预测概率和值写入到csv文件中
            df['probs'] = all_probs
            df['label'] = all_preds
            # 3. 保存csv文件
            df.to_csv(self.output_file_path, index=False, header=False, mode='a')

def label_mapping(label):
    if label == '0':
        label = 'SUPPORTS'
    elif label == '1':
        label = 'REFUTES'
    elif label == '2':
        label = 'NOT_ENOUGH_INFO'
    elif label == '3':
        label = 'DISPUTED'
    return label

def format_data(input_file, unlabelled_claims_path, final_result_path):
    df = pd.read_csv(input_file)
    df["claim_id"] = df["id"].apply(lambda x: x.split(",")[0])
    df["evidence_id"] = df["id"].apply(lambda x: x.split(",")[1])
    with open(unlabelled_claims_path, 'r') as f:  # unlabelled_claims_path可以是dev/test
        # 读取JSON数据 - 字典
        unlabelled_claims = json.load(f)

    new_claims = {}
    for claim in unlabelled_claims:
        label_decisions = {}
        evidences = []
        # 取出df中id中的claim跟unlabelled_claims中claim_text相同的行
        df_claim = df[df["claim_id"] == claim]
        # 放到label_decisions中
        for index, row in df_claim.iterrows():
            evidences.append(row["evidence_id"])
            if row["label"] not in label_decisions:
                # label_decisions[row["label"]] = row["probs"]
                label_decisions[row["label"]] = 1
                # label_decisions[row["label"]] = row["probs"]
            else:
                # label_decisions[row["label"]] += row["probs"]
                label_decisions[row["label"]] += 1
                # if label_decisions[row["label"]] < row["probs"]:
                #     label_decisions[row["label"]] = row["probs"]

        # 找出label_decisions中最大的值
        max_label = max(label_decisions, key=label_decisions.get)
        max_label = label_mapping(str(max_label))
        new_claims[claim] = {}
        new_claims[claim]['claim_text'] = unlabelled_claims[claim]['claim_text']
        new_claims[claim]['claim_label'] = max_label
        new_claims[claim]['evidences'] = evidences
    with open(final_result_path, 'w') as f:
        json.dump(new_claims, f, indent=2)
    print("format_preds done!")



def classify(ready_file_path, unlabelled_claims_path,output_file_path):
    start = time.time()
    maxlen = 280
    gpu = 0
    predictor = ClassifyPredictor(maxlen, gpu)

    temp_file = "data/temp/temp.csv"
    temp_file2 = "data/temp/temp2.csv"
    df = pd.DataFrame(columns=['id','sentence', 'probs', 'label'])
    df.to_csv(temp_file2, index=False, header=True, mode='w')

    # 把retrieve完成并且整理格式后的文件放进来，处理成可以用来classify的格式
    preprocess_data(ready_file_path, temp_file)
    predictor.classify(temp_file, temp_file2)
    format_data(temp_file2, unlabelled_claims_path, output_file_path)

    os.remove(temp_file)
    os.remove(temp_file2)
    end = time.time()
    print("classify time: ", end - start)


if __name__ == "__main__":
    classify("data/retrieve_output/test-claims-predictions_40000_4.json",
                "data/test-claims-unlabelled.json",
             "test-claims-predictions/test-claims-predictions.json")

    # classify("data/retrieve_output/dev-claims-predictions_10000_4.json",
    #             "data/dev-claims.json",
    #          "test-claims-predictions/predictions_dev_10000_4_0.5337_maxvote.json")

"""
    if label == 'SUPPORTS':
        label = 0
    elif label == 'REFUTES':
        label = 1
    elif label == 'NOT_ENOUGH_INFO':
        label = 2
    elif label == 'DISPUTED':
        label = 3
"""