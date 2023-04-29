import csv
import json
import os
import random
import time

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer
from retrieve import SentimentClassifier

# MODEL_FILE = "./models/bert_base_128max_55batch_segmentid_positionid.dat"
MODEL_FILE = "./models/siameseBert_256max_53batch_new_balance_data.dat"

class PDataset(Dataset):

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




class Predictor:
    def __init__(self, maxlen, gpu):


        self.classifier = SentimentClassifier()
        self.classifier.load_state_dict(torch.load(MODEL_FILE))
        self.classifier.eval()
        self.classifier.cuda(gpu)
        self.gpu = gpu

        self.maxlen = maxlen


    def predict(self, predict_dataset_path, output_dataset_path):
        self.output_file_path = output_dataset_path
        self.predict_dataset_path = predict_dataset_path
        self.predict_dataset = PDataset(predict_dataset_path, self.maxlen)
        self.output_file_path = output_dataset_path
        self.dataloader = DataLoader(self.predict_dataset, batch_size=400, shuffle=False,
                                     num_workers=10)  # TODO:增大worker, 减小batch_size

        with torch.no_grad():
            all_preds = []
            all_probs = []
            for seq, attn_masks, segment_ids, position_ids in self.dataloader:
                seq, attn_masks, segment_ids, position_ids = seq.cuda(self.gpu), attn_masks.cuda(self.gpu), \
                                                       segment_ids.cuda(self.gpu), position_ids.cuda(self.gpu)
                logits = self.classifier(seq, attn_masks, segment_ids, position_ids)
                probs = torch.sigmoid(logits.unsqueeze(-1))
                soft_probs = (probs > 0.90).long()  # 0.9是阈值， soft_probs是预测值 i.e. 0 or 1
                all_probs.extend(probs.squeeze().tolist())  # all_probs是所有预测值的概率
                all_preds.extend(soft_probs.squeeze().tolist())  # all_preds是所有预测值的列表


            # 将预测结果写入文件
            df = self.predict_dataset.df
            # 2. 将预测概率和值写入到csv文件中
            df['probs'] = all_probs
            df['label'] = all_preds
            # 选出label为1的数据
            df = df[df['label'] == 1]
            # 3. 保存csv文件
            df.to_csv(self.output_file_path, index=False, header=False, mode='a')






def check_pred(dev_claims_path, output_pred_path):
    # 读取 output_pred_path
    df = pd.read_csv(output_pred_path)
    # 读取 dev_claims_path
    with open(dev_claims_path, 'r') as f:
        # 读取JSON数据 - 字典
        claims = json.load(f)
    # 找出output_pred_path中的label为1的句子
    df_pred = df[df['label'] == 1]
    # 打印dev_claims_path文件中对应的claim的evidences
    claim_id = ''
    print("evidence selected:")
    for index, row in df_pred.iterrows():
        id = row['id']
        probs = row['probs']
        claim_id = id.split(',')[0]
        evidence_id = id.split(',')[1]
        print(evidence_id, " : ", probs)

    true_evidences = claims[claim_id]['evidences']
    print("true_evidences:", true_evidences)


def format_preds(preds_path, unlabelled_claims_path, output_path, k):
    # 将得到的预测格式化
    # 读取 output_pred_path 预测结果
    df = pd.read_csv(preds_path)
    # 找出output_pred_path中的label为1的句子
    df_pred = df[df['label'] == 1]
    # 根据prob排序
    df_pred = df_pred.sort_values(by='probs', ascending=False)

    # 读取 dev_claims_path
    with open(unlabelled_claims_path, 'r') as f: # unlabelled_claims_path可以是dev/test
        # 读取JSON数据 - 字典
        claims = json.load(f)

    # 创建空json
    new_claims = {}
    # 遍历claims 找出所有的claim_id
    for claim_id in claims:
        # 创建空list
        new_claims[claim_id] = {}
        new_claims[claim_id]['claim_text'] = claims[claim_id]['claim_text']
        new_claims[claim_id]['claim_label'] = "NAN"
        new_claims[claim_id]['evidences'] = []
    # 遍历df_pred 选出这个claim_id对应的evidence_id最高的k个
    for index, row in df_pred.iterrows():
        id = row['id']
        probs = row['probs']
        claim_id = id.split(',')[0]
        evidence_id = id.split(',')[1]
        # 如果claim_id在new_claims中
        if claim_id in new_claims and len(new_claims[claim_id]['evidences']) < k:
            # 将evidence_id加入到new_claims中
            new_claims[claim_id]['evidences'].append(evidence_id)

    # 遍历claims，碰到没有evidence的claim，将随机一个evidence加入到claims中（只是以防错误）
    counter = 0
    for claim_id in new_claims:
        if len(new_claims[claim_id]['evidences']) == 0:
            random_num = random.randint(0, 1208827)
            new_claims[claim_id]['evidences'].append(f"evidence-{random_num}")
            print("This claim has no evidence claim_id:", claim_id)
            counter += 1
    print("How many claims that don't have predictions:", counter)
    # 将claims写入到output_path中
    with open(output_path, 'w') as f:
        json.dump(new_claims, f, indent=2)
    print("format_preds done!")


def predict(dataset_for_predict_path, output_dataset_path):
    maxlen = 256
    # predict_dataset_path = "./data/demo_dev_for_predict.csv"
    # output_dataset_path = "./data/demo_dev_for_predict_output2.csv"
    gpu = 0
    chunk_size = 50000
    predictor = Predictor(maxlen, gpu)
    # 清理文件
    if os.path.exists(output_dataset_path):
        os.remove(output_dataset_path)
        print("Removed old file: ", output_dataset_path)

    # 用pandas提取dataset_for_predict_path中的标题头
    df = pd.read_csv(dataset_for_predict_path, nrows=0)
    # 加入标题probs和label
    df['probs'] = 0
    df['label'] = 0
    # 将标题写入到output_dataset_path中
    df.to_csv(output_dataset_path, index=False, header=True, mode='w')

    for i, chunk in enumerate(pd.read_csv(dataset_for_predict_path, chunksize=chunk_size)):
        start = time.time()
        temp_file_path = f"./data/temp{i}.csv"
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
            print("Removed old file: ", temp_file_path)

        chunk.to_csv(temp_file_path, mode='a', index=False)
        print("predicting for chunk:", i, " - data in the file:", temp_file_path)
        predictor.predict(temp_file_path, output_dataset_path)
        print("Done for chunk:", i, " - data in the file:", temp_file_path)
        # 删除临时文件
        os.remove(temp_file_path)
        print("Removed temp file: ", temp_file_path)
        end = time.time()
        print("time cost: ", end - start)


if __name__ == '__main__':
    # dataset_for_predict_path = "./data/demo_dev_claims_evi_pairs_for_predict.csv"
    # output_dataset_path = "./data/retrieve_output/demo_dev_claims_evi_pairs_for_predict_output2.csv"
    #
    # dataset_for_predict_path = "./data/similarity_filtered/test_output_10000.csv"
    # output_dataset_path = "./data/retrieve_output/test_10000.csv"
    #
    # predict(dataset_for_predict_path, output_dataset_path)

    # check_pred("./data/dev-claims.json", "./data/demo_dev_for_predict_output2.csv")

    # format_preds("./data/demo_dev_for_predict_output.csv", "./data/test-claims-unlabelled.json", "./data/test-claims-predictions.json")

    # 整理dev的预测结果
    format_preds("data/retrieve_output/test_10000.csv", "./data/test-claims-unlabelled.json",
                 "data/retrieve_output/test-claims-predictions_10000_4.json", 4)
