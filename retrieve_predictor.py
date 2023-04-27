import csv
import json
import os

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer
from retrieve import SentimentClassifier, positional_encoding

# MODEL_FILE = "./models/bert_base_128max_55batch_segmentid_positionid.dat"
MODEL_FILE = "./models/siameseBert_new_train_data2.dat"

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

        # 对claim和evidence进行分词
        claim_tokens = self.tokenizer.tokenize(claim)
        evidence_tokens = self.tokenizer.tokenize(evidence)

        claim_tokens = ['[CLS]'] + claim_tokens + ['[SEP]']
        evidence_tokens = evidence_tokens + ['[SEP]']
        if len(claim_tokens) < self.maxlen:
            claim_tokens = claim_tokens + ['[PAD]' for _ in range(self.maxlen - len(claim_tokens))]
        else:
            claim_tokens = claim_tokens[:self.maxlen - 1] + ['[SEP]']
        if len(evidence_tokens) < self.maxlen:
            evidence_tokens = evidence_tokens + ['[PAD]' for _ in range(self.maxlen - len(evidence_tokens))]
        else:
            evidence_tokens = evidence_tokens[:self.maxlen - 1] + ['[SEP]']

        # build segment_ids
        segment_ids = [0] * len(claim_tokens) + [1] * len(evidence_tokens)
        segment_ids = torch.tensor(segment_ids)

        sentence_tokens = claim_tokens + evidence_tokens

        tokens_ids = self.tokenizer.convert_tokens_to_ids(sentence_tokens)  # Obtaining the indices of the tokens in the BERT Vocabulary
        tokens_ids_tensor = torch.tensor(tokens_ids)  # Converting the list to a pytorch tensor

        # position tokens
        position_ids = [i for i in range(len(tokens_ids))]
        position_ids = torch.tensor(position_ids)

        # Obtaining the attention mask i.e a tensor containing 1s for no padded tokens and 0s for padded ones
        attn_mask = (tokens_ids_tensor != 0).long()

        return tokens_ids_tensor, attn_mask, segment_ids, position_ids




class Predictor():
    def __init__(self, maxlen, predict_dataset_path, output_dataset_path, gpu):
        self.output_file_path = output_dataset_path
        self.predict_dataset_path = predict_dataset_path
        self.classifier = SentimentClassifier()
        self.classifier.load_state_dict(torch.load(MODEL_FILE))
        self.classifier.eval()
        self.classifier.cuda(gpu)
        self.gpu = gpu

        self.maxlen = maxlen

        self.predict_dataset = PDataset(predict_dataset_path, self.maxlen)
        self.output_file_path = output_dataset_path
        self.dataloader = DataLoader(self.predict_dataset, batch_size=200, shuffle=False,
                                     num_workers=10)  # TODO:增大worker, 减小batch_size

    def predict(self):
        with torch.no_grad():
            all_preds = []
            all_probs = []
            for seq, attn_masks, segment_ids, position_ids in self.dataloader:
                seq, attn_masks, segment_ids, position_ids = seq.cuda(self.gpu), attn_masks.cuda(self.gpu), \
                                                       segment_ids.cuda(self.gpu), position_ids.cuda(self.gpu)
                logits = self.classifier(seq, attn_masks, segment_ids, position_ids)
                probs = torch.sigmoid(logits.unsqueeze(-1))
                soft_probs = (probs > 0.97).long()  # 0.9是阈值， soft_probs是预测值 i.e. 0 or 1
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


def format_preds(preds_path, unlabelled_claims_path, output_path):
    # 将得到的预测格式化
    # 读取 output_pred_path
    df = pd.read_csv(preds_path)
    # 读取 dev_claims_path
    with open(unlabelled_claims_path, 'r') as f:
        # 读取JSON数据 - 字典
        claims = json.load(f)
    # 找出output_pred_path中的label为1的句子
    df_pred = df[df['label'] == 1]
    claim_id = ''
    for index, row in df_pred.iterrows():
        id = row['id']
        probs = row['probs']
        claim_id = id.split(',')[0]
        evidence_id = id.split(',')[1]
        # 将预测的evidence加入到claims中
        claims[claim_id]['evidences'].append(evidence_id)
    # 遍历claims，碰到没有evidence的claim，将随机一个evidence加入到claims中（只是以防错误）
    for claim_id in claims:
        if len(claims[claim_id]['evidences']) == 0:
            claims[claim_id]['evidences'].append("evidence-957389")
        print("This claim has no evidence claim_id:", claim_id)
    # 将claims写入到output_path中
    with open(output_path, 'w') as f:
        json.dump(claims, f, indent=2, ensure_ascii=False)
    print("format_preds done!")


def predict(dataset_for_predict_path, output_dataset_path):
    maxlen = 128
    # predict_dataset_path = "./data/demo_dev_for_predict.csv"
    # output_dataset_path = "./data/demo_dev_for_predict_output2.csv"
    gpu = 0
    chunk_size = 50000
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
        temp_file_path = f"./data/temp{i}.csv"
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
            print("Removed old file: ", temp_file_path)

        chunk.to_csv(temp_file_path, mode='a', index=False)
        print("predicting for chunk:", i, " - data in the file:", temp_file_path)
        predictor = Predictor(maxlen, temp_file_path, output_dataset_path, gpu)
        predictor.predict()
        print("Done for chunk:", i, " - data in the file:", temp_file_path)
        # 删除临时文件
        os.remove(temp_file_path)
        print("Removed temp file: ", temp_file_path)


if __name__ == '__main__':
    # dataset_for_predict_path = "./data/demo_dev_claims_evi_pairs_for_predict.csv"
    # output_dataset_path = "./data/output/demo_dev_claims_evi_pairs_for_predict_output2.csv"
    # dataset_for_predict_path = "./data/demo_dev_for_predict.csv"
    # output_dataset_path = "./data/output/demo_dev_for_predict_output_by_sia_new_data2.csv"
    dataset_for_predict_path = "./data/temp_test.csv"
    output_dataset_path = "./data/output/temp_output.csv"
    predict(dataset_for_predict_path, output_dataset_path)

    # check_pred("./data/dev-claims.json", "./data/demo_dev_for_predict_output2.csv")

    # format_preds("./data/demo_dev_for_predict_output.csv", "./data/test-claims-unlabelled.json", "./data/test-claims-predictions.json")

