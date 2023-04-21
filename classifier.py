import csv
import json

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer
from bert import SentimentClassifier

MODEL_FILE = "./models/bert_base_64max_32batch_segmentid.dat"


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

        # Obtaining the attention mask i.e a tensor containing 1s for no padded tokens and 0s for padded ones
        attn_mask = (tokens_ids_tensor != 0).long()

        return tokens_ids_tensor, attn_mask, segment_ids




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

        predict_dataset = PDataset(predict_dataset_path, self.maxlen)
        self.length = len(predict_dataset)
        self.dataloader = DataLoader(predict_dataset, batch_size=20, shuffle=False, num_workers=2)

    def predict(self):
        print("Predicting...")
        with torch.no_grad():
            all_preds = []
            all_probs = []
            #  下面的
            for seq, attn_masks, segment_ids in self.dataloader:
                seq, attn_masks, segment_ids = seq.cuda(self.gpu), attn_masks.cuda(self.gpu), \
                                                       segment_ids.cuda(self.gpu)
                logits = self.classifier(seq, attn_masks, segment_ids)
                probs = torch.sigmoid(logits.unsqueeze(-1))
                soft_probs = (probs > 0.98).long()  # 0.9是阈值， soft_probs是预测值 i.e. 0 or 1
                all_probs.extend(probs.squeeze().tolist())  # all_probs是所有预测值的概率
                all_preds.extend(soft_probs.squeeze().tolist())  # all_preds是所有预测值的列表


            if len(all_preds) != self.length:
                print("预测值数量不对！")
                return
            print("is writing to csv file...")
            # 将预测值写到原来的csv文件中 - 加一列label
            # 1. 读取原来的csv文件
            df = pd.read_csv(self.predict_dataset_path)
            # 2. 将预测概率和值写入到csv文件中
            df['probs'] = all_probs
            df['label'] = all_preds
            # 3. 保存csv文件
            df.to_csv(self.output_file_path, index=False)


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




if __name__ == '__main__':
    # maxlen = 64
    # predict_dataset_path = "./data/dev_for_predict.csv"
    # output_dataset_path = "./data/dev_for_predict_output.csv"
    # gpu = 0
    # predictor = Predictor(maxlen, predict_dataset_path, output_dataset_path, gpu)
    # predictor.predict()

    check_pred("./data/dev-claims.json", "./data/dev_for_predict_output.csv")

"""
claim-752
evidence-89  :  0.9985496401786804
"Pollution produced from centralised generation of electricity is emitted at a distant power station, rather than \"on site\"."
"""