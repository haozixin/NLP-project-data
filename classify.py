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
    # 打开JSON文件
    with open(train_or_dev_claims_path, 'r') as f:
        # 读取JSON数据 - 字典
        train_claims = json.load(f)

    # 打开evidence.json文件
    with open("data/evidence.json", 'r') as f:
        # 读取JSON数据 - 字典
        evidences = json.load(f)
        num_of_evidences = len(evidences) - 1

    result = {}
    for key, value in train_claims.items():
        evidences_nos = value['evidences']
        for evidences_no in evidences_nos:
            evi_text = evidences[evidences_no]
            label = value['claim_label']
            # 对label {SUPPORTS, REFUTES, NOT_ENOUGH_INFO, DISPUTED} 进行 one-hot 编码
            label = label_mapping(label)
            # claim 编号+evidence 编号 (用，连接) = key
            result[key + ',' + evidences_no] = [value['claim_text'], evi_text, label]

    # 均匀打乱
    result = dict(random.sample(result.items(), len(result)))

    data = [["id", "sentence", "label"]]
    for key, value in result.items():
        data.append([key, value[0] + "[SEP]" + value[1], value[2]])

    with open(f"./classify_data/{new_file_name}", mode='w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerows(data)


def label_mapping(label):
    if label == 'SUPPORTS':
        label = 0
    elif label == 'REFUTES':
        label = 1
    elif label == 'NOT_ENOUGH_INFO':
        label = 2
    elif label == 'DISPUTED':
        label = 3
    return label


class SSTDataset(Dataset):

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
        label = self.df.loc[index, 'label']
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
        return tokens_ids_tensor, attn_mask, segment_ids, position_ids, label


class step2Classifier(nn.Module):

    def __init__(self):
        super(step2Classifier, self).__init__()
        # Instantiating BERT model object
        self.config = BertConfig.from_pretrained('bert-base-uncased')
        # change config
        self.config.architectures = ["SiameseBertModel"]
        # self.config.num_hidden_layers = 6
        # self.config.attention_probs_dropout_prob = 0.3
        # self.config.hidden_dropout_prob = 0.3

        self.bert_layer = BertModel.from_pretrained('bert-base-uncased', config=self.config)


        # Classification layer
        # input dimension is 768 because [CLS] embedding has a dimension of 768
        # retrieve_output dimension is 1 because we're working with a binary classification problem
        self.cls_layer = nn.Linear(768, 4)

    def forward(self, seq, attn_masks, segment_ids, position_ids):
        '''
        Inputs:
            -seq : Tensor of shape [B, T] containing token ids of sequences
            -attn_masks : Tensor of shape [B, T] containing attention masks to be used to avoid contibution of PAD tokens
        '''

        # Feeding the input to BERT model to obtain contextualized representations
        outputs = self.bert_layer(seq, attention_mask=attn_masks, token_type_ids=segment_ids, position_ids=position_ids, return_dict=True)
        cont_reps = outputs.last_hidden_state

        # Obtaining the representation of [CLS] head (the first token)
        cls_rep = cont_reps[:, 0]

        # Feeding cls_rep to the classifier layer
        logits = self.cls_layer(cls_rep)

        return logits

def get_accuracy_from_logits(logits, labels):
    probs = F.softmax(logits, dim=1)
    preds = torch.argmax(probs, dim=1)
    acc = torch.mean((preds == labels).float())
    return acc

def get_f1_from_logits(logits, labels):
    # 对logits进行softmax操作
    probs = F.softmax(logits, dim=1)
    # 获取每个样本的预测类别
    preds = torch.argmax(probs, dim=1).cpu().numpy()

    labels = labels.cpu().numpy()
    f1 = f1_score(labels, preds, average='weighted')
    return f1

def evaluate(net, criterion, dataloader, gpu):
    net.eval()

    mean_acc, mean_loss = 0, 0
    count = 0
    f1_score = 0

    with torch.no_grad():
        for seq, attn_masks, segment_ids, position_ids, labels in dataloader:
            seq, attn_masks, segment_ids, position_ids, labels = seq.cuda(gpu), attn_masks.cuda(gpu), segment_ids.cuda(gpu), position_ids.cuda(gpu), labels.cuda(gpu)
            logits = net(seq, attn_masks, segment_ids, position_ids.cuda(gpu))
            mean_loss += criterion(logits.squeeze(-1), labels).item()
            mean_acc += get_accuracy_from_logits(logits, labels)
            # 计算F1-score
            f1_score += get_f1_from_logits(logits, labels)
            count += 1

    return mean_acc / count, mean_loss / count, f1_score / count

def train(net, criterion, opti, train_loader, dev_loader, max_eps, gpu):
    best_f1 = 0
    st = time.time()
    for ep in range(max_eps):

        net.train()
        for it, (seq, attn_masks, segment_ids, position_ids, labels) in enumerate(train_loader):
            # Clear gradients
            opti.zero_grad()
            # Converting these to cuda tensors
            seq, attn_masks, segment_ids, position_ids, labels = seq.cuda(gpu), attn_masks.cuda(gpu), segment_ids.cuda(gpu), position_ids.cuda(gpu), labels.cuda(gpu)

            # Obtaining the logits from the model
            logits = net(seq, attn_masks, segment_ids, position_ids)

            # Computing loss
            # labels = torch.argmax(labels, dim=1)
            loss = criterion(logits.squeeze(-1), labels)

            # Backpropagating the gradients
            loss.backward()

            # Optimization step
            opti.step()

            if it % 100 == 0:
                acc = get_accuracy_from_logits(logits, labels)
                f1 = get_f1_from_logits(logits, labels)
                print("Iteration {} of epoch {} complete. Loss: {}; Accuracy: {}; f1: {}; Time taken (s): {}"
                      .format(it, ep, loss.item(), acc, f1, (time.time() - st)))
                st = time.time()

        dev_acc, dev_loss, dev_f1 = evaluate(net, criterion, dev_loader, gpu)
        print("Epoch {} complete! Development Accuracy: {}; Development Loss: {}; F1: {};".format(ep, dev_acc, dev_loss, dev_f1))
        if dev_f1 > best_f1:
            print("Best development accuracy improved from {} to {}, saving model...".format(best_f1, dev_f1))
            best_f1 = dev_f1
            torch.save(net.state_dict(), './classify_models/sstcls_{}.dat'.format(ep))



if __name__ == "__main__":
    # preprocess_data("data/train-claims.json", "train_for_classify.csv")

    # ===============================只需要调用一次生成训练集和dev集======================================
    # prepare_pairs_data("data/train-claims.json", "ole_train.csv")
    # prepare_pairs_data("data/dev-claims.json", "old_dev.csv")
    # ====================================================================
    # Creating instances of training and development set
    # maxlen sets the maximum length a sentence can have
    # any sentence longer than this length is truncated to the maxlen size
    train_set = SSTDataset(filename='classify_data/train_for_classify.csv', maxlen=256)
    dev_set = SSTDataset(filename='classify_data/dev_for_classify.csv', maxlen=256)
    # Creating intsances of training and development dataloaders

    train_loader = DataLoader(train_set, batch_size=16, num_workers=1)
    dev_loader = DataLoader(dev_set, batch_size=16, num_workers=1)
    print("Done preprocessing training and development data.")

    gpu = 0  # gpu ID
    print("Creating the sentiment classifier, initialised with pretrained BERT-BASE parameters...")
    net = step2Classifier()
    net.cuda(gpu)  # Enable gpu support for the model
    print("Done creating the sentiment classifier.")


    # =======Defining the loss function and optimizer=======
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=2e-5)

    num_epoch = 2
    # fine-tune the model
    train(net, criterion, optimizer, train_loader, dev_loader, num_epoch, gpu)


# TODO: 试试 BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=5)