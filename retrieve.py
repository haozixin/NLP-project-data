import math

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


class SentimentClassifier(nn.Module):

    def __init__(self):
        super(SentimentClassifier, self).__init__()
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
        # output dimension is 1 because we're working with a binary classification problem
        self.cls_layer = nn.Linear(768, 1)

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
    probs = torch.sigmoid(logits.unsqueeze(-1))
    soft_probs = (probs > 0.5).long()
    acc = (soft_probs.squeeze() == labels).float().mean()
    return acc

def get_f1_from_logits(logits, labels):
    probs = torch.sigmoid(logits.unsqueeze(-1))
    soft_probs = (probs > 0.5).long()
    t = soft_probs.squeeze().cpu()
    labels = labels.cpu()
    f1 = f1_score(labels, t, average='weighted')
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
            mean_loss += criterion(logits.squeeze(-1), labels.float()).item()
            mean_acc += get_accuracy_from_logits(logits, labels)
            # 计算F1-score
            f1_score += get_f1_from_logits(logits, labels)
            count += 1

    return mean_acc / count, mean_loss / count, f1_score / count

def train(net, criterion, opti, train_loader, dev_loader, max_eps, gpu):
    best_acc = 0
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
            loss = criterion(logits.squeeze(-1), labels.float())

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
        if dev_acc > best_acc:
            print("Best development accuracy improved from {} to {}, saving model...".format(best_acc, dev_acc))
            best_acc = dev_acc
            torch.save(net.state_dict(), 'sstcls_{}.dat'.format(ep))



if __name__ == "__main__":
    # ===============================只需要调用一次生成训练集和dev集======================================
    # prepare_pairs_data("data/train-claims.json", "train.csv")
    # prepare_pairs_data("data/dev-claims.json", "dev.csv")
    # ====================================================================
    # Creating instances of training and development set
    # maxlen sets the maximum length a sentence can have
    # any sentence longer than this length is truncated to the maxlen size
    train_set = SSTDataset(filename='data/train.csv', maxlen=64)
    dev_set = SSTDataset(filename='data/dev.csv', maxlen=64)
    # Creating intsances of training and development dataloaders
    # TODO: 交叉验证; num_workers 自己本地算的时候可以调大点；这些值最后都要调优
    train_loader = DataLoader(train_set, batch_size=32, num_workers=2)
    dev_loader = DataLoader(dev_set, batch_size=32, num_workers=2)
    print("Done preprocessing training and development data.")

    gpu = 0  # gpu ID
    print("Creating the sentiment classifier, initialised with pretrained BERT-BASE parameters...")
    net = SentimentClassifier()
    net.cuda(gpu)  # Enable gpu support for the model
    print("Done creating the sentiment classifier.")


    # =======Defining the loss function and optimizer=======
    criterion = nn.BCEWithLogitsLoss()
    opti = optim.Adam(net.parameters(), lr=2e-5)

    num_epoch = 2
    # fine-tune the model
    train(net, criterion, opti, train_loader, dev_loader, num_epoch, gpu)


# TODO: 试试 BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=5)