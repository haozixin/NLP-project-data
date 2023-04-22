from torch.utils.data import Dataset
from transformers import BertTokenizer
import pandas as pd
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from transformers import BertModel
import torch.optim as optim
import time
from preprocess import prepare_pairs_data





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

        return tokens_ids_tensor, attn_mask, segment_ids, label


class SentimentClassifier(nn.Module):

    def __init__(self):
        super(SentimentClassifier, self).__init__()
        # Instantiating BERT model object
        self.bert_layer = BertModel.from_pretrained('bert-base-uncased')  #TODO：可以换'bert-large-uncased'

        # Classification layer
        # input dimension is 768 because [CLS] embedding has a dimension of 768
        # output dimension is 1 because we're working with a binary classification problem
        self.cls_layer = nn.Linear(768, 1)

    def forward(self, seq, attn_masks, segment_ids):
        '''
        Inputs:
            -seq : Tensor of shape [B, T] containing token ids of sequences
            -attn_masks : Tensor of shape [B, T] containing attention masks to be used to avoid contibution of PAD tokens
        '''

        # Feeding the input to BERT model to obtain contextualized representations
        outputs = self.bert_layer(seq, attention_mask=attn_masks, token_type_ids=segment_ids, return_dict=True)
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

def evaluate(net, criterion, dataloader, gpu):
    net.eval()

    mean_acc, mean_loss = 0, 0
    count = 0

    with torch.no_grad():
        for seq, attn_masks, segment_ids, labels in dataloader:
            seq, attn_masks, segment_ids, labels = seq.cuda(gpu), attn_masks.cuda(gpu), segment_ids.cuda(gpu), labels.cuda(gpu)
            logits = net(seq, attn_masks, segment_ids)
            mean_loss += criterion(logits.squeeze(-1), labels.float()).item()
            mean_acc += get_accuracy_from_logits(logits, labels)
            count += 1

    return mean_acc / count, mean_loss / count

def train(net, criterion, opti, train_loader, dev_loader, max_eps, gpu):
    best_acc = 0
    st = time.time()
    for ep in range(max_eps):

        net.train()
        for it, (seq, attn_masks, segment_ids, labels) in enumerate(train_loader):
            # Clear gradients
            opti.zero_grad()
            # Converting these to cuda tensors
            seq, attn_masks, segment_ids, labels = seq.cuda(gpu), attn_masks.cuda(gpu), segment_ids.cuda(gpu), labels.cuda(gpu)

            # Obtaining the logits from the model
            logits = net(seq, attn_masks, segment_ids)

            # Computing loss
            loss = criterion(logits.squeeze(-1), labels.float())

            # Backpropagating the gradients
            loss.backward()

            # Optimization step
            opti.step()

            if it % 100 == 0:
                acc = get_accuracy_from_logits(logits, labels)
                print("Iteration {} of epoch {} complete. Loss: {}; Accuracy: {}; Time taken (s): {}"
                      .format(it, ep, loss.item(), acc, (time.time() - st)))
                st = time.time()

        dev_acc, dev_loss = evaluate(net, criterion, dev_loader, gpu)
        print("Epoch {} complete! Development Accuracy: {}; Development Loss: {}".format(ep, dev_acc, dev_loss))
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
    criterion = nn.BCEWithLogitsLoss()  #TODO: 第二阶段分类时，换损失函数-交叉熵
    opti = optim.Adam(net.parameters(), lr=2e-5)

    num_epoch = 2
    # fine-tune the model
    train(net, criterion, opti, train_loader, dev_loader, num_epoch, gpu)


# TODO: 试试 BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=5)