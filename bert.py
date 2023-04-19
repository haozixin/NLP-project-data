from torch.utils.data import Dataset
from transformers import BertTokenizer
import pandas as pd
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from transformers import BertModel
import torch.optim as optim
import time


class SSTDataset(Dataset):

    def __init__(self, filename, maxlen):

        # Store the contents of the file in a pandas dataframe
        self.df = pd.read_csv(filename, delimiter='\t')

        # Initialize the BERT tokenizer
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        self.maxlen = maxlen

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):

        # Selecting the sentence and label at the specified index in the data frame
        sentence = self.df.loc[index, 'sentence']
        label = self.df.loc[index, 'label']

        # Preprocessing the text to be suitable for BERT
        tokens = self.tokenizer.tokenize(sentence)  # Tokenize the sentence
        tokens = ['[CLS]'] + tokens + [
            '[SEP]']  # Insering the CLS and SEP token in the beginning and end of the sentence
        if len(tokens) < self.maxlen:
            tokens = tokens + ['[PAD]' for _ in range(self.maxlen - len(tokens))]  # Padding sentences
        else:
            tokens = tokens[:self.maxlen - 1] + ['[SEP]']  # Prunning the list to be of specified max length

        tokens_ids = self.tokenizer.convert_tokens_to_ids(
            tokens)  # Obtaining the indices of the tokens in the BERT Vocabulary
        tokens_ids_tensor = torch.tensor(tokens_ids)  # Converting the list to a pytorch tensor

        # Obtaining the attention mask i.e a tensor containing 1s for no padded tokens and 0s for padded ones
        attn_mask = (tokens_ids_tensor != 0).long()

        return tokens_ids_tensor, attn_mask, label


class SentimentClassifier(nn.Module):

    def __init__(self):
        super(SentimentClassifier, self).__init__()
        # Instantiating BERT model object
        self.bert_layer = BertModel.from_pretrained('bert-base-uncased')

        # Classification layer
        # input dimension is 768 because [CLS] embedding has a dimension of 768
        # output dimension is 1 because we're working with a binary classification problem
        self.cls_layer = nn.Linear(768, 1)

    def forward(self, seq, attn_masks):
        '''
        Inputs:
            -seq : Tensor of shape [B, T] containing token ids of sequences
            -attn_masks : Tensor of shape [B, T] containing attention masks to be used to avoid contibution of PAD tokens
        '''

        # Feeding the input to BERT model to obtain contextualized representations
        outputs = self.bert_layer(seq, attention_mask=attn_masks, return_dict=True)
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
        for seq, attn_masks, labels in dataloader:
            seq, attn_masks, labels = seq.cuda(gpu), attn_masks.cuda(gpu), labels.cuda(gpu)
            logits = net(seq, attn_masks)
            mean_loss += criterion(logits.squeeze(-1), labels.float()).item()
            mean_acc += get_accuracy_from_logits(logits, labels)
            count += 1

    return mean_acc / count, mean_loss / count

def train(net, criterion, opti, train_loader, dev_loader, max_eps, gpu):
    best_acc = 0
    st = time.time()
    for ep in range(max_eps):

        net.train()
        for it, (seq, attn_masks, labels) in enumerate(train_loader):
            # Clear gradients
            opti.zero_grad()
            # Converting these to cuda tensors
            seq, attn_masks, labels = seq.cuda(gpu), attn_masks.cuda(gpu), labels.cuda(gpu)

            # Obtaining the logits from the model
            logits = net(seq, attn_masks)

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
    # Creating instances of training and development set
    # maxlen sets the maximum length a sentence can have
    # any sentence longer than this length is truncated to the maxlen size
    train_set = SSTDataset(filename='10-train.tsv', maxlen=30)
    dev_set = SSTDataset(filename='10-dev.tsv', maxlen=30)
    # Creating intsances of training and development dataloaders
    train_loader = DataLoader(train_set, batch_size=64, num_workers=2)
    dev_loader = DataLoader(dev_set, batch_size=64, num_workers=2)
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
