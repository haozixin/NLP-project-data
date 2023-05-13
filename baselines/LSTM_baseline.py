import os

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.metrics import f1_score
from torch.utils.data import Dataset, DataLoader
import time

MODEL_FILE = 'sstcls_0.mdl'
class SSTDataset(Dataset):

    def __init__(self, filename, maxlen):

        # Store the contents of the file in a pandas dataframe
        self.df = pd.read_csv(filename, delimiter=',')

        # Initialize the tokenizer
        # 初始化tokenizer
        words = set()
        for sentence in self.df['sentence']:
            claim, evidence = sentence.split("[SEP]")
            words |= set(claim.split() + evidence.split())
        self.tokenizer = {word: i + 1 for i, word in enumerate(words)}

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

        # 获取句子的token ids
        tokens_ids = [self.tokenizer[word] for word in claim.split() + evidence.split()]

        # Padding or truncating the token ids
        if len(tokens_ids) < self.maxlen:
            tokens_ids += [0] * (self.maxlen - len(tokens_ids))
        else:
            tokens_ids = tokens_ids[:self.maxlen]

        # return as tensors
        tokens_ids_tensor = torch.tensor(tokens_ids, dtype=torch.long)
        label_tensor = torch.tensor(label, dtype=torch.float32)

        return tokens_ids_tensor, label_tensor

class PDataset(Dataset):

    def __init__(self, filename, maxlen):

        # Store the contents of the file in a pandas dataframe
        self.df = pd.read_csv(filename, delimiter=',')

        # Initialize the tokenizer
        # 初始化tokenizer
        words = set()
        for sentence in self.df['sentence']:
            claim, evidence = sentence.split("[SEP]")
            words |= set(claim.split() + evidence.split())
        self.tokenizer = {word: i + 1 for i, word in enumerate(words)}

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

        # 获取句子的token ids
        tokens_ids = [self.tokenizer[word] for word in claim.split() + evidence.split()]

        # Padding or truncating the token ids
        if len(tokens_ids) < self.maxlen:
            tokens_ids += [0] * (self.maxlen - len(tokens_ids))
        else:
            tokens_ids = tokens_ids[:self.maxlen]

        # return as tensors
        tokens_ids_tensor = torch.tensor(tokens_ids, dtype=torch.long)
        # label_tensor = torch.tensor(label, dtype=torch.float32)

        return tokens_ids_tensor

class LSTMClassifier(nn.Module):

    def __init__(self, vocab_size, hidden_dim):
        super(LSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.lstm = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.embedding(x)
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out[:, -1, :]  # 取最后一个时间步的输出
        logits = self.fc(lstm_out)
        return logits.squeeze(-1)


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
        for seq, labels in dataloader:
            seq, labels = seq.cuda(gpu), labels.cuda(gpu)
            logits = net(seq)
            mean_loss += criterion(logits.squeeze(-1), labels).item()
            mean_acc += get_accuracy_from_logits(logits, labels)
            # 计算F1-score
            f1_score += get_f1_from_logits(logits,labels)
            count += 1

    return mean_loss / count, mean_acc / count, f1_score / count

def train(net, criterion, optimizer, train_loader, val_loader, epochs, gpu):
    train_losses, train_accs, val_losses, val_accs, f1_scores = [], [], [], [], []
    best_f1 = 0
    for epoch in range(epochs):
        start_time = time.time()

        net.train()
        train_loss, train_acc = 0, 0
        counter = 0
        for seq, labels in train_loader:
            counter += 1
            seq, labels = seq.cuda(gpu), labels.cuda(gpu)
            optimizer.zero_grad()
            logits = net(seq)
            loss = criterion(logits.squeeze(-1), labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_acc += get_accuracy_from_logits(logits, labels)

            if counter % 100 == 0:
                print(f'Iteration {counter} of Epoch: {epoch}, Train Loss: {train_loss / counter:.4f}, '
                      f'Train Acc: {train_acc / counter:.4f}')

        train_loss /= len(train_loader)
        train_acc /= len(train_loader)
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        val_loss, val_acc, f1_score = evaluate(net, criterion, val_loader, gpu)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        f1_scores.append(f1_score)

        print(f'Epoch: {epoch + 1} Completed, '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, F1-score: {f1_score:.4f}, '
              f'Time: {time.time() - start_time:.2f}s')

        if f1_score > best_f1:
            print("Best development accuracy improved from {} to {}, saving model...".format(best_f1, f1_score))
            best_f1 = f1_score
            torch.save(net.state_dict(), 'sstcls_{}.mdl'.format(epoch))

    return train_losses, train_accs, val_losses, val_accs, f1_scores

class Predictor:
    def __init__(self, maxlen, gpu):
        train_set = SSTDataset('../data/old_train.csv', maxlen=256)

        self.classifier = LSTMClassifier(vocab_size=len(train_set.tokenizer)+1, hidden_dim=512)
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
        self.dataloader = DataLoader(self.predict_dataset, batch_size=800, shuffle=False,
                                     num_workers=20)

        with torch.no_grad():
            all_preds = []
            all_probs = []
            for seq in self.dataloader:
                seq = seq.cuda(self.gpu)
                logits = self.classifier(seq)
                probs = torch.sigmoid(logits.unsqueeze(-1))
                soft_probs = (probs > 0).long()
                t = soft_probs.squeeze().cpu()
                all_preds.extend(t.numpy())
                all_probs.extend(probs.squeeze().cpu().numpy())


            # 将预测结果写入文件
            df = self.predict_dataset.df
            # 2. 将预测概率和值写入到csv文件中
            df['probs'] = all_probs
            df['label'] = all_preds
            # 选出label为1的数据
            df = df[df['label'] == 1]
            # 3. 保存csv文件
            df.to_csv(self.output_file_path, index=False, header=False, mode='a')


def predict(dataset_for_predict_path, output_dataset_path):
    maxlen = 256
    # predict_dataset_path = "./data/demo_dev_for_predict.csv"
    # output_dataset_path = "./data/demo_dev_for_predict_output2.csv"
    gpu = 0
    chunk_size = 10000
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
    # # Initialize the dataset and data loader (note the +1 in vocab size for the 0 padding token)
    # train_set = SSTDataset('../data/old_train.csv', maxlen=256)
    # val_set = SSTDataset('../data/old_dev.csv', maxlen=256)
    # train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    # val_loader = DataLoader(val_set, batch_size=64, shuffle=False)
    # # Initialize the model, optimizer, and criterion
    # model = LSTMClassifier(vocab_size=len(train_set.tokenizer)+1, hidden_dim=512)
    # optimizer = optim.Adam(model.parameters(), lr=1e-3)
    # criterion = nn.BCEWithLogitsLoss()
    #
    # # Use GPU if available
    # gpu = 0 if torch.cuda.is_available() else None
    # if gpu is not None:
    #     model = model.cuda(gpu)
    #     criterion = criterion.cuda(gpu)
    #
    # # Train the model
    # train_losses, train_accs, val_losses, val_accs, f1_scores = train(model, criterion, optimizer,
    #                                                                   train_loader, val_loader,
    #                                                                   epochs=30, gpu=gpu)

    output_dataset_path = "baseline_retrieve_output.csv"
    dataset_for_predict_path = "../data/similarity_filtered/dev_output_10000_include_stopword.csv"

    predict(dataset_for_predict_path, output_dataset_path)