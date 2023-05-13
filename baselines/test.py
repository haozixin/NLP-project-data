import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer

if __name__ == '__main__':



    # 定义数据集类
    class TextDataset(Dataset):
        def __init__(self, texts, labels, tokenizer, max_len):
            self.texts = texts
            self.labels = labels
            self.tokenizer = tokenizer
            self.max_len = max_len

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, index):
            text = self.texts[index]
            label = self.labels[index]
            encoding = self.tokenizer.encode_plus(text,
                                                  add_special_tokens=True,
                                                  max_length=self.max_len,
                                                  padding='max_length',
                                                  return_attention_mask=True,
                                                  return_tensors='pt')
            input_ids = encoding['input_ids'][0]
            attention_mask = encoding['attention_mask'][0]
            return {'input_ids': input_ids, 'attention_mask': attention_mask, 'label': torch.tensor(label)}


    # 读取数据
    data = pd.read_csv('data.csv', encoding='utf-8')
    texts = data['text'].values
    labels = data['label'].values

    # 数据预处理
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    max_len = 100
    dataset = TextDataset(texts, labels, tokenizer, max_len)
    train_set, test_set = train_test_split(dataset, test_size=0.2, random_state=42)
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=32, shuffle=False)


    # 定义模型
    class LSTMClassifier(nn.Module):
        def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim):
            super().__init__()
            self.embedding = nn.Embedding(input_dim, embedding_dim)
            self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
            self.fc = nn.Linear(hidden_dim, output_dim)

        def forward(self, input_ids, attention_mask):
            embedded = self.embedding(input_ids)
            outputs, _ = self.lstm(embedded)
            last_output = outputs[:, -1, :]
            logits = self.fc(last_output)
            return logits


    # 训练模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LSTMClassifier(len(tokenizer), 32, 64, 1).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    num_epochs = 10
    for epoch in range(num_epochs):
        train_loss = 0.0
        train_acc = 0.0
        model.train()
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].float().to(device)
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(batch)
            train_acc += ((outputs > 0) == labels.byte()).sum().item()
        train_loss /= len(train_set)
        train_acc /= len(train_set)

        # 在测试集上评估模型
