import torch
import torch.nn as nn
import torch.optim as optim

# 构建 other_methods 模型
class RNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(RNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 2)

    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])
        return out

# 定义模型参数
vocab_size = 10000
embedding_dim = 100
hidden_dim = 100
lr = 0.001
epochs = 10
batch_size = 32

# 加载数据
train_data = ...
val_data = ...
test_data = ...

# 定义模型、损失函数、优化器
model = RNN(vocab_size, embedding_dim, hidden_dim)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# 训练模型
for epoch in range(epochs):
    model.train()
    for i in range(0, len(train_data), batch_size):
        batch_data = train_data[i:i+batch_size]
        inputs = torch.LongTensor([data[0] for data in batch_data])
        targets = torch.LongTensor([data[1] for data in batch_data])

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

    # 在验证集上测试模型
    model.eval()
    correct = 0
    with torch.no_grad():
        for data in val_data:
            inputs = torch.LongTensor([data[0]])
            target = torch.LongTensor([data[1]])

            output = model(inputs)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()

    accuracy = correct / len(val_data)
    print(f"Epoch {epoch+1}, Val Accuracy: {accuracy:.3f}")

# 在测试集上评估模型
model.eval()
correct = 0
with torch.no_grad():
    for data in test_data:
        inputs = torch.LongTensor([data[0]])
        target = torch.LongTensor([data[1]])

        output = model(inputs)
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()

accuracy = correct / len(test_data)
print(f"Test Accuracy: {accuracy:.3f}")
