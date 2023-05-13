import torch
from transformers import BertTokenizer, BertForSequenceClassification
import torch.optim as optim

# 加载 BERT 模型和 tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# 定义模型参数
lr = 2e-5
epochs = 5
batch_size = 32

# 加载数据
train_data = ...
val_data = ...
test_data = ...

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# 训练模型
for epoch in range(epochs):
    model.train()
    for i in range(0, len(train_data), batch_size):
        batch_data = train_data[i:i+batch_size]
        inputs = tokenizer([data[0] for data in batch_data], padding='max_length', truncation=True, max_length=512, return_tensors='pt')
        targets = torch.LongTensor([data[1] for data in batch_data])

        optimizer.zero_grad()
        outputs = model(inputs['input_ids'], attention_mask=inputs['attention_mask'], labels=targets)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

    # 在验证集上测试模型
    model.eval()
    correct = 0
    with torch.no_grad():
        for data in val_data:
            inputs = tokenizer(data[0], padding='max_length', truncation=True, max_length=512, return_tensors='pt')
            target = torch.LongTensor([data[1]])

            output = model(inputs['input_ids'], attention_mask=inputs['attention_mask'], labels=target)
            pred = output.logits.argmax(dim=1)
            correct += pred.eq(target).sum().item()

    accuracy = correct / len(val_data)
    print(f"Epoch {epoch+1}, Val Accuracy: {accuracy:.3f}")

# 在测试集上评估模型
model.eval()
correct = 0
with torch.no_grad():
    for data in test_data:
        inputs = tokenizer(data[0], padding='max_length', truncation=True, max_length=512, return_tensors='pt')
        target = torch.LongTensor([data[1]])

        output = model(inputs['input_ids'], attention_mask=inputs['attention_mask'], labels=target)
        pred = output.logits.argmax(dim=1)
        correct += pred.eq(target).sum().item()

accuracy = correct / len(test_data)
print(f"Test Accuracy: {accuracy:.3f}")
