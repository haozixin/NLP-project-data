# BaseInformation
Competition - Automated Fact Checking For Climate Science Claims

Secret url: https://codalab.lisn.upsaclay.fr/competitions/12639?secret_key=319e8c8d-ae76-4f3e-8178-72fe526cac85


# Baseline
|  User  |Date of Last Entry|Team Name|Harmonic Mean of F and A |Evidence Retrieval F-score |Claim Classification Accuracy
|:------:|:---:|:---:|:---:|:---:|:---:|
| zenanz |04/21/23|	Baseline|0.12040 (4)|0.07150 (4)|0.38160 (4)|

# Promotion direction

## computing power
The larger the better, slightly

## batch size
The batch_size has an effect on the effect of the model. A smaller batch_size can make the model update parameters more frequently, thus accelerating the model convergence. However, too small batch_size may cause the convergence of the model to be slow or unstable, and may also increase the noise during the training process and affect the generalization ability of the model.
A larger batch_size can make full use of the computational power of hardware devices such as GPU, thus making the model training faster, and also smoothing out some noise to improve the generalization ability of the model. However, too large batch_size will occupy more memory and may lead to bottlenecks in the gradient descent process, resulting in an unstable model training process.
Therefore, when choosing the batch_size, several factors such as the scale of the model, the capacity of the hardware device and the scale of the training data need to be taken into account, and appropriate experiments and adjustments need to be made to find the optimal batch_size value.

## Models
large>base>small
maxlen and batchsize 
1. [Model 1](#model-1) 

条件：
bert-base-uncased; 128 maxlen; 2 epochs; 55 batch; 有segmentId;

同条件下预测结果：
```
claim-752： "[South Australia] has the most expensive electricity in the world."
evidence selected:
evidence-89  :  0.9954309463500975
evidence-508  :  0.9833679795265198
evidence-67732  :  0.9974631071090698
evidence-572512  :  0.9975584745407104
true_evidences: ['evidence-67732', 'evidence-572512']
```

2. [Model 2](#model-2)

条件： ./models/bert_large_128max_18batch_segmentid.dat

同条件下预测结果：
```
evidence selected:
evidence-89  :  0.99286687374115
evidence-243  :  0.9921686053276062
evidence-835  :  0.9916194677352904
evidence-67732  :  0.9973837733268738
evidence-572512  :  0.9974736571311952
true_evidences: ['evidence-67732', 'evidence-572512']
```


3. [Model 3](#model-3)

条件: 
bert-base_64max_32batch_segmentid; 2epochs

同条件下预测结果：
```
evidence selected:
evidence-89  :  0.9985496401786804
evidence-243  :  0.9967302083969116
evidence-508  :  0.9961091876029968
evidence-835  :  0.998505473136902
evidence-67732  :  0.9975588321685792
evidence-572512  :  0.9982074499130248
true_evidences: ['evidence-67732', 'evidence-572512']
```

具体信息（文本）：<br>
**evidence-89**  :  0.9985496401786804 <br>
"Pollution produced from centralised generation of electricity is emitted at a distant power station, rather than \"on site\"." <br>
**evidence-243**: 0.9967302083969116 <br>
"At The Geysers in California, after the first thirty years of power production, the steam supply had depleted and generation was substantially reduced." <br>
**evidence-67732**  :  0.9975588321685792
"[citation needed] South Australia has the highest retail price for electricity in the country."

分析: <br>
由于evidences中存在某些关键字导致 <br><br>

改进方法：
1. 数据增强：对于训练集中的无关句子，可以使用数据增强技术（如随机替换、随机插入、随机删除等）来生成更多的负样本，从而增加模型对无关句子的判别能力，提高模型的鲁棒性。

2. Bert 参数方面 比如：https://huggingface.co/transformers/v3.0.2/model_doc/bert.html#bertconfig

3. Fine-tune不同的BERT模型：可以尝试使用其他的BERT变体或者预训练模型来进行微调，比如RoBERTa、ALBERT等，这些模型在预训练阶段可能已经学习了更多的语义和上下文信息，从而可以提高模型的准确性。

4. 训练数据增加相似度高的evidence 

5. 集成模型：可以将多个不同的模型集成在一起，通过投票或者加权平均的方式进行预测，以提高模型的准确性和鲁棒性。

## Config

### 最初的BertConfig
```
BertConfig {
  "architectures": [
    "BertForMaskedLM"
  ],
  "attention_probs_dropout_prob": 0.1,
  "gradient_checkpointing": false,
  "hidden_act": "gelu",
  "hidden_dropout_prob": 0.1,
  "hidden_size": 768,
  "initializer_range": 0.02,
  "intermediate_size": 3072,
  "layer_norm_eps": 1e-12,
  "max_position_embeddings": 512,
  "model_type": "bert",
  "num_attention_heads": 12,
  "num_hidden_layers": 12,
  "pad_token_id": 0,
  "position_embedding_type": "absolute",
  "transformers_version": "4.28.1",
  "type_vocab_size": 2,
  "use_cache": true,
  "vocab_size": 30522
}
```

1. "architectures"

"BertForMaskedLM" 主要用于训练语言模型，即预测一个句子中被遮挡的词汇。它不是专门用于判断两个句子之间是否有关系的模型，但是可以通过微调（fine-tuning）或在其基础上构建更复杂的模型来实现该任务。

通常用于判断两个句子之间是否有关系的模型是基于 Siamese 架构的模型，例如 "Siamese-BERT"、"Siamese-CNN" 等。这些模型通过将两个句子输入到同一个模型中，然后计算两个句子的相似度得分来判断它们之间是否有关系。其中，Siamese 架构意味着两个句子共享同样的权重参数。

因此，如果您需要判断两个句子之间是否有关系，可以考虑使用基于 Siamese 架构的模型，而不是 "BertForMaskedLM"。

2. "classifier_dropout" + hidden_dropout_prob + attention_probs_dropout_prob

总的来说，分类器中的dropout主要针对模型的过拟合问题，隐藏层中的dropout则更多的是为了模型的泛化性能考虑。
   
3. "gradient_checkpointing": 控制是否使用gradient checkpointing技术来减少内存占用。如果您的模型比较大，可以考虑将该参数设置为True。

4. hidden_size: 隐藏层的神经元数量

5. num_hidden_layers: 隐藏层的层数


[Model 4](#model-4) models/siameseBert_new_train_data.dat
训练时，accuracy低，无法聚合，偏向把对的预测成错的。 new_train_data中的数据不平衡，容易带偏

[Model 5](#model-5) models/siameseBert.dat
其他都没变，只改了模型


## 检查代码更新，输出路径，数据（新的dev和train），TODO里的tips , 参数maxleng和batch_size

- new_train_data, 和new_train_dev中是预测错的数据，人工纠正后要作为新训练数据
- data/demo_evaluation/demo_dev_claims_evi_pairs_for_predict.csv 是一个claim的所有evidences组合
- 新的训练数据 按比例结合正确数据，预测错误后标记数据，和随机数据： x:x:2;   dev 原有的：预测错的：随机的 = 1:1:1
- 
TODO: 
1. 用sentence_transformer做训练数据和dev, 用新的训练数据和原来dev测看有没有变好
2. 训练sentence_transformer模型（报告用）
3. sentence_transformer 直接做预测，做baseline
4. tfidf做baseline2