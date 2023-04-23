# BaseInformation
Competition - Automated Fact Checking For Climate Science Claims

Secret url: https://codalab.lisn.upsaclay.fr/competitions/12639?secret_key=319e8c8d-ae76-4f3e-8178-72fe526cac85


# Baseline
|  User  |Date of Last Entry|Team Name|Harmonic Mean of F and A |Evidence Retrieval F-score |Claim Classification Accuracy
|:------:|:---:|:---:|:---:|:---:|:---:|
| zenanz |04/21/23|	Baseline|0.12040 (4)|0.07150 (4)|0.38160 (4)|

# 提升方向

## 算力
越大越好， 略

## batch size
batch_size对模型的效果有一定的影响。较小的batch_size可以使模型更频繁地更新参数，从而加速模型收敛。但是，过小的batch_size可能会导致模型的收敛速度变慢或不稳定，同时还可能增加训练过程中的噪声，影响模型的泛化能力。
较大的batch_size可以充分利用GPU等硬件设备的计算能力，从而使模型训练速度更快，同时还可以平滑掉一些噪声，提高模型的泛化能力。但是，过大的batch_size会占用更多的内存，并可能导致梯度下降过程中出现瓶颈，导致模型训练过程不稳定。
因此，在选择batch_size时，需要考虑到模型的规模、硬件设备的能力以及训练数据的规模等多个因素，并进行适当的实验和调整，以找到最优的batch_size值。

## Models
large>base>small
maxlen 和 batchsize 越大，模型越大，效果越好
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
