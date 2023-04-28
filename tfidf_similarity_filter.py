import os.path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer
from torch.nn.functional import cosine_similarity
from torch.utils.data import Dataset, DataLoader


# 定义数据集和数据加载器
class MyDataset(Dataset):
    def __init__(self, filename):
        self.filename = filename
        self.df = pd.read_csv(filename, delimiter=',')
        self.tfidf = TfidfVectorizer()
        self.wordLen = 512

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        sentence = self.df.loc[index, 'sentence']
        s1 = sentence.split("[SEP]")[0]
        s2 = sentence.split("[SEP]")[1]



        # tfidf tokenizer
        vector = self.tfidf.fit_transform([s1, s2]).toarray()
        s1 = vector[0]
        s2 = vector[1]

        # # 对向量进行填充或切割
        if len(s1) < self.wordLen:
            s1 = np.pad(s1, (0, self.wordLen - len(s1)), 'constant', constant_values=(0, 0))
            s2 = np.pad(s2, (0, self.wordLen - len(s2)), 'constant', constant_values=(0, 0))
        else:
            s1 = s1[:self.wordLen]
            s2 = s2[:self.wordLen]


        s1 = torch.from_numpy(s1).float()
        s2 = torch.from_numpy(s2).float()
        index_tensor = torch.tensor([index])

        return s1, s2, index_tensor


# 定义模型
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()

    def forward(self, s1_gpu, s2_gpu, index):
        sims = []
        for i in range(len(s1_gpu)):
            sim = cosine_similarity(s1_gpu[i], s2_gpu[i], dim=0)
            sims.append(sim)
        return sims, index


class TfidfSimilarityFilter:
    def __init__(self, filename, k, output_dataset_path):
        self.dataset = MyDataset(filename)
        self.dataloader = DataLoader(self.dataset, batch_size=5, shuffle=False, num_workers=1)
        self.k = k
        self.output_dataset_path = output_dataset_path
        self.filter = MyModel()
        self.filter.cuda(0)

    def calculate(self, temp_file_path):
        # 一次性计算一个claim对应所有的evidence的相似度
        result = []
        for s1, s2, index in self.dataloader:
            s1_gpu, s2_gpu, index_gpu = s1.to('cuda'), s2.to('cuda'), index.to('cuda')
            similarity, index = self.filter(s1_gpu, s2_gpu, index_gpu)
            for i in range(len(similarity)):
                similarity_num = similarity[i].item()
                index_num = index[i].item()
                if similarity_num == 0.0:
                    continue
                instance = (similarity_num, index_num)
                result.append(instance)

        if len(result) != 0:
            # 对result 根据元素的similarity进行排序，取前k个
            result.sort(key=lambda x: x[0], reverse=True)
            result = result[:self.k]

            df_data = pd.read_csv(temp_file_path, delimiter=',')
            # 取出result中的index对应的row
            result = [[df_data.loc[index, "id"], df_data.loc[index, "sentence"], similarity]  for similarity, index in result]

            # 将result中的row写入到output_dataset_path中
            df = pd.DataFrame(data=result, columns=['id', 'sentence', 'similarity'])
            df.to_csv(self.output_dataset_path, index=False, header=False, mode='a')


def filter(pairs_data_path, output_path):
    chunk_size = 15
    k=5

    if os.path.exists(output_path):
        os.remove(output_path)
        print("Remove old file.", output_path)

    df = pd.DataFrame(columns=['id', 'sentence', 'similarity'])
    # 将标题先写入到output_dataset_path中
    df.to_csv(output_path, index=False, header=True, mode='w')

    for i, chunk in enumerate(pd.read_csv(pairs_data_path, chunksize=chunk_size)):
        temp_file_path = f"./data/similarity_filtered/temp/temp{i}.csv"
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
            print("Removed old file: ", temp_file_path)

        chunk.to_csv(temp_file_path, mode='a', index=False)
        print("Calculating for chunk:", i, " - data in the file:", temp_file_path)
        TfidfSimilarityFilter(temp_file_path, k, output_path).calculate(temp_file_path)
        print("Done for chunk:", i, " - data in the file:", temp_file_path)

        # 删除临时文件
        os.remove(temp_file_path)
        print("Removed temp file: ", temp_file_path)


if __name__ == '__main__':
    filter("./data/test_claims_evi_pairs_for_predict.csv", "./data/similarity_filtered/output.csv")
