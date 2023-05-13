import json
import random
import time

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class baseline:
    def __init__(self):
        self.dev_claims = json.load(open("../data/dev-claims.json"))
        self.evidences = json.load(open("../data/evidence.json"))
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.vectorizer.fit(list(self.evidences.values())+[self.dev_claims[c]['claim_text'] for c in self.dev_claims])
        self.evidences_tfidf = self.vectorizer.transform(self.evidences.values())

    def calculate(self, output_path,N):
        print("Start calculating...")
        start = time.time()

        df = pd.DataFrame(columns=['similarity', 'id', 'sentence'])
        df.to_csv(output_path, index=False, header=True, mode='w')

        counter = 0
        for c in self.dev_claims:
            claim_tfidf = self.vectorizer.transform([self.dev_claims[c]['claim_text']])
            similarity = cosine_similarity(claim_tfidf, self.evidences_tfidf).squeeze()
            df = pd.DataFrame({'evidences': self.evidences.keys(), 'similarity': similarity}).sort_values(by=['similarity'], ascending=False)
            potential_relevant_evidences = df.iloc[:N]
            # 加入id 列， id= claim_id + evidence_id
            potential_relevant_evidences.loc[:,'id'] = potential_relevant_evidences['evidences'].apply(lambda x: c + ',' + x)
            # 加入sentence列, sentence = claim_text + evidence_text
            potential_relevant_evidences.loc[:,'sentence'] = self.dev_claims[c]['claim_text'] + '[SEP]' + potential_relevant_evidences['evidences'].apply(lambda x: self.evidences[x])

            # 去掉evidences列
            potential_relevant_evidences = potential_relevant_evidences.drop(columns=['evidences'])

            potential_relevant_evidences.to_csv(output_path, index=False, header=False, mode='a')
            if c == "claim-1834" or c == "claim-871" or c=="claim-139" or c == "claim-1407" or c=="claim-3070" or c=="claim-677" or c=="claim-3063":
                print("----------",c)
            counter += 1
        print("Done for ", counter, " claims.")
        end = time.time()
        print("Time cost: ", end - start)
        print("Done for ", counter, " claims.")

def format_preds(preds_path, unlabelled_claims_path, output_path, k):
    # 将得到的预测格式化
    # 读取 output_pred_path 预测结果
    df = pd.read_csv(preds_path)
    # 找出output_pred_path中的label为1的句子
    # df_pred = df[df['label'] == 1]
    # 根据prob排序
    # df_pred = df_pred.sort_values(by='probs', ascending=False)

    # 读取 dev_claims_path
    with open(unlabelled_claims_path, 'r') as f: # unlabelled_claims_path可以是dev/test
        # 读取JSON数据 - 字典
        claims = json.load(f)

    # 创建空json
    new_claims = {}
    # 遍历claims 找出所有的claim_id
    for claim_id in claims:
        # 创建空list
        new_claims[claim_id] = {}
        new_claims[claim_id]['claim_text'] = claims[claim_id]['claim_text']
        new_claims[claim_id]['claim_label'] = "NAN"
        new_claims[claim_id]['evidences'] = []
    # 遍历df_pred 选出这个claim_id对应的evidence_id最高的k个
    for index, row in df.iterrows():
        id = row['id']
        # probs = row['probs']
        claim_id = id.split(',')[0]
        evidence_id = id.split(',')[1]
        new_claims[claim_id]['evidences'].append(evidence_id)

    # 遍历claims，碰到没有evidence的claim，将随机一个evidence加入到claims中（只是以防错误）
    counter = 0
    for claim_id in new_claims:
        if len(new_claims[claim_id]['evidences']) == 0:
            random_num = random.randint(0, 1208827)
            new_claims[claim_id]['evidences'].append(f"evidence-{random_num}")
            print("This claim has no evidence claim_id:", claim_id)
            counter += 1
    print("How many claims that don't have predictions:", counter)
    # 将claims写入到output_path中
    with open(output_path, 'w') as f:
        json.dump(new_claims, f, indent=2)
    print("format_preds done!")


if __name__ == '__main__':
    # baseline = baseline()
    # baseline.calculate("./data/baseline__first_5.csv", 5)

    format_preds("./data/baseline__first_5.csv", "../data/dev-claims.json", "./data/formated_baseline_first_5.json", 5)
