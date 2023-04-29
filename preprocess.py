import json
import csv
import random
import pandas as pd

# train-claims.json 有1228条数据 最长的332 chars - 49 words
# 拆成claim和evidence数据对后有4122条数据
# evidence.json 有 1208827条数据 最长的是3148 chars - 479 words mean  编号：evidence-1208826
# dev-claims.json 有 154条数据
# test-claims.json 有 153条数据
# {SUPPORTS(520+68=588, 0.424), REFUTES(200+27=227, 0.164), NOT_ENOUGH_INFO(386+41=427, 0.308), DISPUTED(124+19=143, 0.104)}
# 1385
HAS_RELATION = 1
NO_RELATION = 0
NO_RELATION_NUM = 5

def prepare_train_pairs_data(train_or_dev_claims_path, new_file_name):
    # 打开JSON文件
    with open(train_or_dev_claims_path, 'r') as f:
        # 读取JSON数据 - 字典
        train_claims = json.load(f)

    # 打开evidence.json文件
    with open("data/evidence.json", 'r') as f:
        # 读取JSON数据 - 字典
        evidences = json.load(f)
        num_of_evidences = len(evidences)-1

    result = {}
    for key, value in train_claims.items():
        evidences_nos = value['evidences']
        for evidences_no in evidences_nos:
            evi_text = evidences[evidences_no]
            # claim 编号+evidence 编号 (用，连接) = key
            result[key + ',' + evidences_no] = [value['claim_text'], evi_text, HAS_RELATION]
        # 从evidence.json中找出没有被train_claims使用的evidence
        # 用于生成没有关系的数据对, 指定生成个数NO_RELATION_NUM
        counter = 0
        while counter < NO_RELATION_NUM:
            index = random.randint(0, num_of_evidences)
            text_index = list(evidences.keys())[index]
            if text_index in evidences_nos:
                continue
            evi_text = evidences[text_index]
            result[key + ',' + text_index] = [value['claim_text'], evi_text, NO_RELATION]
            counter += 1

    # 均匀打乱数据
    random.seed(69)
    print(f"From {train_or_dev_claims_path} get {len(result)} pairs data after adding no relation pairs.")
    result = dict(random.sample(result.items(), len(result)))

    data = [["id", "sentence", "label"]]
    for key, value in result.items():
        data.append([key, value[0] + '[SEP]' + value[1], value[2]])
    # 将result 写入csv文件
    with open(f"data/{new_file_name}", mode='w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerows(data)

    return result  # 返回字典 - 例子: value = [claim, evidence, label]

def data_analise(path):
    # 用pandas从csv加载数据
    print(f"====================={path}=====================")
    data = pd.read_csv(path)

    # 打印数据的形状
    print("Data Shape: ", data.shape)
    # 打印数据的列名
    print("Data Columns: ", data.columns)
    # 打印数据的统计信息
    print("Data Describe:")
    print(data.describe())

def check_correctness(generate_data_path, train_or_dev_claims_path):
    # 检查生成的训练集和验证集是否正确
    train_data = []
    train_claims = {}
    with open(generate_data_path, 'r', encoding="utf-8-sig") as f:
        next(f)
        train_data = list(csv.reader(f))
    with open(train_or_dev_claims_path, 'r') as f2:
        train_claims = json.load(f2)

    for row in train_data:
        if row[2] == str(HAS_RELATION):
            claim_id, evidence_id = row[0].split(',')
            if evidence_id not in train_claims[claim_id]['evidences']:
                print(f"Error: {row[0]}")
                return
        else:
            claim_id, evidence_id = row[0].split(',')
            if evidence_id in train_claims[claim_id]['evidences']:
                print(f"Error: {row[0]}")
                return

    print("Correct!")


def prepare_test_data(dev_claims_path, new_file_name):
    """
    生成测试集 - 一对一对的数据
    :param dev_claims_path: unlabelled 的数据
    :param new_file_name: 生成的文件 - 为后续预测做准备
    """
    # 打开JSON文件
    with open(dev_claims_path, 'r') as f:
        # 读取JSON数据 - 字典
        predict_claims = json.load(f)

    # 打开evidence.json文件
    with open("data/evidence.json", 'r') as f:
        # 读取JSON数据 - 字典
        evidences = json.load(f)
        num_of_evidences = len(evidences)-1

    # 清空文件，并写入标题
    data = [["id", "sentence"]]
    # 将result 写入csv文件
    with open(f"data/{new_file_name}", mode='w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerows(data)

    # 遍历claims 和 每个evidence 生成新的数据
    counter = 1
    for key, value in predict_claims.items():
        if counter == 1:
            counter += 1
            continue
        result = {}
        print("generate data for key: ", key)
        for evidences_no in evidences.keys():
            evi_text = evidences[evidences_no]
            # claim 编号+evidence 编号 (用，连接) = key
            result[key + ',' + evidences_no] = [value['claim_text'], evi_text]
    # # ======================记得删除================================================
    # # 添加有关系的数据对
    #     evidences_nos = value['evidences']
    #     for evidences_no in evidences_nos:
    #         evi_text = evidences[evidences_no]
    #         # claim 编号+evidence 编号 (用，连接) = key
    #         result[key + ',' + evidences_no] = [value['claim_text'], evi_text]
    #
    # # ======================================================================
        data = []
        for key, value in result.items():
            data.append([key, value[0] + '[SEP]' + value[1]])
        # 将result 写入csv文件
        with open(f"data/{new_file_name}", mode='a', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerows(data)

        break
    print("prepare_test_data Done!")


def sort_output_based_on_probs(file_path):
    df = pd.read_csv(file_path)
    df = df.sort_values(by=['probs'], ascending=False)
    # 只打印两列
    print(df[["id", "probs"]].head(10))
    # print(df[df["id", "probs"]].head(5))


def get_two_from_new_train_data():
    # 每个claim 选两个
    dic_counter = {}

    rows = []
    # 读取csv文件
    with open("data/new_train_data2.csv", mode='r', newline='', encoding='utf-8-sig') as f:
        reader = csv.reader(f)
        for row in reader:
            calim = row[0].split(",")[0]
            if calim not in dic_counter.keys():
                dic_counter[calim] = 1
                rows.append(row)
            elif dic_counter[calim] == 1:
                dic_counter[calim] +=1
                rows.append(row)
            else:
                continue

    df = pd.DataFrame(rows, columns=["id", "sentence", "label"])
    df.to_csv("data/train3.csv", index=False, mode='a', header=False, encoding='utf-8-sig')

def conbine_train_train3():
    df1 = pd.read_csv("data/new_dev_data.csv.csv")
    df2 = pd.read_csv("data/train3.csv")
    df = pd.concat([df1, df2], axis=0)
    # 打乱 - shufftle
    df = df.sample(frac=1)
    df.to_csv("data/train3.csv", index=False, encoding='utf-8-sig')


def prepare_negative_sample(new_train_data_all, output_path, N):
    # 每个claim 选两个
    dic_counter = {}
    rows = []
    # 读取csv文件
    with open(new_train_data_all, mode='r', newline='', encoding='utf-8-sig') as f:
        reader = csv.reader(f)
        for row in reader:
            calim = row[0].split(",")[0]
            if calim not in dic_counter.keys():
                dic_counter[calim] = 1
                rows.append(row)
            elif dic_counter[calim] < N:
                dic_counter[calim] +=1
                rows.append(row)


    # 写入数据rows 到csv文件
    df = pd.DataFrame(rows, columns=["id", "sentence", "label"])
    df.to_csv(output_path, index=False, mode='w', header=False, encoding='utf-8-sig')

def prepare_positive_sample():
    # 生成positive sample并检查negative sample 是不是没有positive sample
    df = pd.read_csv("data/old_dev.csv")
    df = df[df["label"] == 1]
    df.to_csv("data/new_data/dev_positive.csv", index=False, mode='w', header=True, encoding='utf-8-sig')

    # # 检查df 中 的值 的id 应该和negative_sample_for_new_train.csv 中的id完全不一样
    # df_positive_id = df["id"].tolist()
    #
    # df2 = pd.read_csv("data/new_data/negative_sample_for_new_train_3_for_each_claim.csv")
    # df_negative_id = df2["id"].tolist()
    #
    # for id in df_positive_id:
    #     if id in df_negative_id:
    #         print("error")
    #         break


def prepare_negative_sample_for_dev(positive_data, negative_data, output_path):
    with open("./data/train-claims.json", mode='r', newline='', encoding='utf-8-sig') as f: # 根据dev/train 换变量
        dev_claims = json.load(f)

    dic_counter = {}
    repeat_set = set()
    rows = []
    df = pd.read_csv(positive_data)
    # 获取positive_data id中的claim
    positive_claim_id = [i.split(",")[0] for i in df["id"].tolist()]

    # df["id"]中的元素放到repeat_set中
    for id in df["id"].tolist():
        repeat_set.add(id)

    # 放到字典里统计个数
    for id in positive_claim_id:
        if id not in dic_counter.keys():
            dic_counter[id] = 1
        else:
            dic_counter[id] += 1

    with open(negative_data, mode='r', newline='', encoding='utf-8-sig') as f:
        reader = csv.reader(f)
        # 去掉第一行
        next(reader)
        for row in reader:
            calim = row[0].split(",")[0]
            # 如果calim在positive中出现过并且dic_counter中的值还没到0
            if calim in dic_counter.keys() and dic_counter[calim] > 0:
                rows.append(row)
                dic_counter[calim] -= 1
                repeat_set.add(row[0])
            else:
                if calim not in dic_counter.keys():
                    print("error(预测的claim在positive中没出现): ", calim)

    # 随机补全 - dic_counter中值还没到0的claim
    for key in dic_counter.keys():
        if dic_counter[key] > 0:
            # 去evidence中随机组合
            # 打开evidence.json文件
            with open("data/evidence.json", 'r') as f:
                # 读取JSON数据 - 字典
                evidences = json.load(f)
                for round in range(dic_counter[key]):
                    index = random.randint(0, 1208827)
                    text_index = list(evidences.keys())[index]
                    id = key + ',' + text_index
                    if id not in repeat_set:
                        sentence = dev_claims[key]["claim_text"] + '[SEP]' + evidences[text_index]
                        rows.append([id, sentence, 0])
                        dic_counter[key] -= 1
                        repeat_set.add(id)
    # 每个claim 加两个随机的
    for key in dic_counter.keys():
        for round in range(2):
            index = random.randint(0, 1208827)
            text_index = list(evidences.keys())[index]
            id = key + ',' + text_index
            if id not in repeat_set:
                sentence = dev_claims[key]["claim_text"] + '[SEP]' + evidences[text_index]
                rows.append([id, sentence, 0])
                dic_counter[key] -= 1
                repeat_set.add(id)


    # 写入数据rows 到csv文件
    df2 = pd.DataFrame(rows, columns=["id", "sentence", "label"])
    # 混合df1和df2
    df_final = pd.concat([df, df2], axis=0)
    # shuffle
    df_final = df_final.sample(frac=1)

    df_final.to_csv(output_path, index=False, mode='w', header=True, encoding='utf-8-sig')











if __name__=="__main__":
    prepare_negative_sample_for_dev("data/new_data/train_positive.csv", "data/new_data/new_train_negative_data_all.csv", "data/new_data/train.csv")

    # conbine_train_train3()
    # sort_output_based_on_probs("./data/retrieve_output/demo_dev_claims_evi_pairs_for_predict_output.csv")
    # prepare_test_data("data/dev-claims.json", "demo_dev_claims_evi_pairs_for_predict.csv")
    # prepare_test_data("data/test-claims-unlabelled.json", "test_claims_evi_pairs_for_predict.csv")
    # check_correctness("data/ole_train.csv", "data/train-claims.json")
    # check_correctness("data/old_dev.csv", "data/dev-claims.json")





