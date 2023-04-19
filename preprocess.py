import json
import csv
# train-claims.json 有1228条数据 最长的332 chars - 49 words
# 拆成claim和evidence数据对后有4122条数据
# evidence.json 有 1208827条数据 最长的是3148 chars - 479 words
# dev-claims.json 有 154条数据
# test-claims.json 有 153条数据

def prepare_pairs_data(train_or_dev_claims_path, new_file_name):
    # 打开JSON文件
    with open(train_or_dev_claims_path, 'r') as f:
        # 读取JSON数据 - 字典
        train_claims = json.load(f)

    # 打开evidence.json文件
    with open("data/evidence.json", 'r') as f:
        # 读取JSON数据 - 字典
        evidences = json.load(f)

    result = {}
    for key, value in train_claims.items():
        evidences_nos = value['evidences']
        for evidences_no in evidences_nos:
            evi_text = evidences[evidences_no]
            # claim 编号+evidence 编号 (用，连接) = key
            result[key + ',' + evidences_no] = [value['claim_text'], evi_text, value['claim_label']]


    data = [["id", "sentence", "label"]]
    for key, value in result.items():
        data.append([key, value[0] + '[SEP]' + value[1], value[2]])
    # 将result 写入csv文件
    with open(f"data/{new_file_name}", mode='w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerows(data)

    return result  # 返回字典 - 例子: value = [claim, evidence, label]


if __name__=="__main__":
    # 读取JSON文件
    # data = read_json_file("data/test-claims-unlabelled.json")
    value = prepare_pairs_data("data/train-claims.json", "train.csv")
    # 找出claim最长的

    for i in value:
        if len(value[i][0]) == 332:
            print(value[i][0])





