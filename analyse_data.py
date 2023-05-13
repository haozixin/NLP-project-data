import json
import matplotlib.pyplot as plt
import numpy as np


def analyse_data(data_path, field_name):
    """Analyse the data and return the result."""
    # 加载json数据
    with open(data_path, 'r') as f:
        # 读取JSON数据 - 字典
        instance = json.load(f)

    # 统计field_name的平均长度和最大长度 (单词长度)
    word_dict = {}
    total_length = 0
    max_length = 0
    for key in instance:
        length = len(instance[key][field_name].split())
        if length in word_dict:
            word_dict[length] += 1
        else:
            word_dict[length] = 1
        total_length += length
        if length > max_length:
            max_length = length
    avg_length = total_length / len(instance)
    print("avg_length: ", avg_length)
    print("max_length: ", max_length)

    return word_dict

    # # 统计field_name的单词频率， 作图
    # import matplotlib.pyplot as plt
    # # 添加标题和标签
    # plt.title('Claim Text Length Bar Chart in test-claims-unlabelled.json')
    # plt.xlabel('text length')
    # plt.ylabel('frequency')
    #
    # plt.bar(word_dict.keys(), word_dict.values())
    # plt.show()









def analyse_data_for_evidence():
    """Analyse the data and return the result."""
    # 加载json数据
    with open("data/evidence.json", 'r') as f:
        # 读取JSON数据 - 字典
        instance = json.load(f)

    # 统计field_name的平均长度和最大长度 (单词长度)
    total_length = 0
    max_length = 0
    word_dict = {}
    for key in instance:
        length = len(instance[key].split())
        if length in word_dict:
            word_dict[length] += 1
        else:
            word_dict[length] = 1
        total_length += length
        if length > max_length:
            max_length = length
    avg_length = total_length / len(instance)
    print("avg_length: ", avg_length)
    print("max_length: ", max_length)
    return word_dict

    # import matplotlib.pyplot as plt
    # # 添加标题和标签
    # plt.title('Evidence Text Length Bar Chart in evidence.json')
    # plt.xlabel('text length')
    # plt.ylabel('frequency')
    #
    # plt.bar(word_dict.keys(), word_dict.values())
    # plt.show()

def all_chart_in_one():
    # 生成一些数据
    train = analyse_data("data/train-claims.json", "claim_text")
    dev = analyse_data("data/dev-claims.json", "claim_text")
    test = analyse_data("data/test-claims-unlabelled.json", "claim_text")
    evidence = analyse_data_for_evidence()

    train_x = list(train.keys())
    train_y = list(train.values())
    dev_x = list(dev.keys())
    dev_y = list(dev.values())
    test_x = list(test.keys())
    test_y = list(test.values())
    evidence_x = list(evidence.keys())
    evidence_y = list(evidence.values())

    # 创建一个4x1的图像，并将它们分配给4个子图
    fig, axs = plt.subplots(2, 2, figsize=(14, 12))

    # 在每个子图上绘制相应的图形，并设置标题和轴标签
    axs[0][0].bar(train_x, train_y)
    axs[0][0].set_title('Claim Text Length Bar Chart in train-claims.json')
    axs[0][0].set_xlabel('Text Length')
    axs[0][0].set_ylabel('Frequency')

    axs[0][1].bar(dev_x, dev_y)
    axs[0][1].set_title('Claim Text Length Bar Chart in dev-claims.json')
    axs[0][1].set_xlabel('Text Length')
    axs[0][1].set_ylabel('Frequency')

    axs[1][0].bar(test_x, test_y)
    axs[1][0].set_title('Claim Text Length Bar Chart in test-claims-unlabelled.json')
    axs[1][0].set_xlabel('Text Length')
    axs[1][0].set_ylabel('Frequency')

    axs[1][1].bar(evidence_x, evidence_y)
    axs[1][1].set_title('Evidence Text Length Bar Chart in evidence.json')
    axs[1][1].set_xlabel('Text Length')
    axs[1][1].set_ylabel('Frequency')

    # 调整子图之间的间距
    fig.tight_layout()

    # 显示图像
    plt.show()


if __name__=="__main__":
    # analyse_data("data/dev-claims.json", "claim_text")
    # analyse_data("data/test-claims-unlabelled.json", "claim_text")
    # analyse_data_for_evidence()
    all_chart_in_one()