# -*- coding: utf8 -*-
"""
======================================
    Project Name: NLP
    File Name: text_to_json
    Author: czh
    Create Date: 2021/8/19
--------------------------------------
    Change Activity: 
======================================
"""
# 格式转换为{"text": "郑阿姨就赶到文汇路排队拿钱，希望能将缴纳的一万余元学费拿回来，顺便找校方或者教委要个说法。", "label": {"address": {"文汇路": [[6, 8]]}}}
import codecs
import json
import numpy as np
from nlp.processors import get_entities

data_file = "ner/"


def trans_data(data_type):
    file_name = data_file + f"{data_type}.txt"
    lines = []
    with codecs.open(file_name, encoding='utf8') as fr:
        words = []
        labels = []
        for line in fr:
            if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                if words:
                    lines.append({"text": "".join(words), "labels": labels})
                    words = []
                    labels = []
            else:
                splits = line.split(" ")
                words.append(splits[0])
                if len(splits) > 1:
                    labels.append(splits[-1].replace("\n", ""))
                else:
                    # Examples could have no label for mode = "test"
                    labels.append("O")
        if words:
            lines.append({"text": "".join(words), "labels": labels})
    results = []
    for item in lines:
        text = item["text"]
        labels = item["labels"]
        subjects = get_entities(labels, id2label=None, markup='bios')
        label_dict = {}
        for subject in subjects:
            label = subject[0]
            start = subject[1]
            end = subject[2]
            word = text[start: end+1]
            if label not in label_dict:
                label_dict[label] = {}
            if word not in label_dict[label]:
                label_dict[label][word] = []
            label_dict[label][word].append([start, end])
        results.append({"text": text, "label": label_dict})

    return results


def save_datas(datas, data_type):
    file_name = data_file + f"{data_type}.json"
    with codecs.open(file_name, 'w', encoding='utf8') as fw:
        for line in datas:
            line = json.dumps(line, ensure_ascii=False)
            fw.write(line + '\n')


def main():
    train_datas = trans_data("train")
    dev_datas = trans_data("dev")
    test_datas = trans_data("test")

    all_datas = train_datas + dev_datas + test_datas
    np.random.shuffle(all_datas)

    num = len(all_datas)
    print("total num: ", num)
    train_num = int(num * 0.8)
    train_datas = all_datas[:train_num]
    dev_datas = all_datas[train_num: train_num+int(num*0.1)]
    test_datas = all_datas[train_num+int(num*0.1):]

    print("train num: ", len(train_datas))
    print("dev num: ", len(dev_datas))
    print("test num: ", len(test_datas))

    save_datas(train_datas, "train")
    save_datas(dev_datas, "dev")
    save_datas(test_datas, "test")


if __name__ == "__main__":
    main()
