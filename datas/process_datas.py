# -*- coding: utf8 -*-
"""
======================================
    Project Name: NLP
    File Name: process_datas
    Author: czh
    Create Date: 2021/9/23
--------------------------------------
    Change Activity: 
======================================
"""
import os
import codecs
import json
import random


def split_data(input_file, output_file):
    all_samples = []
    with codecs.open(input_file, encoding='utf8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d_line = json.loads(line)
            all_samples.append(d_line)

    total_num = len(all_samples)
    indexs = list(range(total_num))
    random.shuffle(indexs)

    def save_datas(idxs, data_type):
        file_name = os.path.join(output_file, f"{data_type}.json")

        with codecs.open(file_name, 'w', encoding="utf8") as fw:
            for i in idxs:
                line = all_samples[i]
                fw.write(json.dumps(line, ensure_ascii=False) + '\n')

    train_indexs = indexs[: int(total_num*0.8)]
    dev_indexs = indexs[int(total_num*0.8): int(total_num*0.9)]
    test_indexs = indexs[int(total_num*0.9):]

    print("train set length: ", len(train_indexs))
    print("dev set length: ", len(dev_indexs))
    print("test set index: ", len(test_indexs))

    save_datas(train_indexs, "train")
    save_datas(dev_indexs, "dev")
    save_datas(test_indexs, "test")


if __name__ == "__main__":
    split_data('/Users/czh/Documents/qkb_org_name.json', '/Users/czh/Downloads/qkb_org_name/')
