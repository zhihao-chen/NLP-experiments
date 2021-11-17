# -*- coding: utf8 -*-
"""
======================================
    Project Name: BERT-NER-Pytorch
    File Name: json_to_text
    Author: czh
    Create Date: 2021/6/24
--------------------------------------
    Change Activity: 
======================================
"""
import json


file_names = ['train.json', 'dev.json']
for name in file_names:
    print("processing the file {}".format(name))
    prefix = name.split('.')[0]
    with open(name, encoding="utf-8") as fr, open(prefix+'.txt', 'w', encoding='utf8') as fw:
        for line in fr:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            text = data['text']
            tokens = list(text)
            labels = ['O' for _ in tokens]
            try:
                if data.get('label'):
                    for label, dic in data['label'].items():
                        for token, lsts in dic.items():
                            for lst in lsts:
                                labels[lst[0]] = "B-"+label
                                labels[lst[0]+1:lst[-1]+1] = ["I-"+label]*(lst[-1]-lst[0])
            except Exception as e:
                print(data)
                raise e
            for j,t in enumerate(tokens):
                fw.write(t+' '+labels[j]+'\n')
            fw.write('\n')
    print("has processed the file {}".format(name))
