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


def merge_all_match_datas(input_dir, output_dir):
    allow_name_list = ['ATEC', 'BQ', 'LCQMC', 'PAWSX']
    all_train_datas = []
    all_dev_datas = []
    all_test_datas = []
    for file_name in allow_name_list:
        file_dir = os.path.join(input_dir, file_name)
        assert os.path.exists(file_dir)
        train_file_path = os.path.join(file_dir, file_name+'.train.data')
        valid_file_path = os.path.join(file_dir, file_name+'.valid.data')
        test_file_path = os.path.join(file_dir, file_name+'.test.data')

        def load_datas(input_file, data_type):
            assert os.path.exists(input_file)
            assert data_type in input_file
            print(f"loading {input_file}")
            with codecs.open(input_file, encoding='utf8') as fr:
                for line in fr:
                    line = line.strip()
                    if not line:
                        continue
                    sents = line.split('\t')
                    if len(sents) != 3:
                        continue
                    if data_type == 'train':
                        if line not in all_train_datas:
                            all_train_datas.append(line)
                    elif data_type == 'valid':
                        if line not in all_dev_datas:
                            all_dev_datas.append(line)
                    elif data_type == 'test':
                        if line not in all_test_datas:
                            all_test_datas.append(line)
                    else:
                        raise ValueError('[data_type] must be one of [train, valid, test]')

        load_datas(train_file_path, 'train')
        load_datas(valid_file_path, 'valid')
        load_datas(test_file_path, 'test')

    print("train datas num: ", len(all_train_datas))
    print("valid datas num: ", len(all_dev_datas))
    print("test datas num: ", len(all_test_datas))

    def save_datas(datasets, data_type):
        output_file = os.path.join(output_dir, '-'.join(allow_name_list)+f'_{data_type}.txt')
        with codecs.open(output_file, 'w', encoding='utf8') as fw:
            for line in datasets:
                fw.write(line + "\n")

    save_datas(all_train_datas, 'train')
    save_datas(all_dev_datas, 'valid')
    save_datas(all_test_datas, 'test')


def merge_all_nli_datas(input_dir):
    allow_names = ['MNLI', 'SNLI']

    train_datasets = set()
    valid_datasets = set()
    test_datasets = set()
    for name in allow_names:
        dir_path = os.path.join(input_dir, name)
        assert os.path.exists(dir_path)
        train_file_path = os.path.join(dir_path, 'train.json')
        dev_file_path = os.path.join(dir_path, 'dev.json')
        test_file_path = os.path.join(dir_path, 'test.json')

        def load_datas(input_file, data_type):
            assert data_type in input_file
            assert os.path.exists(input_file)
            print(f"loading datas from {input_file}")
            with codecs.open(input_file, encoding='utf8') as fr:
                for line in fr:
                    line = line.strip()
                    if not line:
                        continue
                    if data_type == 'train':
                        train_datasets.add(line)
                    elif data_type == 'dev':
                        valid_datasets.add(line)
                    elif data_type == 'test':
                        test_datasets.add(line)

        load_datas(train_file_path, 'train')
        load_datas(dev_file_path, 'dev')
        load_datas(test_file_path, 'test')

    print("train datas num: ", len(train_datasets))
    print("valid datas num: ", len(valid_datasets))
    print("test datas num: ", len(test_datasets))

    def save_datas(datasets, data_type):
        output_dir = os.path.join(input_dir, '-'.join(allow_names)+f'.{data_type}.json')
        print(f"save datas to {output_dir}")
        with codecs.open(output_dir, 'w', encoding='utf8') as fw:
            for d in datasets:
                fw.write(d+'\n')

    save_datas(train_datasets, 'train')
    save_datas(valid_datasets, 'dev')
    save_datas(test_datasets, 'test')


if __name__ == "__main__":
    # split_data('/Users/czh/Documents/qkb_org_name.json', '/Users/czh/Downloads/qkb_org_name/')
    # merge_all_match_datas("/root/work2/work2/chenzhihao/datasets/chinese-semantics-match-dataset",
    #                       "/root/work2/work2/chenzhihao/datasets/chinese-semantics-match-dataset")
    merge_all_nli_datas("/root/work2/work2/chenzhihao/datasets/chinese-semantics-match-dataset")
