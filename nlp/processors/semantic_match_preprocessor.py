#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: czh
@email:
@date: 2022/6/28 17:00
"""
import codecs
from typing import List
import json

import torch
from torch.utils.data import Dataset


def load_data(data_path: str, seg_tag='\t'):
    """
    只适用txt文件，每行有3列数据
    :param data_path:
    :param seg_tag:
    :return:
    """
    all_data = []
    with codecs.open(data_path, encoding="utf8") as fr:
        for line in fr:
            line = line.strip()
            lst_line = line.split(seg_tag)
            if len(lst_line) != 3:
                continue
            data = [lst_line[0].strip(), lst_line[1].strip(), int(lst_line[2].strip())]
            all_data.append(data)
    return all_data


def load_data_for_snli(data_path: str, return_list=False):
    sent_dict = {}
    sentences = []
    with codecs.open(data_path, encoding="utf8") as fr:
        for line in fr:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            sentences.append(data)
            s1 = data['sentence1'].strip()
            s2 = data['sentence2'].strip()
            label = data['gold_label'].strip()

            if s1 not in sent_dict:
                sent_dict[s1] = {'contradiction': set(), 'entailment': set(), 'neutral': set()}
            sent_dict[s1][label].add(s2)
    if return_list:
        return sentences
    else:
        return sent_dict


class SentDataSet(Dataset):
    def __init__(self, dataset: List, tokenizer, max_seq_length: int, task_type='nli'):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.task_type = task_type

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        if self.task_type.lower() == "nli":
            assert len(data) == 3
            anchor_sent, pos_sent, neg_sent = data
            label = None
        else:
            if len(data) == 3:
                anchor_sent, pos_sent, label = data
            if len(data) == 2:
                anchor_sent, pos_sent = data
                label = None

        anchor_inputs = self.tokenizer(anchor_sent, truncation=True, max_length=self.max_seq_length)
        anchor_input_ids = anchor_inputs['input_ids']
        anchor_attention_mask = anchor_inputs['attention_mask']

        pos_inputs = self.tokenizer(pos_sent, truncation=True, max_length=self.max_seq_length)
        pos_input_ids = pos_inputs['input_ids']
        pos_attention_mask = pos_inputs['attention_mask']

        if self.task_type.lower() == "nli":
            neg_inputs = self.tokenizer(neg_sent, truncation=True, max_length=self.max_seq_length)
            neg_input_ids = neg_inputs['input_ids']
            neg_attention_mask = neg_inputs['attention_mask']
            return {
                'idx': idx,
                'anchor_input_ids': anchor_input_ids,
                'pos_input_ids': pos_input_ids,
                'neg_input_ids': neg_input_ids,
                'anchor_attention_mask': anchor_attention_mask,
                'pos_attention_mask': pos_attention_mask,
                'neg_attention_mask': neg_attention_mask,
                'label': label
                }
        else:
            return {
                'idx': idx,
                'anchor_input_ids': anchor_input_ids,
                'pos_input_ids': pos_input_ids,
                'anchor_attention_mask': anchor_attention_mask,
                'pos_attention_mask': pos_attention_mask,
                'label': label
                    }


def pad_to_maxlen(input_ids, max_len, pad_value=0):
    if len(input_ids) >= max_len:
        input_ids = input_ids[:max_len]
    else:
        input_ids = input_ids + [pad_value] * (max_len - len(input_ids))
    return input_ids


def collate_fn(batch_data):
    anchor_max_len = 0
    pos_max_len = 0
    neg_max_len = 0
    for d in batch_data:
        num_sent = len(d.keys()) - 1
        anchor_max_len = max(anchor_max_len, len(d['anchor_input_ids']))
        pos_max_len = max(pos_max_len, len(d['pos_input_ids']))
        if num_sent == 3:
            neg_max_len = max(neg_max_len, len(d['neg_input_ids']))
    max_len = max(anchor_max_len, pos_max_len, neg_max_len)
    idx_lst = []
    anchor_input_ids_lst = []
    pos_input_ids_lst = []
    neg_input_ids_lst = []
    anchor_attention_mask_lst = []
    pos_attention_mask_lst = []
    neg_attention_mask_lst = []
    label_lst = []
    for item in batch_data:
        idx_lst.append(item['idx'])
        anchor_input_ids_lst.append(pad_to_maxlen(item['anchor_input_ids'], max_len=max_len))
        pos_input_ids_lst.append(pad_to_maxlen(item['pos_input_ids'], max_len=max_len))
        anchor_attention_mask_lst.append(pad_to_maxlen(item['anchor_attention_mask'], max_len=max_len))
        pos_attention_mask_lst.append(pad_to_maxlen(item['pos_attention_mask'], max_len=max_len))
        if neg_max_len > 0:
            neg_input_ids_lst.append(pad_to_maxlen(item['neg_input_ids'], max_len=max_len))
            neg_attention_mask_lst.append(pad_to_maxlen(item['neg_attention_mask'], max_len=max_len))
        if item['label'] is not None:
            label_lst.append(item['label'])

    anchor_input_ids = torch.LongTensor(anchor_input_ids_lst)
    pos_input_ids = torch.LongTensor(pos_input_ids_lst)
    anchor_attention_masks = torch.LongTensor(anchor_attention_mask_lst)
    pos_attention_masks = torch.LongTensor(pos_attention_mask_lst)
    if label_lst:
        labels = torch.LongTensor(label_lst)   # 分类这里为torch.long  回归这里为torch.float
    else:
        labels = None
    if neg_max_len > 0:
        neg_input_ids = torch.LongTensor(neg_input_ids_lst)
        neg_attention_masks = torch.LongTensor(neg_attention_mask_lst)
        return {
            'id_lst': idx_lst,
            'anchor_input_ids': anchor_input_ids,
            'pos_input_ids': pos_input_ids,
            'neg_input_ids': neg_input_ids,
            'anchor_attention_mask': anchor_attention_masks,
            'pos_attention_mask': pos_attention_masks,
            'neg_attention_mask': neg_attention_masks,
            'label': labels
        }
    else:
        return {
            'id_lst': idx_lst,
            'anchor_input_ids': anchor_input_ids,
            'pos_input_ids': pos_input_ids,
            'anchor_attention_mask': anchor_attention_masks,
            'pos_attention_mask': pos_attention_masks,
            'label': labels
        }


class VaSCLDataset(Dataset):
    def __init__(self, dataset, tokenizer, args, padding=True):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.args = args
        self.padding = padding

    def __len__(self):
        return len(self.dataset)

    def get_batch_token(self, text):
        token_feat = self.tokenizer(
            text,
            max_length=self.args.max_seq_length,
            padding='max_length',
            truncation=True
        )
        return token_feat

    def __getitem__(self, idx):
        dataset = self.dataset[idx]
        assert len(dataset) <= 3
        if len(dataset) == 3:
            text1 = dataset[0]
            text2 = dataset[1]
            label = dataset[2]
        else:
            text1 = dataset[0]
            text2 = dataset[0]
            label = None

        feat1 = self.get_batch_token(text1)
        feat2 = self.get_batch_token(text2)
        return {
            'pos_inputs': {
                'input_ids': feat1['input_ids'],
                'attention_mask': feat1['attention_mask']
            },
            'neg_inputs': {
                'input_ids': feat2['input_ids'],
                'attention_mask': feat2['attention_mask']
            },
            'label': label
        }

    def collate_fn(self, batch):
        label = batch[0]['label']
        if label is None:
            batch_input_ids = []
            batch_attention_mask = []
            for item in batch:
                input_ids = [item['pos_inputs']['input_ids'], item['neg_inputs']['input_ids']]
                attention_mask = [item['pos_inputs']['attention_mask'], item['neg_inputs']['attention_mask']]
                batch_input_ids.append(input_ids)
                batch_attention_mask.append(attention_mask)
            return {
                'input_ids': torch.LongTensor(batch_input_ids).to(self.args.device),
                'attention_mask': torch.LongTensor(batch_attention_mask).to(self.args.device)
            }
        else:
            batch_pos_input_ids = []
            batch_pos_attention_mask = []
            batch_neg_input_ids = []
            batch_neg_attention_mask = []
            batch_labels = []
            for item in batch:
                pos_input_ids = item['pos_inputs']['input_ids']
                pos_attention_mask = item['pos_inputs']['attention_mask']

                neg_input_ids = item['neg_inputs']['input_ids']
                neg_attention_mask = item['neg_inputs']['attention_mask']

                batch_pos_input_ids.append(pos_input_ids)
                batch_pos_attention_mask.append(pos_attention_mask)
                batch_neg_input_ids.append(neg_input_ids)
                batch_neg_attention_mask.append(neg_attention_mask)

                batch_labels.append(item['label'])

            return {
                'pos_inputs': {
                    'input_ids': torch.LongTensor(batch_pos_input_ids).to(self.args.device),
                    'attention_mask': torch.LongTensor(batch_pos_attention_mask).to(self.args.device)
                },
                'neg_inputs': {
                    'input_ids': torch.LongTensor(batch_neg_input_ids).to(self.args.device),
                    'attention_mask': torch.LongTensor(batch_neg_attention_mask).to(self.args.device)
                },
                'label': batch_labels
            }
