# -*- coding: utf8 -*-
"""
======================================
    Project Name: NLP
    File Name: dataset
    Author: czh
    Create Date: 2021/11/9
--------------------------------------
    Change Activity: 
======================================
"""
from typing import List

import torch
import numpy as np
from torch.utils.data import Dataset

from nlp.processors.preprocess import Preprocessor


class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.length = len(data)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.length


class NerDataset(MyDataset):
    def __init__(self, data, num_labels, data_type):
        super(NerDataset, self).__init__(data)
        self.num_labels = num_labels
        self.data_type = data_type

    def collate_fn(self, batch_data: List):
        batch_size = len(batch_data)
        sent_lens = list(x.token_len for x in batch_data)
        max_sent_len = max(sent_lens)

        input_ids = torch.zeros((batch_size, max_sent_len), requires_grad=False).long()
        attention_mask = torch.zeros((batch_size, max_sent_len), requires_grad=False).long()
        token_type_ids = torch.zeros((batch_size, max_sent_len), requires_grad=False).long()
        if self.data_type != 'test':
            label_ids = torch.zeros((batch_size, self.num_labels, max_sent_len, max_sent_len),
                                    requires_grad=False).long()
        else:
            label_ids = None

        sample_list = []
        for idx, (seq_len, sample) in enumerate(zip(sent_lens, batch_data)):
            input_ids[idx, :seq_len] = torch.LongTensor(sample.input_ids)
            attention_mask[idx, :seq_len] = torch.LongTensor(sample.attention_mask)
            token_type_ids[idx, :seq_len] = torch.LongTensor(sample.token_type_ids)
            if sample.entity_label_ids is not None and self.data_type != 'test':
                label_ids[idx, :, :seq_len, :seq_len] = torch.LongTensor(sample.entity_label_ids)
            sample_list.append(sample)

        results = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids,
            'labels': label_ids,
            'sample_list': sample_list
        }
        return results


class DataMaker(object):
    def __init__(self, tokenizer, add_special_tokens=True):
        super().__init__()
        self.tokenizer = tokenizer
        self.add_special_tokens = add_special_tokens
        self.preprocessor = Preprocessor(tokenizer, self.add_special_tokens)

    def generate_inputs(self, datas, max_seq_len, ent2id, data_type="train"):
        """生成喂入模型的数据
        Args:
            datas (list): json格式的数据[{'text':'','entity_list':[(start,end,ent_type),()]}]
            max_seq_len (int): 句子最大token数量
            ent2id (dict): ent到id的映射
            data_type (str, optional): data类型. Defaults to "train".
        Returns:
            list: [(sample, input_ids, attention_mask, token_type_ids, labels),(),()...]
        """

        ent_type_size = len(ent2id)  # 实体类别

        all_inputs = []
        for sample in datas:
            inputs = self.tokenizer(
                sample["text"],
                max_length=max_seq_len,
                truncation=True,
                padding='max_length'
            )

            labels = None
            if data_type != "test":
                ent2token_spans = self.preprocessor.get_ent2token_spans(
                    sample["text"], sample["entity_list"]
                )
                labels = np.zeros((ent_type_size, max_seq_len, max_seq_len))
                for start, end, label in ent2token_spans:
                    labels[ent2id[label], start, end] = 1
            inputs["labels"] = labels

            input_ids = torch.tensor(inputs["input_ids"]).long()
            attention_mask = torch.tensor(inputs["attention_mask"]).long()
            token_type_ids = torch.tensor(inputs["token_type_ids"]).long()
            if labels is not None:
                labels = torch.tensor(inputs["labels"]).long()

            sample_input = (sample, input_ids, attention_mask, token_type_ids, labels)

            all_inputs.append(sample_input)
        return all_inputs

    @staticmethod
    def generate_batch(batch_data, data_type="train"):
        sample_list = []
        input_ids_list = []
        attention_mask_list = []
        token_type_ids_list = []
        labels_list = []

        for sample in batch_data:
            sample_list.append(sample[0])
            input_ids_list.append(sample[1])
            attention_mask_list.append(sample[2])
            token_type_ids_list.append(sample[3])
            if data_type != "test":
                labels_list.append(sample[4])

        batch_input_ids = torch.stack(input_ids_list, dim=0)
        batch_attention_mask = torch.stack(attention_mask_list, dim=0)
        batch_token_type_ids = torch.stack(token_type_ids_list, dim=0)
        batch_labels = torch.stack(labels_list, dim=0) if data_type != "test" else None

        return sample_list, batch_input_ids, batch_attention_mask, batch_token_type_ids, batch_labels
