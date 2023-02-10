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
from tqdm import tqdm
from typing import List
from itertools import chain

import torch
import numpy as np
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

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


class CdailQADataset(Dataset):
    """
    一问一答
    参考https://github.com/thu-coai/CDial-GPT/blob/81064865221d3503251d633f3d0b651004c938a6/od/inputters/dataset_wb.py
    """
    SPECIAL_TOKENS = ["[CLS]", "[SEP]", "[speaker1]", "[speaker2]"]
    MODEL_INPUTS = ["input_ids", "lm_labels", "token_type_ids"]

    def __init__(self, dataset, tokenizer, batch_first=True, lm_labels=True):
        super(CdailQADataset, self).__init__()
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.pad = tokenizer.pad_token_id
        self.batch_first = batch_first
        self.lm_labels = lm_labels

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if self.lm_labels:
            question = self.dataset[idx]['question']
            answer = self.dataset[idx]['answer']
        else:
            question = self.dataset[idx]['question']
            answer = []
        return self.process(question, answer)

    def process(self, question, answer, with_eos=True):
        bos, eos, speaker1, speaker2 = self.tokenizer.convert_tokens_to_ids(self.SPECIAL_TOKENS)
        question_ids = self.tokenizer(question, add_special_tokens=False, truncation=False, padding=False)['input_ids']
        answer_ids = self.tokenizer(answer, add_special_tokens=False, truncation=False, padding=False)['input_ids']

        input_ids = [bos, speaker1] + question_ids + [speaker2] + answer_ids + [eos]
        token_type_ids = [bos, speaker1] + [speaker1]*len(question_ids) + [speaker2]*(len(answer_ids)+2)

        # sequence = [[bos]] + [question_ids] + [answer_ids + ([eos] if with_eos else [])]
        # sequence = [sequence[0]] + [[speaker2 if i % 2 else speaker1] + s
        #                             for i, s in enumerate(sequence[1:])]
        # instance = {"input_ids": list(chain(*sequence)),
        #             "token_type_ids": [bos] + [speaker2 if i % 2 else speaker1 for i, s in
        #                                        enumerate(sequence[1:])
        #                                        for _ in s]}
        instance = {'input_ids': input_ids, 'token_type_ids': token_type_ids}
        instance['lm_labels'] = [-1] * len(instance["input_ids"])
        if self.lm_labels:
            # instance["lm_labels"] = ([-1] * sum(len(s) for s in sequence[:-1])) + [-1] + sequence[-1][1:]
            instance['lm_labels'] = [-1] * 2 + [-1] * len(question_ids) + [-1] + answer_ids + [eos]

        return instance

    def collate(self, batch):
        input_ids = pad_sequence(
            [torch.tensor(instance["input_ids"], dtype=torch.long) for instance in batch],
            batch_first=self.batch_first, padding_value=self.pad)
        token_type_ids = pad_sequence(
            [torch.tensor(instance["token_type_ids"], dtype=torch.long) for instance in batch],
            batch_first=self.batch_first, padding_value=self.pad)
        labels = pad_sequence(
            [torch.tensor(instance["lm_labels"], dtype=torch.long) for instance in batch],
            batch_first=self.batch_first, padding_value=-1)
        return {
            'input_ids': input_ids,
            'token_type_ids': token_type_ids,
            'labels': labels
        }


class CdailDataset(Dataset):
    """
    项目原始数据类，处理对话数据
    参考https://github.com/thu-coai/CDial-GPT/blob/81064865221d3503251d633f3d0b651004c938a6/od/inputters/dataset_wb.py
    """
    SPECIAL_TOKENS = ["[CLS]", "[SEP]", "[speaker1]", "[speaker2]"]
    MODEL_INPUTS = ["input_ids", "lm_labels", "token_type_ids"]

    def __init__(self, dataset, tokenizer, max_history=15, max_length=512,
                 batch_first=True, lm_labels=True, with_eos=True):
        super(CdailDataset, self).__init__()
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_history = max_history
        self.max_length = max_length
        self.pad = tokenizer.pad_token_id
        self.batch_first = batch_first
        self.lm_labels = lm_labels
        self.with_eos = with_eos

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        bos, eos, speaker1, speaker2 = self.tokenizer.convert_tokens_to_ids(self.SPECIAL_TOKENS)
        dataset = self.dataset[index][-2 * self.max_history:]

        input_ids = [bos]
        history_ids = []
        for i, h in enumerate(dataset):
            h_ids = self.tokenizer(h, add_special_tokens=False, truncation=False, padding=False)['input_ids']
            if i % 2 == 0:
                sp = speaker1
            else:
                sp = speaker2
            if len(input_ids) + len(h_ids) + 1 <= self.max_length:
                input_ids.append(sp)
                input_ids.extend(h_ids)
                history_ids.append(h_ids)
        if self.lm_labels:
            response_ids = history_ids[-1]
            history_ids = history_ids[:-1]
        else:
            response_ids = []
        sequence = [[bos]] + history_ids + [response_ids + ([eos] if self.with_eos else [])]
        sequence = [sequence[0]] + [[speaker2 if i % 2 else speaker1] + s
                                    for i, s in enumerate(sequence[1:])]
        instance = {}
        instance["input_ids"] = list(chain(*sequence))
        instance["token_type_ids"] = [bos] + [speaker2 if i % 2 else speaker1 for i, s in
                                              enumerate(sequence[1:])
                                              for _ in s]
        instance["lm_labels"] = [-1] * len(instance["input_ids"])
        if self.lm_labels:
            instance["lm_labels"] = ([-1] * sum(len(s) for s in sequence[:-1])) + [-1] + sequence[-1][1:]
        return instance

    def collate(self, batch):
        input_ids = pad_sequence(
            [torch.tensor(instance["input_ids"], dtype=torch.long) for instance in batch],
            batch_first=self.batch_first, padding_value=self.pad)
        token_type_ids = pad_sequence(
            [torch.tensor(instance["token_type_ids"], dtype=torch.long) for instance in batch],
            batch_first=self.batch_first, padding_value=self.pad)
        labels = pad_sequence(
            [torch.tensor(instance["lm_labels"], dtype=torch.long) for instance in batch],
            batch_first=self.batch_first, padding_value=-1)
        return {
            'input_ids': input_ids,
            'token_type_ids': token_type_ids,
            'labels': labels
        }


class CPMDataset(Dataset):
    """
    参考:https://github.com/TsinghuaAI/CPM-1-Finetune/blob/main/finetune_lm.py
    """
    def __init__(self, args, dataset, tokenizer, max_seq_length=512, batch_first=True):
        super(CPMDataset, self).__init__()
        self.args = args
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.batch_first = batch_first

        self.pad_id = tokenizer.pad_token_id
        self.eod_token_id = tokenizer.encode_plus('<eod>', add_special_tokens=False)['input_ids']

        self.samples = self.process(self.dataset)

    def process(self, data):
        samples = []
        for doc in tqdm(data):
            token_ids = self.tokenizer.encode_plus(doc, add_special_tokens=False)['input_ids']
            token_ids += self.eod_token_id
            start = 0
            while start + self.max_seq_length + 1 < len(token_ids):
                samples.append(token_ids[start: start + self.max_seq_length + 1])
                start = start + self.max_seq_length + 1
            if len(token_ids) - start > 1:
                samples.append(token_ids[start:] + [self.pad_id] * (self.max_seq_length + 1 - (len(token_ids) - start)))

        return samples

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # sent = self.dataset[idx]
        # outputs = self.tokenizer.encode_plus(sent, truncation=True, padding=False, max_length=self.max_seq_length+1)
        # input_ids = outputs['input_ids']
        # attention_mask = outputs['attention_mask']
        #
        # input_ids += self.eod_token_id
        # attention_mask += [1]
        #
        # if len(input_ids) < self.max_seq_length+1:
        #     input_ids += [self.pad_id] * (self.max_seq_length+1-len(input_ids))
        #     attention_mask += [0] * (self.max_seq_length+1-len(input_ids))
        #
        # labels = input_ids[1:]
        # loss_mask = [float(i != self.pad_id) for i in labels]
        # instance = {'input_ids': input_ids[:-1], 'attention_mask': attention_mask[:-1],
        #             'labels': labels, 'loss_mask': loss_mask}
        # return instance
        return self.samples[idx]

    def collate(self, batch_data):
        input_ids = pad_sequence(
            [torch.tensor(instance["input_ids"], dtype=torch.long) for instance in batch_data],
            batch_first=self.batch_first, padding_value=self.pad_id)
        attention_mask = pad_sequence(
            [torch.tensor(instance["attention_mask"], dtype=torch.long) for instance in batch_data],
            batch_first=self.batch_first, padding_value=0)
        labels = pad_sequence(
            [torch.tensor(instance["labels"], dtype=torch.long) for instance in batch_data],
            batch_first=self.batch_first, padding_value=self.pad_id)
        loss_mask = pad_sequence(
            [torch.tensor(instance['loss_mask'], dtype=torch.float) for instance in batch_data],
            batch_first=self.batch_first, padding_value=0.0
        )
        batch_sample = {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }
        no_model_sample = {
            'labels': labels,
            'loss_mask': loss_mask
        }
        return batch_sample, no_model_sample

    def collate_fn(self, batch_data):
        bs = len(batch_data)

        # triangle attention mask
        attn_mask = torch.tril(torch.ones((self.max_seq_length, self.max_seq_length))).unsqueeze(0)
        position_ids = torch.arange(self.max_seq_length, dtype=torch.long).unsqueeze(0).repeat(bs, 1)

        if self.args.fp16:
            attn_mask = attn_mask.half()

        # the data that need to go through the model
        batch_sample = {
            "input_ids": torch.ones(bs, self.max_seq_length).long() * self.pad_id,
            # "attention_mask": attn_mask.unsqueeze(1),
            "position_ids": position_ids,
        }

        # the data that do not need to go through the model
        no_model_sample = {
            "labels": torch.ones(bs, self.max_seq_length).long() * self.pad_id,
            "loss_mask": torch.zeros(bs, self.max_seq_length).float()
        }

        for i, samp in enumerate(batch_data):
            assert len(samp) == self.max_seq_length + 1, (len(samp), self.max_seq_length)
            batch_sample["input_ids"][i] = torch.tensor(samp[:-1], dtype=torch.long)
            no_model_sample["labels"][i] = torch.tensor(samp[1:], dtype=torch.long)
            no_model_sample["loss_mask"][i] = (no_model_sample["labels"][i] != self.pad_id).float()

        return batch_sample, no_model_sample
