# -*- coding: utf8 -*-
"""
======================================
    Project Name: NLP
    File Name: ee_seq
    Author: czh
    Create Date: 2021/9/8
--------------------------------------
    Change Activity: 
======================================
"""
# 事件抽取任务转换成序列标注任务
import logging
import os

import torch
from tqdm import tqdm

from nlp.processors.utils_ee import DataProcessor, InputExample, InputFeatures
from nlp.utils.tokenizers import tokenize

logger = logging.getLogger(__name__)


def collate_fn(batch):
    """
    batch should be a list of (sequence, target, length) tuples...
    Returns a padded tensor of sequences sorted from longest to shortest,
    """
    all_input_ids, all_attention_mask, all_token_type_ids, all_lens, all_labels = map(torch.stack, zip(*batch))
    max_len = max(all_lens).item()
    all_input_ids = all_input_ids[:, :max_len]
    all_attention_mask = all_attention_mask[:, :max_len]
    all_token_type_ids = all_token_type_ids[:, :max_len]
    all_labels = all_labels[:, :max_len]
    return all_input_ids, all_attention_mask, all_token_type_ids, all_labels, all_lens


def search(pattern, sequence):
    """从sequence中寻找子串pattern
    如果找到，返回第一个下标；否则返回-1。
    """
    n = len(pattern)
    for i in range(len(sequence)):
        if sequence[i:i + n] == pattern:
            return i
    return -1


def convert_examples_to_features(examples, label2id, max_seq_length, tokenizer, cls_token="[CLS]", sep_token="[SEP]",
                                 pad_token=0, pad_token_segment_id=0, mask_padding_with_zero=True, data_type="train",
                                 do_lower_case=False, task_name='ee'):
    features = []
    for (ex_index, example) in tqdm(enumerate(examples), desc=f"Convert {data_type} examples to features"):
        tokens = tokenize(example.text_a, tokenizer.vocab, do_lower_case)
        if len(tokens) > max_seq_length - 2:
            tokens = tokens[: max_seq_length - 2]
        tokens = [cls_token] + tokens + [sep_token]
        input_len = len(tokens)

        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1 if mask_padding_with_zero else 0] * len(token_ids)
        segment_ids = [0] * len(token_ids)

        if len(token_ids) < max_seq_length:
            padding_num = max_seq_length - len(token_ids)
            token_ids += [pad_token] * padding_num
            input_mask += [0 if mask_padding_with_zero else 1] * padding_num
            segment_ids += [pad_token_segment_id] * padding_num

        labels = [0] * len(token_ids)
        arguments = example.arguments
        if arguments:
            if task_name == 'ee':
                for argument, item in arguments.items():
                    event_type, role = item
                    if (event_type, role) not in label2id or event_type == "OTHER":
                        continue
                    a_token = tokenize(argument, tokenizer.vocab, do_lower_case)
                    a_token_id = tokenizer.convert_tokens_to_ids(a_token)
                    start_index = search(a_token_id, token_ids)
                    if start_index != -1:
                        labels[start_index] = label2id[(event_type, role)] * 2 + 1
                        for i in range(1, len(a_token_id)):
                            labels[start_index + i] = label2id[(event_type, role)] * 2 + 2
            elif task_name == 'ner':
                for argument, item in arguments.items():
                    role = item
                    a_token = tokenize(argument, tokenizer.vocab, do_lower_case)
                    a_token_id = tokenizer.convert_tokens_to_ids(a_token)
                    start_index = search(a_token_id, token_ids)
                    if start_index != -1:
                        labels[start_index] = label2id[role] * 2 + 1
                        for i in range(1, len(a_token_id)):
                            labels[start_index + i] = label2id[role] * 2 + 2
        assert len(token_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(labels) == max_seq_length

        if ex_index < 2:
            logger.info("*** Example ***")
            logger.info("guid: %s", example.guid)
            logger.info("tokens: %s", " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s", " ".join([str(x) for x in token_ids]))
            logger.info("input_mask: %s", " ".join([str(x) for x in input_mask]))
            logger.info("segment_ids: %s", " ".join([str(x) for x in segment_ids]))
            logger.info("label_ids: %s", " ".join([str(x) for x in labels]))

        feature = InputFeatures(input_ids=token_ids,
                                input_mask=input_mask,
                                segment_ids=segment_ids,
                                input_len=input_len,
                                label_ids=labels)
        features.append(feature)
    return features


class EEProcessor(DataProcessor):
    def __init__(self, data_dir):
        self.data_dir = data_dir

    def get_train_examples(self):
        return self.__create_examples(self.read_json(os.path.join(self.data_dir, 'train.json')), 'train')

    def get_dev_examples(self):
        return self.__create_examples(self.read_json(os.path.join(self.data_dir, 'dev.json')), 'dev')

    def get_test_examples(self):
        return self.__create_examples(self.read_json(os.path.join(self.data_dir, 'test.json')), 'test')

    @staticmethod
    def labels():
        label_dicts = [
            {"event_type": "PERUP", "role_list": [{"role": "PER", "name": "晋升人员"}, {"role": "CORG", "name": "当前所在公司"},
                                                  {"role": "CDEP", "name": "当前所在部门"},
                                                  {"role": "CTITLE", "name": "当前职位/职级"},
                                                  {"role": "LORG", "name": "晋升前公司"}, {"role": "LDEP", "name": "晋升前部门"},
                                                  {"role": "LTITLE", "name": "晋升前职位/职级"},
                                                  {"role": "CHA", "name": "主要业务"}], "class": "人员晋升"},
            {"event_type": "BUSEXP", "role_list": [{"role": "ORG", "name": "公司名称"}, {"role": "CHA", "name": "业务方向"},
                                                   {"role": "BTIME", "name": "计划时间"}, {"role": "PER", "name": "主要人员"},
                                                   {"role": "BPTR", "name": "合作伙伴"}], "class": "业务拓展"},
            {"event_type": "ORGFIN",
             "role_list": [{"role": "CIRCLE", "name": "融资轮数"}, {"role": "MONEY", "name": "融资金额"},
                           {"role": "ORG", "name": "融资公司"}, {"role": "INV", "name": "投资方"},
                           {"role": "CHA", "name": "融资业务方向"}], "class": "企业融资"},
            {"event_type": "ADMPEN",
             "role_list": [{"role": "ORG", "name": "处罚公司"}, {"role": "PTYP", "name": "处罚内容/违反条例"},
                           {"role": "MONEY", "name": "处罚金额"}], "class": "行政处罚"},
            {"event_type": "JOBREC",
             "role_list": [{"role": "HCHA", "name": "业务关键词"}, {"role": "HTITLE", "name": "职位关键词"},
                           {"role": "HLEVEL", "name": "职级关键词"}], "class": "职位招聘"},
            {"event_type": "STRCOO", "role_list": [{"role": "ORG", "name": "公司名称"}, {"role": "CHA", "name": "业务方向"}],
             "class": "战略合作"},
            {"event_type": "OTHER", "role_list": [{"role": "ORG", "name": "公司名称"}, {"role": "CHA", "name": "主要业务"},
                                                  {"role": "PER", "name": "晋升人员"}, {"role": "BPRT", "name": "合作伙伴"}],
             "class": "其它"}
        ]
        return label_dicts

    def get_labels(self):
        file_path = os.path.join(self.data_dir, 'event_schema.json')
        if os.path.exists(file_path):
            id2label, label2id, num_labels, event_type_dict = self.read_event_schema(file_name_or_path=file_path)
        else:
            label_dicts = self.labels()
            id2label, label2id, num_labels, event_type_dict = self.read_event_schema(alist=label_dicts)
        return id2label, label2id, num_labels*2+1, event_type_dict

    @staticmethod
    def __create_examples(lines, set_type):
        examples = []
        for i, (text, argument) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = text
            examples.append(InputExample(guid, text_a, argument))
        return examples
