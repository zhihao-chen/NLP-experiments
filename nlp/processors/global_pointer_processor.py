# -*- coding: utf8 -*-
"""
======================================
    Project Name: NLP
    File Name: global_pointer_processor
    Author: czh
    Create Date: 2022/2/9
--------------------------------------
    Change Activity: 
======================================
"""
import os
from typing import List, Union, Tuple, Optional, Dict
from pathlib import Path
import string
import codecs
import json
import pickle
from tqdm import tqdm

from pydantic import BaseModel
import numpy as np
import torch

from nlp.utils.enums import DataType
from nlp.utils.errors import ParseEntityOffsetMappingError
from nlp.processors.preprocess import find_head_idx


SPE = "│"
LETTERS = string.ascii_letters + string.punctuation + "！=·，；。？《》【】"
CHINESE_PUNCTUATION_TRANSLATION_ENGLISH = {'！': "!", '，': ",", '？': "?", '；': ";", '：': ":", '【': "[", '】': "]"}


class EntityItem(BaseModel):
    text: str
    entity_type_id: int
    char_span: Tuple[int, int]
    token_span: Tuple[int, int]


class TupleItem(BaseModel):
    """每个实体以及其对应的关系"""
    relation_id: int
    entities: List[EntityItem]


class InputSample:
    def __init__(self,
                 idx: str,
                 text: str,
                 tokens: List[str],
                 token_len: int,
                 input_ids: List[int],
                 attention_mask: List[int],
                 token_type_ids: List[int],
                 offset_mapping: List[Tuple[int, int]] = None,
                 entity_label_ids=None,
                 token_offset: int = 0,
                 char_offset: int = 0,
                 entities: List = None
                 ):
        self.id = idx
        self.text = text
        self.tokens = tokens
        self.token_len = token_len
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.entity_label_ids = entity_label_ids
        self.offset_mapping = offset_mapping
        self.token_offset = token_offset
        self.char_offset = char_offset
        self.entities = entities


class NerPreProcess:
    def __init__(self,
                 data_dir: Union[str, Path] = None,
                 add_special_tokens=False
                 ):
        self.data_dir = data_dir
        self.add_special_tokens = add_special_tokens
        self.letter_dict = {c: i for i, c in enumerate(LETTERS)}

        entity_types, entity_types_to_name_dict, self.entity2id, self.id2entity = self.get_labels()
        self.num_entity_types = len(entity_types)

        self.tokenizer = None

    def get_labels(self):
        label_file = os.path.join(self.data_dir, 'entity_types.json')
        entity_types = set()
        entity_types_to_name_dict = {}
        try:
            with codecs.open(label_file, encoding='utf8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    item = json.loads(line)
                    tag = item['tag']
                    entity_types.add(tag)
                    entity_types_to_name_dict[tag] = item['name']
        except Exception as e:
            raise e
        else:
            id2entity_file = os.path.join(self.data_dir, 'id2entity.pkl')
            entity2id_file = os.path.join(self.data_dir, 'entity2id.pkl')
            if os.path.exists(id2entity_file):
                with codecs.open(id2entity_file, 'rb') as fr:
                    id2entity = pickle.load(fr)
            else:
                id2entity = {i: e for i, e in enumerate(entity_types)}
                with codecs.open(os.path.join(self.data_dir, 'id2entity.pkl'), 'wb') as fw:
                    pickle.dump(id2entity, fw)
            if os.path.exists(entity2id_file):
                with codecs.open(entity2id_file, 'rb') as fr:
                    entity2id = pickle.load(fr)
            else:
                entity2id = {e: i for i, e in enumerate(entity_types)}
                with codecs.open(os.path.join(self.data_dir, 'entity2id.pkl'), 'wb') as fw:
                    pickle.dump(entity2id, fw)
            return entity_types, entity_types_to_name_dict, entity2id, id2entity

    def prepare_for_text(self, text: str):
        """
        主要为了预处理文本。中文间的空格不应该直接剔除，而英文的空格保持原样，此外，替换完后，为了便于恢复原有文本（每个token一致），
        需要保证修改后的文本与原文本长度一致。
        """
        if len(text) <= 2:
            return text
        new_text = text[0]
        for i in range(1, len(text) - 1):
            c = text[i]
            if not c.strip() and not (text[i - 1] in self.letter_dict or text[i + 1] in self.letter_dict):
                new_text += SPE
            else:
                # c = CHINESE_PUNCTUATION_TRANSLATION_ENGLISH.get(c, c)
                new_text += c
        new_text += text[-1]
        return new_text

    def parser_data_from_json_file(self,
                                   tokenizer,
                                   data_type: DataType = DataType.TRAIN
                                   ) -> List[InputSample]:
        self.tokenizer = tokenizer
        data_file = os.path.join(self.data_dir, f"{data_type.value}.json")
        if not os.path.exists(data_file) or not os.path.isfile(data_file):
            raise ValueError(f"'{data_file}' not exist or not is a file")
        all_samples = []
        error_num = 0
        # TODO：完成数据解析
        with codecs.open(data_file, encoding='utf8') as f:
            for i, line in enumerate(tqdm(f, desc=f"load {data_type.value} data")):
                line = line.strip()
                if not line:
                    continue
                sample = json.loads(line)
                idx = sample.get('id', str(i))
                text = sample['text']
                entity_list = sample['entity_list']

                text_new = self.prepare_for_text(text)
                token_sent = tokenizer(text_new,
                                       add_special_tokens=self.add_special_tokens,
                                       return_offsets_mapping=True)
                input_ids = token_sent["input_ids"]
                attention_mask = token_sent["attention_mask"]
                token_type_ids = token_sent["token_type_ids"]
                offset_mapping = token_sent["offset_mapping"]
                # 补全'[CLS]和'[SEP]''
                input_ids = [self.tokenizer.cls_token_id] + input_ids + [self.tokenizer.sep_token_id]
                attention_mask = [1] + attention_mask + [1]
                token_type_ids = [0] + token_type_ids + [0]
                tokens = self.tokenizer.tokenize(text_new, add_special_tokens=True)

                token_len = len(input_ids)
                labels = np.zeros((self.num_entity_types, token_len, token_len))

                if entity_list:
                    entities = []
                    ent2token_spans = self.get_ent2token_spans(text_new, offset_mapping, entity_list)
                    if not ent2token_spans:
                        error_num += 1
                        continue
                    for ent, token_span in zip(entity_list, ent2token_spans):
                        offset = (ent[0], ent[1])
                        entities.append(
                            EntityItem(
                                text=ent[3],
                                entity_type_id=self.entity2id[ent[2]],
                                char_span=offset,
                                token_span=(token_span[0]+1, token_span[1]+1)  # 因为补上了'[CLS]'， 所以得右移一位
                            )
                        )
                    for start, end, label_id in ent2token_spans:
                        labels[self.entity2id[label_id], start, end] = 1
                else:
                    entities = None
                all_samples.append(
                    InputSample(
                        idx=idx,
                        text=text_new,
                        tokens=tokens,
                        token_len=token_len,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids,
                        entity_label_ids=labels,
                        offset_mapping=offset_mapping,
                        entities=entities
                    )
                )
        print(f"总计 {error_num} 条样本的实体无法对应到token_span")
        return all_samples

    @staticmethod
    def compute_token_offset_of_entity(
            text_offset_mapping: List[Tuple[int, int]],
            entity_offset: Optional[Tuple[int, int]]
    ) -> Tuple[int, int]:
        """
        这一版严格要求样本的实体必须含offset。
        char_offset => token_offset
        :param text_offset_mapping:
        :param entity_offset:
        :return:
        """
        # 空实体直接返回: (0, 0)
        if not entity_offset or entity_offset[0] == entity_offset[1]:
            return 0, 0

        tok_st = tok_ed = -1
        entity_st, entity_ed = entity_offset
        for idx, (st, ed) in enumerate(text_offset_mapping):
            if entity_st == st:
                tok_st = idx
            if entity_ed == ed-1:
                tok_ed = idx

            if 0 <= tok_st <= tok_ed:
                return tok_st, tok_ed

        if not (0 <= tok_st <= tok_ed):
            raise ParseEntityOffsetMappingError("Labeled entity not match tokenizer!")

    @staticmethod
    def get_ent2token_spans(text, offset_mapping, entity_list):
        """实体列表转为token_spans
        Args:
            text (str): 原始文本
            offset_mapping:
            entity_list (list): [(start, end, ent_type),(start, end, ent_type)...]
        """
        ent2token_spans = []
        for ent_span in entity_list:
            ent = ent_span[-1]
            tok_st = tok_ed = -1
            entity_st, entity_ed = ent_span[0], ent_span[1]
            for idx, (st, ed) in enumerate(offset_mapping):
                if entity_st == st:
                    tok_st = idx
                if entity_ed == ed-1:
                    tok_ed = idx
                if 0 <= tok_st <= tok_ed:
                    token_span = (tok_st, tok_ed, ent_span[2])
                    ent2token_spans.append(token_span)
                    break

            if not (0 <= tok_st <= tok_ed):
                print(f'[{ent}] 无法对应到 [{text}] 的token_span，已丢弃')
                ent2token_spans = []
                break

        return ent2token_spans


class CluenerProcessor(NerPreProcess):
    def __init__(self, data_dir: Union[str, Path] = None,
                 add_special_tokens=False):
        super(CluenerProcessor, self).__init__(data_dir, add_special_tokens)

    def get_labels(self):
        entity_types_to_name_dict = {'address': "地址", 'book': "书名", 'company': "公司", 'game': "游戏",
                                     'government': "政府", 'movie': "电影", 'name': "姓名", 'organization': "组织机构",
                                     'position': "职位", 'scene': "景点"}
        entity_types = list(set(entity_types_to_name_dict.keys()))
        id2entity_file = os.path.join(self.data_dir, 'id2entity.pkl')
        entity2id_file = os.path.join(self.data_dir, 'entity2id.pkl')
        if os.path.exists(id2entity_file):
            with codecs.open(id2entity_file, 'rb') as fr:
                id2entity = pickle.load(fr)
        else:
            id2entity = {i: e for i, e in enumerate(entity_types)}
            with codecs.open(os.path.join(self.data_dir, 'id2entity.pkl'), 'wb') as fw:
                pickle.dump(id2entity, fw)
        if os.path.exists(entity2id_file):
            with codecs.open(entity2id_file, 'rb') as fr:
                entity2id = pickle.load(fr)
        else:
            entity2id = {e: i for i, e in enumerate(entity_types)}
            with codecs.open(os.path.join(self.data_dir, 'entity2id.pkl'), 'wb') as fw:
                pickle.dump(entity2id, fw)
        return entity_types, entity_types_to_name_dict, entity2id, id2entity

    def parser_data_from_json_file(self,
                                   tokenizer,
                                   data_type: DataType = DataType.TRAIN
                                   ) -> List[InputSample]:
        self.tokenizer = tokenizer
        data_file = os.path.join(self.data_dir, f"{data_type.value}.json")
        if not os.path.exists(data_file) or not os.path.isfile(data_file):
            raise ValueError(f"'{data_file}' not exist or not is a file")
        all_samples = []
        error_num = 0
        with codecs.open(data_file, encoding='utf8') as f:
            for i, line in enumerate(tqdm(f, desc=f"load {data_type.value} data")):
                line = line.strip()
                if not line:
                    continue
                sample = json.loads(line)
                idx = sample.get('id', str(i))
                text = sample['text']
                label_dict = sample.get('label', {})
                entity_list = []
                if label_dict:
                    for t, adict in label_dict.items():
                        for name, span_list in adict.items():
                            for span in span_list:
                                entity_list.append([span[0], span[1], t, name])

                # text_new = self.prepare_for_text(text)
                text_new = text
                if data_type != DataType.TEST:
                    token_sent = tokenizer(text_new,
                                           add_special_tokens=self.add_special_tokens,
                                           return_offsets_mapping=True)
                    input_ids = token_sent["input_ids"]
                    attention_mask = token_sent["attention_mask"]
                    token_type_ids = token_sent["token_type_ids"]
                    offset_mapping = token_sent["offset_mapping"]
                    # 补全'[CLS]和'[SEP]''
                    input_ids = [self.tokenizer.cls_token_id] + input_ids + [self.tokenizer.sep_token_id]
                    attention_mask = [1] + attention_mask + [1]
                    token_type_ids = [0] + token_type_ids + [0]
                    tokens = self.tokenizer.tokenize(text_new, add_special_tokens=True)
                else:
                    token_sent = tokenizer.encode_plus(text_new, add_special_tokens=True, return_offsets_mapping=True)
                    input_ids = token_sent["input_ids"]
                    attention_mask = token_sent["attention_mask"]
                    token_type_ids = token_sent["token_type_ids"]
                    offset_mapping = token_sent["offset_mapping"]
                    tokens = self.tokenizer.tokenize(text_new, add_special_tokens=True)

                token_len = len(input_ids)
                labels = np.zeros((self.num_entity_types, token_len, token_len))

                if entity_list:
                    entities = []
                    ent2token_spans = self.get_ent2token_spans(text_new, offset_mapping, entity_list)
                    if not ent2token_spans:
                        error_num += 1
                        continue
                    for ent, token_span in zip(entity_list, ent2token_spans):
                        offset = (ent[0], ent[1])
                        entities.append(
                            EntityItem(
                                text=ent[3],
                                entity_type_id=self.entity2id[ent[2]],
                                char_span=offset,
                                token_span=(token_span[0] + 1, token_span[1] + 1)  # 因为补上了'[CLS]'， 所以得右移一位
                            )
                        )
                    for start, end, label_id in ent2token_spans:
                        labels[self.entity2id[label_id], start, end] = 1
                else:
                    entities = None
                all_samples.append(
                    InputSample(
                        idx=idx,
                        text=text_new,
                        tokens=tokens,
                        token_len=token_len,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids,
                        entity_label_ids=labels,
                        offset_mapping=offset_mapping,
                        entities=entities
                    )
                )
            print(f"总计 {error_num} 条样本的实体无法对应到token_span")
            return all_samples

    def get_ent2token_spans(self, text, offset_mapping, entity_list):
        ent2token_spans = []

        text2tokens = self.tokenizer.tokenize(text, add_special_tokens=self.add_special_tokens)

        for ent_span in entity_list:
            ent = text[ent_span[0]:ent_span[1] + 1]
            ent2token = self.tokenizer.tokenize(ent, add_special_tokens=False)

            token_start_index = find_head_idx(text2tokens, ent2token)
            if token_start_index != -1:
                token_end_index = token_start_index + len(ent2token)
                token_span = (token_start_index, token_end_index, ent_span[2])
                ent2token_spans.append(token_span)
            else:
                print(f'[{ent}] 无法对应到 [{text}] 的token_span，已丢弃')
                continue

        return ent2token_spans


def global_pointer_entity_extract(pred_logits: torch.Tensor,
                                  id2entity: Dict[int, str],
                                  entity_type_names: dict) -> List[List[dict]]:
    batch_size = pred_logits.size()[0]
    pred_logits = pred_logits.detach().cpu().numpy()

    pred_list = [[] for i in range(batch_size)]
    for bs, label_id, start, end in zip(*np.where(pred_logits > 0)):
        label = id2entity[label_id]
        label_name = entity_type_names[label]
        res = {'label': label, 'label_name': label_name, 'start': start, 'end': end}
        pred_list[bs].append(res)

    return pred_list
