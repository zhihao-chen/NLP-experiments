# -*- coding: utf8 -*-
"""
======================================
    Project Name: NLP
    File Name: spn4ner_processor
    Author: czh
    Create Date: 2021/11/15
--------------------------------------
    Change Activity: 
======================================
"""
import json
import string
import logging
from pathlib import Path
from typing import List, Optional, Union, Tuple, Dict

import torch
from pydantic import BaseModel
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import BertTokenizerFast
from transformers.file_utils import ModelOutput

from nlp.utils.enums import DataType

LOGGER = logging.getLogger(__name__)
UNKNOWN_RELATION = "__SPN_UNKNOWN_RELATION__"
# 特殊字符，该字符若不在预训练模型的vocab.txt中，会导致后续使用出错。
SPE = "│"


class ParseSpanError(Exception):
    pass


class ParseEntityOffsetMappingError(ParseSpanError):
    pass


class EntityNumNotMatchError(ParseSpanError):
    pass


class EntityItem(BaseModel):
    text: str
    char_span: Tuple[int, int]
    token_span: Tuple[int, int]


class TupleItem(BaseModel):
    """每个实体以及其对应的关系"""
    relation_id: int
    entities: List[EntityItem]


class InputSample(BaseModel):
    id: int
    text: str
    input_ids: List[int]
    attention_mask: List[int]
    token_type_ids: List[int]
    offset_mapping: List[Tuple[int, int]]
    token_offset: int = 0
    char_offset: int = 0
    tuples: Optional[List[TupleItem]] = None


class BatchOutput(ModelOutput):
    input_ids: List[torch.LongTensor]
    attention_mask: List[torch.LongTensor]
    token_type_ids: List[torch.LongTensor]
    targets: List[Dict[str, torch.LongTensor]]
    sent_lens: List[int]
    # 用于标记样本的唯一id，因为原样本会被slide，所以采用id和token_offset作为唯一标识
    sent_idx: List[Tuple[int, int]]
    sample_list: List[InputSample]


class PreProcessor:
    def __init__(self,
                 tokenizer: BertTokenizerFast,
                 relation_labels: List[str],
                 num_entities_in_tuple: int = 2,
                 max_seq_len: int = 424,
                 sliding_len: int = 50,
                 verbose: bool = True):
        self.tokenizer = tokenizer
        self.relation_labels = relation_labels
        self.num_entities_in_tuple = num_entities_in_tuple
        self.max_seq_len = max_seq_len
        self.sliding_len = sliding_len
        self.verbose = verbose

        self.rel2id = {v: k for k, v in enumerate(sorted(set(self.relation_labels + [UNKNOWN_RELATION])))}
        self.id2rel = {v: k for k, v in self.rel2id.items()}
        self.spe = SPE

    def prepare_for_text(self, text: str):
        """
        主要为了预处理文本。中文间的空格不应该直接剔除，而英文的空格保持原样，此外，替换完后，为了便于恢复原有文本（每个token一致），
        需要保证修改后的文本与原文本长度一致。
        """
        if len(text) <= 2:
            return text
        new_text = text[0]
        letters = string.ascii_letters + string.punctuation + "！=·，；。？《》【】"
        for i in range(1, len(text) - 1):
            c = text[i]
            if not c.strip() and not (text[i - 1] in letters or text[i + 1] in letters):
                new_text += self.spe
            else:
                new_text += c
        new_text += text[-1]
        return new_text

    def parse_from_pos_json_files(
            self,
            paths: Union[str, List[str], Path],
            data_type: DataType = DataType.TRAIN,
    ) -> List[InputSample]:
        """
        数据格式如下（每行一个json，entities必须顺序有含义，即每个位置的元素代表不同的含义。必须有text， 可无offset，表示无实体）：
        {
            "text": "今日头条CEO张一鸣参加展会",
            "label": [
                {
                    "relation":"雇佣",
                    "entities": [
                        {
                            "text": "今日头条",
                            "offset": [0, 4]
                        },
                        {
                            "text": "张一鸣",
                            "offset": [4, 7],
                        },
                        {
                            "text": "CEO",
                            "offset": [7, 10]
                        }
                    ]
                }
            ]
        }
        :param paths:
        :param data_type:
        :return:
        """
        if isinstance(paths, (str, Path)):
            paths = [paths]

        total_counter = 0
        valid_counter = 0
        # 正样本计数
        positive_counter = 0
        cnt = 0

        data = []
        for p in paths:
            p = Path(p)
            with p.open() as f:
                if self.verbose:
                    desc = f"preprocessing {data_type.value} file: {p}"
                    loop = enumerate(tqdm(f, desc=desc))
                else:
                    loop = enumerate(f)

                for idx, line in loop:
                    line = line.strip()
                    if not line:
                        continue
                    total_counter += 1
                    line_labeled_data = json.loads(line)
                    text = line_labeled_data["text"]
                    labels = line_labeled_data["label"]
                    try:
                        tokenizer = self.tokenizer
                        token_sent = tokenizer.encode_plus(self.prepare_for_text(text), add_special_tokens=False,
                                                           return_offsets_mapping=True)
                        input_ids = token_sent["input_ids"]
                        attention_mask = token_sent["attention_mask"]
                        token_type_ids = token_sent["token_type_ids"]
                        offset_mapping = token_sent["offset_mapping"]
                        if labels:
                            tuples = []
                            uniq_key = set()
                            for rel in labels:
                                relation_type = rel["relation"]
                                entities = []
                                for ent in rel["entities"]:
                                    # print(ent)
                                    char_span = ent.get("offset") if len(ent.get("offset")) == 2 else (0, 0)
                                    a = EntityItem(
                                            text=ent["text"],
                                            char_span=char_span,  # 如果实体为空，offset需要兼容
                                            token_span=self.compute_token_offset_of_entity(
                                                text_offset_mapping=offset_mapping,
                                                entity_offset=char_span
                                            )
                                        )
                                    entities.append(a)
                                tup = TupleItem(
                                    relation_id=self.rel2id.get(
                                        relation_type,
                                        self.rel2id.get(UNKNOWN_RELATION)
                                    ),
                                    entities=entities
                                )
                                # 去重
                                key_arr = [cnt, tup.relation_id]
                                for ent in tup.entities:
                                    key_arr.append(ent.token_span[0])
                                    key_arr.append(ent.token_span[1])
                                key = tuple(key_arr)
                                if key not in uniq_key:
                                    tuples.append(tup)
                                    uniq_key.add(key)
                                    if len(tup.entities) != self.num_entities_in_tuple:
                                        # 设置的一个元组中应该匹配指定数量的实体
                                        raise EntityNumNotMatchError("tuple entity num not equal to entity_num_in"
                                                                     "_tuple!")
                        else:
                            tuples = None

                        if tuples:
                            positive_counter += 1

                        data.append(InputSample(
                            text=text,
                            id=cnt,
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            offset_mapping=offset_mapping,
                            tuples=tuples
                        ))

                        cnt += 1
                        valid_counter += 1
                    except ParseSpanError as err:
                        if self.verbose:
                            LOGGER.debug(err, exc_info=True)
                            LOGGER.warning(f"parse error with labeled text, ignore text in line num: {idx + 1} in {p}")
                        continue

        sliding_samples = self.split_into_short_samples(data)
        if self.verbose and total_counter > 0:
            LOGGER.info(f"process the {data_type.value} file info (before sliding): "
                        f"valid/total={valid_counter}/{total_counter}={valid_counter / total_counter}")
            if valid_counter > 0:
                LOGGER.info(f"positive samples info (before sliding): "
                            f"positive/valid={positive_counter}/{valid_counter}={positive_counter / valid_counter}")
            LOGGER.info(f"total samples number after sliding: {len(sliding_samples)}")

        return sliding_samples

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
            if entity_ed == ed:
                tok_ed = idx
            if 0 <= tok_st <= tok_ed:
                return tok_st, tok_ed

        if not (0 <= tok_st <= tok_ed):
            raise ParseEntityOffsetMappingError("Labeled entity not match tokenizer!")

    def complete_sample(self, sample: InputSample):
        """补全样本的一些信息，如前后补上[CLS]、[SEP], token_span变化，以及负样本的label构造"""
        input_ids = [self.tokenizer.cls_token_id] + sample.input_ids + [self.tokenizer.sep_token_id]
        attention_mask = [1] + sample.attention_mask + [1]
        token_type_ids = [0] + sample.token_type_ids + [0]
        char_offset = sample.char_offset
        offset_mapping = [(0, 0)] + [(s - char_offset, e - char_offset) for s, e in sample.offset_mapping] + [(0, 0)]
        # 负样本
        if not sample.tuples:
            tuples = [
                TupleItem(
                    relation_id=self.rel2id[UNKNOWN_RELATION],
                    entities=[
                        EntityItem(text="", char_span=(0, 0), token_span=(0, 0))
                        for _ in range(self.num_entities_in_tuple)
                    ]
                ),
            ]
        else:
            tuples = sample.tuples
            for tup in tuples:
                is_unknown = tup.relation_id == self.rel2id[UNKNOWN_RELATION]
                for ent in tup.entities:
                    if ent.token_span[0] == ent.token_span[1] or is_unknown:
                        ent.token_span = (0, 0)
                    else:
                        # 加上了[CLS]，应该偏移一位
                        ent.token_span = (ent.token_span[0] + 1, ent.token_span[1] + 1)

        return InputSample(
            id=sample.id,
            text=sample.text,
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            token_offset=sample.token_offset,
            char_offset=sample.char_offset,
            offset_mapping=offset_mapping,
            tuples=tuples
        )

    def split_into_short_samples(self, samples: List[InputSample]):
        """分拆样本，超过长度的slide，并补全负样本label"""
        new_samples = []
        for sample in samples:

            # [CLS]、[SEP]
            if len(sample.input_ids) + 2 <= self.max_seq_len:
                new_samples.append(self.complete_sample(sample))
                continue

            split_sample_list = []
            tokens = self.tokenizer.convert_ids_to_tokens(sample.input_ids)
            for start_ind in range(0, len(sample.input_ids), self.sliding_len):
                while "##" in tokens[start_ind]:
                    start_ind -= 1

                end_ind = start_ind + self.max_seq_len - 2  # start_ind和end_ind是token的起止index
                char_span_list = sample.offset_mapping[start_ind:end_ind]
                char_level_span = [char_span_list[0][0], char_span_list[-1][1]]  # char_level_span是char的起止
                sub_text = sample.text[char_level_span[0]:char_level_span[1]]
                token_offset = start_ind
                char_offset = char_level_span[0]
                if sample.tuples:
                    sub_tup_list = []
                    for tup in sample.tuples:
                        rel_id = tup.relation_id
                        sub_ent_list = []
                        for ent in tup.entities:
                            # 优先考虑空实体输出的情况
                            if ent.token_span[0] == ent.token_span[1]:
                                new_ent = EntityItem(
                                    text=ent.text,
                                    char_span=(0, 0),
                                    token_span=(0, 0)
                                )
                                sub_ent_list.append(new_ent)
                            elif ent.token_span[0] >= start_ind and ent.token_span[1] <= end_ind:
                                token_span = (ent.token_span[0] - token_offset, ent.token_span[1] - token_offset)
                                new_ent = EntityItem(
                                    text=ent.text,
                                    char_span=(ent.char_span[0] - char_offset, ent.char_span[1] - char_offset),
                                    token_span=token_span
                                )

                                sub_ent_list.append(new_ent)
                            else:
                                # 截断破坏关系的终止
                                break
                        # 若截断破坏关系抽取了，输出应改为空
                        if len(sub_ent_list) != len(tup.entities):
                            continue

                        sub_tup_list.append(TupleItem(
                            relation_id=rel_id,
                            entities=sub_ent_list
                        ))
                else:
                    sub_tup_list = None

                new_sample = InputSample(
                    id=sample.id,
                    text=sub_text,
                    token_offset=token_offset,
                    char_offset=char_offset,
                    tuples=sub_tup_list,
                    token_type_ids=sample.token_type_ids[token_offset:end_ind],
                    attention_mask=sample.attention_mask[token_offset:end_ind],
                    input_ids=sample.input_ids[token_offset:end_ind],
                    offset_mapping=char_span_list
                )
                split_sample_list.append(self.complete_sample(new_sample))
                # 已经计算到最后了，就不再接着滑动窗口
                if end_ind > len(sample.input_ids):
                    break

            new_samples.extend(split_sample_list)

        return new_samples


class RelDataset(Dataset):
    def __init__(self, samples: List[InputSample]):
        self.features = samples

    def __getitem__(self, index):
        return self.features[index]

    def __len__(self):
        return len(self.features)

    @staticmethod
    def collate_fn(batch_data: List[InputSample]):
        batch_size = len(batch_data)
        sent_lens = list(map(lambda x: len(x.input_ids), batch_data))
        max_sent_len = max(sent_lens)

        input_ids = torch.zeros((batch_size, max_sent_len), requires_grad=False).long()
        attention_mask = torch.zeros((batch_size, max_sent_len), requires_grad=False).long()
        token_type_ids = torch.zeros((batch_size, max_sent_len), requires_grad=False).long()

        targets = []
        sent_idx = []

        sample_list = []
        for idx, (seq_len, sample) in enumerate(zip(sent_lens, batch_data)):
            input_ids[idx, :seq_len] = torch.LongTensor(sample.input_ids)
            attention_mask[idx, :seq_len] = torch.LongTensor(sample.attention_mask)
            token_type_ids[idx, :seq_len] = torch.LongTensor(sample.token_type_ids)

            idx_key = (sample.id, sample.token_offset)
            sent_idx.append(idx_key)
            sample_list.append(sample)

            relation_ids = []
            entity_span_ids = []
            for t in sample.tuples:
                relation_ids.append(t.relation_id)
                ent_span_ids = []
                for e in t.entities:
                    ent_span_ids.append(e.token_span)
                entity_span_ids.append(ent_span_ids)

            targets.append({
                "relation": torch.tensor(relation_ids, dtype=torch.long, requires_grad=False),
                "entity": torch.tensor(entity_span_ids, dtype=torch.long, requires_grad=False)
            })

        return BatchOutput(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            targets=targets,
            sent_lens=sent_lens,
            sent_idx=sent_idx,
            sample_list=sample_list
        )
