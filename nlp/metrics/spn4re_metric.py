# -*- coding: utf8 -*-
"""
======================================
    Project Name: NLP
    File Name: spn4re_metric
    Author: czh
    Create Date: 2021/11/15
--------------------------------------
    Change Activity: 
======================================
"""
import logging
from collections import defaultdict
from typing import Dict, List, Tuple

from nlp.utils.factory import PredTuple, GoldTuple

LOGGER = logging.getLogger(__name__)


def metric(pred: Dict[Tuple[int, int], List[PredTuple]], gold: Dict[Tuple[int, int], List[GoldTuple]],
           verbose: bool = True):
    assert pred.keys() == gold.keys()
    gold_num = 0
    rel_num = 0
    ent_num = 0
    right_num = 0
    pred_num = 0
    for sent_idx in pred:
        gold_num += len(gold[sent_idx])
        pred_correct_num = 0

        pred_i = list(set(
            [
                (ele.rel, tuple([(e.start_index, e.end_index) for e in ele.ents if e]))
                for ele in pred[sent_idx]
            ]
        ))

        pred_num += len(pred_i)
        pred_rel_dict = defaultdict(int)
        pred_ent_dict = defaultdict(int)

        gold_rel_dict = defaultdict(int)
        gold_ent_dict = defaultdict(int)

        gold_i = []
        for ele in gold[sent_idx]:
            ent = tuple([(e.start_index, e.end_index) for e in ele.ents])
            gold_i.append((ele.rel, ent))
            gold_rel_dict[ele.rel] += 1
            gold_ent_dict[ent] += 1

        # 参考https://github.com/DianboWork/SPN4RE/issues/15#issuecomment-866662471
        # 主要按照2个list求共同元素个数，且pred中重复元素个数不能超过gold
        for ele in pred_i:
            if ele in gold_i:
                right_num += 1
                pred_correct_num += 1
            rel, ent = ele
            if rel in gold_rel_dict:
                pred_rel_dict[rel] = min(pred_rel_dict[rel] + 1, gold_rel_dict[rel])

            if ent in gold_ent_dict:
                pred_ent_dict[ent] = min(pred_ent_dict[ent] + 1, gold_ent_dict[ent])

        rel_num += sum(pred_rel_dict.values())
        ent_num += sum(pred_ent_dict.values())

    if pred_num == 0:
        precision = -1
        r_p = -1
        e_p = -1
    else:
        precision = (right_num + 0.0) / pred_num
        e_p = (ent_num + 0.0) / pred_num
        r_p = (rel_num + 0.0) / pred_num

    if gold_num == 0:
        recall = -1
        r_r = -1
        e_r = -1
    else:
        recall = (right_num + 0.0) / gold_num
        e_r = ent_num / gold_num
        r_r = rel_num / gold_num

    if (precision == -1) or (recall == -1) or (precision + recall) <= 0.:
        f_measure = -1
    else:
        f_measure = 2 * precision * recall / (precision + recall)

    if (e_p == -1) or (e_r == -1) or (e_p + e_r) <= 0.:
        e_f = -1
    else:
        e_f = 2 * e_r * e_p / (e_p + e_r)

    if (r_p == -1) or (r_r == -1) or (r_p + r_r) <= 0.:
        r_f = -1
    else:
        r_f = 2 * r_p * r_r / (r_r + r_p)

    if verbose:
        LOGGER.info(f"gold_num={gold_num}, pred_num={pred_num}, right_num={right_num}, "
                    f"relation_right_num={rel_num}, entity_right_num={ent_num}")
        LOGGER.info(f"precision={precision}, recall={recall}, f1_value={f_measure}")
        LOGGER.info(f"rel_precision={r_p}, rel_recall={r_r}, rel_f1_value={r_f}")
        LOGGER.info(f"ent_precision={e_p}, ent_recall={e_r}, ent_f1_value={e_f}")
        LOGGER.info("--------------------")
    return {"precision": precision, "recall": recall, "f1": f_measure}


def num_metric(
        pred: Dict[Tuple[int, int], List[PredTuple]],
        gold: Dict[Tuple[int, int], List[GoldTuple]],
        verbose: bool = True
):
    test_1, test_2, test_3, test_4, test_other = [], [], [], [], []
    for sent_idx in gold:
        if len(gold[sent_idx]) == 1:
            test_1.append(sent_idx)
        elif len(gold[sent_idx]) == 2:
            test_2.append(sent_idx)
        elif len(gold[sent_idx]) == 3:
            test_3.append(sent_idx)
        elif len(gold[sent_idx]) == 4:
            test_4.append(sent_idx)
        else:
            test_other.append(sent_idx)

    pred_1 = get_key_val(pred, test_1)
    gold_1 = get_key_val(gold, test_1)
    pred_2 = get_key_val(pred, test_2)
    gold_2 = get_key_val(gold, test_2)
    pred_3 = get_key_val(pred, test_3)
    gold_3 = get_key_val(gold, test_3)
    pred_4 = get_key_val(pred, test_4)
    gold_4 = get_key_val(gold, test_4)
    pred_other = get_key_val(pred, test_other)
    gold_other = get_key_val(gold, test_other)
    # pred_other = dict((key, vals) for key, vals in pred.items() if key in test_other)
    # gold_other = dict((key, vals) for key, vals in gold.items() if key in test_other)
    if verbose:
        LOGGER.info("--*--*--Num of Gold tuple is 1--*--*--")
    _ = metric(pred_1, gold_1)
    if verbose:
        LOGGER.info("--*--*--Num of Gold tuple is 2--*--*--")
    _ = metric(pred_2, gold_2)
    if verbose:
        LOGGER.info("--*--*--Num of Gold tuple is 3--*--*--")
    _ = metric(pred_3, gold_3)
    if verbose:
        LOGGER.info("--*--*--Num of Gold tuple is 4--*--*--")
    _ = metric(pred_4, gold_4)
    if verbose:
        LOGGER.info("--*--*--Num of Gold tuple is greater than or equal to 5--*--*--")
    _ = metric(pred_other, gold_other)


def overlap_metric(
        pred: Dict[Tuple[int, int], List[PredTuple]],
        gold: Dict[Tuple[int, int], List[GoldTuple]],
        num_entities_in_tuple: int = 2,
        verbose: bool = True
):
    normal_idx, multi_label_idx, overlap_idx = [], [], []
    for sent_idx in gold:
        tuples = gold[sent_idx]
        if is_normal_tuple(tuples, num_entities_in_tuple):
            normal_idx.append(sent_idx)
        else:
            if is_multi_label(tuples):
                multi_label_idx.append(sent_idx)
            if is_overlapping(tuples, num_entities_in_tuple):
                overlap_idx.append(sent_idx)
    pred_normal = get_key_val(pred, normal_idx)
    gold_normal = get_key_val(gold, normal_idx)
    pred_multilabel = get_key_val(pred, multi_label_idx)
    gold_multilabel = get_key_val(gold, multi_label_idx)
    pred_overlap = get_key_val(pred, overlap_idx)
    gold_overlap = get_key_val(gold, overlap_idx)
    if verbose:
        LOGGER.info("--*--*--Normal tuples--*--*--")
    _ = metric(pred_normal, gold_normal)
    if verbose:
        LOGGER.info("--*--*--Multiply label tuples--*--*--")
    _ = metric(pred_multilabel, gold_multilabel)
    if verbose:
        LOGGER.info("--*--*--Overlapping tuples--*--*--")
    _ = metric(pred_overlap, gold_overlap)


def is_normal_tuple(tuples: List[GoldTuple], num_entities_in_tuple: int = 2):
    """是否是正常元组，即元组中实体无重复使用"""
    entities = set()
    for tup in tuples:
        for ent in tup.ents:
            entities.add((ent.start_index, ent.end_index))
    return len(entities) == num_entities_in_tuple * len(tuples)


def is_multi_label(tuples: List[GoldTuple]):
    """是否是多关系（重叠关系），即一个元组实体相同，实体关系却不一样"""
    entity_pair = [tuple([(e.start_index, e.end_index) for e in tup.ents]) for tup in tuples]
    return len(entity_pair) != len(set(entity_pair))


def is_overlapping(tuples: List[GoldTuple], num_entities_in_tuple: int = 2):
    """有重叠实体"""
    entity_pair = [tuple([(e.start_index, e.end_index) for e in tup.ents]) for tup in tuples]
    entity_pair = set(entity_pair)
    entities = []
    for pair in entity_pair:
        for ent_pair in pair:
            entities.append(ent_pair)
    entities = set(entities)
    return len(entities) != num_entities_in_tuple * len(entity_pair)


def get_key_val(dict_1, list_1):
    dict_2 = dict()
    for ele in list_1:
        dict_2.update({ele: dict_1[ele]})
    return dict_2
