#!/usr/bin/env python
# -*- coding:utf-8 _*-

from typing import List, Dict, Tuple

import torch

from nlp.utils.enums import RunMode
from nlp.utils.factory import PredEntity, PredTuple, PredRelation, GoldTuple, GoldEntity


def _get_best_indexes(logits, n_best_size):
    """Get the n-best logits from a list."""
    index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)
    best_indexes = []
    for i in range(len(index_and_score)):
        if i >= n_best_size:
            break
        best_indexes.append(index_and_score[i][0])
    return best_indexes


def generate_span(
        start_logits,
        end_logits,
        seq_lens,
        sent_indices,
        num_generated_tuples,
        n_best_size,
        max_span_length,
):
    output = {}
    start_probs = start_logits.softmax(-1)
    end_probs = end_logits.softmax(-1)
    start_probs = start_probs.cpu().tolist()
    end_probs = end_probs.cpu().tolist()
    for (start_prob, end_prob, seq_len, sent_idx) in zip(start_probs, end_probs, seq_lens, sent_indices):
        sent_idx = tuple(sent_idx)
        output[sent_idx] = {}
        for tuple_id in range(num_generated_tuples):
            predictions = []
            start_indexes = _get_best_indexes(start_prob[tuple_id], n_best_size)
            end_indexes = _get_best_indexes(end_prob[tuple_id], n_best_size)
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # We could hypothetically create invalid predictions, e.g., predict
                    # that the start of the span is in the sentence. We throw out all
                    # invalid predictions.
                    # 如果为0，表示为空
                    if not (start_index == end_index == 0):
                        if start_index > (seq_len - 1):
                            continue
                        if end_index >= seq_len:  # 等于最后一个[SEP]
                            continue
                        if end_index <= start_index:
                            continue
                        length = end_index - start_index + 1
                        if length > max_span_length:
                            continue

                    predictions.append(
                        PredEntity(
                            start_index=start_index,
                            end_index=end_index,
                            start_prob=start_prob[tuple_id][start_index],
                            end_prob=end_prob[tuple_id][end_index],
                        )
                    )
            output[sent_idx][tuple_id] = predictions
    return output


def generate_relation(
        pred_rel_logits,
        sent_indices,
        num_generated_tuples
):
    rel_probs, pred_rels = torch.max(pred_rel_logits.softmax(-1), dim=2)
    rel_probs = rel_probs.cpu().tolist()
    pred_rels = pred_rels.cpu().tolist()
    output = {}
    for (rel_prob, pred_rel, sent_idx) in zip(rel_probs, pred_rels, sent_indices):
        sent_idx = tuple(sent_idx)
        output[sent_idx] = {}
        for tuple_id in range(num_generated_tuples):
            output[sent_idx][tuple_id] = PredRelation(
                rel=pred_rel[tuple_id],
                rel_prob=rel_prob[tuple_id]
            )
    return output


def generate_tuple(
        outputs,
        sent_lens,
        sent_idx,
        num_generated_tuples,
        n_best_size,
        max_span_length,
        allow_null_entities_in_tuple: List[int],
        num_classes: int,
        run_mode: RunMode = RunMode.INFER
) -> Dict[Tuple[int, int], List[PredTuple]]:
    pred_ents_arr = [
        generate_span(
            head,
            tail,
            sent_lens,
            sent_idx,
            num_generated_tuples,
            n_best_size,
            max_span_length
        ) for head, tail in outputs["pred_ent_logits"]]
    pred_rel_dict = generate_relation(outputs['pred_rel_logits'], sent_idx, num_generated_tuples)
    tuples = {}
    for sent_idx in pred_rel_dict:
        tuples[sent_idx] = []
        for tuple_id in range(num_generated_tuples):
            pred_rel = pred_rel_dict[sent_idx][tuple_id]
            pred_ents = [
                r[sent_idx][tuple_id]
                for r in pred_ents_arr
            ]
            item = generate_strategy(pred_rel, pred_ents, allow_null_entities_in_tuple, num_classes, run_mode=run_mode)
            if item:
                tuples[sent_idx].append(item)
    return tuples


def generate_strategy(
        pred_rel: PredRelation,
        pred_ents: List[List[PredEntity]],
        allow_null_entities_in_tuple: List[int],
        num_classes: int,
        run_mode: RunMode = RunMode.INFER
):
    if pred_rel.rel != num_classes:
        if all([i for i in pred_ents]):
            pents = []
            for idx, ents in enumerate(pred_ents):
                if run_mode == RunMode.INFER:
                    for ent in ents:
                        # 如果不允许为空，那么意味着开始等于结束，这种应该先丢弃
                        if allow_null_entities_in_tuple[idx] == 0 and ent.start_index == ent.end_index:
                            continue
                        else:
                            break
                else:
                    ent = ents[0]
                # if run_mode == RunMode.INFER:
                #     if ent.end_prob != ent.start_prob and ent.start_prob <= 0.5 and ent.end_prob <= 0.5:
                #         return None
                pents.append(ent)
            return PredTuple(
                rel=pred_rel.rel,
                rel_prob=pred_rel.rel_prob,
                ents=pents
            )


def formulate_gold(target, sent_indices) -> Dict[Tuple[int, int], List[GoldTuple]]:
    gold = {}
    for i in range(len(sent_indices)):
        sent_idx = tuple(sent_indices[i])
        gold[sent_idx] = []
        for j in range(len(target[i]["relation"])):
            gold[sent_idx].append(
                GoldTuple(
                    rel=target[i]["relation"][j].item(),
                    ents=[
                        GoldEntity(start_index=head.item(), end_index=tail.item())
                        for head, tail in target[i]["entity"][j]
                    ]
                )
            )
    return gold
