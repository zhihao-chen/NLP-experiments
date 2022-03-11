# -*- coding: utf8 -*-
"""
======================================
    Project Name: NLP
    File Name: metric
    Author: czh
    Create Date: 2021/8/9
--------------------------------------
    Change Activity: 
======================================
"""
# 评价指标
import copy
from collections import Counter
import difflib
from _collections import defaultdict

import numpy as np
from sklearn.metrics import auc
import torch
# from pattern.text.en import lexeme, lemma

from nlp.processors.utils_ner import get_entities
from nlp.processors.utils_ee import join_segs


class MetricsCalculator(object):
    def __init__(self, id2label=None):
        super().__init__()
        self.id2label = id2label
        self.origins = []
        self.founds = []
        self.rights = []
        self.reset()

    def set_id2labels(self, id2label):
        self.id2label = id2label

    def reset(self):
        self.origins = []
        self.founds = []
        self.rights = []

    @staticmethod
    def compute(origin, found, right):
        recall = 0 if origin == 0 else (right / origin)
        precision = 0 if found == 0 else (right / found)
        f1 = 0. if recall + precision == 0 else (2 * precision * recall) / (precision + recall)
        return recall, precision, f1

    def result(self):
        class_info = {}
        origin_counter = Counter([self.id2label[x[1]] for x in self.origins])
        found_counter = Counter([self.id2label[x[1]] for x in self.founds])
        right_counter = Counter([self.id2label[x[1]] for x in self.rights])
        for type_, count in origin_counter.items():
            origin = count
            found = found_counter.get(type_, 0)
            right = right_counter.get(type_, 0)
            recall, precision, f1 = self.compute(origin, found, right)
            class_info[type_] = {'precision': round(precision, 4),
                                 'recall': round(recall, 4),
                                 'f1': round(f1, 4),
                                 'gold_num': origin,
                                 'pred_num': found,
                                 'right_num': right}
        origin = len(self.origins)
        found = len(self.founds)
        right = len(self.rights)
        recall, precision, f1 = self.compute(origin, found, right)
        return {'precision': precision, 'recall': recall, 'f1': f1}, class_info

    def update(self, y_true, y_pred):
        y_pred = y_pred.cpu().numpy()
        y_true = y_true.cpu().numpy()
        pred = []
        true = []
        for b, l, start, end in zip(*np.where(y_pred > 0)):
            pred.append((b, l, start, end))
        for b, l, start, end in zip(*np.where(y_true > 0)):
            true.append((b, l, start, end))
        self.origins.extend(true)
        self.founds.extend(pred)
        self.rights.extend([pre_entity for pre_entity in pred if pre_entity in true])

    @staticmethod
    def get_sample_f1(y_pred, y_true):
        y_pred = torch.gt(y_pred, 0).float()
        return 2 * torch.sum(y_true * y_pred) / torch.sum(y_true + y_pred)

    @staticmethod
    def get_sample_precision(y_pred, y_true):
        y_pred = torch.gt(y_pred, 0).float()
        return torch.sum(y_pred[y_true == 1]) / (y_pred.sum() + 1)

    @staticmethod
    def get_evaluate_fpr(y_pred, y_true):
        y_pred = y_pred.cpu().numpy()
        y_true = y_true.cpu().numpy()
        pred = []
        true = []
        for b, l, start, end in zip(*np.where(y_pred > 0)):
            pred.append((b, l, start, end))
        for b, l, start, end in zip(*np.where(y_true > 0)):
            true.append((b, l, start, end))

        r = set(pred)
        t = set(true)
        x = len(r & t)
        y = len(r) if r else 1e-10
        z = len(t) if t else 1e-10
        f1, precision, recall = 2 * x / (y + z), x / y, x / z
        return f1, precision, recall


def get_prf_scores(correct_num, pred_num, gold_num, eval_type):
    """
    get precision, recall, and F1 score
    :param correct_num:
    :param pred_num:
    :param gold_num:
    :param eval_type:
    :return:
    """
    if correct_num == pred_num == gold_num == 0:
        return 1.2333, 1.2333, 1.2333  # highlight this info by illegal outputs instead of outputting 0.

    minimum = 1e-20
    precision = correct_num / (pred_num + minimum)
    recall = correct_num / (gold_num + minimum)
    f1 = 2 * precision * recall / (precision + recall + minimum)

    results = {f"{eval_type}_precision": round(precision, 5),
               f"{eval_type}_recall": round(recall, 5),
               f"{eval_type}_f1": round(f1, 5)}
    return results


class SeqEntityScore(object):
    """
    ner metric
    """
    def __init__(self, id2label, markup='bios'):
        self.id2label = id2label
        self.markup = markup
        self.origins = []
        self.founds = []
        self.rights = []

        self.reset()

    def reset(self):
        self.origins = []
        self.founds = []
        self.rights = []

    @staticmethod
    def compute(origin, found, right):
        recall = 0 if origin == 0 else (right / origin)
        precision = 0 if found == 0 else (right / found)
        f1 = 0. if recall + precision == 0 else (2 * precision * recall) / (precision + recall)
        return recall, precision, f1

    def result(self):
        class_info = {}
        origin_counter = Counter([x[0] for x in self.origins])
        found_counter = Counter([x[0] for x in self.founds])
        right_counter = Counter([x[0] for x in self.rights])
        for type_, count in origin_counter.items():
            origin = count
            found = found_counter.get(type_, 0)
            right = right_counter.get(type_, 0)
            recall, precision, f1 = self.compute(origin, found, right)
            class_info[type_] = {"precision": round(precision, 4), 'recall': round(recall, 4), 'f1': round(f1, 4),
                                 "gold_num": origin, "pred_num": found, "right_num": right}
        origin = len(self.origins)
        found = len(self.founds)
        right = len(self.rights)
        recall, precision, f1 = self.compute(origin, found, right)
        return {'precision': precision, 'recall': recall, 'f1': f1}, class_info

    def update(self, label_paths, pred_paths):
        """
        labels_paths: [[],[],[],....]
        pred_paths: [[],[],[],.....]

        :param label_paths:
        :param pred_paths:
        :return:
        Example:
            labels_paths = [['O', 'O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
            pred_paths = [['O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        """
        for label_path, pre_path in zip(label_paths, pred_paths):
            label_entities = get_entities(label_path, self.id2label, self.markup)
            pre_entities = get_entities(pre_path, self.id2label, self.markup)
            self.origins.extend(label_entities)
            self.founds.extend(pre_entities)
            self.rights.extend([pre_entity for pre_entity in pre_entities if pre_entity in label_entities])


class SpanEntityScore(object):
    """
    ner metric
    """
    def __init__(self, id2label):
        self.id2label = id2label
        self.origins = []
        self.founds = []
        self.rights = []
        self.reset()

    def reset(self):
        self.origins = []
        self.founds = []
        self.rights = []

    @staticmethod
    def compute(origin, found, right):
        recall = 0 if origin == 0 else (right / origin)
        precision = 0 if found == 0 else (right / found)
        f1 = 0. if recall + precision == 0 else (2 * precision * recall) / (precision + recall)
        return recall, precision, f1

    def result(self):
        class_info = {}
        origin_counter = Counter([self.id2label[x[0]] for x in self.origins])
        found_counter = Counter([self.id2label[x[0]] for x in self.founds])
        right_counter = Counter([self.id2label[x[0]] for x in self.rights])
        for type_, count in origin_counter.items():
            origin = count
            found = found_counter.get(type_, 0)
            right = right_counter.get(type_, 0)
            recall, precision, f1 = self.compute(origin, found, right)
            class_info[type_] = {"precision": round(precision, 4), 'recall': round(recall, 4), 'f1': round(f1, 4),
                                 "gold_num": origin, "pred_num": found, "right_num": right}
        origin = len(self.origins)
        found = len(self.founds)
        right = len(self.rights)
        recall, precision, f1 = self.compute(origin, found, right)
        return {'precision': precision, 'recall': recall, 'f1': f1}, class_info

    def update(self, true_subject, pred_subject):
        self.origins.extend(true_subject)
        self.founds.extend(pred_subject)
        self.rights.extend([pre_entity for pre_entity in pred_subject if pre_entity in true_subject])


class TPLinkerMetricsCalculator(object):
    def __init__(self, handshaking_tagger):
        self.handshaking_tagger = handshaking_tagger

    @staticmethod
    def get_sample_accuracy(pred, truth):
        """
        计算所有抽取字段都正确的样本比例
        即该batch的输出与truth全等的样本比例
        """
        # (batch_size, ..., seq_len, tag_size) -> (batch_size, ..., seq_len)
        pred_id = torch.argmax(pred, dim=-1)
        # (batch_size, ..., seq_len) -> (batch_size, )，把每个sample压成一条seq
        pred_id = pred_id.view(pred_id.size()[0], -1)
        truth = truth.view(truth.size()[0], -1)

        # (batch_size, )，每个元素是pred与truth之间tag相同的数量
        correct_tag_num = torch.sum(torch.eq(truth, pred_id).float(), dim=1)

        # seq维上所有tag必须正确，所以correct_tag_num必须等于seq的长度才算一个correct的sample
        sample_acc_ = torch.eq(correct_tag_num, torch.ones_like(correct_tag_num) * truth.size()[-1]).float()
        sample_acc = torch.mean(sample_acc_)
        return sample_acc

    def get_rel_cpg(self, sample_list, tok2char_span_list, batch_pred_ent_shaking_outputs,
                    batch_pred_head_rel_shaking_outputs, batch_pred_tail_rel_shaking_outputs, pattern="only_head_text"):
        batch_pred_ent_shaking_tag = torch.argmax(batch_pred_ent_shaking_outputs, dim=-1)
        batch_pred_head_rel_shaking_tag = torch.argmax(batch_pred_head_rel_shaking_outputs, dim=-1)
        batch_pred_tail_rel_shaking_tag = torch.argmax(batch_pred_tail_rel_shaking_outputs, dim=-1)

        correct_num, pred_num, gold_num = 0, 0, 0
        for ind in range(len(sample_list)):
            sample = sample_list[ind]
            text = sample["text"]
            tok2char_span = tok2char_span_list[ind]
            pred_ent_shaking_tag = batch_pred_ent_shaking_tag[ind]
            pred_head_rel_shaking_tag = batch_pred_head_rel_shaking_tag[ind]
            pred_tail_rel_shaking_tag = batch_pred_tail_rel_shaking_tag[ind]

            pred_rel_list = self.handshaking_tagger.decode_rel_fr_shaking_tag(text, pred_ent_shaking_tag,
                                                                              pred_head_rel_shaking_tag,
                                                                              pred_tail_rel_shaking_tag, tok2char_span)
            gold_rel_list = sample["relation_list"]

            if pattern == "only_head_index":
                gold_rel_set = set(["{}\u2E80{}\u2E80{}".format(rel["subj_tok_span"][0], rel["predicate"],
                                                                rel["obj_tok_span"][0]) for rel in gold_rel_list])
                pred_rel_set = set(["{}\u2E80{}\u2E80{}".format(rel["subj_tok_span"][0], rel["predicate"],
                                                                rel["obj_tok_span"][0]) for rel in pred_rel_list])
            elif pattern == "whole_span":
                gold_rel_set = set(["{}\u2E80{}\u2E80{}\u2E80{}\u2E80{}".format(rel["subj_tok_span"][0],
                                                                                rel["subj_tok_span"][1],
                                                                                rel["predicate"],
                                                                                rel["obj_tok_span"][0],
                                                                                rel["obj_tok_span"][1]) for rel in
                                    gold_rel_list])
                pred_rel_set = set(["{}\u2E80{}\u2E80{}\u2E80{}\u2E80{}".format(rel["subj_tok_span"][0],
                                                                                rel["subj_tok_span"][1],
                                                                                rel["predicate"],
                                                                                rel["obj_tok_span"][0],
                                                                                rel["obj_tok_span"][1]) for rel in
                                    pred_rel_list])
            elif pattern == "whole_text":
                gold_rel_set = set(["{}\u2E80{}\u2E80{}".format(rel["subject"], rel["predicate"],
                                                                rel["object"]) for rel in gold_rel_list])
                pred_rel_set = set(["{}\u2E80{}\u2E80{}".format(rel["subject"], rel["predicate"],
                                                                rel["object"]) for rel in pred_rel_list])
            elif pattern == "only_head_text":
                gold_rel_set = set(["{}\u2E80{}\u2E80{}".format(rel["subject"].split(" ")[0], rel["predicate"],
                                                                rel["object"].split(" ")[0]) for rel in gold_rel_list])
                pred_rel_set = set(["{}\u2E80{}\u2E80{}".format(rel["subject"].split(" ")[0], rel["predicate"],
                                                                rel["object"].split(" ")[0]) for rel in pred_rel_list])
            else:
                raise ValueError("'pattern' must be one of the list: "
                                 "['only_head_index', 'whole_span', 'whole_text', 'only_head_text']")

            for rel_str in pred_rel_set:
                if rel_str in gold_rel_set:
                    correct_num += 1

            pred_num += len(pred_rel_set)
            gold_num += len(gold_rel_set)

        return correct_num, pred_num, gold_num

    @staticmethod
    def get_prf_scores(correct_num, pred_num, gold_num):
        minimini = 1e-10
        precision = correct_num / (pred_num + minimini)
        recall = correct_num / (gold_num + minimini)
        f1 = 2 * precision * recall / (precision + recall + minimini)
        return precision, recall, f1


class TPLinkerPlusMetricsCalculator(object):
    """
    TPLinker metrics
    """
    def __init__(self, shaking_tagger):
        self.shaking_tagger = shaking_tagger
        self.last_weights = None  # for exponential moving averaging

    def ghm(self, gradient, bins=10, beta=0.9):
        """
        gradient_norm: gradient_norms of all examples in this batch; (batch_size, shaking_seq_len)
        """
        avg = torch.mean(gradient)
        std = torch.std(gradient) + 1e-12
        gradient_norm = torch.sigmoid((gradient - avg) / std)  # normalization and pass through sigmoid to 0 ~ 1.

        min_, max_ = torch.min(gradient_norm), torch.max(gradient_norm)
        gradient_norm = (gradient_norm - min_) / (max_ - min_)
        gradient_norm = torch.clamp(gradient_norm, 0, 0.9999999)  # ensure elements in gradient_norm != 1.

        example_sum = torch.flatten(gradient_norm).size()[0]  # N

        # calculate weights
        current_weights = torch.zeros(bins).to(gradient.device)
        hits_vec = torch.zeros(bins).to(gradient.device)
        count_hits = 0  # coungradient_normof hits
        for i in range(bins):
            bar = float((i + 1) / bins)
            hits = torch.sum((gradient_norm <= bar)) - count_hits  # noqa
            count_hits += hits
            hits_vec[i] = hits.item()
            current_weights[i] = example_sum / bins / (hits.item() + example_sum / bins)
        # EMA: exponential moving averaging
        if self.last_weights is None:
            self.last_weights = torch.ones(bins).to(gradient.device)  # init by ones
        current_weights = self.last_weights * beta + (1 - beta) * current_weights
        self.last_weights = current_weights

        # weights4examples: pick weights for all examples
        weight_pk_idx = (gradient_norm / (1 / bins)).long()[:, :, None]
        weights_rp = current_weights[None, None, :].repeat(gradient_norm.size()[0], gradient_norm.size()[1], 1)
        weights4examples = torch.gather(weights_rp, -1, weight_pk_idx).squeeze(-1)
        weights4examples /= torch.sum(weights4examples)
        return weights4examples * gradient  # return weighted gradients

    # loss func
    def _multilabel_categorical_crossentropy(self, y_pred, y_true, ghm=True):
        """
        y_pred: (batch_size, shaking_seq_len, type_size)
        y_true: (batch_size, shaking_seq_len, type_size)
        y_true and y_pred have the same shape，elements in y_true are either 0 or 1，
             1 tags positive classes，0 tags negtive classes(means tok-pair does not have this type of link).
        """
        y_pred = (1 - 2 * y_true) * y_pred  # -1 -> pos classes, 1 -> neg classes
        y_pred_neg = y_pred - y_true * 1e12  # mask the pred oudtuts of pos classes
        y_pred_pos = y_pred - (1 - y_true) * 1e12  # mask the pred oudtuts of neg classes
        zeros = torch.zeros_like(y_pred[..., :1])  # st - st
        y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
        y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
        neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
        pos_loss = torch.logsumexp(y_pred_pos, dim=-1)

        if ghm:
            return (self.ghm(neg_loss + pos_loss, bins=1000)).sum()
        else:
            return (neg_loss + pos_loss).mean()

    def loss_func(self, y_pred, y_true, ghm):
        return self._multilabel_categorical_crossentropy(y_pred, y_true, ghm=ghm)

    @staticmethod
    def get_sample_accuracy(pred, truth):
        """
        计算该batch的pred与truth全等的样本比例
        """
        # # (batch_size, ..., seq_len, tag_size) -> (batch_size, ..., seq_len)
        #  pred = torch.argmax(pred, dim = -1)
        # (batch_size, ..., seq_len) -> (batch_size, seq_len)
        pred = pred.view(pred.size()[0], -1)
        truth = truth.view(truth.size()[0], -1)

        # (batch_size, )，每个元素是pred与truth之间tag相同的数量
        correct_tag_num = torch.sum(torch.eq(truth, pred).float(), dim=1)

        # seq维上所有tag必须正确，所以correct_tag_num必须等于seq的长度才算一个correct的sample
        sample_acc_ = torch.eq(correct_tag_num, torch.ones_like(correct_tag_num) * truth.size()[-1]).float()
        sample_acc = torch.mean(sample_acc_)

        return sample_acc

    @staticmethod
    def get_mark_sets_event(event_list):
        trigger_iden_set, trigger_class_set, arg_iden_set, arg_class_set = set(), set(), set(), set()
        for event in event_list:
            event_type = event["trigger_type"]
            trigger_offset = event["trigger_tok_span"]
            trigger_iden_set.add("{}\u2E80{}".format(trigger_offset[0], trigger_offset[1]))
            trigger_class_set.add("{}\u2E80{}\u2E80{}".format(event_type, trigger_offset[0], trigger_offset[1]))
            for arg in event["argument_list"]:
                argument_offset = arg["tok_span"]
                argument_role = arg["type"]
                arg_iden_set.add(
                    "{}\u2E80{}\u2E80{}\u2E80{}\u2E80{}".format(event_type, trigger_offset[0], trigger_offset[1],
                                                                argument_offset[0], argument_offset[1]))
                arg_class_set.add("{}\u2E80{}\u2E80{}\u2E80{}\u2E80{}\u2E80{}".format(event_type, trigger_offset[0],
                                                                                      trigger_offset[1],
                                                                                      argument_offset[0],
                                                                                      argument_offset[1],
                                                                                      argument_role))

        return trigger_iden_set, trigger_class_set, arg_iden_set, arg_class_set

    @staticmethod
    def get_mark_sets_rel(rel_list, ent_list, pattern="only_head_text"):
        if pattern == "only_head_index":
            rel_set = set(
                ["{}\u2E80{}\u2E80{}".format(rel["subj_tok_span"][0], rel["predicate"], rel["obj_tok_span"][0]) for rel
                 in rel_list])
            ent_set = set(["{}\u2E80{}".format(ent["tok_span"][0], ent["type"]) for ent in ent_list])

        elif pattern == "whole_span":
            rel_set = set(["{}\u2E80{}\u2E80{}\u2E80{}\u2E80{}".format(rel["subj_tok_span"][0], rel["subj_tok_span"][1],
                                                                       rel["predicate"], rel["obj_tok_span"][0],
                                                                       rel["obj_tok_span"][1]) for rel in rel_list])
            ent_set = set(
                ["{}\u2E80{}\u2E80{}".format(ent["tok_span"][0], ent["tok_span"][1], ent["type"]) for ent in ent_list])

        elif pattern == "whole_text":
            rel_set = set(
                ["{}\u2E80{}\u2E80{}".format(rel["subject"], rel["predicate"], rel["object"]) for rel in rel_list])
            ent_set = set(["{}\u2E80{}".format(ent["text"], ent["type"]) for ent in ent_list])

        elif pattern == "only_head_text":
            rel_set = set(["{}\u2E80{}\u2E80{}".format(rel["subject"].split(" ")[0], rel["predicate"],
                                                       rel["object"].split(" ")[0]) for rel in rel_list])
            ent_set = set(["{}\u2E80{}".format(ent["text"].split(" ")[0], ent["type"]) for ent in ent_list])
        else:
            raise ValueError("'pattern' must be one of the list: "
                             "['only_head_index', 'whole_span', 'whole_text', 'only_head_text']")

        return rel_set, ent_set

    @staticmethod
    def _cal_cpg(pred_set, gold_set, cpg):
        """
        cpg is a list: [correct_num, pred_num, gold_num]
        """
        for mark_str in pred_set:
            if mark_str in gold_set:
                cpg[0] += 1
        cpg[1] += len(pred_set)
        cpg[2] += len(gold_set)

    def cal_rel_cpg(self, pred_rel_list, pred_ent_list, gold_rel_list, gold_ent_list, ere_cpg_dict, pattern):
        """
        ere_cpg_dict = {
                "rel_cpg": [0, 0, 0],
                "ent_cpg": [0, 0, 0],
                }
        pattern: metric pattern
        """
        gold_rel_set, gold_ent_set = self.get_mark_sets_rel(gold_rel_list, gold_ent_list, pattern)
        pred_rel_set, pred_ent_set = self.get_mark_sets_rel(pred_rel_list, pred_ent_list, pattern)

        self._cal_cpg(pred_rel_set, gold_rel_set, ere_cpg_dict["rel_cpg"])
        self._cal_cpg(pred_ent_set, gold_ent_set, ere_cpg_dict["ent_cpg"])

    def cal_event_cpg(self, pred_event_list, gold_event_list, ee_cpg_dict):
        """
        ee_cpg_dict = {
            "trigger_iden_cpg": [0, 0, 0],
            "trigger_class_cpg": [0, 0, 0],
            "arg_iden_cpg": [0, 0, 0],
            "arg_class_cpg": [0, 0, 0],
        }
        """
        (pred_trigger_iden_set, pred_trigger_class_set,
         pred_arg_iden_set, pred_arg_class_set) = self.get_mark_sets_event(pred_event_list)

        (gold_trigger_iden_set, gold_trigger_class_set,
         gold_arg_iden_set, gold_arg_class_set) = self.get_mark_sets_event(gold_event_list)

        self._cal_cpg(pred_trigger_iden_set, gold_trigger_iden_set, ee_cpg_dict["trigger_iden_cpg"])
        self._cal_cpg(pred_trigger_class_set, gold_trigger_class_set, ee_cpg_dict["trigger_class_cpg"])
        self._cal_cpg(pred_arg_iden_set, gold_arg_iden_set, ee_cpg_dict["arg_iden_cpg"])
        self._cal_cpg(pred_arg_class_set, gold_arg_class_set, ee_cpg_dict["arg_class_cpg"])

    def get_cpg(self, sample_list,
                tok2char_span_list,
                batch_pred_shaking_tag,
                pattern="only_head_text"):
        """
        return correct number, predict number, gold number (cpg)
        """

        ee_cpg_dict = {
            "trigger_iden_cpg": [0, 0, 0],
            "trigger_class_cpg": [0, 0, 0],
            "arg_iden_cpg": [0, 0, 0],
            "arg_class_cpg": [0, 0, 0],
        }
        ere_cpg_dict = {
            "rel_cpg": [0, 0, 0],
            "ent_cpg": [0, 0, 0],
        }

        # go through all sentences
        for ind in range(len(sample_list)):
            sample = sample_list[ind]
            text = sample["text"]
            tok2char_span = tok2char_span_list[ind]
            pred_shaking_tag = batch_pred_shaking_tag[ind]

            pred_rel_list, pred_ent_list = self.shaking_tagger.decode_rel(text,
                                                                          pred_shaking_tag,
                                                                          tok2char_span)  # decoding
            gold_rel_list = sample["relation_list"]
            gold_ent_list = sample["entity_list"]

            if pattern == "event_extraction":
                pred_event_list = self.shaking_tagger.trans2ee(pred_rel_list, pred_ent_list)  # transform to event list
                gold_event_list = sample["event_list"]
                self.cal_event_cpg(pred_event_list, gold_event_list, ee_cpg_dict)
            else:
                self.cal_rel_cpg(pred_rel_list, pred_ent_list, gold_rel_list, gold_ent_list, ere_cpg_dict, pattern)

        if pattern == "event_extraction":
            return ee_cpg_dict
        else:
            return ere_cpg_dict

    @staticmethod
    def get_prf_scores(correct_num, pred_num, gold_num):
        minimini = 1e-12
        precision = correct_num / (pred_num + minimini)
        recall = correct_num / (gold_num + minimini)
        f1 = 2 * precision * recall / (precision + recall + minimini)
        return precision, recall, f1


class Extraction:
    """
    Stores sentence, single predicate and corresponding arguments.
    """
    def __init__(self, pred, head_pred_index, sent, confidence, question_dist='', index=-1):
        self.pred = pred
        self.head_pred_index = head_pred_index
        self.sent = sent
        self.args = []
        self.confidence = confidence
        self.matched = []
        self.questions = {}
        self.is_mwp = False
        self.question_dist = question_dist
        self.index = index

    def add_arg(self, arg):
        self.args.append(arg)


class OIEMetrics:
    PREPS = ['above', 'across', 'against', 'along', 'among', 'around', 'at', 'before', 'behind', 'below', 'beneath',
             'beside', 'between', 'by', 'for', 'from', 'in', 'into', 'near', 'of', 'off', 'on', 'to', 'toward', 'under',
             'upon', 'with', 'within']

    @staticmethod
    def trans_2extra_obj(data):
        text2spo = dict()
        for sample in data:
            text = sample["text"]
            for spo in sample["open_spo_list"]:
                pred = ""
                pred_prefix = ""
                pred_suffix = ""
                subj = ""
                obj = ""
                other_args = []
                for arg in spo:
                    if arg["type"] == "predicate":
                        pred = arg["text"]
                    elif arg["type"] == "predicate_prefix":
                        pred_prefix = arg["text"]
                    elif arg["type"] == "predicate_suffix":
                        pred_suffix = arg["text"]
                    elif arg["type"] == "subject":
                        subj = arg["text"]
                    elif arg["type"] == "object":
                        obj = arg["text"]
                    else:
                        other_args.append(arg["text"])

                pred_comp = join_segs([pred_prefix, pred, pred_suffix])
                extr_spo = Extraction(pred=pred_comp,
                                      head_pred_index=None,
                                      sent=text,
                                      confidence=1.)
                extr_spo.add_arg(subj)
                object_ = join_segs([obj] + [arg_text for arg_text in other_args])
                extr_spo.add_arg(object_)

                text2spo.setdefault(text, []).append(extr_spo)

        return text2spo

    @staticmethod
    def linient_tuple_match(ref, ex):
        precision = [0, 0]  # 0 out of 0 predicted words match
        recall = [0, 0]  # 0 out of 0 reference words match

        # If, for each part, any word is the same as a reference word, then it's a match.

        def my_lemma(word):
            if word[-2:] in {"'s", "ly", "'t"}:
                word = word[:-2]
            word = lemma(word)
            return word

        predicted_words = [my_lemma(w) for w in ex.pred.split()]
        gold_words = [my_lemma(w) for w in ref.pred.split()]
        precision[1] += len(predicted_words)
        recall[1] += len(gold_words)

        matching_words = 0
        for w in gold_words:
            if w in predicted_words:
                matching_words += 1
                predicted_words.remove(w)

        if matching_words == 0:
            return [0, 0]  # t <-> gt is not a match

        precision[0] += matching_words
        recall[0] += matching_words

        for i in range(len(ref.args)):
            gold_words = [my_lemma(w) for w in ref.args[i].split()]
            recall[1] += len(gold_words)
            if len(ex.args) <= i:
                if i < 2:
                    return [0, 0]  # changed
                else:
                    continue
            predicted_words = [my_lemma(w) for w in ex.args[i].split()]
            precision[1] += len(predicted_words)
            matching_words = 0
            for w in gold_words:
                if w in predicted_words:
                    matching_words += 1
                    predicted_words.remove(w)

            precision[0] += matching_words
            # Currently this slightly penalises systems when the reference
            # reformulates the sentence words, because the reformulation doesn't
            # match the predicted word. It's a one-wrong-word penalty to precision,
            # to all systems that correctly extracted the reformulated word.
            recall[0] += matching_words

        if precision[1] == 0:
            prec = 0
        else:
            prec = 1.0 * precision[0] / precision[1]
        if recall[1] == 0:
            rec = 0
        else:
            rec = 1.0 * recall[0] / recall[1]

        return [prec, rec]

    @staticmethod
    def binary_linient_tuple_match(ref, ex):
        if len(ref.args) >= 2:
            r = copy.copy(ref)
            r.args = [ref.args[0], ' '.join(ref.args[1:])]
        else:
            r = ref
        if len(ex.args) >= 2:
            e = copy.copy(ex)
            e.args = [ex.args[0], ' '.join(ex.args[1:])]
        else:
            e = ex

        stright_match = OIEMetrics.linient_tuple_match(r, e)

        said_type_reln = lexeme("say") + lexeme("tell") + lexeme("add")
        said_type_sentence = False
        for said_verb in said_type_reln:
            if said_verb in ref.pred:
                said_type_sentence = True
                break
        if not said_type_sentence:
            return stright_match
        else:
            if len(ex.args) >= 2:
                e = copy.copy(ex)
                e.args = [' '.join(ex.args[1:]), ex.args[0]]
            else:
                e = ex
            reverse_match = OIEMetrics.linient_tuple_match(r, e)

            return max(stright_match, reverse_match)

    @staticmethod
    def compare_oie4(pred_data, gold_data, matching_func, binary=False, strategy='sm'):
        pred_data = OIEMetrics.trans_2extra_obj(pred_data)
        gold_data = OIEMetrics.trans_2extra_obj(gold_data)

        if binary:
            pred_data = OIEMetrics.binarize(pred_data)
            gold_data = OIEMetrics.binarize(gold_data)
        # taking all distinct values of confidences as thresholds
        confidence_thresholds = set()
        for sent in pred_data:
            for predicted_ex in pred_data[sent]:
                confidence_thresholds.add(predicted_ex.confidence)

        confidence_thresholds = sorted(list(confidence_thresholds))
        num_conf = len(confidence_thresholds)

        p = np.zeros(num_conf)
        pl = np.zeros(num_conf)
        r = np.zeros(num_conf)
        rl = np.zeros(num_conf)

        for sent, goldExtractions in gold_data.items():
            if sent in pred_data:
                predicted_extractions = pred_data[sent]
            else:
                predicted_extractions = []
                # continue # Uncomment if you want to ignore gold_data sentences with no predictions

            scores = [[None for _ in predicted_extractions] for __ in goldExtractions]

            for i, goldEx in enumerate(goldExtractions):
                for j, predictedEx in enumerate(predicted_extractions):
                    score = matching_func(goldEx, predictedEx)
                    scores[i][j] = score

            # OPTIMISED GLOBAL MATCH
            sent_confidences = [extraction.confidence for extraction in predicted_extractions]
            sent_confidences.sort()
            prev_c = 0
            for conf in sent_confidences:
                c = confidence_thresholds.index(conf)
                ext_indices = []
                for ext_indx, extraction in enumerate(predicted_extractions):
                    if extraction.confidence >= conf:
                        ext_indices.append(ext_indx)

                # ksk mod
                recall_numerator = 0
                if strategy == 'sm':
                    for i, row in enumerate(scores):
                        max_recall_row = max([row[ext_indx][1] for ext_indx in ext_indices], default=0)  # noqa
                        recall_numerator += max_recall_row

                precision_numerator = 0
                selected_rows = []
                selected_cols = []
                num_precision_matches = min(len(scores), len(ext_indices))
                for t in range(num_precision_matches):
                    matched_row = -1
                    matched_col = -1
                    matched_precision = -1  # initialised to <0 so that it updates whenever precision is 0 as well
                    for i in range(len(scores)):
                        if i in selected_rows:
                            continue
                        for ext_indx in ext_indices:
                            if ext_indx in selected_cols:
                                continue
                            if scores[i][ext_indx][0] > matched_precision:  # noqa
                                matched_precision = scores[i][ext_indx][0]  # noqa
                                matched_row = i
                                matched_col = ext_indx

                    if matched_col == -1 or matched_row == -1:
                        raise Exception("error in CaRB, matched row/col is -1")

                    selected_rows.append(matched_row)
                    selected_cols.append(matched_col)
                    precision_numerator += scores[matched_row][matched_col][0]  # noqa

                # ksk mod
                if strategy == 'ss':
                    recall_numerator = 0
                    selected_rows = []
                    selected_cols = []
                    num_recall_matches = min(len(scores), len(ext_indices))
                    for t in range(num_recall_matches):
                        matched_row = -1
                        matched_col = -1
                        matched_recall = -1  # initialised to <0 so that it updates whenever recall is 0 as well
                        for i in range(len(scores)):
                            if i in selected_rows:
                                continue
                            for ext_indx in ext_indices:
                                if ext_indx in selected_cols:
                                    continue
                                if scores[i][ext_indx][1] > matched_recall:  # noqa
                                    matched_recall = scores[i][ext_indx][1]  # noqa
                                    matched_row = i
                                    matched_col = ext_indx

                        if matched_col == -1 or matched_row == -1:
                            raise Exception("error in CaRB, matched row/col is -1")

                        selected_rows.append(matched_row)
                        selected_cols.append(matched_col)
                        recall_numerator += scores[matched_row][matched_col][1]  # noqa

                p[prev_c:c + 1] += precision_numerator
                pl[prev_c:c + 1] += len(ext_indices)
                # pl[prev_c:c+1] += num_precision_matches
                r[prev_c:c + 1] += recall_numerator
                rl[prev_c:c + 1] += len(scores)

                prev_c = c + 1

            # for indices beyond the maximum sentence confidence,
            # len(scores) has to be added to the denominator of recall
            rl[prev_c:] += len(scores)

        prec_scores = [a / b if b > 0 else 1 for a, b in zip(p, pl)]
        rec_scores = [a / b if b > 0 else 0 for a, b in zip(r, rl)]

        f1s = [OIEMetrics.f1(p, r) for p, r in zip(prec_scores, rec_scores)]

        try:
            optimal_idx = np.nanargmax(f1s)
            optimal = (np.round(prec_scores[optimal_idx], 4),
                       np.round(rec_scores[optimal_idx], 4),
                       np.round(f1s[optimal_idx], 4),
                       confidence_thresholds[optimal_idx])
            zero_conf_point = (np.round(prec_scores[0], 4),
                               np.round(rec_scores[0], 4),
                               np.round(f1s[0], 4),
                               confidence_thresholds[0])
        except ValueError:
            # When there is no prediction
            optimal = (0, 0, 0, 0)
            zero_conf_point = (0, 0, 0, 0)

        # In order to calculate auc, we need to add the point corresponding to precision=1 , recall=0 to the PR-curve
        temp_rec_scores = rec_scores.copy()
        temp_prec_scores = prec_scores.copy()
        temp_rec_scores.append(0)
        temp_prec_scores.append(1)

        if len(f1s) > 0:
            return np.round(auc(temp_rec_scores, temp_prec_scores),
                            4), optimal, zero_conf_point
        else:
            # When there is no prediction
            return 0, (0, 0, 0, 0), (0, 0, 0, 0)

    @staticmethod
    def binarize(extrs):
        res = defaultdict(lambda: [])
        for sent, extr in extrs.items():
            for ex in extr:
                # Add (a1, r, a2)
                temp = copy.copy(ex)
                temp.args = ex.args[:2]
                res[sent].append(temp)

                if len(ex.args) <= 2:
                    continue

                # Add (a1, r a2 , a3 ...)
                for arg in ex.args[2:]:
                    temp.args = [ex.args[0]]
                    temp.pred = ex.pred + ' ' + ex.args[1]
                    words = arg.split()

                    # Add preposition of arg to rel
                    if words[0].lower() in OIEMetrics.PREPS:
                        temp.pred += ' ' + words[0]
                        words = words[1:]
                    temp.args.append(' '.join(words))
                    res[sent].append(temp)

        return res

    @staticmethod
    def f1(prec, rec):
        return 2 * prec * rec / (prec + rec + 1e-20)

    @staticmethod
    def trans(spo):
        new_spo = {}
        for arg in spo:
            if arg["type"] != "object":
                new_spo[arg["type"]] = arg
            else:
                if "object" not in new_spo:
                    new_spo["object"] = []
                new_spo["object"].append(arg)
        return new_spo

    @staticmethod
    def match(predicted_ex, gold_ex):
        match_score = 0
        element_num = 1e-20

        for key in set(predicted_ex.keys()).union(set(gold_ex.keys())):
            if key != "object":
                element_num += 1
            else:
                predicted_obj_num = len(predicted_ex["object"]) if "object" in predicted_ex else 0
                gold_obj_num = len(
                    gold_ex["object"]) if "object" in gold_ex else 0
                element_num += max(predicted_obj_num, gold_obj_num)

        for tp in predicted_ex:
            if tp in gold_ex:
                if tp != "object":
                    match_score += difflib.SequenceMatcher(
                        None, predicted_ex[tp]["text"],
                        gold_ex[tp]["text"]).ratio()
                else:
                    min_object_num = min(len(predicted_ex["object"]), len(gold_ex["object"]))
                    for idx in range(min_object_num):
                        match_score += difflib.SequenceMatcher(
                            None, predicted_ex["object"][idx]["text"],
                            gold_ex["object"][idx]["text"]).ratio()

        return match_score / element_num

    @staticmethod
    def compare_saoke(pred_data, gold_data, threshold):
        # 读每个ins，计算每个pair的相似性，
        total_correct_num = 0
        total_gold_num = 0
        total_pred_num = 0

        for sample_idx, pred_sample in enumerate(pred_data):
            gold_sample = gold_data[sample_idx]
            pred_spo_list = pred_sample["open_spo_list"]
            # gold_spo_list4debug = gold_sample["open_spo_list"]
            gold_spo_list = copy.deepcopy(gold_sample["open_spo_list"])

            pred_num = len(pred_spo_list)
            gold_num = len(gold_spo_list)

            total_gold_num += gold_num
            total_pred_num += pred_num

            correct_num = 0
            for predicted_ex in pred_spo_list:
                ex_score = 0
                hit_idx = None
                for spo_idx, gold_ex in enumerate(
                        gold_spo_list):
                    match_score = OIEMetrics.match(OIEMetrics.trans(predicted_ex), OIEMetrics.trans(gold_ex))
                    if match_score > ex_score:
                        ex_score = match_score
                        hit_idx = spo_idx
                if ex_score > threshold:
                    correct_num += 1
                    del gold_spo_list[hit_idx]
            total_correct_num += correct_num

        return total_correct_num, total_pred_num, total_gold_num

