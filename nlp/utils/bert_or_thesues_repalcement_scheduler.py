# -*- coding: utf8 -*-
"""
======================================
    Project Name: NLP
    File Name: bert_or_thesues_repalcement_scheduler
    Author: czh
    Create Date: 2021/8/12
--------------------------------------
    Change Activity: 
======================================
"""
# bert_of_thesues 的replacement scheduler
from nlp.models.distill_model import BertThesuesEncoder


class ConstantReplacementScheduler:
    def __init__(self, bert_encoder: BertThesuesEncoder, replacing_rate, replacing_steps=None):
        self.bert_encoder = bert_encoder
        self.replacing_rate = replacing_rate
        self.replacing_steps = replacing_steps
        self.step_counter = 0
        self.bert_encoder.set_replacing_rate(replacing_rate)

    def step(self):
        self.step_counter += 1
        if self.replacing_steps is None or self.replacing_rate == 1.0:
            return self.replacing_rate
        else:
            if self.step_counter >= self.replacing_steps:
                self.bert_encoder.set_replacing_rate(1.0)
                self.replacing_rate = 1.0
            return self.replacing_rate


class LinearReplacementScheduler:
    def __init__(self, bert_encoder: BertThesuesEncoder, base_replacing_rate, k):
        self.bert_encoder = bert_encoder
        self.base_replacing_rate = base_replacing_rate
        self.step_counter = 0
        self.k = k
        self.bert_encoder.set_replacing_rate(base_replacing_rate)

    def step(self):
        self.step_counter += 1
        current_replacing_rate = min(self.k * self.step_counter + self.base_replacing_rate, 1.0)
        self.bert_encoder.set_replacing_rate(current_replacing_rate)
        return current_replacing_rate
