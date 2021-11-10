# -*- coding: utf8 -*-
"""
======================================
    Project Name: NLP
    File Name: bert_for_ee
    Author: czh
    Create Date: 2021/9/8
--------------------------------------
    Change Activity: 
======================================
"""
import torch.nn as nn

from transformers import BertModel, BertPreTrainedModel, BertConfig, BertTokenizer

from nlp.models.nezha import NeZhaModel, NeZhaConfig
from nlp.models.layers import CRF


# 参考苏剑林的方法，https://github.com/bojone/lic2020_baselines/blob/master/ee.py
class BertCRFForDuEE1Su(BertPreTrainedModel):
    def __init__(self, config, train_config):
        super(BertCRFForDuEE1Su, self).__init__(config)
        if train_config.model_type == "bert":
            self.bert = BertModel(config)
        elif train_config.model_type == "nezha":
            self.bert = NeZhaModel(config)
        else:
            raise ValueError("'model_type' must be 'bert' or 'nezha'")

        self.use_lstm = train_config.use_lstm
        self.dropout = nn.Dropout(train_config.dropout_rate)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.crf = CRF(num_tags=config.num_labels, batch_first=True)
        if self.use_lstm:
            self.lstm = nn.LSTM(input_size=config.hidden_size, hidden_size=config.hidden_size // 2,
                                num_layers=1, bidirectional=True, batch_first=True)
        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        sequence_output = outputs[0]
        if self.use_lstm:
            sequence_output, _ = self.lstm(sequence_output)
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        outputs = (logits,)
        if labels is not None:
            loss = -1 * self.crf(emissions=logits, tags=labels, mask=attention_mask)
            outputs = (loss,)+outputs
        return outputs  # (loss), scores


MODEL_TYPE_CLASSES = {
    "bert": (BertConfig, BertTokenizer, BertCRFForDuEE1Su),
    "nezha": (NeZhaConfig, BertTokenizer, BertCRFForDuEE1Su)
}
