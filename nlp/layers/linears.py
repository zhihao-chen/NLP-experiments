# -*- coding: utf8 -*-
"""
======================================
    Project Name: NLP
    File Name: linears
    Author: czh
    Create Date: 2021/11/15
--------------------------------------
    Change Activity: 
======================================
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as func
from torch.nn.parameter import Parameter


class Linears(nn.Module):
    def __init__(self, input_dim: int, output_dim: int = 1, bias: bool = True):
        super().__init__()
        self.fn1 = nn.Linear(input_dim, input_dim)
        self.fn2 = nn.Linear(input_dim, input_dim)
        self.fn3 = nn.Linear(input_dim, output_dim, bias=bias)

        nn.init.orthogonal_(self.fn1.weight, gain=1)
        nn.init.orthogonal_(self.fn2.weight, gain=1)
        nn.init.orthogonal_(self.fn3.weight, gain=1)

    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor):
        logits = self.fn3(torch.tanh(
            self.fn1(hidden_states).unsqueeze(2) + self.fn2(encoder_hidden_states).unsqueeze(1)
        )).squeeze()
        return logits


class EntityLinears(nn.Module):
    def __init__(self, input_dim: int, output_dim: int = 1, bias: bool = True):
        super().__init__()
        self.head = Linears(input_dim=input_dim, output_dim=output_dim, bias=bias)
        self.tail = Linears(input_dim=input_dim, output_dim=output_dim, bias=bias)

    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor):
        # [bsz, num_triples, seq_len, output_dim]
        return self.head(hidden_states, encoder_hidden_states), self.tail(hidden_states, encoder_hidden_states)


class FeedForwardNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_rate=0):
        super(FeedForwardNetwork, self).__init__()
        self.dropout_rate = dropout_rate
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x_proj = func.dropout(func.relu(self.linear1(x)), p=self.dropout_rate, training=self.training)
        x_proj = self.linear2(x_proj)
        return x_proj


class PoolerStartLogits(nn.Module):
    """
    bert_ner_span
    """
    def __init__(self, hidden_size, num_classes):
        super(PoolerStartLogits, self).__init__()
        self.dense = nn.Linear(hidden_size, num_classes)

    def forward(self, hidden_states):
        x = self.dense(hidden_states)
        return x


class PoolerEndLogits(nn.Module):
    """
    bert_ner_span
    """
    def __init__(self, hidden_size, num_classes):
        super(PoolerEndLogits, self).__init__()
        self.dense_0 = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()
        self.LayerNorm = nn.LayerNorm(hidden_size)
        self.dense_1 = nn.Linear(hidden_size, num_classes)

    def forward(self, hidden_states, start_positions=None):
        x = self.dense_0(torch.cat([hidden_states, start_positions], dim=-1))
        x = self.activation(x)
        x = self.LayerNorm(x)
        x = self.dense_1(x)
        return x


class MultiNonLinearClassifier(nn.Module):
    def __init__(self, hidden_size, num_label, dropout_rate, act_func="gelu", intermediate_hidden_size=None):
        super(MultiNonLinearClassifier, self).__init__()
        self.num_label = num_label
        self.intermediate_hidden_size = hidden_size if intermediate_hidden_size is None else intermediate_hidden_size
        self.classifier1 = nn.Linear(hidden_size, self.intermediate_hidden_size)
        self.classifier2 = nn.Linear(self.intermediate_hidden_size, self.num_label)
        self.dropout = nn.Dropout(dropout_rate)
        self.act_func = act_func

    def forward(self, input_features):
        features_output1 = self.classifier1(input_features)
        if self.act_func == "gelu":
            features_output1 = func.gelu(features_output1)
        elif self.act_func == "relu":
            features_output1 = func.relu(features_output1)
        elif self.act_func == "tanh":
            features_output1 = func.tanh(features_output1)
        else:
            raise ValueError
        features_output1 = self.dropout(features_output1)
        features_output2 = self.classifier2(features_output1)
        return features_output2


class SingleLinearClassifier(nn.Module):
    def __init__(self, hidden_size, num_label):
        super(SingleLinearClassifier, self).__init__()
        self.num_label = num_label
        self.classifier = nn.Linear(hidden_size, num_label)

    def forward(self, input_features):
        features_output = self.classifier(input_features)
        return features_output


class BERTTaggerClassifier(nn.Module):
    def __init__(self, hidden_size, num_label, dropout_rate, act_func="gelu", intermediate_hidden_size=None):
        super(BERTTaggerClassifier, self).__init__()
        self.num_label = num_label
        self.intermediate_hidden_size = hidden_size if intermediate_hidden_size is None else intermediate_hidden_size
        self.classifier1 = nn.Linear(hidden_size, self.intermediate_hidden_size)
        self.classifier2 = nn.Linear(self.intermediate_hidden_size, self.num_label)
        self.dropout = nn.Dropout(dropout_rate)
        self.act_func = act_func

    def forward(self, input_features):
        features_output1 = self.classifier1(input_features)
        if self.act_func == "gelu":
            features_output1 = func.gelu(features_output1)
        elif self.act_func == "relu":
            features_output1 = func.relu(features_output1)
        elif self.act_func == "tanh":
            features_output1 = func.tanh(features_output1)
        else:
            raise ValueError
        features_output1 = self.dropout(features_output1)
        features_output2 = self.classifier2(features_output1)
        return features_output2


class ClassifierLayer(nn.Module):
    # https://github.com/Akeepers/LEAR/blob/master/utils/model_utils.py
    def __init__(self, class_num, out_features, bias=True):
        super(ClassifierLayer, self).__init__()
        self.class_num = class_num
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(class_num, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(class_num))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, inputs):
        x = torch.mul(inputs, self.weight)
        # (class_num, 1)
        x = torch.sum(x, -1)  # [-1, class_num]
        if self.bias is not None:
            x = x + self.bias
        return x

    def extra_repr(self):
        return 'class_num={}, out_features={}, bias={}'.format(
            self.class_num, self.out_features, self.bias is not None)


class MultiNonLinearClassifierForMultiLabel(nn.Module):
    # https://github.com/Akeepers/LEAR/blob/master/utils/model_utils.py
    def __init__(self, hidden_size, num_label, dropout_rate):
        super(MultiNonLinearClassifierForMultiLabel, self).__init__()
        self.num_label = num_label
        self.classifier1 = nn.Linear(hidden_size, hidden_size)
        self.classifier2 = ClassifierLayer(num_label, hidden_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, input_features):
        features_output1 = self.classifier1(input_features)
        features_output1 = func.gelu(features_output1)
        features_output1 = self.dropout(features_output1)
        features_output2 = self.classifier2(features_output1)
        return features_output2
