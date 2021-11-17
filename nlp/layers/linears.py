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
import torch
import torch.nn as nn
import torch.nn.functional as func


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

