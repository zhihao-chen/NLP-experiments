# -*- coding: utf8 -*-
"""
======================================
    Project Name: NLP
    File Name: transformer
    Author: czh
    Create Date: 2021/5/28
--------------------------------------
    Change Activity: 
======================================
"""
# 实现一些transformer的方法，主要参考自苏神的代码

from copy import deepcopy

import torch
import torch.nn as nn


class Transformer(nn.Module):
    """
    只使用transformer的encoder
    """
    def __init__(self, d_model,  # 输入特征大小
                 num_head,  # attention的头数
                 num_hidden_layers,  # transformer总层数
                 dim_feedforward=2048,  # feedforward的隐层维度
                 dropout=0.1,  # encoder的dropout
                 activation='relu',  # feedforward的激活函数，relu或gelu
                 layer_norm_eps=1e-5,
                 batch_first=False  # If True, then the input and output tensors are provided as (batch, seq, feature).
                 ):
        super(Transformer, self).__init__()
        self.transformer_encoder = nn.TransformerEncoderLayer(d_model, num_head, dim_feedforward, dropout, activation)
        self.encoder = nn.ModuleList([deepcopy(self.transformer_encoder) for _ in range(num_hidden_layers)])


class AttentionLayer(nn.Module):
    """
    多头注意力机制
    """
    def __init__(self, nb_head, hidden_dim, dropout_rate=0.5):
        super(AttentionLayer, self).__init__()
        self.num_attention_heads = nb_head
        self.multi_head_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=nb_head, dropout=dropout_rate)

    def forward(self, q, k, v, mask=None):
        if mask:
            att_output, att_output_weights = self.multi_head_attn(q, k, v, attn_mask=mask)
        else:
            att_output, att_output_weights = self.multi_head_attn(q, k, v)
        return att_output


class SelfAttention(nn.Module):
    """多头注意力机制
    """
    def __init__(self, nb_head, size_per_head, input_size):
        super(SelfAttention, self).__init__()
        self.nb_head = nb_head
        self.size_per_head = size_per_head
        self.out_dim = nb_head * size_per_head
        q_in_dim = input_size
        k_in_dim = input_size
        v_in_dim = input_size
        self.q_linear = nn.Linear(q_in_dim, self.out_dim)
        self.k_linear = nn.Linear(k_in_dim, self.out_dim)
        self.v_linear = nn.Linear(v_in_dim, self.out_dim)

    @staticmethod
    def mask(x, mask, mode='mul'):
        if mask is None:
            return x
        else:
            for _ in range(x.dim() - mask.dim()):
                mask = torch.unsqueeze(mask, mask.dim())
            if mode == 'mul':
                return x * mask
            else:
                return x - (1 - mask) * 1e10

    def forward(self, inputs):
        q, k, v = inputs[:3]
        v_mask, q_mask = None, None
        if len(inputs) > 3:
            v_mask = inputs[3]
            if len(inputs) > 4:
                q_mask = inputs[4]
        # 线性变换
        qw1 = self.q_linear(q)
        kw1 = self.k_linear(k)
        vw1 = self.v_linear(v)
        # 形状变换
        qw2 = qw1.reshape(-1, qw1.shape[1], self.nb_head, self.size_per_head)
        kw2 = kw1.reshape(-1, kw1.shape[1], self.nb_head, self.size_per_head)
        vw2 = vw1.reshape(-1, vw1.shape[1], self.nb_head, self.size_per_head)
        # 维度置换
        qw = qw2.permute(0, 2, 1, 3)
        kw = kw2.permute(0, 2, 1, 3)
        vw = vw2.permute(0, 2, 1, 3)
        # Attention
        a1 = torch.matmul(qw, kw.permute(0, 1, 3, 2)) / self.size_per_head ** 0.5
        a2 = a1.permute(0, 3, 2, 1)
        a3 = self.mask(a2, v_mask, 'add')
        a4 = a3.permute(0, 3, 2, 1)
        a = torch.softmax(a4, dim=-1)
        # 完成输出
        o1 = torch.matmul(a, vw)
        o2 = o1.permute(0, 2, 1, 3)
        o3 = o2.reshape(-1, o2.shape[1], self.out_dim)
        o = self.mask(o3, q_mask, 'mul')
        return o


