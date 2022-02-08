# -*- coding: utf8 -*-
"""
======================================
    Project Name: NLP
    File Name: global_pointer
    Author: czh
    Create Date: 2021/11/15
--------------------------------------
    Change Activity: 
======================================
"""
import torch
import torch.nn as nn

from nlp.layers.position_embeddings import SinusoidalPositionEmbedding


class GlobalPointer(nn.Module):
    """
    全局指针模块
    将序列的每个(start, end)作为整体来进行判断
    https://kexue.fm/archives/8373
    """
    def __init__(self, encoder, ent_type_size, inner_dim, rope=True):
        # encodr: RoBerta-Large as encoder
        # inner_dim: 64
        # ent_type_size: ent_cls_num
        super().__init__()
        self.encoder = encoder
        self.ent_type_size = ent_type_size
        self.inner_dim = inner_dim
        self.hidden_size = encoder.config.hidden_size
        self.dense = nn.Linear(self.hidden_size, self.ent_type_size * self.inner_dim * 2)

        self.rope = rope
        self.device = torch.device('cpu')

    def sinusoidal_position_embedding(self, batch_size, seq_len, output_dim):
        position_ids = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(-1)

        indices = torch.arange(0, output_dim // 2, dtype=torch.float)
        indices = torch.pow(10000, -2 * indices / output_dim)
        embeddings = position_ids * indices
        embeddings = torch.stack([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        embeddings = embeddings.repeat((batch_size, *([1] * len(embeddings.shape))))
        embeddings = torch.reshape(embeddings, (batch_size, seq_len, output_dim))
        embeddings = embeddings.to(self.device)
        return embeddings

    def forward(self, input_ids, attention_mask, token_type_ids):
        self.device = input_ids.device

        context_outputs = self.encoder(input_ids, attention_mask, token_type_ids)
        # last_hidden_state:(batch_size, seq_len, hidden_size)
        last_hidden_state = context_outputs[0]

        batch_size = last_hidden_state.size()[0]
        seq_len = last_hidden_state.size()[1]

        # outputs:(batch_size, seq_len, ent_type_size*inner_dim*2)
        outputs = self.dense(last_hidden_state)
        outputs = torch.split(outputs, self.inner_dim * 2, dim=-1)
        # outputs:(batch_size, seq_len, ent_type_size, inner_dim*2)
        outputs = torch.stack(outputs, dim=-2)
        # qw,kw:(batch_size, seq_len, ent_type_size, inner_dim)
        qw, kw = outputs[..., :self.inner_dim], outputs[..., self.inner_dim:]
        if self.rope:
            # pos_emb:(batch_size, seq_len, inner_dim)
            pos_emb = self.sinusoidal_position_embedding(batch_size, seq_len, self.inner_dim)
            # cos_pos,sin_pos: (batch_size, seq_len, 1, inner_dim)
            cos_pos = pos_emb[..., None, 1::2].repeat_interleave(2, dim=-1)
            sin_pos = pos_emb[..., None, ::2].repeat_interleave(2, dim=-1)
            qw2 = torch.stack([-qw[..., 1::2], qw[..., ::2]], -1)
            qw2 = qw2.reshape(qw.shape)
            qw = qw * cos_pos + qw2 * sin_pos
            kw2 = torch.stack([-kw[..., 1::2], kw[..., ::2]], -1)
            kw2 = kw2.reshape(kw.shape)
            kw = kw * cos_pos + kw2 * sin_pos
        # logits:(batch_size, ent_type_size, seq_len, seq_len)
        logits = torch.einsum('bmhd,bnhd->bhmn', qw, kw)

        # padding mask
        pad_mask = attention_mask.unsqueeze(1).unsqueeze(1).expand(batch_size, self.ent_type_size, seq_len, seq_len)
        logits = logits * pad_mask - (1 - pad_mask) * 1e12

        # 排除下三角
        mask = torch.tril(torch.ones_like(logits), -1)
        logits = logits - mask * 1e12

        return logits / self.inner_dim ** 0.5


class EfficientGlobalPointer(GlobalPointer):
    def __init__(self, encoder, ent_type_size, head_size=64, rope=True, tril_mask=True):
        """
        改进后的global-pointer，https://kexue.fm/archives/8877
        :param encoder: 预训练的bert模型
        :param ent_type_size:
        :param head_size:
        :param rope:
        :param tril_mask: 是否排除下三角
        """
        super(EfficientGlobalPointer, self).__init__(encoder, ent_type_size, head_size, rope)
        self.head_size = head_size
        self.tril_mask = tril_mask

        self.dense_1 = nn.Linear(self.hidden_size, self.head_size*2, bias=True)
        self.dense_2 = nn.Linear(self.head_size*2, self.ent_type_size*2, bias=True)

    def forward(self, input_ids, attention_mask, token_type_ids):
        self.device = input_ids.device

        context_outputs = self.encoder(input_ids, attention_mask, token_type_ids)
        # last_hidden_state:[batch_size, seq_len, hidden_size]
        last_hidden_state = context_outputs.last_hidden_state

        batch_size = last_hidden_state.size()[0]
        seq_len = last_hidden_state.size()[1]

        # outputs:(batch_size, seq_len, head_size*2)
        outputs = self.dense_1(last_hidden_state)

        qw, kw = outputs[..., ::2], outputs[..., 1::2]

        if self.rope:
            pos = SinusoidalPositionEmbedding(self.head_size, 'zero')(outputs).to(self.device)
            cos_pos = pos[..., 1::2].repeat(1, 1, 2)
            sin_pos = pos[..., ::2].repeat(1, 1, 2)
            qw2 = torch.stack([-qw[..., 1::2], qw[..., ::2]], -1)
            qw2 = qw2.reshape(qw.shape)
            qw = qw * cos_pos + qw2 * sin_pos
            kw2 = torch.stack([-kw[..., 1::2], kw[..., ::2]], -1)
            kw2 = kw2.reshape(kw.shape)
            kw = kw * cos_pos + kw2 * sin_pos
        # 计算內积
        logits = torch.einsum('bmd, bnd -> bmn', qw, kw) / self.head_size ** 0.5
        bias = torch.einsum('bnh -> bhn', self.dense_2(outputs)) / 2
        logits = logits[:, None] + bias[:, ::2, None] + bias[:, 1::2, :, None]

        # padding mask
        pad_mask = attention_mask.unsqueeze(1).unsqueeze(1).expand(batch_size, self.ent_type_size, seq_len, seq_len)
        logits = logits * pad_mask - (1 - pad_mask) * 1e12

        # 排除下三角
        if self.tril_mask:
            mask = torch.tril(torch.ones_like(logits), -1)
            logits = logits - mask * 1e12

        return logits
