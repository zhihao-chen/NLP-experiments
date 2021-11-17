# -*- coding: utf8 -*-
"""
======================================
    Project Name: NLP
    File Name: layers
    Author: czh
    Create Date: 2021/6/2
--------------------------------------
    Change Activity: 
======================================
"""
import math
from typing import Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as func


class Lambda(nn.Module):
    def __init__(self, func_):
        super(Lambda, self).__init__()
        self.func = func_

    @classmethod
    def forward(cls, x):
        return cls.func(x)


class PositionEmbedding(nn.Module):
    """定义可训练的位置Embedding
    """
    def __init__(
        self,
        input_dim,
        output_dim,
        merge_mode='add',
        hierarchical: Union[float, bool] = None,
        embeddings_initializer='zeros',
        custom_position_ids=False,
        a=0.0,  # uniform_ 的a
        b=1.0,  # uniform_的b
        mean=0.0,  # norm_的mean
        std=1.0,  # norm_的std
        gain=1.0  # xavier_uniform_ or xavier_normal_的gain
    ):
        super(PositionEmbedding, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.merge_mode = merge_mode
        self.hierarchical = hierarchical
        self.embeddings_initializer = embeddings_initializer
        self.custom_position_ids = custom_position_ids
        self.a = a
        self.b = b
        self.mean = mean
        self.std = std
        self.gain = gain

        self.embedding = nn.Embedding(input_dim, output_dim)

        self.init_embedding()

    def init_embedding(self):
        if self.embeddings_initializer == 'zeros':
            nn.init.zeros_(self.embedding.weight)
        elif self.embeddings_initializer == 'ones':
            nn.init.ones_(self.embedding.weight)
        elif self.embeddings_initializer == 'uniform':
            nn.init.uniform_(self.embedding.weight, a=self.a, b=self.b)
        elif self.embeddings_initializer == 'normal':
            nn.init.normal_(self.embedding.weight, mean=self.mean, std=self.std)
        elif self.embeddings_initializer == 'xavier_uniform':
            nn.init.xavier_uniform_(self.embedding.weight, gain=self.gain)
        elif self.embeddings_initializer == 'xavier_normal':
            nn.init.xavier_normal_(self.embedding.weight, gain=self.gain)
        else:
            nn.init.zeros_(self.embedding.weight)

    def forward(self, inputs):
        """
        如果custom_position_ids，那么第二个输入为自定义的位置id
        """
        input_shape = inputs.size()
        batch_size, seq_len = input_shape[0], input_shape[1]
        if self.custom_position_ids:
            inputs, position_ids = inputs
            if 'int' not in str(position_ids.dtype):
                position_ids = position_ids.to(torch.int32)
        else:
            position_ids = torch.arange(0, seq_len, dtype=torch.int32)[None]

        embedding = self.embedding(position_ids)
        print(embedding.size())
        # 位置编码的层次分解  https://kexue.fm/archives/7947
        if self.hierarchical:
            alpha = 0.4 if self.hierarchical is True else self.hierarchical
            embeddings = embedding - alpha * embedding[:1]
            embeddings = embeddings / (1 - alpha)
            index_a = (position_ids // self.input_dim).to(torch.long)
            index_b = (position_ids % self.input_dim).to(torch.long)
            embeddings_x = embeddings[:, index_a[0], :]
            embeddings_y = embeddings[:, index_b[0], :]
            embeddings = alpha * embeddings_x + (1 - alpha) * embeddings_y
        else:
            if self.custom_position_ids:
                embeddings = embedding[position_ids, :]
            else:
                embeddings = embedding[:, :seq_len]

        if self.merge_mode == 'add':
            return inputs + embeddings
        elif self.merge_mode == 'mul':
            return inputs * (embeddings + 1.0)
        elif self.merge_mode == 'zero':
            return embeddings
        else:
            if not self.custom_position_ids:
                embeddings = torch.tile(embeddings, [batch_size, 1, 1])
            return torch.cat([inputs, embeddings])

    def compute_output_shape(self, input_shape):
        if self.custom_position_ids:
            input_shape = input_shape[0]

        if self.merge_mode in ['add', 'mul', 'zero']:
            return input_shape[:2] + (self.output_dim,)
        else:
            return input_shape[:2] + (input_shape[2] + self.output_dim,)

    def get_config(self):
        config = {
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'merge_mode': self.merge_mode,
            'hierarchical': self.hierarchical,
            'custom_position_ids': self.custom_position_ids,
            'a': self.a,
            'b': self.b,
            'mean': self.mean,
            'std': self.std,
            'gain': self.gain
        }
        return config


class LMMask(object):
    """
    定义下三角Attention Mask（语言模型用）
    """
    @staticmethod
    def lm_mask(seq_len, pad_len):
        """
        通过idxs序列的比较来得到对应的mask
        :param seq_len: 序列长度，包括[cls],....,[sep]
        :param pad_len: pad的长度，pad部分不参与计算
        """
        idxs = torch.arange(0, seq_len+pad_len)
        mask = idxs[None, :] <= idxs[:, None]
        mask = mask.to(torch.float)
        if pad_len > 0:
            mask[-pad_len:, :] = 0.0
        return -(1 - mask[None, None]) * 1e12


class UniLMMask(object):
    """
    定义UniLM的Attention Mask（Seq2Seq模型用）
    其中source和target的分区，由segment_ids来表示。
    UniLM: https://arxiv.org/abs/1905.03197
    """
    @staticmethod
    def unilm_mask(segment_ids, pad_len=0):
        """
        通过idxs序列的比较来得到对应的mask
        :param segment_ids: [CLS] 你 想 吃 啥 [SEP] 白 切 鸡 [SEP][pad][pad]。[0,0,0,0，0，0，1，1,1,1,1,1],0表示句子source,
        1表示属于句子target。source中包括[cls]和[sep]标记，target中包括[sep]标记
        :param pad_len:pad的长度，pad部分不参与计算
        """
        # segment_ids = [batch_size, seq_len]
        idxs = torch.cumsum(segment_ids, dim=1)
        mask = idxs[:, None, :] <= idxs[:, :, None]
        mask = mask.to(torch.float)
        if pad_len > 0:
            mask[:, -pad_len:, :] = 0
        return -(1 - mask[:, None]) * 1e12


class SinusoidalPositionEmbedding(nn.Module):
    """定义Sin-Cos位置Embedding
    """
    def __init__(self, output_dim, merge_mode='add', custom_position_ids=False):
        super(SinusoidalPositionEmbedding, self).__init__()
        self.output_dim = output_dim
        self.merge_mode = merge_mode
        self.custom_position_ids = custom_position_ids

    def forward(self, inputs):
        """
        如果custom_position_ids，那么第二个输入为自定义的位置id
        :param inputs:
        :return:
        """
        # print(inputs.size())
        batch_size, seq_len = inputs.size()[:2]
        if self.custom_position_ids:
            inputs, position_ids = inputs
            if "float" not in position_ids.dtype:
                position_ids = position_ids.to(torch.float)
        else:
            position_ids = torch.arange(0, seq_len, dtype=torch.float)[None]

        indices = torch.arange(0, self.output_dim//2, dtype=torch.float)
        indices = torch.pow(10000.0, -2 * indices / self.output_dim)
        # print(position_ids.size(), indices.size())
        embeddings = torch.einsum('bn,d->bnd', position_ids, indices)
        embeddings = torch.stack([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        embeddings = embeddings.view(-1, seq_len, self.output_dim)

        if self.merge_mode == 'add':
            return inputs + embeddings
        elif self.merge_mode == 'mul':
            return inputs * (embeddings + 1.0)
        elif self.merge_mode == 'zero':
            return embeddings
        else:
            if not self.custom_position_ids:
                embeddings = torch.tile(embeddings, [batch_size, 1, 1])
            return torch.cat([inputs, embeddings])


class SinCosPositionEmbedding(nn.Module):
    """
    sin-cos 位置编码
    """
    def __init__(self, output_dim, max_seq_len=512, p=0.5):
        super(SinCosPositionEmbedding, self).__init__()
        self.output_dim = output_dim
        self.max_seq_len = max_seq_len
        self.dropout = nn.Dropout(p=p)

    def forward(self, inputs):
        bs, seq_len = inputs.size()[:2]
        pe = torch.zeros(seq_len, self.output_dim)
        position = torch.arange(0, seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0.0, self.output_dim, 2) *
                             torch.tensor(-(math.log(10000.0) / self.output_dim)))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        return self.dropout(pe)


class NeZHARelativePositionsEncoding(nn.Module):
    def __init__(self, length, depth, max_relative_position=127):
        super(NeZHARelativePositionsEncoding, self).__init__()
        vocab_size = max_relative_position * 2 + 1
        range_vec = torch.arange(length)
        range_mat = range_vec.repeat(length).view(length, length)
        distance_mat = range_mat - torch.t(range_mat)
        distance_mat_clipped = torch.clamp(distance_mat, -max_relative_position, max_relative_position)
        final_mat = distance_mat_clipped + max_relative_position

        embeddings_table = torch.zeros(vocab_size, depth)
        position = torch.arange(0, vocab_size, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, depth, 2).float() * (-math.log(10000.0) / depth))
        embeddings_table[:, 0::2] = torch.sin(position * div_term)
        embeddings_table[:, 1::2] = torch.cos(position * div_term)
        embeddings_table = embeddings_table.unsqueeze(0).transpose(0, 1).squeeze(1)

        flat_relative_positions_matrix = final_mat.view(-1)
        one_hot_relative_positions_matrix = torch.nn.functional.one_hot(flat_relative_positions_matrix,
                                                                        num_classes=vocab_size).float()
        positions_encoding = torch.matmul(one_hot_relative_positions_matrix, embeddings_table)
        my_shape = list(final_mat.size())
        my_shape.append(depth)
        positions_encoding = positions_encoding.view(my_shape)
        self.register_buffer('positions_encoding', positions_encoding)

    def forward(self, length):
        return self.positions_encoding[:length, :length, :]


class RelativePositionEmbedding(nn.Module):
    """
    相对位置编码
    来自论文：https://arxiv.org/abs/1803.02155
    """
    def __init__(self, input_dim, output_dim, embeddings_initializer='zeros', **kwargs):
        super(RelativePositionEmbedding, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.embeddings_initializer = embeddings_initializer

        self.embedding = nn.Embedding(self.input_dim, self.output_dim)

        self.init_embedding()

    def init_embedding(self):
        if self.embeddings_initializer == 'zeros':
            nn.init.zeros_(self.embedding.weight)
        elif self.embeddings_initializer == 'ones':
            nn.init.ones_(self.embedding.weight)
        elif self.embeddings_initializer == 'uniform':
            nn.init.uniform_(self.embedding.weight, a=self.a, b=self.b)
        elif self.embeddings_initializer == 'normal':
            nn.init.normal_(self.embedding.weight, mean=self.mean, std=self.std)
        elif self.embeddings_initializer == 'xavier_uniform':
            nn.init.xavier_uniform_(self.embedding.weight, gain=self.gain)
        elif self.embeddings_initializer == 'xavier_normal':
            nn.init.xavier_normal_(self.embedding.weight, gain=self.gain)
        else:
            nn.init.zeros_(self.embedding.weight)

    def compute_position_ids(self, inputs):
        q = inputs
        v = inputs
        # 计算位置差
        q_idxs = torch.arange(0, q.size(1), dtype=torch.int32)
        q_idxs = torch.unsqueeze(q_idxs, dim=1)
        v_idxs = torch.arange(0, v.size(1), dtype=torch.int32)
        v_idxs = torch.unsqueeze(v_idxs, dim=0)
        pos_ids = v_idxs - q_idxs
        # 后处理操作
        max_position = (self.input_dim - 1) // 2
        pos_ids = torch.clamp(pos_ids, -max_position, max_position)
        pos_ids = pos_ids + max_position
        return pos_ids

    def forward(self, inputs):
        # batch_size, seq_len = inputs.size()
        pos_ids = self.compute_position_ids(inputs)
        embeddings = self.embedding(pos_ids)
        # embeddings = embeddings.view(batch_size, seq_len, self.output_dim)
        return embeddings


class RelativePositionEmbeddingT5(RelativePositionEmbedding):
    """Google T5的相对位置编码
    来自论文：https://arxiv.org/abs/1910.10683
    """
    def __init__(
        self,
        input_dim,
        output_dim,
        max_distance=128,
        bidirectional=True,
        embeddings_initializer='zeros',
        **kwargs
    ):
        super(RelativePositionEmbeddingT5, self).__init__(input_dim, output_dim, embeddings_initializer, **kwargs)
        self.max_distance = max_distance
        self.bidirectional = bidirectional

    def compute_position_ids(self, inputs):
        """
        T5的相对位置分桶（直接翻译自官方T5源码）
        """
        q = inputs
        v = inputs
        # 计算位置差
        q_idxs = torch.arange(0, q.size(1), dtype=torch.int32)
        q_idxs = torch.unsqueeze(q_idxs, dim=1)
        v_idxs = torch.arange(0, v.size(1), dtype=torch.int32)
        v_idxs = torch.unsqueeze(v_idxs, dim=0)
        pos_ids = v_idxs - q_idxs

        # 后处理操作
        num_buckets, max_distance = self.input_dim, self.max_distance
        ret = 0
        n = -pos_ids
        if self.bidirectional:
            num_buckets //= 2
            ret += torch.less(n, 0).to(torch.int32) * num_buckets
            n = torch.abs(n)
        else:
            n = torch.maximum(n, torch.zeros(1))
        max_exact = num_buckets // 2
        is_small = torch.less(n, max_exact)
        tmp = torch.log(n.float() / max_exact) / np.log(max_distance / max_exact) * (num_buckets - max_exact)
        val_if_large = max_exact + tmp.to(torch.int32)
        val_if_large = torch.minimum(val_if_large, torch.tensor([num_buckets - 1]))
        ret += torch.where(is_small, n.int(), val_if_large.int())
        return ret


# Copied from transformers.models.marian.modeling_marian.MarianSinusoidalPositionalEmbedding with Marian->RoFormer
class RoFormerSinusoidalPositionalEmbedding(nn.Embedding):
    """This module produces sinusoidal positional embeddings of any length."""

    def __init__(self, num_positions: int, embedding_dim: int):
        super().__init__(num_positions, embedding_dim)
        self.weight = self._init_weight(self.weight)  # noqa

    @staticmethod
    def _init_weight(out: nn.Parameter) -> nn.Parameter:
        """
        Identical to the XLM create_sinusoidal_embeddings except features are not interleaved. The cos features are in
        the 2nd half of the vector. [dim // 2:]
        """
        n_pos, dim = out.shape
        position_enc = np.array(
            [
                [pos / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)]
                for pos in range(n_pos)
            ]
        )
        out.requires_grad = False  # set early to avoid an error in pytorch-1.8+
        sentinel = dim // 2 if dim % 2 == 0 else (dim // 2) + 1
        out[:, 0:sentinel] = torch.FloatTensor(np.sin(position_enc[:, 0::2]))
        out[:, sentinel:] = torch.FloatTensor(np.cos(position_enc[:, 1::2]))
        out.detach_()
        return out

    @torch.no_grad()
    def forward(self, input_ids, past_key_values_length: int = 0):
        """`input_ids_shape` is expected to be [bsz x seqlen]."""
        bsz, seq_len = input_ids.size()[:2]
        positions = torch.arange(
            past_key_values_length,
            past_key_values_length + seq_len,
            dtype=torch.long,
            device=self.weight.device,
        )
        return super().forward(positions)
