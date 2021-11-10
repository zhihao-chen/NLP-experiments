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
from typing import List, Optional, Union

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

        self.RoPE = rope
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
        if self.RoPE:
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


class CRF(nn.Module):
    """Conditional random field.
    This module implements a conditional random field [LMP01]_. The forward computation
    of this class computes the log likelihood of the given sequence of tags and
    emission score tensor. This class also has `~CRF.decode` method which finds
    the best tag sequence given an emission score tensor using `Viterbi algorithm`_.
    Args:
        num_tags: Number of tags.
        batch_first: Whether the first dimension corresponds to the size of a minibatch.
    Attributes:
        start_transitions (`~torch.nn.Parameter`): Start transition score tensor of size
            ``(num_tags,)``.
        end_transitions (`~torch.nn.Parameter`): End transition score tensor of size
            ``(num_tags,)``.
        transitions (`~torch.nn.Parameter`): Transition score tensor of size
            ``(num_tags, num_tags)``.
    .. [LMP01] Lafferty, J., McCallum, A., Pereira, F. (2001).
       "Conditional random fields: Probabilistic models for segmenting and
       labeling sequence data". *Proc. 18th International Conf. on Machine
       Learning*. Morgan Kaufmann. pp. 282–289.
    .. _Viterbi algorithm: https://en.wikipedia.org/wiki/Viterbi_algorithm
    """

    def __init__(self, num_tags: int, batch_first: bool = False) -> None:
        if num_tags <= 0:
            raise ValueError(f'invalid number of tags: {num_tags}')
        super().__init__()
        self.num_tags = num_tags
        self.batch_first = batch_first
        self.start_transitions = nn.Parameter(torch.empty(num_tags))
        self.end_transitions = nn.Parameter(torch.empty(num_tags))
        self.transitions = nn.Parameter(torch.empty(num_tags, num_tags))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize the transition parameters.
        The parameters will be initialized randomly from a uniform distribution
        between -0.1 and 0.1.
        """
        nn.init.uniform_(self.start_transitions, -0.1, 0.1)
        nn.init.uniform_(self.end_transitions, -0.1, 0.1)
        nn.init.uniform_(self.transitions, -0.1, 0.1)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(num_tags={self.num_tags})'

    def forward(self, emissions: torch.Tensor,
                tags: torch.LongTensor,
                mask: Optional[torch.ByteTensor] = None,
                reduction: str = 'mean') -> torch.Tensor:
        """Compute the conditional log likelihood of a sequence of tags given emission scores.
        Args:
            emissions (`~torch.Tensor`): Emission score tensor of size
                ``(seq_length, batch_size, num_tags)`` if ``batch_first`` is ``False``,
                ``(batch_size, seq_length, num_tags)`` otherwise.
            tags (`~torch.LongTensor`): Sequence of tags tensor of size
                ``(seq_length, batch_size)`` if ``batch_first`` is ``False``,
                ``(batch_size, seq_length)`` otherwise.
            mask (`~torch.ByteTensor`): Mask tensor of size ``(seq_length, batch_size)``
                if ``batch_first`` is ``False``, ``(batch_size, seq_length)`` otherwise.
            reduction: Specifies  the reduction to apply to the output:
                ``none|sum|mean|token_mean``. ``none``: no reduction will be applied.
                ``sum``: the output will be summed over batches. ``mean``: the output will be
                averaged over batches. ``token_mean``: the output will be averaged over tokens.
        Returns:
            `~torch.Tensor`: The log likelihood. This will have size ``(batch_size,)`` if
            reduction is ``none``, ``()`` otherwise.
        """
        if reduction not in ('none', 'sum', 'mean', 'token_mean'):
            raise ValueError(f'invalid reduction: {reduction}')
        if mask is None:
            mask = torch.ones_like(tags, dtype=torch.uint8, device=tags.device)
        if mask.dtype != torch.uint8:
            mask = mask.byte()
        self._validate(emissions, tags=tags, mask=mask)

        if self.batch_first:
            emissions = emissions.transpose(0, 1)
            tags = tags.transpose(0, 1)
            mask = mask.transpose(0, 1)

        # shape: (batch_size,)
        numerator = self._compute_score(emissions, tags, mask)
        # shape: (batch_size,)
        denominator = self._compute_normalizer(emissions, mask)
        # shape: (batch_size,)
        llh = numerator - denominator

        if reduction == 'none':
            return llh
        if reduction == 'sum':
            return llh.sum()
        if reduction == 'mean':
            return llh.mean()
        return llh.sum() / mask.float().sum()

    def decode(self, emissions: torch.Tensor,
               mask: Optional[torch.ByteTensor] = None,
               nbest: Optional[int] = None,
               pad_tag: Optional[int] = None) -> List[List[List[int]]]:
        """Find the most likely tag sequence using Viterbi algorithm.
        Args:
            emissions (`~torch.Tensor`): Emission score tensor of size
                ``(seq_length, batch_size, num_tags)`` if ``batch_first`` is ``False``,
                ``(batch_size, seq_length, num_tags)`` otherwise.
            mask (`~torch.ByteTensor`): Mask tensor of size ``(seq_length, batch_size)``
                if ``batch_first`` is ``False``, ``(batch_size, seq_length)`` otherwise.
            nbest (`int`): Number of most probable paths for each sequence
            pad_tag (`int`): Tag at padded positions. Often input varies in length and
                the length will be padded to the maximum length in the batch. Tags at
                the padded positions will be assigned with a padding tag, i.e. `pad_tag`
        Returns:
            A PyTorch tensor of the best tag sequence for each batch of shape
            (nbest, batch_size, seq_length)
        """
        if nbest is None:
            nbest = 1
        if mask is None:
            mask = torch.ones(emissions.shape[:2], dtype=torch.uint8,
                              device=emissions.device)
        if mask.dtype != torch.uint8:
            mask = mask.byte()
        self._validate(emissions, mask=mask)

        if self.batch_first:
            emissions = emissions.transpose(0, 1)
            mask = mask.transpose(0, 1)

        if nbest == 1:
            return self._viterbi_decode(emissions, mask, pad_tag).unsqueeze(0)
        return self._viterbi_decode_nbest(emissions, mask, nbest, pad_tag)

    def _validate(self, emissions: torch.Tensor,
                  tags: Optional[torch.LongTensor] = None,
                  mask: Optional[torch.ByteTensor] = None) -> None:
        if emissions.dim() != 3:
            raise ValueError(f'emissions must have dimension of 3, got {emissions.dim()}')
        else:
            if emissions.size()[2] != self.num_tags:
                raise ValueError(
                    f'expected last dimension of emissions is {self.num_tags}, '
                    f'got {emissions.size()[2]}')

        if tags is not None:
            if emissions.shape[:2] != tags.shape:
                raise ValueError(
                    'the first two dimensions of emissions and tags must match, '
                    f'got {tuple(emissions.shape[:2])} and {tuple(tags.shape)}')

        if mask is not None:
            if emissions.shape[:2] != mask.shape:
                raise ValueError(
                    'the first two dimensions of emissions and mask must match, '
                    f'got {tuple(emissions.shape[:2])} and {tuple(mask.shape)}')
            no_empty_seq = not self.batch_first and mask[0].all()
            no_empty_seq_bf = self.batch_first and mask[:, 0].all()
            if not no_empty_seq and not no_empty_seq_bf:
                raise ValueError('mask of the first timestep must all be on')

    def _compute_score(self, emissions: torch.Tensor,
                       tags: torch.LongTensor,
                       mask: torch.ByteTensor) -> torch.Tensor:
        # emissions: (seq_length, batch_size, num_tags)
        # tags: (seq_length, batch_size)
        # mask: (seq_length, batch_size)
        seq_length, batch_size = tags.shape
        mask = mask.float()

        # Start transition score and first emission
        # shape: (batch_size,)
        score = self.start_transitions[tags[0]]
        score += emissions[0, torch.arange(batch_size), tags[0]]

        for i in range(1, seq_length):
            # Transition score to next tag, only added if next timestep is valid (mask == 1)
            # shape: (batch_size,)
            score += self.transitions[tags[i - 1], tags[i]] * mask[i]

            # Emission score for next tag, only added if next timestep is valid (mask == 1)
            # shape: (batch_size,)
            score += emissions[i, torch.arange(batch_size), tags[i]] * mask[i]

        # End transition score
        # shape: (batch_size,)
        seq_ends = mask.long().sum(dim=0) - 1
        # shape: (batch_size,)
        last_tags = tags[seq_ends, torch.arange(batch_size)]
        # shape: (batch_size,)
        score += self.end_transitions[last_tags]

        return score

    def _compute_normalizer(self, emissions: torch.Tensor,
                            mask: torch.ByteTensor) -> torch.Tensor:
        # emissions: (seq_length, batch_size, num_tags)
        # mask: (seq_length, batch_size)
        seq_length = emissions.size()[0]

        # Start transition score and first emission; score has size of
        # (batch_size, num_tags) where for each batch, the j-th column stores
        # the score that the first timestep has tag j
        # shape: (batch_size, num_tags)
        score = self.start_transitions + emissions[0]

        for i in range(1, seq_length):
            # Broadcast score for every possible next tag
            # shape: (batch_size, num_tags, 1)
            broadcast_score = score.unsqueeze(2)

            # Broadcast emission score for every possible current tag
            # shape: (batch_size, 1, num_tags)
            broadcast_emissions = emissions[i].unsqueeze(1)

            # Compute the score tensor of size (batch_size, num_tags, num_tags) where
            # for each sample, entry at row i and column j stores the sum of scores of all
            # possible tag sequences so far that end with transitioning from tag i to tag j
            # and emitting
            # shape: (batch_size, num_tags, num_tags)
            next_score = broadcast_score + self.transitions + broadcast_emissions

            # Sum over all possible current tags, but we're in score space, so a sum
            # becomes a log-sum-exp: for each sample, entry i stores the sum of scores of
            # all possible tag sequences so far, that end in tag i
            # shape: (batch_size, num_tags)
            next_score = torch.logsumexp(next_score, dim=1)

            # Set score to the next score if this timestep is valid (mask == 1)
            # shape: (batch_size, num_tags)
            score = torch.where(mask[i].unsqueeze(1), next_score, score)

        # End transition score
        # shape: (batch_size, num_tags)
        score += self.end_transitions

        # Sum (log-sum-exp) over all possible tags
        # shape: (batch_size,)
        return torch.logsumexp(score, dim=1)

    def _viterbi_decode(self, emissions: torch.FloatTensor,
                        mask: torch.ByteTensor,
                        pad_tag: Optional[int] = None) -> List[List[int]]:
        # emissions: (seq_length, batch_size, num_tags)
        # mask: (seq_length, batch_size)
        # return: (batch_size, seq_length)
        if pad_tag is None:
            pad_tag = 0

        device = emissions.device
        seq_length, batch_size = mask.shape

        # Start transition and first emission
        # shape: (batch_size, num_tags)
        score = self.start_transitions + emissions[0]
        history_idx = torch.zeros((seq_length, batch_size, self.num_tags),
                                  dtype=torch.long, device=device)
        oor_idx = torch.zeros((batch_size, self.num_tags),
                              dtype=torch.long, device=device)
        oor_tag = torch.full((seq_length, batch_size), pad_tag,
                             dtype=torch.long, device=device)

        # - score is a tensor of size (batch_size, num_tags) where for every batch,
        #   value at column j stores the score of the best tag sequence so far that ends
        #   with tag j
        # - history_idx saves where the best tags candidate transitioned from; this is used
        #   when we trace back the best tag sequence
        # - oor_idx saves the best tags candidate transitioned from at the positions
        #   where mask is 0, i.e. out of range (oor)

        # Viterbi algorithm recursive case: we compute the score of the best tag sequence
        # for every possible next tag
        for i in range(1, seq_length):
            # Broadcast viterbi score for every possible next tag
            # shape: (batch_size, num_tags, 1)
            broadcast_score = score.unsqueeze(2)

            # Broadcast emission score for every possible current tag
            # shape: (batch_size, 1, num_tags)
            broadcast_emission = emissions[i].unsqueeze(1)

            # Compute the score tensor of size (batch_size, num_tags, num_tags) where
            # for each sample, entry at row i and column j stores the score of the best
            # tag sequence so far that ends with transitioning from tag i to tag j and emitting
            # shape: (batch_size, num_tags, num_tags)
            next_score = broadcast_score + self.transitions + broadcast_emission

            # Find the maximum score over all possible current tag
            # shape: (batch_size, num_tags)
            next_score, indices = next_score.max(dim=1)

            # Set score to the next score if this timestep is valid (mask == 1)
            # and save the index that produces the next score
            # shape: (batch_size, num_tags)
            score = torch.where(mask[i].unsqueeze(-1), next_score, score)
            indices = torch.where(mask[i].unsqueeze(-1), indices, oor_idx)
            history_idx[i - 1] = indices

        # End transition score
        # shape: (batch_size, num_tags)
        end_score = score + self.end_transitions
        _, end_tag = end_score.max(dim=1)

        # shape: (batch_size,)
        seq_ends = mask.long().sum(dim=0) - 1

        # insert the best tag at each sequence end (last position with mask == 1)
        history_idx = history_idx.transpose(1, 0).contiguous()
        history_idx.scatter_(1, seq_ends.view(-1, 1, 1).expand(-1, 1, self.num_tags),
                             end_tag.view(-1, 1, 1).expand(-1, 1, self.num_tags))
        history_idx = history_idx.transpose(1, 0).contiguous()

        # The most probable path for each sequence
        best_tags_arr = torch.zeros((seq_length, batch_size),
                                    dtype=torch.long, device=device)
        best_tags = torch.zeros(batch_size, 1, dtype=torch.long, device=device)
        for idx in range(seq_length - 1, -1, -1):
            best_tags = torch.gather(history_idx[idx], 1, best_tags)
            best_tags_arr[idx] = best_tags.data.view(batch_size)

        return torch.where(mask, best_tags_arr, oor_tag).transpose(0, 1)

    def _viterbi_decode_nbest(self, emissions: torch.FloatTensor,
                              mask: torch.ByteTensor,
                              nbest: int,
                              pad_tag: Optional[int] = None) -> List[List[List[int]]]:
        # emissions: (seq_length, batch_size, num_tags)
        # mask: (seq_length, batch_size)
        # return: (nbest, batch_size, seq_length)
        if pad_tag is None:
            pad_tag = 0

        device = emissions.device
        seq_length, batch_size = mask.shape

        # Start transition and first emission
        # shape: (batch_size, num_tags)
        score = self.start_transitions + emissions[0]
        history_idx = torch.zeros((seq_length, batch_size, self.num_tags, nbest),
                                  dtype=torch.long, device=device)
        oor_idx = torch.zeros((batch_size, self.num_tags, nbest),
                              dtype=torch.long, device=device)
        oor_tag = torch.full((seq_length, batch_size, nbest), pad_tag,
                             dtype=torch.long, device=device)

        # + score is a tensor of size (batch_size, num_tags) where for every batch,
        #   value at column j stores the score of the best tag sequence so far that ends
        #   with tag j
        # + history_idx saves where the best tags candidate transitioned from; this is used
        #   when we trace back the best tag sequence
        # - oor_idx saves the best tags candidate transitioned from at the positions
        #   where mask is 0, i.e. out of range (oor)

        # Viterbi algorithm recursive case: we compute the score of the best tag sequence
        # for every possible next tag
        for i in range(1, seq_length):
            if i == 1:
                broadcast_score = score.unsqueeze(-1)
                broadcast_emission = emissions[i].unsqueeze(1)
                # shape: (batch_size, num_tags, num_tags)
                next_score = broadcast_score + self.transitions + broadcast_emission
            else:
                broadcast_score = score.unsqueeze(-1)
                broadcast_emission = emissions[i].unsqueeze(1).unsqueeze(2)
                # shape: (batch_size, num_tags, nbest, num_tags)
                next_score = broadcast_score + self.transitions.unsqueeze(1) + broadcast_emission

            # Find the top `nbest` maximum score over all possible current tag
            # shape: (batch_size, nbest, num_tags)
            next_score, indices = next_score.view(batch_size, -1, self.num_tags).topk(nbest, dim=1)

            if i == 1:
                score = score.unsqueeze(-1).expand(-1, -1, nbest)
                indices = indices * nbest

            # convert to shape: (batch_size, num_tags, nbest)
            next_score = next_score.transpose(2, 1)
            indices = indices.transpose(2, 1)

            # Set score to the next score if this timestep is valid (mask == 1)
            # and save the index that produces the next score
            # shape: (batch_size, num_tags, nbest)
            score = torch.where(mask[i].unsqueeze(-1).unsqueeze(-1), next_score, score)
            indices = torch.where(mask[i].unsqueeze(-1).unsqueeze(-1), indices, oor_idx)
            history_idx[i - 1] = indices

        # End transition score shape: (batch_size, num_tags, nbest)
        end_score = score + self.end_transitions.unsqueeze(-1)
        _, end_tag = end_score.view(batch_size, -1).topk(nbest, dim=1)

        # shape: (batch_size,)
        seq_ends = mask.long().sum(dim=0) - 1

        # insert the best tag at each sequence end (last position with mask == 1)
        history_idx = history_idx.transpose(1, 0).contiguous()
        history_idx.scatter_(1, seq_ends.view(-1, 1, 1, 1).expand(-1, 1, self.num_tags, nbest),
                             end_tag.view(-1, 1, 1, nbest).expand(-1, 1, self.num_tags, nbest))
        history_idx = history_idx.transpose(1, 0).contiguous()

        # The most probable path for each sequence
        best_tags_arr = torch.zeros((seq_length, batch_size, nbest),
                                    dtype=torch.long, device=device)
        best_tags = torch.arange(nbest, dtype=torch.long, device=device) \
                         .view(1, -1).expand(batch_size, -1)
        for idx in range(seq_length - 1, -1, -1):
            best_tags = torch.gather(history_idx[idx].view(batch_size, -1), 1, best_tags)
            best_tags_arr[idx] = best_tags.data.view(batch_size, -1) // nbest

        return torch.where(mask.unsqueeze(-1), best_tags_arr, oor_tag).permute(2, 1, 0)


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
    def __init__(self, hidden_size, num_classes):
        super(PoolerStartLogits, self).__init__()
        self.dense = nn.Linear(hidden_size, num_classes)

    def forward(self, hidden_states):
        x = self.dense(hidden_states)
        return x


class PoolerEndLogits(nn.Module):
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
