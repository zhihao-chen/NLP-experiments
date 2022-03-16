# -*- coding: utf8 -*-
"""
======================================
    Project Name: NLP
    File Name: utils
    Author: czh
    Create Date: 2021/8/6
--------------------------------------
    Change Activity: 
======================================
"""
import torch
import torch.nn as nn
import torch.nn.functional as func
from torch.nn.parameter import Parameter
import numpy as np


class LayerNorm(nn.Module):
    def __init__(self, input_dim, cond_dim=0, center=True, scale=True, epsilon=None, conditional=False,
                 hidden_units=None, hidden_initializer='xaiver'):
        """
        :param input_dim: inputs.shape[-1]
        :param cond_dim: cond.shape[-1]
        :param center:
        :param scale:
        :param epsilon:
        :param conditional: 如果为True，则是条件LayerNorm
        :param hidden_units:
        :param hidden_initializer:
        """
        super(LayerNorm, self).__init__()
        self.center = center
        self.scale = scale
        self.conditional = conditional
        self.hidden_units = hidden_units
        self.hidden_initializer = hidden_initializer
        self.epsilon = epsilon or 1e-12
        self.input_dim = input_dim
        self.cond_dim = cond_dim

        if self.center:
            self.beta = Parameter(torch.zeros(input_dim))
        if self.scale:
            self.gamma = Parameter(torch.ones(input_dim))

        if self.conditional:
            if self.hidden_units is not None:
                self.hidden_dense = nn.Linear(in_features=self.cond_dim, out_features=self.hidden_units, bias=False)
            if self.center:
                self.beta_dense = nn.Linear(in_features=self.cond_dim, out_features=input_dim, bias=False)
            if self.scale:
                self.gamma_dense = nn.Linear(in_features=self.cond_dim, out_features=input_dim, bias=False)

        self.initialize_weights()

    def initialize_weights(self):

        if self.conditional:
            if self.hidden_units is not None:
                if self.hidden_initializer == 'normal':
                    torch.nn.init.normal(self.hidden_dense.weight)
                elif self.hidden_initializer == 'xavier':  # glorot_uniform
                    torch.nn.init.xavier_uniform_(self.hidden_dense.weight)
            # 下面这两个为什么都初始化为0呢?
            # 为了防止扰乱原来的预训练权重，两个变换矩阵可以全零初始化（单层神经网络可以用全零初始化，连续的多层神经网络才不应当用全零初始化），
            # 这样在初始状态，模型依然保持跟原来的预训练模型一致。
            if self.center:
                torch.nn.init.constant_(self.beta_dense.weight, 0)
            if self.scale:
                torch.nn.init.constant_(self.gamma_dense.weight, 0)

    def forward(self, inputs, cond=None):
        """
            如果是条件Layer Norm，则cond不是None
        """
        gamma = 1
        beta = 0
        if self.conditional:
            if self.hidden_units is not None:
                cond = self.hidden_dense(cond)
            # for _ in range(K.ndim(inputs) - K.ndim(cond)): # K.ndim: 以整数形式返回张量中的轴数。
            # TODO: 这两个为什么有轴数差呢？ 为什么在 dim=1 上增加维度??
            # 为了保持维度一致，cond可以是（batch_size, cond_dim）
            for _ in range(len(inputs.shape) - len(cond.shape)):
                cond = cond.unsqueeze(1)  # cond = K.expand_dims(cond, 1)

            # cond在加入beta和gamma之前做一次线性变换，以保证与input维度一致
            if self.center:
                beta = self.beta_dense(cond) + self.beta
            if self.scale:
                gamma = self.gamma_dense(cond) + self.gamma
        else:
            if self.center:
                beta = self.beta
            if self.scale:
                gamma = self.gamma

        outputs = inputs
        if self.center:
            mean = torch.mean(outputs, dim=-1).unsqueeze(-1)
            outputs = outputs - mean
        if self.scale:
            variance = torch.mean(outputs ** 2, dim=-1).unsqueeze(-1)
            std = (variance + self.epsilon) ** 2
            outputs = outputs / std
            outputs = outputs * gamma
        if self.center:
            outputs = outputs + beta

        return outputs


def sequence_masking(x: torch.Tensor, mask: torch.Tensor, value=0.0, axis=None):
    """为序列条件mask的函数
    mask: 形如(batch_size, seq_len)的0-1矩阵；
    value: mask部分要被替换成的值，可以是'-inf'或'inf'；
    axis: 序列所在轴，默认为1；
    """
    if mask is None:
        return x
    else:
        if mask.dtype != x.dtype:
            mask = mask.to(x.dtype)
        if value == '-inf':
            value = -1e12
        elif value == 'inf':
            value = 1e12
        if axis is None:
            axis = 1
        elif axis < 0:
            axis = x.ndim + axis
        assert axis > 0, 'axis must be greater than 0'
        for _ in range(axis - 1):
            mask = torch.unsqueeze(mask, 1)
        for _ in range(x.ndim - mask.ndim):
            mask = torch.unsqueeze(mask, mask.ndim)
        return x * mask + value * (1 - mask)


def _generate_relative_positions_matrix(length, max_relative_position,
                                        cache=False):
    """Generates matrix of relative positions between inputs."""
    if not cache:
        range_vec = torch.arange(length)
        range_mat = range_vec.repeat(length).view(length, length)
        distance_mat = range_mat - torch.t(range_mat)
    else:
        distance_mat = torch.arange(-length + 1, 1, 1).unsqueeze(0)

    distance_mat_clipped = torch.clamp(distance_mat, -max_relative_position, max_relative_position)
    final_mat = distance_mat_clipped + max_relative_position

    return final_mat


def _generate_relative_positions_embeddings(seq_length, embed_dim, max_relative_position=127):
    vocab_size = max_relative_position * 2 + 1
    range_vec = torch.arange(seq_length)
    range_mat = range_vec.repeat(seq_length).view(seq_length, seq_length)
    distance_mat = range_mat - torch.t(range_mat)
    distance_mat_clipped = torch.clamp(distance_mat, -max_relative_position, max_relative_position)
    final_mat = distance_mat_clipped + max_relative_position
    embeddings_table = np.zeros([vocab_size, embed_dim])
    for pos in range(vocab_size):
        for i in range(embed_dim // 2):
            embeddings_table[pos, 2 * i] = np.sin(pos / np.power(10000, 2 * i / embed_dim))
            embeddings_table[pos, 2 * i + 1] = np.cos(pos / np.power(10000, 2 * i / embed_dim))

    embeddings_table_tensor = torch.tensor(embeddings_table).float()
    flat_relative_positions_matrix = final_mat.view(-1)
    one_hot_relative_positions_matrix = func.one_hot(flat_relative_positions_matrix,
                                                     num_classes=vocab_size).float()
    embeddings = torch.matmul(one_hot_relative_positions_matrix, embeddings_table_tensor)
    my_shape = list(final_mat.size())
    my_shape.append(embed_dim)
    embeddings = embeddings.view(my_shape)
    # print(embeddings.shape)
    return embeddings


# Test:
# print(_generate_relative_positions_embeddings(6, 32, 4)[0, 0, :])


class HandshakingKernel(nn.Module):
    """
    TPLinker 方法
    """
    def __init__(self, hidden_size, shaking_type, inner_enc_type):
        super().__init__()
        self.shaking_type = shaking_type
        if shaking_type == "cat":
            self.combine_fc = nn.Linear(hidden_size * 2, hidden_size)
        elif shaking_type == "cat_plus":
            self.combine_fc = nn.Linear(hidden_size * 3, hidden_size)
        elif shaking_type == "cln":
            self.tp_cln = LayerNorm(hidden_size, hidden_size, conditional=True)
        elif shaking_type == "cln_plus":
            self.tp_cln = LayerNorm(hidden_size, hidden_size, conditional=True)
            self.inner_context_cln = LayerNorm(hidden_size, hidden_size, conditional=True)

        self.inner_enc_type = inner_enc_type
        if inner_enc_type == "mix_pooling":
            self.lamtha = Parameter(torch.rand(hidden_size))
        elif inner_enc_type == "lstm":
            self.inner_context_lstm = nn.LSTM(hidden_size, hidden_size, num_layers=1, bidirectional=False,
                                              batch_first=True)

    def enc_inner_hiddens(self, seq_hiddens, inner_enc_type="lstm"):
        # seq_hiddens: (batch_size, seq_len, hidden_size)
        def pool(seqence, pooling_type):
            if pooling_type == "mean_pooling":
                pooling = torch.mean(seqence, dim=-2)  # (batch_size, hidden_size)
            elif pooling_type == "max_pooling":
                pooling, _ = torch.max(seqence, dim=-2)  # (batch_size, hidden_size)
            elif pooling_type == "mix_pooling":
                pooling = self.lamtha * torch.mean(seqence, dim=-2) + (1 - self.lamtha) * torch.max(seqence, dim=-2)[0]
            else:
                raise ValueError("'pooling_type must be one of the list: "
                                 "['mean_pooling', 'max_pooling', 'mix_pooling']'")
            return pooling

        if "pooling" in inner_enc_type:
            inner_context = torch.stack(
                [pool(seq_hiddens[:, :i + 1, :], inner_enc_type) for i in range(seq_hiddens.size()[1])], dim=1)
        elif inner_enc_type == "lstm":
            inner_context, _ = self.inner_context_lstm(seq_hiddens)
        else:
            raise ValueError("'inner_enc_type' must be one of the list: "
                             "['mean_pooling', 'max_pooling', 'mix_pooling', 'lstm']")

        return inner_context

    def forward(self, seq_hiddens):
        """
        seq_hiddens: (batch_size, seq_len, hidden_size)
        return:
            shaking_hiddenss: (batch_size, (1 + seq_len) * seq_len / 2, hidden_size) (32, 5+4+3+2+1, 5)
        """
        seq_len = seq_hiddens.size()[-2]
        shaking_hiddens_list = []
        for ind in range(seq_len):
            hidden_each_step = seq_hiddens[:, ind, :]
            visible_hiddens = seq_hiddens[:, ind:, :]  # ind: only look back
            repeat_hiddens = hidden_each_step[:, None, :].repeat(1, seq_len - ind, 1)

            if self.shaking_type == "cat":
                shaking_hiddens = torch.cat([repeat_hiddens, visible_hiddens], dim=-1)
                shaking_hiddens = torch.tanh(self.combine_fc(shaking_hiddens))
            elif self.shaking_type == "cat_plus":
                inner_context = self.enc_inner_hiddens(visible_hiddens, self.inner_enc_type)
                shaking_hiddens = torch.cat([repeat_hiddens, visible_hiddens, inner_context], dim=-1)
                shaking_hiddens = torch.tanh(self.combine_fc(shaking_hiddens))
            elif self.shaking_type == "cln":
                shaking_hiddens = self.tp_cln(visible_hiddens, repeat_hiddens)
            elif self.shaking_type == "cln_plus":
                inner_context = self.enc_inner_hiddens(visible_hiddens, self.inner_enc_type)
                shaking_hiddens = self.tp_cln(visible_hiddens, repeat_hiddens)
                shaking_hiddens = self.inner_context_cln(shaking_hiddens, inner_context)
            else:
                raise ValueError("'shaking_type' must be one of the list: "
                                 "['cat', 'cat_plus', 'cln', 'cln_plus']")

            shaking_hiddens_list.append(shaking_hiddens)
        long_shaking_hiddens = torch.cat(shaking_hiddens_list, dim=1)
        return long_shaking_hiddens


class MyMaths:
    @staticmethod
    def handshaking_len2matrix_size(hsk_len):
        matrix_size = int((2 * hsk_len + 0.25) ** 0.5 - 0.5)
        return matrix_size


class MyMatrix:
    @staticmethod
    def get_shaking_idx2matrix_idx(matrix_size):
        """
        :param matrix_size:
        :return: a list mapping shaking sequence points to matrix points
        """
        shaking_idx2matrix_idx = [(ind, end_ind) for ind in range(matrix_size) for end_ind in
                                  list(range(matrix_size))[ind:]]
        return shaking_idx2matrix_idx

    @staticmethod
    def get_matrix_idx2shaking_idx(matrix_size):
        """
        :param matrix_size:
        :return: a matrix mapping matrix points to shaking sequence points
        """
        matrix_idx2shaking_idx = [[0 for _ in range(matrix_size)] for _ in range(matrix_size)]
        shaking_idx2matrix_idx = MyMatrix.get_shaking_idx2matrix_idx(matrix_size)
        for shaking_ind, matrix_ind in enumerate(shaking_idx2matrix_idx):
            matrix_idx2shaking_idx[matrix_ind[0]][matrix_ind[1]] = shaking_ind
        return matrix_idx2shaking_idx

    @staticmethod
    def mirror(shaking_seq):
        """
        copy upper region to lower region
        :param shaking_seq:
        :return:
        """
        batch_size, handshaking_seq_len, hidden_size = shaking_seq.size()

        matrix_size = MyMaths.handshaking_len2matrix_size(handshaking_seq_len)
        map_ = MyMatrix.get_matrix_idx2shaking_idx(matrix_size)
        mirror_select_ids = [map_[i][j] if i <= j else map_[j][i] for i in range(matrix_size) for j in
                             range(matrix_size)]
        mirror_select_vec = torch.tensor(mirror_select_ids).to(shaking_seq.device)

        matrix = torch.index_select(shaking_seq, dim=1, index=mirror_select_vec)
        matrix = matrix.view(batch_size, matrix_size, matrix_size, hidden_size)
        return matrix

    @staticmethod
    def upper_reg2seq(ori_tensor):
        """
        drop lower triangular part and flat upper triangular part to sequence
        :param ori_tensor: (batch_size, matrix_size, matrix_size, hidden_size)
        :return: (batch_size, matrix_size + ... + 1, hidden_size)
        """
        tensor = ori_tensor.permute(0, 3, 1, 2).contiguous()
        uppder_ones = torch.ones([tensor.size()[-2], tensor.size()[-1]]).long().triu().to(ori_tensor.device)
        upper_diag_ids = torch.nonzero(uppder_ones.view(-1), as_tuple=False).view(-1)
        # flat_tensor: (batch_size, matrix_size * matrix_size, hidden_size)
        flat_tensor = tensor.view(tensor.size()[0], tensor.size()[1], -1).permute(0, 2, 1)
        tensor_upper = torch.index_select(flat_tensor, dim=1, index=upper_diag_ids)
        return tensor_upper

    @staticmethod
    def lower_reg2seq(ori_tensor):
        """
        drop upper triangular part and flat lower triangular part to sequence
        :param ori_tensor: (batch_size, matrix_size, matrix_size, hidden_size)
        :return: (batch_size, matrix_size + ... + 1, hidden_size)
        """
        tensor = ori_tensor.permute(0, 3, 1, 2).contiguous()
        lower_ones = torch.ones([tensor.size()[-2], tensor.size()[-1]]).long().tril().to(ori_tensor.device)
        lower_diag_ids = torch.nonzero(lower_ones.view(-1), as_tuple=False).view(-1)
        # flat_tensor: (batch_size, matrix_size * matrix_size, hidden_size)
        flat_tensor = tensor.view(tensor.size()[0], tensor.size()[1], -1).permute(0, 2, 1)
        tensor_lower = torch.index_select(flat_tensor, dim=1, index=lower_diag_ids)
        return tensor_lower

    @staticmethod
    def shaking_seq2matrix(sequence):
        """
        map sequence tensor to matrix tensor; only upper region has values, pad 0 to the lower region
        :param sequence:
        :return:
        """
        # sequence: (batch_size, seq_len, hidden_size)
        batch_size, seq_len, hidden_size = sequence.size()
        matrix_size = MyMaths.handshaking_len2matrix_size(seq_len)
        map_ = MyMatrix.get_matrix_idx2shaking_idx(matrix_size)
        index_ids = [map_[i][j] if i <= j else seq_len for i in range(matrix_size) for j in range(matrix_size)]
        sequence_w_ze = func.pad(sequence, (0, 0, 0, 1), "constant", 0)
        index_tensor = torch.LongTensor(index_ids).to(sequence.device)
        long_seq = torch.index_select(sequence_w_ze, dim=1, index=index_tensor)
        return long_seq.view(batch_size, matrix_size, matrix_size, hidden_size)


class SingleSourceHandshakingKernel(nn.Module):
    def __init__(self, hidden_size, shaking_type, only_look_after=True, distance_emb_dim=-1):
        super().__init__()
        self.shaking_types = shaking_type.split("+")
        self.only_look_after = only_look_after
        cat_length = 0

        if "cat" in self.shaking_types:
            self.cat_fc = nn.Linear(hidden_size * 2, hidden_size)
            cat_length += hidden_size

        if "cmm" in self.shaking_types:
            self.cat_fc = nn.Linear(hidden_size * 4, hidden_size)
            self.guide_fc = nn.Linear(hidden_size, hidden_size)
            self.vis_fc = nn.Linear(hidden_size, hidden_size)
            cat_length += hidden_size
        if "mul" in self.shaking_types:
            self.guide_fc = nn.Linear(hidden_size, hidden_size)
            self.vis_fc = nn.Linear(hidden_size, hidden_size)
            self.mul_fc = nn.Linear(hidden_size, hidden_size)
        if "cln" in self.shaking_types:
            self.tp_cln = LayerNorm(hidden_size, hidden_size, conditional=True)
            cat_length += hidden_size

        if "lstm" in self.shaking_types:
            assert only_look_after is True
            self.lstm4span = nn.LSTM(hidden_size,
                                     hidden_size,
                                     num_layers=1,
                                     bidirectional=False,
                                     batch_first=True)
            cat_length += hidden_size

        elif "gru" in self.shaking_types:
            assert only_look_after is True
            self.lstm4span = nn.GRU(hidden_size,
                                    hidden_size,
                                    num_layers=1,
                                    bidirectional=False,
                                    batch_first=True)
            cat_length += hidden_size

        if "bilstm" in self.shaking_types:
            assert only_look_after is True
            self.lstm4span = nn.LSTM(hidden_size,
                                     hidden_size // 2,
                                     num_layers=1,
                                     bidirectional=False,
                                     batch_first=True)
            self.lstm4span_back = nn.LSTM(hidden_size,
                                          hidden_size // 2,
                                          num_layers=1,
                                          bidirectional=False,
                                          batch_first=True)
            cat_length += hidden_size
        elif "bigru" in self.shaking_types:
            assert only_look_after is True
            self.lstm4span = nn.GRU(hidden_size,
                                    hidden_size // 2,
                                    num_layers=1,
                                    bidirectional=False,
                                    batch_first=True)
            self.lstm4span_back = nn.GRU(hidden_size,
                                         hidden_size // 2,
                                         num_layers=1,
                                         bidirectional=False,
                                         batch_first=True)
            cat_length += hidden_size

        if "biaffine" in self.shaking_types:
            self.biaffine = nn.Bilinear(hidden_size, hidden_size, hidden_size)
            cat_length += hidden_size

        self.distance_emb_dim = distance_emb_dim
        if distance_emb_dim > 0:
            self.dist_emb = nn.Embedding(512, distance_emb_dim)
            self.dist_ids_matrix = None  # for cache
            cat_length += distance_emb_dim

        self.aggr_fc = nn.Linear(cat_length, hidden_size)

    def forward(self, seq_hiddens):
        """
        seq_hiddens: (batch_size, seq_len, hidden_size_x)
        return:
            if only look after:
                shaking_hiddenss: (batch_size, (1 + seq_len) * seq_len / 2, hidden_size); e.g. (32, 5+4+3+2+1, 5)
            else:
                shaking_hiddenss: (batch_size, seq_len * seq_len, hidden_size)
        """
        # seq_len = seq_hiddens.size()[1]
        batch_size, seq_len, vis_hidden_size = seq_hiddens.size()

        guide = seq_hiddens[:, :, None, :].repeat(1, 1, seq_len, 1)
        visible = guide.permute(0, 2, 1, 3)
        feature_pre_list = []

        if self.only_look_after:
            if len({"lstm", "bilstm", "gru", "bigru"}.intersection(self.shaking_types)) > 0:
                # batch_size, _, matrix_size, vis_hidden_size = visible.size()
                # mask lower triangle part
                upper_visible = visible.permute(0, 3, 1, 2).triu().permute(0, 2, 3, 1).contiguous()

                # visible4lstm: (batch_size * matrix_size, matrix_size, hidden_size)
                visible4lstm = upper_visible.view(batch_size * seq_len, seq_len, -1)
                span_pre, _ = self.lstm4span(visible4lstm)
                span_pre = span_pre.view(batch_size, seq_len, seq_len, -1)

                if len({"bilstm", "bigru"}.intersection(self.shaking_types)) > 0:
                    # mask upper triangle part
                    lower_visible = visible.permute(0, 3, 1, 2).tril().permute(0, 2, 3, 1).contiguous()
                    visible4lstm_back = lower_visible.view(batch_size * seq_len, seq_len, -1)

                    visible4lstm_back = torch.flip(visible4lstm_back, [1, ])
                    span_pre_back, _ = self.lstm4span_back(visible4lstm_back)
                    span_pre_back = torch.flip(span_pre_back, [1, ])
                    span_pre_back = span_pre_back.view(batch_size, seq_len, seq_len, -1)
                    span_pre_back = span_pre_back.permute(0, 2, 1, 3)
                    span_pre = torch.cat([span_pre, span_pre_back], dim=-1)

                # drop lower triangle and convert matrix to sequence
                # span_pre: (batch_size, shaking_seq_len, hidden_size)
                span_pre = MyMatrix.upper_reg2seq(span_pre)
                feature_pre_list.append(span_pre)

            # guide, visible: (batch_size, shaking_seq_len, hidden_size)
            guide = MyMatrix.upper_reg2seq(guide)
            visible = MyMatrix.upper_reg2seq(visible)

        if "cat" in self.shaking_types:
            tp_cat_pre = torch.cat([guide, visible], dim=-1)
            tp_cat_pre = torch.relu(self.cat_fc(tp_cat_pre))
            feature_pre_list.append(tp_cat_pre)

        if "cmm" in self.shaking_types:  # cat and multiple
            tp_cat_pre = torch.cat([guide, visible,
                                    torch.abs(guide - visible),
                                    torch.mul(self.guide_fc(guide), self.vis_fc(visible))], dim=-1)
            tp_cat_pre = torch.relu(self.cat_fc(tp_cat_pre))
            feature_pre_list.append(tp_cat_pre)

        if "cln" in self.shaking_types:
            tp_cln_pre = self.tp_cln(visible, guide)
            feature_pre_list.append(tp_cln_pre)

        if "biaffine" in self.shaking_types:
            biaffine_pre = self.biaffine(guide, visible)
            biaffine_pre = torch.relu(biaffine_pre)
            feature_pre_list.append(biaffine_pre)

        if self.distance_emb_dim > 0:
            if self.dist_ids_matrix is None or \
                    self.dist_ids_matrix.size()[0] != batch_size or \
                    self.dist_ids_matrix.size()[1] != seq_len:  # need to update cached distance ids
                t = torch.arange(0, seq_len).to(seq_hiddens.device)[:, None].repeat(1, seq_len)
                self.dist_ids_matrix = torch.abs(t - t.permute(1, 0)).long()[None, :, :].repeat(batch_size, 1, 1)
                if self.only_look_after:  # matrix to handshaking seq
                    self.dist_ids_matrix = MyMatrix.upper_reg2seq(
                        self.dist_ids_matrix[:, :, :, None]).view(batch_size, -1)
            dist_embeddings = self.dist_emb(self.dist_ids_matrix)
            feature_pre_list.append(dist_embeddings)

        output_hiddens = self.aggr_fc(torch.cat(feature_pre_list, dim=-1))
        return output_hiddens


class CrossLSTM(nn.Module):
    def __init__(self,
                 in_feature_dim=None,
                 out_feature_dim=None,
                 num_layers=1,
                 hv_comb_type="cat"
                 ):
        super().__init__()
        self.vertical_lstm = nn.LSTM(in_feature_dim,
                                     out_feature_dim // 2,
                                     num_layers=num_layers,
                                     bidirectional=True,
                                     batch_first=True)
        self.horizontal_lstm = nn.LSTM(in_feature_dim,
                                       out_feature_dim // 2,
                                       num_layers=num_layers,
                                       bidirectional=True,
                                       batch_first=True)

        self.hv_comb_type = hv_comb_type
        if hv_comb_type == "cat":
            self.combine_fc = nn.Linear(out_feature_dim * 2, out_feature_dim)
        elif hv_comb_type == "add":
            pass
        elif hv_comb_type == "interpolate":
            self.lamtha = Parameter(torch.rand(out_feature_dim))  # [0, 1)

    def forward(self, matrix):
        # matrix: (batch_size, matrix_ver_len, matrix_hor_len, hidden_size)
        batch_size, matrix_ver_len, matrix_hor_len, hidden_size = matrix.size()
        hor_context, _ = self.horizontal_lstm(matrix.view(-1, matrix_hor_len, hidden_size))
        hor_context = hor_context.view(batch_size, matrix_ver_len, matrix_hor_len, hidden_size)

        ver_context, _ = self.vertical_lstm(
            matrix.permute(0, 2, 1, 3).contiguous().view(-1, matrix_ver_len, hidden_size))
        ver_context = ver_context.view(batch_size, matrix_hor_len, matrix_ver_len, hidden_size)
        ver_context = ver_context.permute(0, 2, 1, 3)

        comb_context = None
        if self.hv_comb_type == "cat":
            comb_context = torch.relu(self.combine_fc(torch.cat([hor_context, ver_context], dim=-1)))
        elif self.hv_comb_type == "interpolate":
            comb_context = self.lamtha * hor_context + (1 - self.lamtha) * ver_context
        elif self.hv_comb_type == "add":
            comb_context = (hor_context + ver_context) / 2

        return comb_context


class CrossConv(nn.Module):
    def __init__(self,
                 channel_dim,
                 hor_dim,
                 ver_dim
                 ):
        super(CrossConv, self).__init__()
        self.alpha = Parameter(torch.randn([channel_dim, hor_dim, 1]))
        self.beta = Parameter(torch.randn([channel_dim, 1, ver_dim]))

    def forward(self, matrix_tensor):
        # matrix_tensor: (batch_size, ver_dim, hor_dim, hidden_size)
        # hor_cont: (batch_size, hidden_size (channel dim), ver_dim, 1)
        hor_cont = torch.matmul(matrix_tensor.permute(0, 3, 1, 2), self.alpha)
        # ver_cont: (batch_size, hidden_size, 1, hor_dim)
        ver_cont = torch.matmul(self.beta, matrix_tensor.permute(0, 3, 1, 2))
        # cross_context: (batch_size, ver_dim, hor_dim, hidden_size)
        cross_context = torch.matmul(hor_cont, ver_cont).permute(0, 2, 3, 1)
        return cross_context


class CrossPool(nn.Module):
    def __init__(self, hidden_size):
        super(CrossPool, self).__init__()
        self.lamtha = Parameter(torch.rand(hidden_size))

    def mix_pool(self, tensor, dim):
        return self.lamtha * torch.mean(tensor, dim=dim) + (1 - self.lamtha) * torch.max(tensor, dim=dim)[0]

    def forward(self, matrix_tensor):
        # matrix_tensor: (batch_size, ver_dim, hor_dim, hidden_size)
        # hor_cont: (batch_size, hidden_size, ver_dim, 1)
        hor_cont = self.mix_pool(matrix_tensor, dim=2)[:, :, None, :].permute(0, 3, 1, 2)

        # ver_cont: (batch_size, hidden_size, 1, hor_dim)
        ver_cont = self.mix_pool(matrix_tensor, dim=1)[:, None, :, :].permute(0, 3, 1, 2)

        # cross_context: (batch_size, ver_dim, hor_dim, hidden_size)
        cross_context = torch.matmul(hor_cont, ver_cont).permute(0, 2, 3, 1)
        return cross_context


class EdgeUpdate(nn.Module):
    def __init__(self, hidden_dim, dim_e, dropout_ratio=0.5):
        super(EdgeUpdate, self).__init__()
        self.hidden_dim = hidden_dim
        self.dim_e = dim_e
        self.dropout = dropout_ratio
        self.W = nn.Linear(self.hidden_dim * 2 + self.dim_e, self.dim_e)

    def forward(self, edge, node1, node2):
        """
        :param edge: [batch, seq, seq, dim_e]
        :param node1: [batch, seq, seq, dim]
        :param node2: [batch, seq, seq, dim]
        :return:
        """

        node = torch.cat([node1, node2], dim=-1)  # [batch, seq, seq, dim * 2]
        edge = self.W(torch.cat([edge, node], dim=-1))
        return edge  # [batch, seq, seq, dim_e]


class GraphConvLayer(nn.Module):
    """ A GCN module operated on dependency graphs. """

    def __init__(self, dep_embed_dim, gcn_dim, pooling='avg'):
        super(GraphConvLayer, self).__init__()

        self.gcn_dim = gcn_dim
        self.dep_embed_dim = dep_embed_dim
        self.pooling = pooling

        self.W = nn.Linear(self.gcn_dim, self.gcn_dim)
        self.highway = EdgeUpdate(gcn_dim, self.dep_embed_dim, dropout_ratio=0.5)

    def forward(self, weight_adj, node_hiddens):
        """
        :param weight_adj: [batch, seq, seq, dim_e]
        :param node_hiddens: [batch, seq, dim]
        :return:
        """

        batch, seq, dim = node_hiddens.shape
        weight_adj = weight_adj.permute(0, 3, 1, 2)  # [batch, dim_e, seq, seq]

        node_hiddens = node_hiddens.unsqueeze(1).expand(batch, self.dep_embed_dim, seq, dim)
        ax = torch.matmul(weight_adj, node_hiddens)  # [batch, dim_e, seq, dim]
        if self.pooling == 'avg':
            ax = ax.mean(dim=1)
        elif self.pooling == 'max':
            ax, _ = ax.max(dim=1)
        elif self.pooling == 'sum':
            ax = ax.sum(dim=1)
        # Ax: [batch, seq, dim]

        gcn_outputs = self.W(ax)
        weights_gcn_outputs = func.relu(gcn_outputs)

        node_outputs = weights_gcn_outputs
        # Edge update weight_adj[batch, dim_e, seq, seq]
        weight_adj = weight_adj.permute(0, 2, 3, 1).contiguous()  # [batch, seq, seq, dim_e]
        node_outputs1 = node_outputs.unsqueeze(1).expand(batch, seq, seq, dim)
        node_outputs2 = node_outputs1.permute(0, 2, 1, 3).contiguous()
        edge_outputs = self.highway(weight_adj, node_outputs1, node_outputs2)
        return edge_outputs, node_outputs


class Indexer:
    def __init__(self, tag2id, max_seq_len, spe_tag_dict):
        self.tag2id = tag2id
        self.max_seq_len = max_seq_len
        self.spe_tag_dict = spe_tag_dict

    def index_tag_list_w_matrix_pos(self, tags):
        """
        :param tags: [[pos_i, pos_j, tag1], [pos_i, pos_j, tag2], ...]
        :return:
        """
        for t in tags:
            if t[2] in self.tag2id:
                t[2] = self.tag2id[t[2]]
            else:
                t[2] = self.spe_tag_dict["[UNK]"]
        return tags

    @staticmethod
    def pad2length(tags, padding_tag, length):
        if len(tags) < length:
            tags.extend([padding_tag] * (length - len(tags)))
        return tags[:length]

    def index_tag_list(self, tags):
        """
        tags: [t1, t2, t3, ...]
        """
        tag_ids = []
        for t in tags:
            if t not in self.tag2id:
                tag_ids.append(self.spe_tag_dict["[UNK]"])
            else:
                tag_ids.append(self.tag2id[t])

        if len(tag_ids) < self.max_seq_len:
            tag_ids.extend([self.spe_tag_dict["[PAD]"]] * (self.max_seq_len - len(tag_ids)))

        return tag_ids[:self.max_seq_len]

    @staticmethod
    def get_shaking_idx2matrix_idx(matrix_size):
        return MyMatrix.get_shaking_idx2matrix_idx(matrix_size)

    @staticmethod
    def get_matrix_idx2shaking_idx(matrix_size):
        return MyMatrix.get_matrix_idx2shaking_idx(matrix_size)

    @staticmethod
    def points2multilabel_shaking_seq(points, matrix_size, tag_size):
        """
        Convert points to a shaking sequence tensor

        points: [(start_ind, end_ind, tag_id), ]
        return:
            shaking_seq: (shaking_seq_len, tag_size)
        """
        matrix_idx2shaking_idx = Indexer.get_matrix_idx2shaking_idx(matrix_size)
        shaking_seq_len = matrix_size * (matrix_size + 1) // 2
        shaking_seq = torch.zeros(shaking_seq_len, tag_size).long()
        for sp in points:
            shaking_idx = matrix_idx2shaking_idx[sp[0]][sp[1]]
            shaking_seq[shaking_idx][sp[2]] = 1
        return shaking_seq

    @staticmethod
    def points2multilabel_shaking_seq_batch(batch_points, matrix_size, tag_size):
        """
        Convert points to a shaking sequence tensor in batch (for training tags)

        batch_points: a batch of points, [points1, points2, ...]
            points: [(start_ind, end_ind, tag_id), ]
        return:
            batch_shaking_seq: (batch_size_train, shaking_seq_len, tag_size)
        """
        matrix_idx2shaking_idx = Indexer.get_matrix_idx2shaking_idx(matrix_size)
        shaking_seq_len = matrix_size * (matrix_size + 1) // 2
        batch_shaking_seq = torch.zeros(len(batch_points), shaking_seq_len, tag_size).long()
        for batch_id, points in enumerate(batch_points):
            for sp in points:
                shaking_idx = matrix_idx2shaking_idx[sp[0]][sp[1]]
                batch_shaking_seq[batch_id][shaking_idx][sp[2]] = 1
        return batch_shaking_seq

    @staticmethod
    def points2shaking_seq_batch(batch_points, matrix_size):
        """
        Convert points to a shaking sequence tensor

        batch_points: a batch of points, [points1, points2, ...]
            points: [(start_ind, end_ind, tag_id), ]
        return:
            batch_shaking_seq: (batch_size_train, shaking_seq_len)
        """
        matrix_idx2shaking_idx = Indexer.get_matrix_idx2shaking_idx(matrix_size)
        shaking_seq_len = matrix_size * (matrix_size + 1) // 2
        batch_shaking_seq = torch.zeros(len(batch_points), shaking_seq_len).long()
        for batch_id, points in enumerate(batch_points):
            for sp in points:
                try:
                    shaking_idx = matrix_idx2shaking_idx[sp[0]][sp[1]]
                except Exception as e:
                    raise e
                else:
                    batch_shaking_seq[batch_id][shaking_idx] = sp[2]
        return batch_shaking_seq

    @staticmethod
    def points2matrix_batch(batch_points, matrix_size):
        """
        Convert points to a matrix tensor

        batch_points: a batch of points, [points1, points2, ...]
            points: [(start_ind, end_ind, tag_id), ]
        return:
            batch_matrix: (batch_size_train, matrix_size, matrix_size)
        """
        batch_matrix = torch.zeros(len(batch_points), matrix_size, matrix_size).long()
        for batch_id, points in enumerate(batch_points):
            for pt in points:
                batch_matrix[batch_id][pt[0]][pt[1]] = pt[2]
        return batch_matrix

    @staticmethod
    def points2multilabel_matrix_batch(batch_points, matrix_size, tag_size):
        """
        Convert points to a matrix tensor for multi-label tasks

        batch_points: a batch of points, [points1, points2, ...]
            points: [(i, j, tag_id), ]
        return:
            batch_matrix: shape: (batch_size_train, matrix_size, matrix_size, tag_size) # element 0 or 1
        """
        batch_matrix = torch.zeros(len(batch_points), matrix_size, matrix_size, tag_size).long()
        for batch_id, points in enumerate(batch_points):
            for pt in points:
                batch_matrix[batch_id][pt[0]][pt[1]][pt[2]] = 1
        return batch_matrix

    @staticmethod
    def shaking_seq2points(shaking_tag):
        """
        shaking_tag -> points
        shaking_tag: shape: (shaking_seq_len, tag_size)
        points: [(start_ind, end_ind, tag_id), ]
        """
        points = []
        shaking_seq_len = shaking_tag.size()[0]
        matrix_size = int((2 * shaking_seq_len + 0.25) ** 0.5 - 0.5)
        shaking_idx2matrix_idx = Indexer.get_shaking_idx2matrix_idx(matrix_size)
        nonzero_points = torch.nonzero(shaking_tag, as_tuple=False)
        for point in nonzero_points:
            shaking_idx, tag_idx = point[0].item(), point[1].item()
            pos1, pos2 = shaking_idx2matrix_idx[shaking_idx]
            point = (pos1, pos2, tag_idx)
            points.append(point)
        return points

    @staticmethod
    def matrix2points(matrix_tag):
        """
        matrix_tag -> points
        matrix_tag: shape: (matrix_size, matrix_size, tag_size)
        points: [(i, j, tag_id), ]
        """
        points = []
        nonzero_points = torch.nonzero(matrix_tag, as_tuple=False)
        for point in nonzero_points:
            i, j, tag_idx = point[0].item(), point[1].item(), point[2].item()
            point = (i, j, tag_idx)
            points.append(point)
        return points
