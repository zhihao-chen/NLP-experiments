# -*- coding: utf8 -*-
"""
======================================
    Project Name: NLP
    File Name: bert_for_relation_extraction
    Author: czh
    Create Date: 2021/8/13
--------------------------------------
    Change Activity: 
======================================
"""
import math
from typing import Optional

import torch
import torch.nn as nn
from transformers import BertModel
from nlp.models.bert_model import BertPreTrainedModel

from nlp.models.model_util import HandshakingKernel
from nlp.layers.global_pointer import GlobalPointer, EfficientGlobalPointer


class GlobalPointerForRel(nn.Module):
    def __init__(self, config,
                 entity_types_num, relation_num,
                 head_size=64,
                 encoder_model_path=None,
                 rope=True,
                 tril_mask=True,
                 efficient=False,
                 bert_frozen=False):
        """
        global pointer用于关系抽取，基于GlobalPointer的仿TPLinker设计
        :param config:
        :param encoder_model_path:
        :param entity_types_num: 实体类型数量
        :param relation_num:
        :param head_size:
        :param rope:
        :param tril_mask:
        :param bert_frozen:
        """
        super(GlobalPointerForRel, self).__init__()
        if encoder_model_path:
            self.encoder = BertModel.from_pretrained(encoder_model_path)
        else:
            self.encoder = BertModel(config)
        self.rope = rope
        self.tril_mask = tril_mask
        if efficient:
            gp_link = EfficientGlobalPointer
        else:
            gp_link = GlobalPointer

        self.entity_model = gp_link(
                                   ent_type_size=entity_types_num,
                                   head_size=head_size,
                                   hidden_size=config.hidden_size,
                                   rope=rope,
                                   tril_mask=tril_mask
                                   )
        self.rel_head_model = gp_link(
                                     ent_type_size=relation_num,
                                     head_size=head_size,
                                     hidden_size=config.hidden_size,
                                     rope=False,
                                     tril_mask=False
                                     )
        self.rel_tail_model = gp_link(
                                     ent_type_size=relation_num,
                                     head_size=head_size,
                                     hidden_size=config.hidden_size,
                                     rope=False,
                                     tril_mask=False)
        if bert_frozen:
            self.encoder.embeddings.word_embeddings.weight.requires_grad = False
            self.encoder.embeddings.position_embeddings.weight.requires_grad = False
            self.encoder.embeddings.token_type_embeddings.weight.requires_grad = False

    def forward(self, input_ids, attention_mask, token_type_ids):
        context_outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        last_hidden_state = context_outputs[0]
        entity_logit = self.entity_model(last_hidden_state=last_hidden_state, attention_mask=attention_mask)
        head_logit = self.rel_head_model(last_hidden_state=last_hidden_state, attention_mask=attention_mask)
        tail_logit = self.rel_tail_model(last_hidden_state=last_hidden_state, attention_mask=attention_mask)

        outputs = {'entity_logits': entity_logit, 'head_logits': head_logit, 'tail_logits': tail_logit}
        # outputs = {'entity_logits': entity_logit}
        return outputs


class Casrel(BertPreTrainedModel):
    """
    A Novel Cascade Binary Tagging Framework for Relational Triple Extraction
    https://github.com/weizhepei/CasRel
    """
    def __init__(self, config, bert_config):
        super(Casrel, self).__init__(bert_config)
        self.config = config
        self.bert_dim = 768
        self.bert_encoder = BertModel(bert_config)
        self.sub_heads_linear = nn.Linear(self.bert_dim, 1)
        self.sub_tails_linear = nn.Linear(self.bert_dim, 1)
        self.obj_heads_linear = nn.Linear(self.bert_dim, self.config.rel_num)
        self.obj_tails_linear = nn.Linear(self.bert_dim, self.config.rel_num)

        self.apply(self.init_bert_weights)

    def get_objs_for_specific_sub(self, sub_head_mapping, sub_tail_mapping, encoded_text):
        # [batch_size, 1, bert_dim]
        sub_head = torch.matmul(sub_head_mapping, encoded_text)
        # [batch_size, 1, bert_dim]
        sub_tail = torch.matmul(sub_tail_mapping, encoded_text)
        # [batch_size, 1, bert_dim]
        sub = (sub_head + sub_tail) / 2
        # [batch_size, seq_len, bert_dim]
        encoded_text = encoded_text + sub
        # [batch_size, seq_len, rel_num]
        pred_obj_heads = self.obj_heads_linear(encoded_text)
        pred_obj_heads = torch.sigmoid(pred_obj_heads)
        # [batch_size, seq_len, rel_num]
        pred_obj_tails = self.obj_tails_linear(encoded_text)
        pred_obj_tails = torch.sigmoid(pred_obj_tails)
        return pred_obj_heads, pred_obj_tails

    def get_encoded_text(self, token_ids, mask):
        # [batch_size, seq_len, bert_dim(768)]
        encoded_text = self.bert_encoder(token_ids, attention_mask=mask)[0]
        return encoded_text

    def get_subs(self, encoded_text):
        # [batch_size, seq_len, 1]
        pred_sub_heads = self.sub_heads_linear(encoded_text)
        pred_sub_heads = torch.sigmoid(pred_sub_heads)
        # [batch_size, seq_len, 1]
        pred_sub_tails = self.sub_tails_linear(encoded_text)
        pred_sub_tails = torch.sigmoid(pred_sub_tails)
        return pred_sub_heads, pred_sub_tails

    def forward(self, data):
        # [batch_size, seq_len]
        token_ids = data['token_ids']
        # [batch_size, seq_len]
        mask = data['mask']
        # [batch_size, seq_len, bert_dim(768)]
        encoded_text = self.get_encoded_text(token_ids, mask)
        # [batch_size, seq_len, 1]
        pred_sub_heads, pred_sub_tails = self.get_subs(encoded_text)
        # [batch_size, 1, seq_len]
        sub_head_mapping = data['sub_head'].unsqueeze(1)
        # [batch_size, 1, seq_len]
        sub_tail_mapping = data['sub_tail'].unsqueeze(1)
        # [batch_size, seq_len, rel_num]
        pred_obj_heads, pred_obj_tails = self.get_objs_for_specific_sub(sub_head_mapping,
                                                                        sub_tail_mapping, encoded_text)
        return pred_sub_heads, pred_sub_tails, pred_obj_heads, pred_obj_tails


class TPLinkerBert(nn.Module):
    def __init__(self, encoder,
                 rel_size,
                 shaking_type,
                 inner_enc_type,
                 dist_emb_size,
                 ent_add_dist,
                 rel_add_dist
                 ):
        super().__init__()
        self.encoder = encoder
        hidden_size = encoder.config.hidden_size

        self.ent_fc = nn.Linear(hidden_size, 2)
        self.head_rel_fc_list = [nn.Linear(hidden_size, 3) for _ in range(rel_size)]
        self.tail_rel_fc_list = [nn.Linear(hidden_size, 3) for _ in range(rel_size)]

        for ind, fc in enumerate(self.head_rel_fc_list):
            self.register_parameter("weight_4_head_rel{}".format(ind), fc.weight)  # noqa
            self.register_parameter("bias_4_head_rel{}".format(ind), fc.bias)
        for ind, fc in enumerate(self.tail_rel_fc_list):
            self.register_parameter("weight_4_tail_rel{}".format(ind), fc.weight)  # noqa
            self.register_parameter("bias_4_tail_rel{}".format(ind), fc.bias)

        # handshaking kernel, 将上三角矩阵平铺成序列
        self.handshaking_kernel = HandshakingKernel(hidden_size, shaking_type, inner_enc_type)

        # distance embedding
        self.dist_emb_size = dist_emb_size
        self.dist_embbedings = None  # it will be set in the first forwarding

        self.ent_add_dist = ent_add_dist
        self.rel_add_dist = rel_add_dist

    def forward(self, input_ids, attention_mask, token_type_ids):
        # input_ids, attention_mask, token_type_ids: (batch_size, seq_len)
        context_outputs = self.encoder(input_ids, attention_mask, token_type_ids)
        # last_hidden_state: (batch_size, seq_len, hidden_size)
        last_hidden_state = context_outputs[0]

        # shaking_hiddens: (batch_size, 1 + ... + seq_len, hidden_size)
        shaking_hiddens = self.handshaking_kernel(last_hidden_state)
        shaking_hiddens4ent = shaking_hiddens
        shaking_hiddens4rel = shaking_hiddens

        # add distance embeddings if it is set，位置编码，采用的是transformer的相对位置编码设计，该embedding是计算出来的，而不是学习出来的
        if self.dist_emb_size != -1:
            # set self.dist_embbedings
            hidden_size = shaking_hiddens.size()[-1]
            if self.dist_embbedings is None:
                dist_emb = torch.zeros([self.dist_emb_size, hidden_size]).to(shaking_hiddens.device)
                for d in range(self.dist_emb_size):
                    for i in range(hidden_size):
                        if i % 2 == 0:
                            dist_emb[d][i] = math.sin(d / 10000 ** (i / hidden_size))
                        else:
                            dist_emb[d][i] = math.cos(d / 10000 ** ((i - 1) / hidden_size))
                seq_len = input_ids.size()[1]
                dist_embbeding_segs = []

                # 将位置编码矩阵转成上三角矩阵，再平铺成序列
                for after_num in range(seq_len, 0, -1):
                    dist_embbeding_segs.append(dist_emb[:after_num, :])
                self.dist_embbedings = torch.cat(dist_embbeding_segs, dim=0)

            if self.ent_add_dist:
                shaking_hiddens4ent = shaking_hiddens + self.dist_embbedings[None, :, :].repeat(
                    shaking_hiddens.size()[0], 1, 1)
            if self.rel_add_dist:
                shaking_hiddens4rel = shaking_hiddens + self.dist_embbedings[None, :, :].repeat(
                    shaking_hiddens.size()[0], 1, 1)

        ent_shaking_outputs = self.ent_fc(shaking_hiddens4ent)

        head_rel_shaking_outputs_list = []
        for fc in self.head_rel_fc_list:
            head_rel_shaking_outputs_list.append(fc(shaking_hiddens4rel))

        tail_rel_shaking_outputs_list = []
        for fc in self.tail_rel_fc_list:
            tail_rel_shaking_outputs_list.append(fc(shaking_hiddens4rel))

        head_rel_shaking_outputs = torch.stack(head_rel_shaking_outputs_list, dim=1)
        tail_rel_shaking_outputs = torch.stack(tail_rel_shaking_outputs_list, dim=1)

        return ent_shaking_outputs, head_rel_shaking_outputs, tail_rel_shaking_outputs


class TPLinkerBiLSTM(nn.Module):
    def __init__(self, init_word_embedding_matrix,
                 emb_dropout_rate,
                 enc_hidden_size,
                 dec_hidden_size,
                 rnn_dropout_rate,
                 rel_size,
                 shaking_type,
                 inner_enc_type,
                 dist_emb_size,
                 ent_add_dist,
                 rel_add_dist):
        super().__init__()
        self.word_embeds = nn.Embedding.from_pretrained(init_word_embedding_matrix, freeze=False)
        self.emb_dropout = nn.Dropout(emb_dropout_rate)
        self.enc_lstm = nn.LSTM(init_word_embedding_matrix.size()[-1],
                                enc_hidden_size // 2,
                                num_layers=1,
                                bidirectional=True,
                                batch_first=True)
        self.dec_lstm = nn.LSTM(enc_hidden_size,
                                dec_hidden_size // 2,
                                num_layers=1,
                                bidirectional=True,
                                batch_first=True)
        self.rnn_dropout = nn.Dropout(rnn_dropout_rate)

        hidden_size = dec_hidden_size

        self.ent_fc = nn.Linear(hidden_size, 2)
        self.head_rel_fc_list = [nn.Linear(hidden_size, 3) for _ in range(rel_size)]
        self.tail_rel_fc_list = [nn.Linear(hidden_size, 3) for _ in range(rel_size)]

        for ind, fc in enumerate(self.head_rel_fc_list):
            self.register_parameter("weight_4_head_rel{}".format(ind), fc.weight)  # noqa
            self.register_parameter("bias_4_head_rel{}".format(ind), fc.bias)
        for ind, fc in enumerate(self.tail_rel_fc_list):
            self.register_parameter("weight_4_tail_rel{}".format(ind), fc.weight)  # noqa
            self.register_parameter("bias_4_tail_rel{}".format(ind), fc.bias)

        # handshaking kernel
        self.handshaking_kernel = HandshakingKernel(hidden_size, shaking_type, inner_enc_type)

        # distance embedding
        self.dist_emb_size = dist_emb_size
        self.dist_embbedings = None  # it will be set in the first forwarding

        self.ent_add_dist = ent_add_dist
        self.rel_add_dist = rel_add_dist

    def forward(self, input_ids):
        # input_ids: (batch_size, seq_len)
        # embedding: (batch_size, seq_len, emb_dim)
        embedding = self.word_embeds(input_ids)
        embedding = self.emb_dropout(embedding)
        # lstm_outputs: (batch_size, seq_len, enc_hidden_size)
        lstm_outputs, _ = self.enc_lstm(embedding)
        lstm_outputs = self.rnn_dropout(lstm_outputs)
        # lstm_outputs: (batch_size, seq_len, dec_hidden_size)
        lstm_outputs, _ = self.dec_lstm(lstm_outputs)
        lstm_outputs = self.rnn_dropout(lstm_outputs)

        # shaking_hiddens: (batch_size, 1 + ... + seq_len, hidden_size)
        shaking_hiddens = self.handshaking_kernel(lstm_outputs)
        shaking_hiddens4ent = shaking_hiddens
        shaking_hiddens4rel = shaking_hiddens

        # add distance embeddings if it is set
        if self.dist_emb_size != -1:
            # set self.dist_embbedings
            hidden_size = shaking_hiddens.size()[-1]
            if self.dist_embbedings is None:
                dist_emb = torch.zeros([self.dist_emb_size, hidden_size]).to(shaking_hiddens.device)
                for d in range(self.dist_emb_size):
                    for i in range(hidden_size):
                        if i % 2 == 0:
                            dist_emb[d][i] = math.sin(d / 10000 ** (i / hidden_size))
                        else:
                            dist_emb[d][i] = math.cos(d / 10000 ** ((i - 1) / hidden_size))
                seq_len = input_ids.size()[1]
                dist_embbeding_segs = []
                for after_num in range(seq_len, 0, -1):
                    dist_embbeding_segs.append(dist_emb[:after_num, :])
                self.dist_embbedings = torch.cat(dist_embbeding_segs, dim=0)

            if self.ent_add_dist:
                shaking_hiddens4ent = shaking_hiddens + self.dist_embbedings[None, :, :].repeat(
                    shaking_hiddens.size()[0], 1, 1)
            if self.rel_add_dist:
                shaking_hiddens4rel = shaking_hiddens + self.dist_embbedings[None, :, :].repeat(
                    shaking_hiddens.size()[0], 1, 1)

        ent_shaking_outputs = self.ent_fc(shaking_hiddens4ent)

        head_rel_shaking_outputs_list = []
        for fc in self.head_rel_fc_list:
            head_rel_shaking_outputs_list.append(fc(shaking_hiddens4rel))

        tail_rel_shaking_outputs_list = []
        for fc in self.tail_rel_fc_list:
            tail_rel_shaking_outputs_list.append(fc(shaking_hiddens4rel))

        head_rel_shaking_outputs = torch.stack(head_rel_shaking_outputs_list, dim=1)
        tail_rel_shaking_outputs = torch.stack(tail_rel_shaking_outputs_list, dim=1)

        return ent_shaking_outputs, head_rel_shaking_outputs, tail_rel_shaking_outputs


class TPLinkerPlusBert(nn.Module):
    def __init__(self, encoder,
                 tag_size,
                 shaking_type,
                 inner_enc_type,
                 tok_pair_sample_rate=1):
        super().__init__()
        self.encoder = encoder
        self.tok_pair_sample_rate = tok_pair_sample_rate

        shaking_hidden_size = encoder.config.hidden_size

        self.fc = nn.Linear(shaking_hidden_size, tag_size)

        # handshaking kernel
        self.handshaking_kernel = HandshakingKernel(shaking_hidden_size, shaking_type, inner_enc_type)

    def forward(self, input_ids,
                attention_mask,
                token_type_ids
                ):
        # input_ids, attention_mask, token_type_ids: (batch_size, seq_len)
        context_outputs = self.encoder(input_ids, attention_mask, token_type_ids)
        # last_hidden_state: (batch_size, seq_len, hidden_size)
        last_hidden_state = context_outputs[0]

        # seq_len = last_hidden_state.size()[1]
        # shaking_hiddens: (batch_size, shaking_seq_len, hidden_size)
        shaking_hiddens = self.handshaking_kernel(last_hidden_state)

        sampled_tok_pair_indices = None
        if self.training:
            # randomly sample segments of token pairs
            shaking_seq_len = shaking_hiddens.size()[1]
            segment_len = int(shaking_seq_len * self.tok_pair_sample_rate)
            seg_num = math.ceil(shaking_seq_len // segment_len)
            start_ind = torch.randint(seg_num, []) * segment_len
            end_ind = min(start_ind + segment_len, shaking_seq_len)
            # sampled_tok_pair_indices: (batch_size, ~segment_len) ~end_ind - start_ind <= segment_len
            sampled_tok_pair_indices = torch.arange(start_ind, end_ind)[None, :].repeat(shaking_hiddens.size()[0], 1)
            # sampled_tok_pair_indices = torch.randint(shaking_seq_len, (shaking_hiddens.size()[0], segment_len))
            sampled_tok_pair_indices = sampled_tok_pair_indices.to(shaking_hiddens.device)

            # sampled_tok_pair_indices will tell model what token pairs should be fed into fcs
            # shaking_hiddens: (batch_size, ~segment_len, hidden_size)
            index = sampled_tok_pair_indices[:, :, None].repeat(1, 1, shaking_hiddens.size()[-1])
            shaking_hiddens = shaking_hiddens.gather(1, index)

        # outputs: (batch_size, segment_len, tag_size) or (batch_size, shaking_seq_len, tag_size)
        outputs = self.fc(shaking_hiddens)

        return outputs, sampled_tok_pair_indices


class TPLinkerPlusBiLSTM(nn.Module):
    def __init__(self, init_word_embedding_matrix,
                 emb_dropout_rate,
                 enc_hidden_size,
                 dec_hidden_size,
                 rnn_dropout_rate,
                 tag_size,
                 shaking_type,
                 inner_enc_type,
                 tok_pair_sample_rate=1
                 ):
        super().__init__()
        self.word_embeds = nn.Embedding.from_pretrained(init_word_embedding_matrix, freeze=False)
        self.emb_dropout = nn.Dropout(emb_dropout_rate)
        self.enc_lstm = nn.LSTM(init_word_embedding_matrix.size()[-1],
                                enc_hidden_size // 2,
                                num_layers=1,
                                bidirectional=True,
                                batch_first=True)
        self.dec_lstm = nn.LSTM(enc_hidden_size,
                                dec_hidden_size // 2,
                                num_layers=1,
                                bidirectional=True,
                                batch_first=True)
        self.rnn_dropout = nn.Dropout(rnn_dropout_rate)
        self.tok_pair_sample_rate = tok_pair_sample_rate

        shaking_hidden_size = dec_hidden_size

        self.fc = nn.Linear(shaking_hidden_size, tag_size)

        # handshaking kernel
        self.handshaking_kernel = HandshakingKernel(shaking_hidden_size, shaking_type, inner_enc_type)

    def forward(self, input_ids):
        # input_ids: (batch_size, seq_len)
        # embedding: (batch_size, seq_len, emb_dim)
        embedding = self.word_embeds(input_ids)
        embedding = self.emb_dropout(embedding)
        # lstm_outputs: (batch_size, seq_len, enc_hidden_size)
        lstm_outputs, _ = self.enc_lstm(embedding)
        lstm_outputs = self.rnn_dropout(lstm_outputs)
        # lstm_outputs: (batch_size, seq_len, dec_hidden_size)
        lstm_outputs, _ = self.dec_lstm(lstm_outputs)
        lstm_outputs = self.rnn_dropout(lstm_outputs)

        # seq_len = lstm_outputs.size()[1]
        # shaking_hiddens: (batch_size, shaking_seq_len, dec_hidden_size)
        shaking_hiddens = self.handshaking_kernel(lstm_outputs)

        sampled_tok_pair_indices = None
        if self.training:
            # randomly sample segments of token pairs
            shaking_seq_len = shaking_hiddens.size()[1]
            segment_len = int(shaking_seq_len * self.tok_pair_sample_rate)
            seg_num = math.ceil(shaking_seq_len // segment_len)
            start_ind = torch.randint(seg_num, []) * segment_len
            end_ind = min(start_ind + segment_len, shaking_seq_len)
            # sampled_tok_pair_indices: (batch_size, ~segment_len) ~end_ind - start_ind <= segment_len
            sampled_tok_pair_indices = torch.arange(start_ind, end_ind)[None, :].repeat(shaking_hiddens.size()[0], 1)
            # sampled_tok_pair_indices = torch.randint(shaking_hiddens, (shaking_hiddens.size()[0], segment_len))
            sampled_tok_pair_indices = sampled_tok_pair_indices.to(shaking_hiddens.device)

            # sampled_tok_pair_indices will tell model what token pairs should be fed into fcs
            # shaking_hiddens: (batch_size, ~segment_len, hidden_size)
            index = sampled_tok_pair_indices[:, :, None].repeat(1, 1, shaking_hiddens.size()[-1])
            shaking_hiddens = shaking_hiddens.gather(1, index)

        # outputs: (batch_size, segment_len, tag_size) or (batch_size, shaking_hiddens, tag_size)
        outputs = self.fc(shaking_hiddens)
        return outputs, sampled_tok_pair_indices
