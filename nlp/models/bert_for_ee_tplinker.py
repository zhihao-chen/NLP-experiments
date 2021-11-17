# -*- coding: utf8 -*-
"""
======================================
    Project Name: NLP
    File Name: bert_for_ee_tplinker
    Author: czh
    Create Date: 2021/9/23
--------------------------------------
    Change Activity: 
======================================
"""
# 采用tplinker方式实现事件抽取
import logging
from abc import ABCMeta, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as func
from transformers import BertModel

from nlp.models.model_util import GraphConvLayer, Indexer, MyMatrix, SingleSourceHandshakingKernel, LayerNorm


class IEModel(nn.Module, metaclass=ABCMeta):
    def __init__(self,
                 tagger,
                 metrics_cal,
                 ner_tag_emb_config=None,
                 pos_tag_emb_config=None,
                 subwd_encoder_config=None,
                 dep_config=None
                 ):
        super().__init__()
        self.tagger = tagger
        self.metrics_cal = metrics_cal
        self.cat_hidden_size = 0
        self.word_encoder_config = None
        self.char_encoder_config = None

        # count bp steps
        self.bp_steps = 0

        # ner bio
        self.ner_tag_emb_config = ner_tag_emb_config
        if ner_tag_emb_config is not None:
            ner_tag_num = ner_tag_emb_config["ner_tag_num"]
            ner_tag_emb_dim = ner_tag_emb_config["emb_dim"]
            ner_tag_emb_dropout = ner_tag_emb_config["emb_dropout"]
            self.ner_tag_emb = nn.Embedding(ner_tag_num, ner_tag_emb_dim)
            self.ner_tag_emb_dropout = nn.Dropout(p=ner_tag_emb_dropout)
            self.cat_hidden_size += ner_tag_emb_dim

        # pos
        self.pos_tag_emb_config = pos_tag_emb_config
        if pos_tag_emb_config is not None:
            pos_tag_num = pos_tag_emb_config["pos_tag_num"]
            pos_tag_emb_dim = pos_tag_emb_config["emb_dim"]
            pos_tag_emb_dropout = pos_tag_emb_config["emb_dropout"]
            self.pos_tag_emb = nn.Embedding(pos_tag_num, pos_tag_emb_dim)
            self.pos_tag_emb_dropout = nn.Dropout(p=pos_tag_emb_dropout)
            self.pos_tag_emb_dim = pos_tag_emb_dim
            self.cat_hidden_size += pos_tag_emb_dim
            # pos_tag_hsk_ids
            if pos_tag_emb_config["hsk_emb"]:
                self.pos_tag_emb4hsk = nn.Embedding(pos_tag_num, pos_tag_emb_dim)
                self.pos_tag_emb4hsk_dropout = nn.Dropout(p=pos_tag_emb_dropout)

        # subword_encoder
        self.subwd_encoder_config = subwd_encoder_config
        if subwd_encoder_config is not None:
            bert_path = subwd_encoder_config["pretrained_model_path"]
            bert_finetune = subwd_encoder_config["finetune"]
            self.use_last_k_layers_bert = subwd_encoder_config["use_last_k_layers"]
            self.bert = BertModel.from_pretrained(bert_path)
            if not bert_finetune:  # if train without finetuning bert
                for param in self.bert.parameters():
                    param.requires_grad = False
            self.cat_hidden_size += self.bert.config.hidden_size

        # dependencies
        self.dep_config = dep_config
        if dep_config is not None:
            self.dep_type_num = dep_config["dep_type_num"]
            self.dep_type_emb_dim = dep_config["dep_type_emb_dim"]
            dep_type_emb_dropout = dep_config["emb_dropout"]
            self.dep_type_emb = nn.Embedding(self.dep_type_num, self.dep_type_emb_dim)
            self.dep_type_emb_dropout = nn.Dropout(p=dep_type_emb_dropout)
            # dep_hnt_matrix
            if dep_config["hnt_emb"]:
                self.dep_type_hnt_emb = nn.Embedding(self.dep_type_num * 2, self.dep_type_emb_dim)
                self.dep_type_hnt_emb_dropout = nn.Dropout(p=dep_type_emb_dropout)

            self.use_gcn4deprel = dep_config["gcn"]
            if self.use_gcn4deprel:
                # GCN
                dep_gcn_dim = dep_config["gcn_dim"]
                dep_gcn_dropout = dep_config["gcn_dropout"]
                dep_gcn_layer_num = dep_config["gcn_layer_num"]

                # aggregate fc
                self.aggr_fc4gcn = nn.Linear(self.cat_hidden_size, dep_gcn_dim)
                self.gcn_layers = nn.ModuleList()
                self.dep_gcn_dropout = nn.Dropout(dep_gcn_dropout)
                for _ in range(dep_gcn_layer_num):
                    self.gcn_layers.append(GraphConvLayer(self.dep_type_emb_dim, dep_gcn_dim, "avg"))
                    self.cat_hidden_size += dep_gcn_dim

    def get_base_features(self,
                          char_input_ids=None,
                          word_input_ids=None,
                          subword_input_ids=None,
                          attention_mask=None,
                          token_type_ids=None,
                          ner_tag_ids=None,
                          pos_tag_ids=None,
                          dep_adj_matrix=None,
                          pos_tag_hsk_ids=None,
                          dep_hnt_matrix=None,
                          ):

        # features
        features = []
        feature_dict = {}

        # ner tag
        if self.ner_tag_emb_config is not None:
            ner_tag_embeddings = self.ner_tag_emb(ner_tag_ids)
            ner_tag_embeddings = self.ner_tag_emb_dropout(ner_tag_embeddings)
            features.append(ner_tag_embeddings)
            feature_dict["ner_tag_embeddings"] = ner_tag_embeddings

        # pos tag
        if self.pos_tag_emb_config is not None:
            pos_tag_embeddings = self.pos_tag_emb(pos_tag_ids)
            pos_tag_embeddings = self.pos_tag_emb_dropout(pos_tag_embeddings)
            features.append(pos_tag_embeddings)
            feature_dict["pos_tag_embeddings"] = pos_tag_embeddings
            # pos_tag_hsk_ids
            if self.pos_tag_emb_config["hsk_emb"]:
                pos_tag_embeddings4hsk = self.pos_tag_emb4hsk(pos_tag_hsk_ids)
                pos_tag_embeddings4hsk = self.pos_tag_emb4hsk_dropout(pos_tag_embeddings4hsk)
                feature_dict["pos_tag_embeddings4hsk"] = pos_tag_embeddings4hsk

        # char
        if self.char_encoder_config is not None:
            # char_input_ids: (batch_size, seq_len * max_char_num_in_subword)
            # char_input_emb/char_hiddens: (batch_size, seq_len * max_char_num_in_subword, char_emb_dim)
            # char_conv_output: (batch_size, seq_len, char_emb_dim)
            char_input_emb = self.char_emb(char_input_ids)
            char_input_emb = self.char_emb_dropout(char_input_emb)
            char_hiddens, _ = self.char_lstm_l1(char_input_emb)
            char_hiddens, _ = self.char_lstm_l2(self.char_lstm_dropout(char_hiddens))
            char_conv_output = self.char_cnn(char_hiddens.permute(0, 2, 1)).permute(0, 2, 1)
            features.append(char_conv_output)
            feature_dict["char_conv_output"] = char_conv_output

        # word
        if self.word_encoder_config is not None:
            # word_input_ids: (batch_size, seq_len)
            # word_input_emb/word_hiddens: batch_size_train, seq_len, word_emb_dim)
            word_input_emb = self.word_emb(word_input_ids)
            word_input_emb = self.word_emb_dropout(word_input_emb)
            features.append(word_input_emb)
            word_fts = self.aggr_fc4fusion_word_level_fts(torch.cat(features, dim=-1))
            word_hiddens, _ = self.word_lstm_l1(word_fts)
            word_hiddens, _ = self.word_lstm_l2(self.word_lstm_dropout(word_hiddens))
            features.append(word_hiddens)
            feature_dict["word_hiddens"] = word_hiddens

        # subword
        if self.subwd_encoder_config is not None:
            # subword_input_ids, attention_mask, token_type_ids: (batch_size, seq_len)
            context_outputs = self.bert(input_ids=subword_input_ids,
                                        attention_mask=attention_mask,
                                        token_type_ids=token_type_ids,
                                        return_dict=True,
                                        output_hidden_states=True,
                                        output_attentions=True)
            # self.attn_tuple = context_outputs[3] if len(context_outputs) >= 4 else None  # noqa
            self.attn_tuple = context_outputs.attentions  # noqa
            # hidden_states = context_outputs[2]
            hidden_states = context_outputs.hidden_states
            # subword_hiddens: (batch_size, seq_len, hidden_size)
            subword_hiddens = torch.mean(torch.stack(list(hidden_states)[-self.use_last_k_layers_bert:], dim=0),
                                         dim=0)
            features.append(subword_hiddens)
            feature_dict["subword_hiddens"] = subword_hiddens

        # dependencies
        if self.dep_config is not None:
            # dep_adj_matrix: (batch_size, seq_len, seq_len)
            # -> deprel_embeddings: (batch_size, seq_len, seq_len, dep_type_emb_dim)
            deprel_embeddings = self.dep_type_emb(dep_adj_matrix)
            deprel_embeddings = self.dep_type_emb_dropout(deprel_embeddings)
            feature_dict["deprel_embeddings"] = deprel_embeddings
            # dep_hnt_matrix
            if self.dep_config["hnt_emb"]:
                deprel_hnt_embeddings = self.dep_type_hnt_emb(dep_hnt_matrix)
                deprel_hnt_embeddings = self.dep_type_emb_dropout(deprel_hnt_embeddings)
                feature_dict["deprel_hnt_embeddings"] = deprel_hnt_embeddings

            if self.use_gcn4deprel:
                deprel_embeddings_trans = torch.transpose(deprel_embeddings, 1, 2)  # (multi -> one) => (one: multi)
                # deprel_embeddings: (batch_size, seq_len, seq_len, dep_emb_dim)
                weight_adj = self.dep_type_emb_dropout(deprel_embeddings_trans)
                gcn_output = self.aggr_fc4gcn(torch.cat(features, dim=-1))

                gcn_output_list = []
                for gcn_l in self.gcn_layers:
                    weight_adj, gcn_output = gcn_l(weight_adj, gcn_output)  # [batch, seq, dim]
                    gcn_output = self.dep_gcn_dropout(gcn_output)
                    weight_adj = self.dep_gcn_dropout(weight_adj)
                    gcn_output_list.append(gcn_output)
                features.extend(gcn_output_list)
                feature_dict["gcn_output_list"] = gcn_output_list

        # concatenated features
        # concatenated_hiddens: (batch_size, seq_len, concatenated_size)
        cat_hiddens = torch.cat(features, dim=-1)

        return cat_hiddens, feature_dict

    def forward(self):
        if self.training:
            self.bp_steps += 1

    def generate_batch(self, batch_data):
        assert len(batch_data) > 0

        batch_dict = {
            "sample_list": [sample for sample in batch_data],
        }
        seq_length = len(batch_data[0]["features"]["tok2char_span"])

        if self.subwd_encoder_config is not None:
            subword_input_ids_list = []
            attention_mask_list = []
            token_type_ids_list = []
            for sample in batch_data:
                features = sample["features"]
                subword_input_ids_list.append(torch.LongTensor(features["subword_input_ids"]
                                                               if "subword_input_ids" in features
                                                               else features["word_input_ids"]))
                attention_mask_list.append(torch.LongTensor(features["attention_mask"]))
                token_type_ids_list.append(torch.LongTensor(features["token_type_ids"]))
            batch_dict["subword_input_ids"] = torch.stack(subword_input_ids_list, dim=0)
            batch_dict["attention_mask"] = torch.stack(attention_mask_list, dim=0)
            batch_dict["token_type_ids"] = torch.stack(token_type_ids_list, dim=0)

        if self.word_encoder_config is not None:
            word_input_ids_list = [torch.LongTensor(sample["features"]["word_input_ids"]) for sample in batch_data]
            batch_dict["word_input_ids"] = torch.stack(word_input_ids_list, dim=0)

        if self.char_encoder_config is not None:
            char_input_ids_list = [torch.LongTensor(sample["features"]["char_input_ids"]) for sample in batch_data]
            batch_dict["char_input_ids"] = torch.stack(char_input_ids_list, dim=0)

        if self.ner_tag_emb_config is not None:
            ner_tag_ids_list = [torch.LongTensor(sample["features"]["ner_tag_ids"]) for sample in batch_data]
            batch_dict["ner_tag_ids"] = torch.stack(ner_tag_ids_list, dim=0)

        if self.pos_tag_emb_config is not None:
            pos_tag_ids_list = [torch.LongTensor(sample["features"]["pos_tag_ids"]) for sample in batch_data]
            batch_dict["pos_tag_ids"] = torch.stack(pos_tag_ids_list, dim=0)
            if self.pos_tag_emb_config["hsk_emb"]:
                pos_tag_points_batch = [sample["features"]["pos_tag_points"] for sample in batch_data]
                batch_dict["pos_tag_hsk_ids"] = Indexer.points2shaking_seq_batch(pos_tag_points_batch, seq_length)

        if self.dep_config is not None:
            dep_matrix_points_batch = [sample["features"]["dependency_points"] for sample in batch_data]
            batch_dict["dep_adj_matrix"] = Indexer.points2matrix_batch(dep_matrix_points_batch, seq_length)
            if self.dep_config["hnt_emb"]:
                dep_matrix_hnt_points_batch = [sample["features"]["deprel_points_hnt"] for sample in batch_data]
                batch_dict["dep_hnt_matrix"] = Indexer.points2matrix_batch(dep_matrix_hnt_points_batch, seq_length)

        # >>>>>>>>>>>>>>>>>>>>>> add annotation points >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        self.tagger.tag(batch_data)
        # batch_dict["golden_tags"] need to be set by inheritors
        return batch_dict

    @abstractmethod
    def pred_output2pred_tag(self, pred_output):
        """
        output to tag id
        :param pred_output: the output of the forward function
        :return:
        """
        pass

    @abstractmethod
    def get_metrics(self, pred_outputs, gold_tags):
        """
        :param pred_outputs: the outputs of the forward function
        :param gold_tags: golden tags from batch_dict["gold_tags"]
        :return:
        """
        pass


class TpLinkerForEE(IEModel):
    def __init__(self,
                 tagger=None,
                 metrics_cal=None,
                 handshaking_kernel_config=None,
                 ent_dim=None,
                 rel_dim=None,
                 use_attns4rel=False,
                 span_len_emb_dim=0,
                 emb_ent_info2rel=False,
                 golden_ent_cla_guide=False,
                 loss_func="bce_loss",  # mce_loss, pred_threshold=0.
                 loss_weight_recover_steps=None,
                 loss_weight=.5,
                 init_loss_weight=.5,
                 pred_threshold=0.,
                 **kwargs,
                 ):
        super().__init__(tagger, metrics_cal, **kwargs)
        self.ent_tag_size, self.rel_tag_size = tagger.get_tag_size()
        self.loss_func = loss_func
        self.loss_weight_recover_steps = loss_weight_recover_steps
        self.metrics_cal = metrics_cal
        self.loss_weight = loss_weight
        self.init_loss_weight = init_loss_weight
        self.pred_threshold = pred_threshold

        inp_dim = self.cat_hidden_size

        self.aggr_fc4ent_hsk = nn.Linear(inp_dim, ent_dim)
        self.aggr_fc4rel_hsk = nn.Linear(inp_dim, rel_dim)

        # handshaking kernel
        ent_shaking_type = handshaking_kernel_config["ent_shaking_type"]
        ent_dist_emb_dim = handshaking_kernel_config.get("ent_dist_emb_dim", -1)
        rel_shaking_type = handshaking_kernel_config["rel_shaking_type"]
        rel_dist_emb_dim = handshaking_kernel_config.get("rel_dist_emb_dim", -1)

        self.ent_handshaking_kernel = SingleSourceHandshakingKernel(ent_dim,
                                                                    ent_shaking_type,
                                                                    only_look_after=True,
                                                                    distance_emb_dim=ent_dist_emb_dim
                                                                    )
        self.rel_handshaking_kernel = SingleSourceHandshakingKernel(rel_dim,
                                                                    rel_shaking_type,
                                                                    only_look_after=False,
                                                                    distance_emb_dim=rel_dist_emb_dim
                                                                    )

        self.use_attns4rel = use_attns4rel
        if use_attns4rel:
            self.attns_fc = nn.Linear(self.bert.config.num_hidden_layers * self.bert.config.num_attention_heads,
                                      rel_dim,
                                      )

        self.span_len_emb_dim = span_len_emb_dim
        if span_len_emb_dim > 0:
            self.span_len_emb = nn.Embedding(512, span_len_emb_dim)
            ent_dim += span_len_emb_dim
            self.span_len_seq = None  # for cache

        self.emb_ent_info2rel = emb_ent_info2rel
        self.golden_ent_cla_guide = golden_ent_cla_guide
        if emb_ent_info2rel:
            # span type: 0, 1 (spans end with this token, or spans start with this token)
            span_type_emb_dim = 32
            self.span_type_emb = nn.Embedding(2, span_type_emb_dim)
            self.span_type_matrix = None

            # ent_tag_emb
            ent_tag_emb_dim = 32
            self.ent_tag_emb = nn.Linear(self.ent_tag_size, ent_tag_emb_dim)
            tp_dim = 2 * (ent_dim + ent_tag_emb_dim + span_type_emb_dim)
            self.cln4rel_guide = LayerNorm(rel_dim, tp_dim, conditional=True)

        # decoding fc
        if self.pos_tag_emb_config is not None and self.pos_tag_emb_config["hsk_emb"]:
            ent_dim += self.pos_tag_emb_dim
        if self.dep_config is not None:
            rel_dim += self.dep_type_emb_dim
            if self.dep_config["hnt_emb"]:
                rel_dim += self.dep_type_emb_dim
        self.ent_fc = nn.Linear(ent_dim, self.ent_tag_size)
        self.rel_fc = nn.Linear(rel_dim, self.rel_tag_size)

    def generate_batch(self, batch_data):
        seq_length = len(batch_data[0]["features"]["tok2char_span"])
        batch_dict = super(TpLinkerForEE, self).generate_batch(batch_data)
        # tags
        batch_ent_points = [sample["ent_points"] for sample in batch_data]
        batch_rel_points = [sample["rel_points"] for sample in batch_data]
        batch_dict["golden_tags"] = [Indexer.points2multilabel_shaking_seq_batch(batch_ent_points,
                                                                                 seq_length,
                                                                                 self.ent_tag_size,
                                                                                 ),
                                     Indexer.points2multilabel_matrix_batch(batch_rel_points,
                                                                            seq_length,
                                                                            self.rel_tag_size,
                                                                            ),
                                     ]
        if self.golden_ent_cla_guide:
            batch_dict["golden_ent_class_guide"] = Indexer.points2multilabel_shaking_seq_batch(batch_ent_points,
                                                                                               seq_length,
                                                                                               self.ent_tag_size,
                                                                                               )
        return batch_dict

    def get_tok_pre(self, ent_hs_hiddens, ent_class_guide):
        """
        :param ent_hs_hiddens: (batch_size, shaking_seq_len, ent_hidden_size)
        :param ent_class_guide: (batch_size, shaking_seq_len, ent_type_size)
        :return: tok_hiddens: (batch_size, seq_len, ent_type_size + ent_hidden_size)
        """
        ent_class_guide = ent_class_guide.float()
        # ent_class_matrix: (batch_size, seq_len, seq_len, ent_type_size)
        ent_class_matrix = MyMatrix.mirror(ent_class_guide)
        # ent_hiddens_matrix: (batch_size, seq_len, seq_len, ent_hidden_size)
        ent_hiddens_matrix = MyMatrix.mirror(ent_hs_hiddens)

        batch_size, seq_len, _, _ = ent_hiddens_matrix.size()

        # span type: 0 or 1 (spans end with this token, or spans start with this token)
        if self.span_type_matrix is None or \
                self.span_type_matrix.size()[0] != batch_size or \
                self.span_type_matrix.size()[1] != seq_len:
            self.span_type_matrix = torch.ones([seq_len, seq_len]).to(ent_hs_hiddens.device).triu().long()[None, :, :]
            self.span_type_matrix = self.span_type_matrix.repeat(batch_size, 1, 1)
        span_type_emb = self.span_type_emb(self.span_type_matrix)

        # ent_type_num_at_this_span: (batch_size, seq_len, seq_len, 1)
        ent_type_num_at_this_span = torch.sum(ent_class_matrix, dim=-1)[:, :, :, None]

        # weight4rel: (batch_size, seq_len, seq_len, 1)
        # weight_at_this_span = ent_type_num_at_this_span * 100 + 1
        # weight4rel = weight_at_this_span / torch.sum(weight_at_this_span, dim=-2)[:, :, None, :]
        weight4rel = func.softmax(ent_type_num_at_this_span, dim=-2)

        # boundary_tok_pre: (batch_size, seq_len, ent_hidden_size + ent_type_size + span_type_emb_dim)
        span_pre = torch.cat([ent_hiddens_matrix,
                              self.ent_tag_emb(ent_class_matrix),
                              span_type_emb],
                             dim=-1)
        boundary_tok_pre = torch.sum(span_pre * weight4rel, dim=-2)

        return boundary_tok_pre

    def get_ent_guide4rel(self, ent_hs_hiddens, ent_class_guide):
        """
        :param ent_hs_hiddens: (batch_size, shaking_seq_len, ent_hidden_size)
        :param ent_class_guide: (batch_size, shaking_seq_len, ent_type_size)
        :return: ent_guide4rel: (batch_size, seq_len, seq_len, 2 * (ent_type_size + ent_hidden_size))
        """
        tok_pre = self.get_tok_pre(ent_hs_hiddens, ent_class_guide)
        seq_len = tok_pre.size()[1]
        boundary_tok_pre_repeat = tok_pre[:, :, None, :].repeat(1, 1, seq_len, 1)
        boundary_tok_inter_pre = torch.cat([boundary_tok_pre_repeat,
                                            boundary_tok_pre_repeat.permute(0, 2, 1, 3)],
                                           dim=-1)
        return boundary_tok_inter_pre

    def forward(self, **kwargs):
        super(TpLinkerForEE, self).forward()
        if self.emb_ent_info2rel and self.golden_ent_cla_guide:
            del kwargs["golden_ent_class_guide"]

        inp_hiddens, feature_dict = self.get_base_features(**kwargs)
        batch_size, seq_len, _ = inp_hiddens.size()

        ent_hiddens = self.aggr_fc4ent_hsk(inp_hiddens)
        rel_hiddens = self.aggr_fc4rel_hsk(inp_hiddens)

        # ent_hs_hiddens: (batch_size, shaking_seq_len, hidden_size)
        # rel_hs_hiddens: (batch_size, seq_len, seq_len, hidden_size)
        ent_hs_hiddens = self.ent_handshaking_kernel(ent_hiddens)
        rel_hs_hiddens = self.rel_handshaking_kernel(rel_hiddens)

        # attentions: (batch_size, layers * heads, seg_len, seq_len)
        if self.use_attns4rel:
            if self.attn_tuple is None:
                logging.warning("Failed to get bert attention tuple! "
                                "Can not use attentions for relation matrix. "
                                "Please add output_attentions=true to your config.json!")
            attns = torch.cat(self.attn_tuple, dim=1).permute(0, 2, 3, 1)
            attns = self.attns_fc(attns)
            rel_hs_hiddens += attns

        # span len
        if self.span_len_emb_dim > 0:
            if self.span_len_seq is None or \
                    self.span_len_seq.size()[0] != batch_size or \
                    self.span_len_seq.size()[1] != seq_len:
                t = torch.arange(0, seq_len).to(ent_hs_hiddens.device)[:, None].repeat(1, seq_len)
                span_len_matrix = torch.abs(t - t.permute(1, 0)).long()[None, :, :].repeat(batch_size, 1, 1)
                self.span_len_seq = MyMatrix.upper_reg2seq(span_len_matrix[:, :, :, None]).view(batch_size, -1)
            span_len_emb = self.span_len_emb(self.span_len_seq)
            ent_hs_hiddens = torch.cat([ent_hs_hiddens, span_len_emb], dim=-1)

        # pos_tag_embeddings4hsk
        if self.pos_tag_emb_config is not None and self.pos_tag_emb_config["hsk_emb"]:
            ent_hs_hiddens = torch.cat([ent_hs_hiddens, feature_dict["pos_tag_embeddings4hsk"]], dim=-1)
        pred_ent_output = self.ent_fc(ent_hs_hiddens)

        # embed entity info into relation hiddens
        if self.emb_ent_info2rel:
            # ent_class_guide: (batch_size, shaking_seq_len, ent_type_size)
            if not self.training or not self.golden_ent_cla_guide:
                ent_class_guide = self.pred_output2pred_tag(pred_ent_output)
                boundary_tok_inter_pre = self.get_ent_guide4rel(ent_hs_hiddens, ent_class_guide)
                rel_hs_hiddens = self.cln4rel_guide(rel_hs_hiddens, boundary_tok_inter_pre)

        if self.dep_config is not None:
            rel_hs_hiddens = torch.cat([rel_hs_hiddens, feature_dict["deprel_embeddings"]], dim=-1)
            # deprel_hnt_embeddings
            if self.dep_config["hnt_emb"]:
                rel_hs_hiddens = torch.cat([rel_hs_hiddens, feature_dict["deprel_hnt_embeddings"]], dim=-1)
        pred_rel_output = self.rel_fc(rel_hs_hiddens)
        return pred_ent_output, pred_rel_output

    def pred_output2pred_tag(self, pred_output):
        tag = (pred_output > self.pred_threshold).long()
        return tag

    def get_metrics(self, pred_outputs, gold_tags):
        ent_pred_out, rel_pred_out, ent_gold_tag, rel_gold_tag = pred_outputs[0], pred_outputs[1], gold_tags[0], \
                                                                 gold_tags[1]
        ent_pred_tag = self.pred_output2pred_tag(ent_pred_out)
        rel_pred_tag = self.pred_output2pred_tag(rel_pred_out)

        # weights
        total_steps = self.loss_weight_recover_steps + 1  # + 1 avoid division by zero error
        current_step = self.bp_steps

        init_ent_w, init_rel_w = self.init_loss_weight, 1 - self.init_loss_weight
        stable_ent_w, stable_rel_w = self.loss_weight, 1 - self.loss_weight
        if init_ent_w > stable_ent_w:
            # decrease to stable in total_steps
            dif = init_ent_w - stable_ent_w
            step_weight = dif * current_step / total_steps
            w_ent = max(init_ent_w - step_weight, stable_ent_w)
            w_rel = min(init_rel_w + step_weight, stable_rel_w)
        else:
            # increase to stable in total_steps
            dif = stable_ent_w - init_ent_w
            step_weight = dif * current_step / total_steps
            w_ent = min(init_ent_w + step_weight, stable_ent_w)
            w_rel = max(init_rel_w - step_weight, stable_rel_w)

        # loss function
        loss_func = None
        if self.loss_func == "bce_loss":
            loss_func = self.metrics_cal.bce_loss
        elif self.loss_func == "mce_loss":
            def loss_func(pred_out, gold_tag):
                return self.metrics_cal.multilabel_categorical_crossentropy(pred_out, gold_tag, self.bp_steps)

        loss = w_ent * loss_func(ent_pred_out, ent_gold_tag) + w_rel * loss_func(rel_pred_out, rel_gold_tag)

        return {
            "loss": loss,
            "ent_seq_acc": self.metrics_cal.get_tag_seq_accuracy(ent_pred_tag, ent_gold_tag),
            "rel_seq_acc": self.metrics_cal.get_tag_seq_accuracy(rel_pred_tag, rel_gold_tag),
        }
