#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: czh
@email:
@date: 2022/6/27 15:25
"""
# 包括sentence_bert,SimCSE,ConSERT,Bert_Whitening,EsimCSE,CoSENT
import random

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
import torch.nn.functional as nnf
from transformers import BertModel, BertPreTrainedModel, RobertaPreTrainedModel, RobertaModel
from transformers.models.bert.modeling_bert import BertLMPredictionHead
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import batch_to_device

from nlp.metrics.triplet_distance_metric import cosin, euclidean, manhattan
from nlp.utils.util import weighted_sum, masked_softmax


class Pooler(nn.Module):
    """
    Parameter-free poolers to get the sentence embedding
    'cls': [CLS] representation with BERT/RoBERTa's MLP pooler.
    'cls_before_pooler': [CLS] representation without the original MLP pooler.
    'avg': average of the last layers' hidden states at each token.
    'avg_top2': average of the last two layers.
    'avg_first_last': average of the first and the last layers.
    """

    def __init__(self, pooler_type):
        super().__init__()
        self.pooler_type = pooler_type
        assert self.pooler_type in ["cls", "cls_before_pooler", "last-avg", "last2avg", "first-last-avg", "pooler"], \
            "unrecognized pooling type %s" % self.pooler_type

    def forward(self, attention_mask, outputs):
        last_hidden = outputs.last_hidden_state
        pooler_output = outputs.pooler_output
        hidden_states = outputs.hidden_states

        if self.pooler_type in ['cls_before_pooler', 'cls']:
            return last_hidden[:, 0]
        elif self.pooler_type == "last-avg":
            return (last_hidden * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
        elif self.pooler_type == "first-last-avg":
            first_hidden = hidden_states[0]
            last_hidden = hidden_states[-1]
            pooled_result = ((first_hidden + last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(1)
            pooled_result = pooled_result / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        elif self.pooler_type == "last2avg":
            second_last_hidden = hidden_states[-2]
            last_hidden = hidden_states[-1]
            pooled_result = ((last_hidden + second_last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(1)
            pooled_result /= attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        elif self.pooler_type == "pooler":
            return pooler_output
        else:
            raise NotImplementedError


class MLPLayer(nn.Module):
    """
    Head for getting sentence representations over RoBERTa/BERT's CLS representation.
    """

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = self.activation(x)

        return x


class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp


def cl_init(cls, config):
    """
    Contrastive learning class init function.
    """
    cls.pooler_type = cls.model_args.pooler_type
    cls.pooler = Pooler(cls.model_args.pooler_type)
    if cls.model_args.pooler_type == "cls":
        cls.mlp = MLPLayer(config)
    cls.sim = Similarity(temp=cls.model_args.temp)
    cls.init_weights()


class SoftmaxAttention(nn.Module):
    """
    Attention layer taking premises and hypotheses encoded by an RNN as input
    and computing the soft attention between their elements.
    The dot product of the encoded vectors in the premises and hypotheses is
    first computed. The softmax of the result is then used in a weighted sum
    of the vectors of the premises for each element of the hypotheses, and
    conversely for the elements of the premises.
    """

    def forward(self,
                premise_batch,
                premise_mask,
                hypothesis_batch,
                hypothesis_mask):
        """
        Args:
            premise_batch: A batch of sequences of vectors representing the
                premises in some NLI task. The batch is assumed to have the
                size (batch, sequences, vector_dim).
            premise_mask: A mask for the sequences in the premise batch, to
                ignore padding data in the sequences during the computation of
                the attention.
            hypothesis_batch: A batch of sequences of vectors representing the
                hypotheses in some NLI task. The batch is assumed to have the
                size (batch, sequences, vector_dim).
            hypothesis_mask: A mask for the sequences in the hypotheses batch,
                to ignore padding data in the sequences during the computation
                of the attention.
        Returns:
            attended_premises: The sequences of attention vectors for the
                premises in the input batch.
            attended_hypotheses: The sequences of attention vectors for the
                hypotheses in the input batch.
        """
        # Dot product between premises and hypotheses in each sequence of
        # the batch.
        similarity_matrix = premise_batch.bmm(hypothesis_batch.transpose(2, 1)
                                                              .contiguous())

        # Softmax attention weights.
        prem_hyp_attn = masked_softmax(similarity_matrix, hypothesis_mask)
        hyp_prem_attn = masked_softmax(similarity_matrix.transpose(1, 2)
                                                        .contiguous(),
                                       premise_mask)

        # Weighted sums of the hypotheses for the the premises attention,
        # and vice-versa for the attention of the hypotheses.
        attended_premises = weighted_sum(hypothesis_batch,
                                         prem_hyp_attn,
                                         premise_mask)
        attended_hypotheses = weighted_sum(premise_batch,
                                           hyp_prem_attn,
                                           hypothesis_mask)

        return attended_premises, attended_hypotheses


class SBERTModel(nn.Module):
    """
    sentence_bert,参考https://github.com/shawroad/Semantic-Textual-Similarity-Pytorch
    """

    def __init__(self,
                 bert_model_path=None,
                 num_labels=None,
                 bert_config=None,
                 object_type="classification",
                 triplet_margin=5,
                 distance_type="euclidean"
                 ):
        super(SBERTModel, self).__init__()
        if bert_model_path:
            self.bert = BertModel.from_pretrained(bert_model_path)
        else:
            self.bert = BertModel(bert_config)
        self.object_type = object_type
        self.num_labels = num_labels
        self.triplet_margin = triplet_margin
        assert object_type in ["classification", "regression", "triplet"]
        if object_type == "classification":
            self.classifier = nn.Linear(bert_config.hidden_size * 3, num_labels)
        elif object_type == "triplet":
            # loss = max(||anchor - positive|| - ||anchor - negative|| + margin, 0).
            if distance_type == "cosine":
                self.distance_func = cosin
            elif distance_type == "euclidean":
                self.distance_func = euclidean
            elif distance_type == "manhattan":
                self.distance_func = manhattan
            else:
                raise ValueError("['distance_type'] must be one of ['cosine', 'euclidean', 'manhattan']")

    @staticmethod
    def get_embedding(output, pooling_strategy):
        if pooling_strategy == 'first-last-avg':
            # 第一层和最后一层的隐层取出  然后经过平均池化
            # hidden_states列表有13个hidden_state，第一个其实是embeddings，第二个元素才是第一层的hidden_state
            first = output.hidden_states[1]
            last = output.hidden_states[-1]
            seq_length = first.size(1)  # 序列长度

            first_avg = torch.avg_pool1d(first.transpose(1, 2), kernel_size=seq_length).squeeze(-1)  # batch, hid_size
            last_avg = torch.avg_pool1d(last.transpose(1, 2), kernel_size=seq_length).squeeze(-1)  # batch, hid_size
            final_encoding = torch.cat([first_avg.unsqueeze(1), last_avg.unsqueeze(1)], dim=1).transpose(1, 2)
            final_encoding = torch.avg_pool1d(final_encoding, kernel_size=2).squeeze(-1)
            return final_encoding

        elif pooling_strategy == 'last-avg':
            sequence_output = output.last_hidden_state  # (batch_size, max_len, hidden_size)
            seq_length = sequence_output.size(1)
            final_encoding = torch.avg_pool1d(sequence_output.transpose(1, 2), kernel_size=seq_length).squeeze(-1)
            return final_encoding

        elif pooling_strategy == "cls":
            sequence_output = output.last_hidden_state
            cls = sequence_output[:, 0]  # [b,d]
            return cls

        elif pooling_strategy == "pooler":
            pooler_output = output.pooler_output  # [b,d]
            return pooler_output
        else:
            raise ValueError("'pooling_strategy' must one of [first-last-avg, last-avg, cls, pooler]")

    def forward(self, anchor_input_ids, pos_input_ids, neg_input_ids=None, pooling_strategy='cls'):
        anchor_attention_mask = torch.ne(anchor_input_ids, 0)
        pos_attention_mask = torch.ne(pos_input_ids, 0)

        anchor_output = self.bert(anchor_input_ids, anchor_attention_mask, output_hidden_states=True)
        anchor_embedding = self.get_embedding(anchor_output, pooling_strategy)

        pos_output = self.bert(pos_input_ids, pos_attention_mask, output_hidden_states=True)
        pos_embedding = self.get_embedding(pos_output, pooling_strategy)

        if self.object_type == "classification":
            diff = torch.abs(anchor_embedding - pos_embedding)
            concat_vector = torch.cat([anchor_embedding, pos_embedding, diff], dim=-1)

            logits = self.classifier(concat_vector)
        elif self.object_type == "regression":
            logits = torch.cosine_similarity(anchor_embedding, pos_embedding)
        elif self.object_type == "triplet":
            if neg_input_ids is not None:
                neg_attention_mask = torch.ne(neg_input_ids, 0)
                neg_output = self.bert(neg_input_ids, neg_attention_mask, output_hiddien_states=True)
                neg_embedding = self.get_embedding(neg_output, pooling_strategy)

                distance_pos = self.distance_func(anchor_embedding, pos_embedding)
                distance_neg = self.distance_func(anchor_embedding, neg_embedding)

                logits = nnf.relu(distance_pos - distance_neg + self.triplet_margin)
            else:
                raise ValueError(f"'{neg_input_ids}' is None")
        else:
            raise ValueError(f"not known '{self.object_type}'")

        return logits

    def encode(self, input_ids, pooling_strategy='cls'):
        attention_mask = torch.ne(input_ids, 0)
        output = self.bert(input_ids, attention_mask, output_hidden_states=True)
        embedding = self.get_embedding(output, pooling_strategy)
        return embedding


class PairSupConForSBERT(nn.Module):
    """
    清华的PairSCL用于sentence_bert架构中
    参考:https://github.com/THU-BPM/PairSCL/blob/main/bert_model.py
    """
    def __init__(self,
                 bert_model_path=None,
                 num_labels=None,
                 bert_config=None
                 ):
        super(PairSupConForSBERT, self).__init__()
        if bert_model_path:
            self.bert = BertModel.from_pretrained(bert_model_path)
        else:
            self.bert = BertModel(bert_config)
        self.num_labels = num_labels
        self.bert_config = bert_config if bert_config is not None else self.bert.config
        self.dim_mlp = self.bert_config.hidden_size
        self.softmax_attention = SoftmaxAttention()
        self.dropout = nn.Dropout(self.bert.config.hidden_dropout_prob)
        self.projection = nn.Sequential(
            nn.Linear(4*self.dim_mlp, self.dim_mlp),
            nn.ReLU())
        self.pooler = nn.Sequential(nn.Linear(4 * self.dim_mlp, self.dim_mlp),
                                    self.bert.pooler)
        self.head = nn.Sequential(nn.Linear(self.dim_mlp, self.dim_mlp),
                                  nn.ReLU(inplace=True))
        self.fc_sup = nn.Linear(self.dim_mlp, 128)
        self.fc_ce = nn.Linear(self.dim_mlp, num_labels)

    def forward(self, anchor_input_ids, pos_input_ids, pooling_strategy='cls'):
        anchor_attention_mask = torch.ne(anchor_input_ids, 0)
        pos_attention_mask = torch.ne(pos_input_ids, 0)
        # [bs, seq_len, dim]
        anchor_embed = self.bert(anchor_input_ids, anchor_attention_mask, output_hidden_states=True)[0]
        # [bs, seq_len, dim]
        pos_embed = self.bert(pos_input_ids, pos_attention_mask, output_hidden_states=True)[0]

        # [bs, seq_len, dim]
        attended_anchor, attended_pos = self.softmax_attention(anchor_embed,
                                                               anchor_attention_mask,
                                                               pos_embed,
                                                               pos_attention_mask)
        # [bs, seq_len, 4*dim]
        enhanced_anchor = torch.cat([anchor_embed,
                                     attended_anchor,
                                     anchor_embed - attended_anchor,
                                     anchor_embed * attended_anchor], dim=-1)
        enhanced_pos = torch.cat([pos_embed,
                                  attended_pos,
                                  pos_embed - attended_pos,
                                  pos_embed * attended_pos], dim=-1)
        assert enhanced_anchor.size(2) == 4*self.dim_mlp, f"{enhanced_anchor.size()}\t{self.dim_mlp}"
        assert enhanced_pos.size(2) == 4*self.dim_mlp, f"{enhanced_pos.size()}\t{self.dim_mlp}"

        projected_anchor = self.projection(enhanced_anchor)
        projected_pos = self.projection(enhanced_pos)
        pair_embeds = torch.cat([projected_anchor,
                                 projected_pos,
                                 projected_anchor-projected_pos,
                                 projected_anchor*projected_pos], dim=-1)
        pair_output = self.pooler(pair_embeds)
        feat = self.head(pair_output)
        ce_feature = nnf.normalize(self.fc_ce(feat), dim=-1)
        scl_feature = nnf.normalize(self.fc_sup(feat), dim=-1)
        return ce_feature, scl_feature

    @staticmethod
    def get_embedding(output, pooling_strategy):
        if pooling_strategy == 'first-last-avg':
            # 第一层和最后一层的隐层取出  然后经过平均池化
            # hidden_states列表有13个hidden_state，第一个其实是embeddings，第二个元素才是第一层的hidden_state
            first = output.hidden_states[1]
            last = output.hidden_states[-1]
            seq_length = first.size(1)  # 序列长度

            first_avg = torch.avg_pool1d(first.transpose(1, 2), kernel_size=seq_length).squeeze(-1)  # batch, hid_size
            last_avg = torch.avg_pool1d(last.transpose(1, 2), kernel_size=seq_length).squeeze(-1)  # batch, hid_size
            final_encoding = torch.cat([first_avg.unsqueeze(1), last_avg.unsqueeze(1)], dim=1).transpose(1, 2)
            final_encoding = torch.avg_pool1d(final_encoding, kernel_size=2).squeeze(-1)
            return final_encoding

        elif pooling_strategy == 'last-avg':
            sequence_output = output.last_hidden_state  # (batch_size, max_len, hidden_size)
            seq_length = sequence_output.size(1)
            final_encoding = torch.avg_pool1d(sequence_output.transpose(1, 2), kernel_size=seq_length).squeeze(-1)
            return final_encoding

        elif pooling_strategy == "cls":
            sequence_output = output.last_hidden_state
            cls = sequence_output[:, 0]  # [b,d]
            return cls

        elif pooling_strategy == "pooler":
            pooler_output = output.pooler_output  # [b,d]
            return pooler_output
        else:
            raise ValueError("'pooling_strategy' must one of [first-last-avg, last-avg, cls, pooler]")

    def encode(self, input_ids, pooling_strategy='cls'):
        attention_mask = torch.ne(input_ids, 0)
        output = self.bert(input_ids, attention_mask, output_hidden_states=True)
        embedding = self.get_embedding(output, pooling_strategy=pooling_strategy)
        return embedding


class PairSupConBert(nn.Module):
    """
    参考：https://github.com/amazon-science/sentence-representations/blob/main/PairSupCon/models/Transformers.py
    """
    def __init__(self, bert_name_or_path=None, config=None, num_labels=2, feat_dim=128):
        super(PairSupConBert, self).__init__()
        if bert_name_or_path is not None:
            self.bert = BertModel.from_pretrained(bert_name_or_path)
        elif config is not None:
            self.bert = BertModel(config)
        else:
            raise ValueError("The argument of [bert_name_or_path] or [config] not be empty")
        self.emb_size = self.bert.config.hidden_size
        self.num_classes = num_labels
        self.feat_dim = feat_dim

        # Sentence-Bert style input for training the pairwise classification head
        self.classify_head = nn.Linear(3 * self.emb_size, self.num_classes)

        self.contrast_head = nn.Sequential(
            nn.Linear(self.emb_size, self.emb_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.emb_size, self.feat_dim))

    def forward(self, premise_input_ids, hypothesis_input_ids, task_type):
        premise_attention_mask = torch.ne(premise_input_ids, 0)
        hypothesis_attention_mask = torch.ne(hypothesis_input_ids, 0)
        # get the mean embeddings
        bert_output_1 = self.bert.forward(premise_input_ids, premise_attention_mask, output_hidden_states=True)
        bert_output_2 = self.bert.forward(hypothesis_input_ids, hypothesis_attention_mask, output_hidden_states=True)

        mean_output_1 = self.get_mean_embeddings(bert_output_1)  # [bs, hidden_size]
        mean_output_2 = self.get_mean_embeddings(bert_output_2)  # [bs, hidden_size]

        if task_type == "classification":
            class_pred = self.classify_pred(mean_output_1, mean_output_2)
            return class_pred
        elif task_type == "contrastive":
            cnst_feat1, cnst_feat2 = self.contrast_logits(mean_output_1, mean_output_2)
            return cnst_feat1, cnst_feat2
        else:
            # PairSupCon Objective
            class_pred = self.classify_pred(mean_output_1, mean_output_2)
            cnst_feat1, cnst_feat2 = self.contrast_logits(mean_output_1, mean_output_2)
            return class_pred, cnst_feat1, cnst_feat2

    def contrast_logits(self, embd1, embd2):
        feat1 = nnf.normalize(self.contrast_head(embd1), dim=1)  # [bs, feat_dim]
        feat2 = nnf.normalize(self.contrast_head(embd2), dim=1)
        return feat1, feat2

    def classify_pred(self, embd1, embd2):
        embeddings = torch.cat([embd1, embd2, torch.abs(embd1 - embd2)], dim=-1)
        return self.classify_head(embeddings)

    @staticmethod
    def get_mean_embeddings(output):
        sequence_output = output.last_hidden_state  # (batch_size, max_len, hidden_size)
        seq_length = sequence_output.size(1)
        final_encoding = torch.avg_pool1d(sequence_output.transpose(1, 2), kernel_size=seq_length).squeeze(-1)
        return final_encoding

    def encode(self, input_ids):
        attention_mask = torch.ne(input_ids, 0)
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        embedding = self.get_mean_embeddings(bert_output)
        return embedding


class SimCSEModel(nn.Module):
    """
    参考官方代码https://github.com/princeton-nlp/SimCSE
    """
    def __init__(self, model_args,
                 bert_config,
                 bert_model_path=None,
                 pooling_strategy="cls"):
        super(SimCSEModel, self).__init__()
        self.bert_config = bert_config
        self.model_args = model_args
        self.pooler_type = pooling_strategy
        if bert_model_path:
            self.bert = BertModel.from_pretrained(bert_model_path)
        else:
            self.bert = BertModel(config=bert_config)

        self.pooler = Pooler(pooler_type=self.pooler_type)

        if self.pooler_type == 'cls':
            self.mlp = MLPLayer(config=bert_config)

        if self.model_args['do_mlm']:
            self.lm_head = BertLMPredictionHead(bert_config)

        self.sim_func = Similarity(temp=self.model_args['temp'])

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                output_attentions=None,
                mlm_input_ids=None,
                mlm_labels=None):
        ori_input_ids = input_ids
        batch_size = input_ids.size(0)
        # Number of sentences in one instance
        # 2: pair instance; 3: pair instance with a hard negative
        num_sent = input_ids.size(1)

        mlm_outputs = None
        # Flatten input for encoding
        input_ids = input_ids.view((-1, input_ids.size(-1)))  # (bs * num_sent, len)
        attention_mask = attention_mask.view((-1, attention_mask.size(-1)))  # (bs*num_sent, len)
        if token_type_ids is not None:
            token_type_ids = token_type_ids.view((-1, token_type_ids.size(-1)))  # (bs * num_sent, len)

        # get raw embeddings
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=True if self.pooler_type in ['last2avg', 'first-last-avg'] else False,
            return_dict=True
        )

        # MLM auxiliary objective
        if mlm_input_ids is not None:
            mlm_input_ids = mlm_input_ids.view((-1, mlm_input_ids.size(-1)))
            mlm_outputs = self.bert(
                mlm_input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=True if self.pooler_type in ['last2avg', 'first-last-avg'] else False,
                return_dict=True,
            )

        # Pooling
        pooler_output = self.pooler(attention_mask, outputs)
        pooler_output = pooler_output.view((batch_size, num_sent, pooler_output.size(-1)))  # (bs, num_sent, hidden)

        # If using "cls", we add an extra MLP layer
        # (same as BERT's original implementation) over the representation.
        if self.pooler_type == "cls":
            pooler_output = self.mlp(pooler_output)

        # Separate representation
        z1, z2 = pooler_output[:, 0], pooler_output[:, 1]

        # Hard negative
        if num_sent == 3:
            z3 = pooler_output[:, 2]

        # Gather all embeddings if using distributed training
        if dist.is_initialized() and self.training:
            # Gather hard negative
            if num_sent >= 3:
                z3_list = [torch.zeros_like(z3) for _ in range(dist.get_world_size())]
                dist.all_gather(tensor_list=z3_list, tensor=z3.contiguous())
                z3_list[dist.get_rank()] = z3
                z3 = torch.cat(z3_list, 0)

            # Dummy vectors for allgather
            z1_list = [torch.zeros_like(z1) for _ in range(dist.get_world_size())]
            z2_list = [torch.zeros_like(z2) for _ in range(dist.get_world_size())]
            # Allgather
            dist.all_gather(tensor_list=z1_list, tensor=z1.contiguous())
            dist.all_gather(tensor_list=z2_list, tensor=z2.contiguous())

            # Since allgather results do not have gradients, we replace the
            # current process's corresponding embeddings with original tensors
            z1_list[dist.get_rank()] = z1
            z2_list[dist.get_rank()] = z2
            # Get full batch embeddings: (bs x N, hidden)
            z1 = torch.cat(z1_list, 0)
            z2 = torch.cat(z2_list, 0)

        cos_sim = self.sim_func(z1.unsqueeze(1), z2.unsqueeze(0))
        # Hard negative
        if num_sent >= 3:
            z1_z3_cos = self.sim_func(z1.unsqueeze(1), z3.unsqueeze(0))
            cos_sim = torch.cat([cos_sim, z1_z3_cos], 1)

        labels = torch.arange(cos_sim.size(0)).long().to(self.model_args['device'])
        loss_fct = nn.CrossEntropyLoss()

        # Calculate loss with hard negatives
        if num_sent == 3:
            # Note that weights are actually logits of weights
            z3_weight = self.model_args['hard_negative_weight']
            weights = torch.tensor(
                [[0.0] * (cos_sim.size(-1) - z1_z3_cos.size(-1)) + [0.0] * i + [z3_weight] + [0.0] * (
                            z1_z3_cos.size(-1) - i - 1) for i in range(z1_z3_cos.size(-1))]
            ).to(self.model_args['device'])
            cos_sim = cos_sim + weights

        loss = loss_fct(cos_sim, labels)

        # Calculate loss for MLM
        if mlm_outputs is not None and mlm_labels is not None:
            mlm_labels = mlm_labels.view(-1, mlm_labels.size(-1))
            prediction_scores = self.lm_head(mlm_outputs.last_hidden_state)
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.bert_config.vocab_size), mlm_labels.view(-1))
            loss = loss + self.model_args['mlm_weight'] * masked_lm_loss

        return {
            'logits': cos_sim,
            'loss': loss,
            'hidden_states': outputs.hidden_states,
            'attentions': outputs.attentions,
        }

    def get_sent_embed(self,
                       input_ids=None,
                       attention_mask=None,
                       token_type_ids=None,
                       position_ids=None,
                       head_mask=None,
                       inputs_embeds=None,
                       output_attentions=None,
                       ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=True if self.pooler_type in ['last2avg', 'first-last-avg'] else False,
            return_dict=True,
        )

        pooler_output = self.pooler(attention_mask, outputs)
        if self.pooler_type == "cls" and not self.model_args['mlp_only_train']:
            pooler_output = self.mlp(pooler_output)

        return pooler_output


class ConSERTSentenceTransformer(SentenceTransformer):
    """
    参考https://github.com/zhoujx4/NLP-Series-sentence-embeddings
    """
    def __init__(self, model_name_or_path=None, modules=None, device=None, cache_folder=None, cutoff_rate=0.15,
                 close_dropout=False):
        SentenceTransformer.__init__(self, model_name_or_path, modules, device, cache_folder)
        self.cutoff_rate = cutoff_rate
        self.close_dropout = close_dropout

    def smart_batching_collate(self, batch):
        """
        Transforms a batch from a SmartBatchingDataset to a batch of tensors for the model
        Here, batch is a list of tuples: [(tokens, label), ...]
        :param batch:
            a batch from a SmartBatchingDataset
        :return:
            a batch of tensors for the model
        """
        num_texts = len(batch[0].texts)
        texts = [[] for _ in range(num_texts)]
        labels = []

        for example in batch:
            for idx, text in enumerate(example.texts):
                texts[idx].append(text)

            labels.append(example.label)

        labels = torch.tensor(labels).to(self._target_device)

        sentence_features = []
        for idx in range(num_texts):
            tokenized = self.tokenize(texts[idx])
            batch_to_device(tokenized, self._target_device)
            sentence_features.append(tokenized)

        sentence_features[1] = self.shuffle_and_cutoff(sentence_features[1])

        return sentence_features, labels

    def shuffle_and_cutoff(self, sentence_feature):
        input_ids, attention_mask = sentence_feature['input_ids'], sentence_feature['attention_mask']
        bsz, seq_len = input_ids.shape
        shuffled_input_ids = []
        cutoff_attention_mask = []
        for bsz_id in range(bsz):
            sample_mask = attention_mask[bsz_id]
            num_tokens = sample_mask.sum().int().item()
            cur_input_ids = input_ids[bsz_id]
            if 102 not in cur_input_ids:  # tip:
                indexes = list(range(num_tokens))[1:]
                random.shuffle(indexes)
                indexes = [0] + indexes  # 保证第一个位置是0
            else:
                indexes = list(range(num_tokens))[1:-1]
                random.shuffle(indexes)
                indexes = [0] + indexes + [num_tokens - 1]  # 保证第一个位置是0，最后一个位置是SEP不变
            rest_indexes = list(range(num_tokens, seq_len))
            total_indexes = indexes + rest_indexes
            shuffled_input_id = input_ids[bsz_id][total_indexes]
            # print(shuffled_input_id,indexes)
            if self.cutoff_rate > 0.0:
                sample_len = max(int(num_tokens * (1 - self.cutoff_rate)),
                                 1)  # if true_len is 32, cutoff_rate is 0.15 then sample_len is 27
                # start_id random select from (0,6)，避免删除CLS
                start_id = np.random.randint(1,
                                             high=num_tokens - sample_len + 1)
                cutoff_mask = [1] * seq_len
                for idx in range(start_id, start_id + sample_len):
                    cutoff_mask[idx] = 0  # 这些位置是0，bool之后就变成了False，而masked_fill是选择True的位置替换为value的
                cutoff_mask[0] = 0  # 避免CLS被替换
                cutoff_mask[num_tokens - 1] = 0  # 避免SEP被替换
                cutoff_mask = torch.ByteTensor(cutoff_mask).bool().to(input_ids.device)
                shuffled_input_id = shuffled_input_id.masked_fill(cutoff_mask, value=0).to(input_ids.device)
                sample_mask = sample_mask.masked_fill(cutoff_mask, value=0).to(input_ids.device)

            shuffled_input_ids.append(shuffled_input_id)
            cutoff_attention_mask.append(sample_mask)
        shuffled_input_ids = torch.vstack(shuffled_input_ids)
        cutoff_attention_mask = torch.vstack(cutoff_attention_mask)
        return {"input_ids": shuffled_input_ids, "attention_mask": cutoff_attention_mask,
                "token_type_ids": sentence_feature["token_type_ids"]}


class ConSERTV1(nn.Module):
    """
    from https://github.com/shawroad/Semantic-Textual-Similarity-Pytorch/blob/main/ConSERT/model.py
    """
    def __init__(self, args,
                 bert_config=None,
                 bert_model_path=None,
                 temperature=0.05,
                 cutoff_rate=0.15,
                 close_dropout=True):
        super(ConSERTV1, self).__init__()
        self.config = bert_config
        self.modle_args = args
        if bert_model_path:
            self.bert = BertModel.from_pretrained(bert_model_path, config=self.config)
        else:
            self.bert = BertModel(bert_config)

        self.temperature = temperature
        self.cutoff_rate = cutoff_rate
        self.close_dropout = close_dropout
        self.loss_fct = nn.CrossEntropyLoss()

    @staticmethod
    def cal_cos_sim(embedding1, embedding2):
        embedding1_norm = nnf.normalize(embedding1, p=2, dim=1)
        embedding2_norm = nnf.normalize(embedding2, p=2, dim=1)
        return torch.mm(embedding1_norm, embedding2_norm.transpose(0, 1))  # (batch_size, batch_size)

    def shuffle_and_cutoff(self, input_ids, attention_mask):
        bsz, seq_len = input_ids.size()
        shuffled_input_ids = []
        cutoff_attention_mask = []

        for bsz_id in range(bsz):
            # 1. 将当前样本的所有token打乱
            sample_mask = attention_mask[bsz_id]
            num_tokens = sample_mask.sum().item()  # 当前这个样本的真实长度

            cur_input_ids = input_ids[bsz_id]  # 当前的input_ids
            if 102 not in cur_input_ids:
                indexes = list(range(num_tokens))[1:]
                random.shuffle(indexes)  # 打乱位置
                indexes = [0] + indexes  # 保证第一个位置是0
            else:
                indexes = list(range(num_tokens))[1:-1]
                random.shuffle(indexes)
                indexes = [0] + indexes + [num_tokens - 1]  # 保证第一个位置是0，最后一个位置是SEP不变
            rest_indexes = list(range(num_tokens, seq_len))  # 这里是保证padding的位置 还是那些零 位置没变
            total_indexes = indexes + rest_indexes
            shuffled_input_id = input_ids[bsz_id][total_indexes]  # 相当于是把token打乱了。

            # 2. 随机遮挡一些token
            if self.cutoff_rate > 0.0:
                # 随机选一个开始位置 然后从开始位置遮住sample_len的长度文本
                sample_len = max(int(num_tokens * (1 - self.cutoff_rate)),
                                 1)  # if true_len is 32, cutoff_rate is 0.15 then sample_len is 27
                # start_id random select from (0,6)，避免删除CLS
                start_id = np.random.randint(1,
                                             high=num_tokens - sample_len + 1)
                cutoff_mask = [1] * seq_len
                for idx in range(start_id, start_id + sample_len):
                    cutoff_mask[idx] = 0  # 这些位置是0，bool之后就变成了False，而masked_fill是选择True的位置替换为value的

                cutoff_mask[0] = 0  # 避免CLS被替换
                cutoff_mask[num_tokens - 1] = 0  # 避免SEP被替换
                cutoff_mask = torch.ByteTensor(cutoff_mask).bool().cuda()
                shuffled_input_id = shuffled_input_id.masked_fill(cutoff_mask, value=0).cuda()
                sample_mask = sample_mask.masked_fill(cutoff_mask, value=0).cuda()
            shuffled_input_id = shuffled_input_id.view(1, -1)
            sample_mask = sample_mask.view(1, -1)

            shuffled_input_ids.append(shuffled_input_id)
            cutoff_attention_mask.append(sample_mask)

        shuffled_input_ids = torch.cat(shuffled_input_ids, dim=0)
        cutoff_attention_mask = torch.cat(cutoff_attention_mask, dim=0)
        return shuffled_input_ids, cutoff_attention_mask

    def forward(self, input_ids, attention_mask):
        s1_embedding = self.bert(input_ids, attention_mask, output_hidden_states=True).last_hidden_state[:, 0]

        input_ids2, attention_mask2 = torch.clone(input_ids), torch.clone(attention_mask)
        shuffle_input_ids, cutoff_attention_mask = self.shuffle_and_cutoff(input_ids2, attention_mask2)

        orig_attention_probs_dropout_prob = self.config.attention_probs_dropout_prob
        orig_hidden_dropout_prob = self.config.hidden_dropout_prob

        if self.close_dropout:
            self.config.attention_probs_dropout_prob = 0.0
            self.config.hidden_dropout_prob = 0.0
        s2_embedding = self.bert(shuffle_input_ids,
                                 cutoff_attention_mask,
                                 output_hidden_states=True).last_hidden_state[:, 0]

        if self.close_dropout:
            self.config.attention_probs_dropout_prob = orig_attention_probs_dropout_prob
            self.config.hidden_dropout_prob = orig_hidden_dropout_prob

        cos_sim = self.cal_cos_sim(s1_embedding, s2_embedding) / self.temperature

        batch_size = cos_sim.size(0)
        assert cos_sim.size() == (batch_size, batch_size)
        labels = torch.arange(batch_size).cuda()
        loss = self.loss_fct(cos_sim, labels)
        return {
            'logits': cos_sim,
            'loss': loss
        }

    def encode(self, input_ids, attention_mask):
        s1_embedding = self.bert(input_ids,
                                 attention_mask,
                                 output_hidden_states=True).last_hidden_state[:, 0]
        return s1_embedding


class ESimCSE(nn.Module):
    """
    https://github.com/shawroad/Semantic-Textual-Similarity-Pytorch/blob/main/ESimCSE/model.py
    """
    def __init__(self, bert_config, bert_name_or_path=None, q_size=128, dup_rate=0.32, temperature=0.05, gamma=0.99):
        super(ESimCSE, self).__init__()
        self.config = bert_config
        if bert_name_or_path:
            self.bert = BertModel.from_pretrained(bert_name_or_path, config=self.config)
        else:
            self.bert = BertModel(self.config)

        # 下面这个是为了获取上一个batch中的样本编码向量
        self.moco_config = bert_config
        self.moco_config.hidden_dropout_prob = 0.0  # 不用dropout
        self.moco_config.attention_probs_dropout_prob = 0.0  # 不用dropout
        if bert_name_or_path:
            self.moco_bert = BertModel.from_pretrained(bert_name_or_path, config=self.moco_config)
        else:
            self.moco_bert = BertModel(self.moco_config)

        self.gamma = gamma
        self.q = []  # 积攒负样本的队列
        self.q_size = q_size  # 队列长度
        self.dup_rate = dup_rate  # 数据增广的比例
        self.temperature = temperature  # 损失 热度
        self.loss_fct = nn.CrossEntropyLoss()

    @staticmethod
    def cal_cos_sim(embedding1, embedding2):
        embedding1_norm = nnf.normalize(embedding1, p=2, dim=1)
        embedding2_norm = nnf.normalize(embedding2, p=2, dim=1)
        return torch.mm(embedding1_norm, embedding2_norm.transpose(0, 1))  # (batch_size, batch_size)

    def word_repetition(self, input_ids, attention_mask):
        batch_size, seq_len = input_ids.size()
        input_ids, attention_mask = input_ids.cpu().tolist(), attention_mask.cpu().tolist()
        repetitied_input_ids, repetitied_attention_mask = [], []
        rep_seq_len = seq_len
        for batch_id in range(batch_size):
            # 一个一个序列进行处理
            sample_mask = attention_mask[batch_id]
            actual_len = sum(sample_mask)  # 计算当前序列的真实长度

            cur_input_ids = input_ids[batch_id]
            # 随机选取dup_len个token
            dup_len = random.randint(a=0, b=max(2, int(self.dup_rate * actual_len)))  # dup_rate越大  可能重复的token越多 否则越少
            dup_word_index = random.sample(list(range(1, actual_len)), k=dup_len)  # 采样出dup_len个token  然后下面进行重复
            r_input_id, r_attention_mask = [], []
            for index, word_id in enumerate(cur_input_ids):
                if index in dup_word_index:
                    r_input_id.append(word_id)
                    r_attention_mask.append(sample_mask[index])
                r_input_id.append(word_id)
                r_attention_mask.append(sample_mask[index])

            after_dup_len = len(r_input_id)
            repetitied_input_ids.append(r_input_id)
            repetitied_attention_mask.append(r_attention_mask)

            assert after_dup_len == dup_len + seq_len
            if after_dup_len > rep_seq_len:
                rep_seq_len = after_dup_len

        for i in range(batch_size):
            after_dup_len = len(repetitied_input_ids[i])
            pad_len = rep_seq_len - after_dup_len
            repetitied_input_ids[i] += [0] * pad_len
            repetitied_attention_mask[i] += [0] * pad_len

        repetitied_input_ids = torch.tensor(repetitied_input_ids, dtype=torch.long).cuda()
        repetitied_attention_mask = torch.tensor(repetitied_attention_mask, dtype=torch.long).cuda()
        return repetitied_input_ids, repetitied_attention_mask

    def forward(self, input_ids1, attention_mask1):
        # 这里直接取CLS向量 也可用其他的方式
        s1_embedding = self.bert(input_ids1, attention_mask1, output_hidden_states=True).last_hidden_state[:, 0]

        # 给当前输入的样本拷贝一个正样本
        input_ids2, attention_mask2 = torch.clone(input_ids1), torch.clone(attention_mask1)
        input_ids2, attention_mask2 = self.word_repetition(input_ids2, attention_mask2)  # 数据增广 重复某些字
        s2_embedding = self.bert(input_ids2, attention_mask2, output_hidden_states=True).last_hidden_state[:, 0]

        # 计算cos
        cos_sim = self.cal_cos_sim(s1_embedding, s2_embedding) / self.temperature  # (batch_size, batch_size)

        batch_size = cos_sim.size(0)
        assert cos_sim.size() == (batch_size, batch_size)

        negative_samples = None
        if len(self.q) > 0:
            # 从队列中取出负样本
            negative_samples = torch.cat(self.q[:self.q_size], dim=0)  # (q_size, 768)

        if len(self.q) + batch_size >= self.q_size:
            # 这个批次的样本准备加入到负样本队列  测试一下  加入进去 是否超过最大队列长度 如果超过 将队头多余的出队
            del self.q[:batch_size]

        # 将当前batch作为负样本 加入到负样本队列
        with torch.no_grad():
            self.q.append(
                self.moco_bert(input_ids1, attention_mask1, output_hidden_states=True).last_hidden_state[:, 0])

        labels = torch.arange(batch_size).cuda()
        if negative_samples is not None:
            batch_size += negative_samples.size(0)  # batch_size + 负样本的个数
            cos_sim_with_neg = self.cal_cos_sim(s1_embedding,
                                                negative_samples) / self.temperature  # 当前batch和之前负样本的cos (N, M)
            cos_sim = torch.cat([cos_sim, cos_sim_with_neg], dim=1)  # (N, N+M)

        for encoder_param, moco_encoder_param in zip(self.bert.parameters(), self.moco_bert.parameters()):
            moco_encoder_param.data = self.gamma * moco_encoder_param.data + (1. - self.gamma) * encoder_param.data

        loss = self.loss_fct(cos_sim, labels)
        return loss

    def encode(self, input_ids, attention_mask):
        s1_embedding = self.bert(input_ids, attention_mask, output_hidden_states=True).last_hidden_state[:, 0]
        return s1_embedding


class CoSENT(nn.Module):
    """
    https://github.com/shawroad/Semantic-Textual-Similarity-Pytorch/blob/main/CoSENT/model.py
    """
    def __init__(self, bert_config, model_name_or_path=None, pooler_type='cls'):
        super(CoSENT, self).__init__()
        self.config = bert_config
        if model_name_or_path:
            self.bert = BertModel.from_pretrained(model_name_or_path, config=self.config)
        else:
            self.bert = BertModel(config=self.config)

        self.pooler = Pooler(pooler_type=pooler_type)

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(input_ids, attention_mask, token_type_ids, output_hidden_states=True)
        pooler_output = self.pooler(attention_mask, outputs)

        return pooler_output


class VaSCLRoBERTa(RobertaPreTrainedModel):
    """
    参考：https://github.com/amazon-research/sentence-representations/blob/main/VaSCL/models/Transformers.py
    """
    def __init__(self, config):
        super(VaSCLRoBERTa, self).__init__(config)
        self.bert = RobertaModel(config)
        self.emb_size = self.bert.config.hidden_size
        self.feat_dim = 128

        self.contrast_head = nn.Sequential(
            nn.Linear(self.emb_size, self.emb_size, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(self.emb_size, self.feat_dim, bias=False))

    def forward(self, input_ids, attention_mask, topk=16, task_type="train"):
        if task_type == "evaluate":
            return self.get_mean_embeddings(input_ids, attention_mask)
        else:
            input_ids_1, input_ids_2 = torch.unbind(input_ids, dim=1)
            attention_mask_1, attention_mask_2 = torch.unbind(attention_mask, dim=1)

            mean_output_1 = self.get_mean_embeddings(input_ids_1, attention_mask_1)
            mean_output_2 = self.get_mean_embeddings(input_ids_2, attention_mask_2)
            inner_prod = torch.mm(mean_output_1, mean_output_1.t().contiguous())

            # estimate the neighborhood of input example
            batch_size = input_ids.shape[0]
            mask = torch.eye(batch_size, dtype=torch.bool).to(mean_output_1.device)
            inner_prod_neg = inner_prod.masked_select(~mask).view(batch_size, -1)
            topk_inner, hard_indices = torch.topk(inner_prod_neg, k=topk, dim=-1)

            cnst_feat1, cnst_feat2 = self.contrast_logits(mean_output_1, mean_output_2)
            return mean_output_1, hard_indices, cnst_feat1, cnst_feat2

    def get_mean_embeddings(self, input_ids, attention_mask):
        bert_output = self.bert.forward(input_ids=input_ids, attention_mask=attention_mask)
        attention_mask = attention_mask.unsqueeze(-1)
        mean_output = torch.sum(bert_output[0] * attention_mask, dim=1) / torch.sum(attention_mask, dim=1)
        return mean_output

    def contrast_logits(self, embd1, embd2=None):
        feat1 = nnf.normalize(self.contrast_head(embd1), dim=1)
        if embd2 is not None:
            feat2 = nnf.normalize(self.contrast_head(embd2), dim=1)
            return feat1, feat2
        else:
            return feat1


class VaSCLBERT(BertPreTrainedModel):
    """
    参考：https://github.com/amazon-research/sentence-representations/blob/main/VaSCL/models/Transformers.py
    """
    def __init__(self, config):
        super(VaSCLBERT, self).__init__(config)
        self.bert = BertModel(config)
        self.emb_size = self.bert.config.hidden_size
        self.feat_dim = 128

        self.contrast_head = nn.Sequential(
            nn.Linear(self.emb_size, self.emb_size, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(self.emb_size, self.feat_dim, bias=False))

    def forward(self, input_ids, attention_mask, topk=16, task_type="train"):
        if task_type == "evaluate":
            return self.get_mean_embeddings(input_ids)
        else:
            input_ids_1, input_ids_2 = torch.unbind(input_ids, dim=1)
            attention_mask_1, attention_mask_2 = torch.unbind(attention_mask, dim=1)

            output_1 = self.bert(input_ids_1, attention_mask_2, output_hidden_states=True)
            output_2 = self.bert(input_ids_2, attention_mask_2, output_hidden_states=True)

            mean_output_1 = self.get_mean_embeddings(output_1)
            mean_output_2 = self.get_mean_embeddings(output_2)

            inner_prod = torch.mm(mean_output_1, mean_output_1.t().contiguous())

            # estimate the neighborhood of input example
            batch_size = input_ids.shape[0]
            mask = torch.eye(batch_size, dtype=torch.bool).to(mean_output_1.device)
            inner_prod_neg = inner_prod.masked_select(~mask).view(batch_size, -1)
            topk_inner, hard_indices_unidir = torch.topk(inner_prod_neg, k=topk, dim=-1)

            cnst_feat1, cnst_feat2 = self.contrast_logits(mean_output_1, mean_output_2)
            return mean_output_1, hard_indices_unidir, cnst_feat1, cnst_feat2

    @staticmethod
    def get_mean_embeddings(output):
        sequence_output = output.last_hidden_state  # (batch_size, max_len, hidden_size)
        seq_length = sequence_output.size(1)
        final_encoding = torch.avg_pool1d(sequence_output.transpose(1, 2), kernel_size=seq_length).squeeze(-1)
        return final_encoding

    def encode(self, input_ids, attention_mask):
        # attention_mask = torch.ne(input_ids, 0)
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        embedding = self.get_mean_embeddings(bert_output)
        return embedding

    def contrast_logits(self, embd1, embd2=None):
        feat1 = nnf.normalize(self.contrast_head(embd1), dim=-1)
        if embd2 is not None:
            feat2 = nnf.normalize(self.contrast_head(embd2), dim=-1)
            return feat1, feat2
        else:
            return feat1


class CrossEncoder(nn.Module):
    def __init__(self, config, model_name_or_path=None, num_labels=2, dropout_rate=0.5):
        super(CrossEncoder, self).__init__()
        self.config = config
        if model_name_or_path is not None:
            self.bert = BertModel.from_pretrained(model_name_or_path)
        else:
            self.bert = BertModel(config=config)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.classifier = nn.Linear(self.config.hidden_size, num_labels)
        self.activate_func = nn.Softmax(dim=-1)

    def forward(self, input_ids, attention_mask, token_type_ids):
        output = self.bert(input_ids, attention_mask, token_type_ids, output_hidden_states=True)
        sequence_output = output.last_hidden_state
        cls_embed = sequence_output[:, 0]  # [b,d]
        cls_embed = self.dropout(cls_embed)
        logit = self.activate_func(self.classifier(cls_embed))
        return logit

