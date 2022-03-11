# -*- coding: utf8 -*-
"""
======================================
    Project Name: NLP
    File Name: bert_for_ner
    Author: czh
    Create Date: 2021/8/11
--------------------------------------
    Change Activity: 
======================================
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as func
from transformers import (BertPreTrainedModel, BertModel, AlbertModel, AlbertPreTrainedModel, RoFormerModel)

from nlp.losses.loss import LabelSmoothingCrossEntropy, FocalLoss
from nlp.layers.crf import CRF
from nlp.layers.linears import PoolerStartLogits, PoolerEndLogits, MultiNonLinearClassifier, BERTTaggerClassifier
from nlp.layers.layer import LabelFusionLayerForToken
from nlp.models.nezha import NeZhaModel
from nlp.layers.global_pointer import GlobalPointer, EfficientGlobalPointer
from nlp.losses.loss import global_pointer_crossentropy


class BertMRCForNER(nn.Module):
    # https://github.com/qiufengyuyi/sequence_tagging/blob/master/models/bert_mrc.py
    def __init__(self, bert_config, encoder_model_path=None, num_labels=2, dropout_rate=0.5, soft_label=False):
        super(BertMRCForNER, self).__init__()
        if encoder_model_path:
            self.encoder = BertModel.from_pretrained(encoder_model_path)
        else:
            self.encoder = BertModel(bert_config)
        self.soft_label = soft_label
        self.dropout_layer = nn.Dropout(dropout_rate)
        self.start_fc = PoolerStartLogits(bert_config.hidden_size, num_labels)
        if self.soft_label:
            self.end_fc = PoolerEndLogits(bert_config.hidden_size + num_labels, num_labels)
        else:
            self.end_fc = PoolerEndLogits(bert_config.hidden_size + 1, num_labels)
        self.init_weights()

    def forward(self, input_ids, attention_mask, token_type_ids):
        encoder_output = self.encoder(input_ids, attention_mask, token_type_ids)
        last_hidden_state = encoder_output.last_hidden_state
        sequence_out = self.dropout_layer(last_hidden_state)

        start_logits = self.start_fc(sequence_out)
        label_logits = func.softmax(start_logits, -1)
        if not self.soft_label:
            label_logits = torch.argmax(label_logits, -1).unsqueeze(2).float()

        end_logits = self.end_fc(sequence_out, label_logits)
        # outputs = (start_logits, end_logits,) + outputs[2:]
        outputs = {"start_logits": start_logits, "end_logits": end_logits}
        return outputs


class BertQueryNER(BertPreTrainedModel):
    # https://github.com/ShannonAI/mrc-for-flat-nested-ner
    def __init__(self, config, mrc_dropout=0.1, classifier_intermediate_hidden_size=1024):
        super(BertQueryNER, self).__init__(config)
        self.bert = BertModel(config)

        self.start_outputs = nn.Linear(config.hidden_size, 1)
        self.end_outputs = nn.Linear(config.hidden_size, 1)
        self.span_embedding = MultiNonLinearClassifier(
            config.hidden_size * 2, 1, mrc_dropout,
            intermediate_hidden_size=classifier_intermediate_hidden_size
        )

        self.hidden_size = config.hidden_size

        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        """
        Args:
            input_ids: bert input tokens, tensor of shape [seq_len]
            token_type_ids: 0 for query, 1 for context, tensor of shape [seq_len]
            attention_mask: attention mask, tensor of shape [seq_len]
        Returns:
            start_logits: start/non-start probs of shape [seq_len]
            end_logits: end/non-end probs of shape [seq_len]
            match_logits: start-end-match probs of shape [seq_len, 1]
        """

        bert_outputs = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)

        sequence_heatmap = bert_outputs[0]  # [batch, seq_len, hidden]
        batch_size, seq_len, hid_size = sequence_heatmap.size()

        start_logits = self.start_outputs(sequence_heatmap).squeeze(-1)  # [batch, seq_len, 1]
        end_logits = self.end_outputs(sequence_heatmap).squeeze(-1)  # [batch, seq_len, 1]

        # for every position $i$ in sequence, should concate $j$ to
        # predict if $i$ and $j$ are start_pos and end_pos for an entity.
        # [batch, seq_len, seq_len, hidden]
        start_extend = sequence_heatmap.unsqueeze(2).expand(-1, -1, seq_len, -1)
        # [batch, seq_len, seq_len, hidden]
        end_extend = sequence_heatmap.unsqueeze(1).expand(-1, seq_len, -1, -1)
        # [batch, seq_len, seq_len, hidden*2]
        span_matrix = torch.cat([start_extend, end_extend], 3)
        # [batch, seq_len, seq_len]
        span_logits = self.span_embedding(span_matrix).squeeze(-1)
        outputs = {'start_logits': start_logits, 'end_logits': end_logits, 'span_logits': span_logits}
        return outputs


class BertTagger(BertPreTrainedModel):
    # https://github.com/ShannonAI/mrc-for-flat-nested-ner
    def __init__(self, config, num_labels,
                 hidden_dropout_prob=0.1,
                 classifier_dropout=0.1,
                 classifier_sign="multi_nonlinear",
                 classifier_act_func="gelu",
                 classifier_intermediate_hidden_size=1024):
        super(BertTagger, self).__init__(config)
        self.bert = BertModel(config)

        self.num_labels = num_labels
        self.hidden_size = config.hidden_size
        self.dropout = nn.Dropout(hidden_dropout_prob)
        if classifier_sign == "multi_nonlinear":
            self.classifier = BERTTaggerClassifier(self.hidden_size, self.num_labels,
                                                   classifier_dropout,
                                                   act_func=classifier_act_func,
                                                   intermediate_hidden_size=classifier_intermediate_hidden_size)
        else:
            self.classifier = nn.Linear(self.hidden_size, self.num_labels)

        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None,):
        last_bert_layer, pooled_output = self.bert(input_ids, token_type_ids, attention_mask)
        last_bert_layer = last_bert_layer.view(-1, self.hidden_size)
        last_bert_layer = self.dropout(last_bert_layer)
        logits = self.classifier(last_bert_layer)
        outputs = {'logits': logits}
        return outputs


class LearForNer(nn.Module):
    # 对MRC for ner的改进
    # https://github.com/akeepers/lear
    def __init__(self, bert_config,
                 num_label,
                 encoder_model_path=None,
                 dropout=0.1,
                 classifier_intermediate_hidden_size=1024,
                 label_ann_vocab_size=10000,
                 label_embedding_file=None,
                 soft_label=False,
                 use_label_embedding=False,
                 use_span_embedding=False,
                 bert_frozen=False,
                 do_attn=True,
                 do_add=False,
                 do_average_pooling=False,
                 device=torch.device('cpu')
                 ):
        super(LearForNer, self).__init__()
        if encoder_model_path:
            self.encoder = BertModel.from_pretrained(encoder_model_path)
        else:
            self.encoder = BertModel(bert_config)

        self.do_attn = do_attn
        self.do_add = do_add
        self.do_average_pooling = do_average_pooling

        self.dropout_layer = nn.Dropout(dropout)
        self.num_label = num_label
        self.hidden_size = bert_config.hidden_size

        # self.start_classifier = PoolerStartLogits(bert_config.hidden_size, num_label)
        # if soft_label:
        #     self.end_classifier = PoolerEndLogits(bert_config.hidden_size+num_label, num_label)
        # else:
        #     self.end_classifier = PoolerEndLogits(bert_config.hidden_size+1, num_label)
        self.start_classifier = nn.Linear(self.hidden_size, self.num_label)
        self.end_classifier = nn.Linear(self.hidden_size, self.num_label)
        self.label_fusing_layer = LabelFusionLayerForToken(
            hidden_size=self.hidden_size,
            label_num=self.num_label,
            label_emb_size=200 if use_label_embedding else self.hidden_size
           )
        self.use_label_embedding = use_label_embedding
        if use_label_embedding:
            self.label_ann_vocab_size = label_ann_vocab_size
            self.label_embedding_layer = nn.Embedding(label_ann_vocab_size, 200)
            glove_embs = torch.from_numpy(np.load(label_embedding_file, allow_pickle=True)).to(device)
            self.label_embedding_layer.weight.data.copy_(glove_embs)
        self.use_span_embedding = use_span_embedding
        if use_span_embedding:
            self.span_embedding = MultiNonLinearClassifier(
                self.hidden_size * 2, self.num_label, dropout,
                intermediate_hidden_size=classifier_intermediate_hidden_size
            )

        if bert_frozen:
            self.encoder.embeddings.word_embeddings.weight.requires_grad = False
            self.encoder.embeddings.position_embeddings.weight.requires_grad = False
            self.encoder.embeddings.token_type_embeddings.weight.requires_grad = False

    def forward(self, input_ids, label_token_ids, attention_mask=None, token_type_ids=None,
                label_attention_mask=None, label_token_type_ids=None, return_score=False):
        batch_size, seq_len = input_ids.size()
        embeds = self.encoder(input_ids, attention_mask, token_type_ids)
        last_hidden_state = embeds.last_hidden_state
        encode_text = self.dropout_layer(last_hidden_state)

        if self.use_label_embedding:
            label_embeddings = self.label_embedding_layer(label_token_ids)
        else:
            label_output = self.encoder(label_token_ids, label_attention_mask, label_token_type_ids)
            label_embeddings = label_output.last_hidden_state

        fused_results = self.label_fusing_layer(
            encode_text, label_embeddings, attention_mask, label_input_mask=label_attention_mask,
            return_scores=return_score, use_attn=self.do_attn, do_add=self.do_add, average_pooling=self.average_pooling)
        fused_feature = fused_results[0]

        start_logits = self.start_classifier(fused_feature)
        end_logits = self.end_classifier(fused_feature)
        outputs = {'start_logits': start_logits, 'end_logits': end_logits}

        if self.use_span_embedding:
            # for every position $i$ in sequence, should concate $j$ to
            # predict if $i$ and $j$ are start_pos and end_pos for an entity.
            # [batch, seq_len, seq_len, class_num, hidden]
            start_extend = fused_feature.unsqueeze(2).expand(-1, -1, seq_len, -1)
            # [batch, seq_len, seq_len, class_num, hidden]
            end_extend = fused_feature.unsqueeze(1).expand(-1, seq_len, -1, -1)
            # [batch, seq_len, seq_len, class_num, hidden*2]
            span_matrix = torch.cat([start_extend, end_extend], 3)
            # [batch, seq_len, seq_len, class_num]
            span_logits = self.span_embedding(span_matrix)
            outputs['span_logits'] = span_logits
        if return_score:
            outputs['scores'] = fused_results[-1]
        return outputs


class GlobalPointerForNER(nn.Module):
    def __init__(self, config, num_labels, encoder_model_path=None,
                 head_size=64, efficient=False, rope=True, tril_mask=True, bert_frozen=False):
        """
        使用global pointer网络结构做ner任务
        :param config:
        :param encoder_model_path:
        :param num_labels:
        :param head_size:
        :param efficient: 是否使用EfficientGlobalPointer
        :param rope: 是否使用rope旋转位置编码
        :param tril_mask: 是否去掉下三角
        :param bert_frozen: 是否freeze bert参数
        """
        super(GlobalPointerForNER, self).__init__()
        if encoder_model_path:
            self.encoder = BertModel.from_pretrained(encoder_model_path)
        else:
            self.encoder = BertModel(config)
        if efficient:
            self.model = EfficientGlobalPointer(encoder=self.encoder,
                                                ent_type_size=num_labels,
                                                head_size=head_size,
                                                rope=rope,
                                                tril_mask=tril_mask)
        else:
            self.model = GlobalPointer(encoder=self.encoder,
                                       ent_type_size=num_labels,
                                       head_size=head_size,
                                       rope=rope,
                                       tril_mask=tril_mask)

        if bert_frozen:
            self.encoder.embeddings.word_embeddings.weight.requires_grad = False
            self.encoder.embeddings.position_embeddings.weight.requires_grad = False
            self.encoder.embeddings.token_type_embeddings.weight.requires_grad = False

    def forward(self, input_ids, attention_mask, token_type_ids, labels=None):
        logits = self.model(input_ids, attention_mask, token_type_ids)
        output = {"logits": logits}
        if labels is not None:
            loss = global_pointer_crossentropy(logits, labels)
            output["loss"] = loss
        return output


class BertSoftmaxForNer(BertPreTrainedModel):
    def __init__(self, config, train_args):
        super(BertSoftmaxForNer, self).__init__(config)
        self.num_labels = train_args.num_labels
        if train_args.model_type == 'bert':
            self.bert = BertModel(config)
        elif train_args.model_type == 'nezha':
            self.bert = NeZhaModel(config)
        elif train_args.model_type == 'roformer':
            self.bert = RoFormerModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)
        self.loss_type = train_args.loss_type

        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        # outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here
        outputs = {"logits": logits}
        if labels is not None:
            assert self.loss_type in ['lsr', 'focal', 'ce']
            if self.loss_type == 'lsr':
                loss_fct = LabelSmoothingCrossEntropy(ignore_index=0)
            elif self.loss_type == 'focal':
                loss_fct = FocalLoss(ignore_index=0)
            else:
                loss_fct = nn.CrossEntropyLoss(ignore_index=0)
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs["loss"] = loss
        return outputs


class BertCrfForNer(BertPreTrainedModel):
    def __init__(self, config, train_args):
        super(BertCrfForNer, self).__init__(config)
        if train_args.model_type == 'bert':
            self.bert = BertModel(config)
        elif train_args.model_type == 'nezha':
            self.bert = NeZhaModel(config)
        elif train_args.model_type == 'roformer':
            self.bert = RoFormerModel(config)
        self.use_lstm = train_args.use_lstm
        self.dropout = nn.Dropout(train_args.dropout_rate)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.crf = CRF(num_tags=config.num_labels, batch_first=True)
        self.activation = nn.Sigmoid()
        if self.use_lstm:
            self.lstm = nn.LSTM(input_size=config.hidden_size, hidden_size=config.hidden_size//2,
                                num_layers=1, bidirectional=True, batch_first=True)
        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        sequence_output = outputs[0]
        if self.use_lstm:
            sequence_output, _ = self.lstm(sequence_output)
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        outputs = {"logits": logits}
        if labels is not None:
            loss = -1 * self.crf(emissions=logits, tags=labels, mask=attention_mask)
            outputs["loss"] = loss
        return outputs


class BertSpanForNer(BertPreTrainedModel):
    def __init__(self, config, train_args):
        super(BertSpanForNer, self).__init__(config)
        self.soft_label = train_args.soft_label
        self.num_labels = train_args.num_labels
        self.loss_type = train_args.loss_type
        if train_args.model_type == 'bert':
            self.bert = BertModel(config)
        elif train_args.model_type == 'nezha':
            self.bert = NeZhaModel(config)
        elif train_args.model_type == 'roformer':
            self.bert = RoFormerModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.start_fc = PoolerStartLogits(config.hidden_size, self.num_labels)
        if self.soft_label:
            self.end_fc = PoolerEndLogits(config.hidden_size + self.num_labels, self.num_labels)
        else:
            self.end_fc = PoolerEndLogits(config.hidden_size + 1, self.num_labels)
        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, start_positions=None, end_positions=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        start_logits = self.start_fc(sequence_output)
        if start_positions is not None and self.training:
            if self.soft_label:
                batch_size = input_ids.size(0)
                seq_len = input_ids.size(1)
                label_logits = torch.FloatTensor(batch_size, seq_len, self.num_labels)
                label_logits.zero_()
                label_logits = label_logits.to(input_ids.device)
                label_logits.scatter_(2, start_positions.unsqueeze(2), 1)
            else:
                label_logits = start_positions.unsqueeze(2).float()
        else:
            label_logits = func.softmax(start_logits, -1)
            if not self.soft_label:
                label_logits = torch.argmax(label_logits, -1).unsqueeze(2).float()
        end_logits = self.end_fc(sequence_output, label_logits)
        # outputs = (start_logits, end_logits,) + outputs[2:]
        outputs = {"start_logits": start_logits, "end_logits": end_logits}

        if start_positions is not None and end_positions is not None:
            assert self.loss_type in ['lsr', 'focal', 'ce']
            if self.loss_type == 'lsr':
                loss_fct = LabelSmoothingCrossEntropy()
            elif self.loss_type == 'focal':
                loss_fct = FocalLoss()
            else:
                loss_fct = nn.CrossEntropyLoss()
            start_logits = start_logits.view(-1, self.num_labels)
            end_logits = end_logits.view(-1, self.num_labels)
            active_loss = attention_mask.view(-1) == 1
            active_start_logits = start_logits[active_loss]
            active_end_logits = end_logits[active_loss]

            active_start_labels = start_positions.view(-1)[active_loss]
            active_end_labels = end_positions.view(-1)[active_loss]

            start_loss = loss_fct(active_start_logits, active_start_labels)
            end_loss = loss_fct(active_end_logits, active_end_labels)
            total_loss = (start_loss + end_loss) / 2
            # outputs = (total_loss,) + outputs
            outputs["loss"] = total_loss
        return outputs


class AlbertSoftmaxForNer(AlbertPreTrainedModel):
    def __init__(self, config):
        super(AlbertSoftmaxForNer, self).__init__(config)
        self.num_labels = config.num_labels
        self.loss_type = config.loss_type
        self.bert = AlbertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                            position_ids=position_ids, head_mask=head_mask)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        # outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here
        outputs = {"logits": logits}
        if labels is not None:
            assert self.loss_type in ['lsr', 'focal', 'ce']
            if self.loss_type == 'lsr':
                loss_fct = LabelSmoothingCrossEntropy(ignore_index=0)
            elif self.loss_type == 'focal':
                loss_fct = FocalLoss(ignore_index=0)
            else:
                loss_fct = nn.CrossEntropyLoss(ignore_index=0)
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            # outputs = (loss,) + outputs
            outputs["loss"] = loss
        return outputs  # (loss), scores, (hidden_states), (attentions)


class AlbertCrfForNer(AlbertPreTrainedModel):
    def __init__(self, config):
        super(AlbertCrfForNer, self).__init__(config)
        self.bert = AlbertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.crf = CRF(num_tags=config.num_labels, batch_first=True)
        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        # outputs = (logits,)
        outputs = {"logits": logits}
        if labels is not None:
            loss = self.crf(emissions=logits, tags=labels, mask=attention_mask)
            loss = -1 * loss
            outputs["loss"] = loss
        return outputs  # (loss), scores


class AlbertSpanForNer(AlbertPreTrainedModel):
    def __init__(self, config,):
        super(AlbertSpanForNer, self).__init__(config)
        self.soft_label = config.soft_label
        self.num_labels = config.num_labels
        self.loss_type = config.loss_type
        self.bert = AlbertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.start_fc = PoolerStartLogits(config.hidden_size, self.num_labels)
        if self.soft_label:
            self.end_fc = PoolerEndLogits(config.hidden_size + self.num_labels, self.num_labels)
        else:
            self.end_fc = PoolerEndLogits(config.hidden_size + 1, self.num_labels)
        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, start_positions=None, end_positions=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        start_logits = self.start_fc(sequence_output)
        if start_positions is not None and self.training:
            if self.soft_label:
                batch_size = input_ids.size(0)
                seq_len = input_ids.size(1)
                label_logits = torch.FloatTensor(batch_size, seq_len, self.num_labels)
                label_logits.zero_()
                label_logits = label_logits.to(input_ids.device)
                label_logits.scatter_(2, start_positions.unsqueeze(2), 1)
            else:
                label_logits = start_positions.unsqueeze(2).float()
        else:
            label_logits = func.softmax(start_logits, -1)
            if not self.soft_label:
                label_logits = torch.argmax(label_logits, -1).unsqueeze(2).float()
        end_logits = self.end_fc(sequence_output, label_logits)
        # outputs = (start_logits, end_logits,) + outputs[2:]
        outputs = {"start_logits": start_logits, "end_logits": end_logits}
        if start_positions is not None and end_positions is not None:
            assert self.loss_type in ['lsr', 'focal', 'ce']
            if self.loss_type == 'lsr':
                loss_fct = LabelSmoothingCrossEntropy()
            elif self.loss_type == 'focal':
                loss_fct = FocalLoss()
            else:
                loss_fct = nn.CrossEntropyLoss()
            start_logits = start_logits.view(-1, self.num_labels)
            end_logits = end_logits.view(-1, self.num_labels)
            active_loss = attention_mask.view(-1) == 1
            active_start_logits = start_logits[active_loss]
            active_start_labels = start_positions.view(-1)[active_loss]
            active_end_logits = end_logits[active_loss]
            active_end_labels = end_positions.view(-1)[active_loss]

            start_loss = loss_fct(active_start_logits, active_start_labels)
            end_loss = loss_fct(active_end_logits, active_end_labels)
            total_loss = (start_loss + end_loss) / 2
            # outputs = (total_loss,) + outputs
            outputs["loss"] = total_loss
        return outputs
