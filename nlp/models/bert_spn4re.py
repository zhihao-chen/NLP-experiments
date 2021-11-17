# -*- coding: utf8 -*-
"""
======================================
    Project Name: NLP
    File Name: spn4re
    Author: czh
    Create Date: 2021/11/15
--------------------------------------
    Change Activity: 
======================================
"""
from typing import List, Dict, Union, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as func
from transformers import BertModel
from transformers import PretrainedConfig
from transformers.models.bert.modeling_bert import BertIntermediate, BertOutput, BertAttention
from scipy.optimize import linear_sum_assignment

from nlp.layers.linears import EntityLinears
from nlp.utils.enums import MatcherType, RunMode
from nlp.utils.functions import generate_tuple


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network
    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(
            self,
            entity_loss_weight: List[float],
            relation_loss_weight: float = 1.0,
            matcher: MatcherType = MatcherType.AVG
    ):
        super().__init__()
        self.cost_relation = relation_loss_weight
        self.cost_entity = entity_loss_weight
        self.matcher = matcher

    @torch.no_grad()
    def forward(
            self,
            outputs: Dict[str, Union[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]],
            targets: List[Dict[str, Union[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]]]
    ):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_rel_logits": Tensor of dim [batch_size, num_generated_tuples, num_classes] with the classification logits
                 "pred_ent_logits": entity size * 2 * Tensor of dim [batch_size, num_generated_tuples, seq_len] with the predicted index logits
            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict.
                 eg.{"relation": LongTensor(5), "entity": [LongTensor(1,3), LongTensor(3,5), LongTensor(5,9)]
        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_generated_tuples, num_gold_tuples)
        """
        bsz, num_generated_tuples = outputs["pred_rel_logits"].shape[:2]
        # We flatten to compute the cost matrices in a batch
        pred_rel = outputs["pred_rel_logits"].flatten(0, 1).softmax(-1)  # [bsz * num_generated_tuples, num_classes]
        gold_rel = torch.cat([v["relation"] for v in targets])
        # after masking the pad token
        pred_ent: List[Tuple[torch.Tensor, torch.Tensor]] = [
            (head.flatten(0, 1).softmax(-1), end.flatten(0, 1).softmax(-1))
            for head, end in outputs["pred_ent_logits"]
        ]  # entity_size * 2 * tensor [bsz * num_generated_tuples, num_classes]

        entity_size = len(pred_ent)
        gold_ent: List[Tuple[torch.Tensor, torch.Tensor]] = [
            (
                torch.cat([v["entity"][:, i, 0] for v in targets if len(v["entity"]) > 0]),
                torch.cat([v["entity"][:, i, 1] for v in targets if len(v["entity"]) > 0])
            )
            for i in range(entity_size)
        ]

        if self.matcher == MatcherType.AVG:
            cost = -self.cost_relation * pred_rel[:, gold_rel] - torch.sum(
                torch.stack([
                    self.cost_entity[i] * 1 / 2 * (
                            pred_ent[i][0][:, gold_ent[i][0]] + pred_ent[i][1][:, gold_ent[i][1]])
                    for i in range(entity_size)
                ]),
                dim=0
            )

        elif self.matcher == MatcherType.MIN:
            cost = torch.cat([
                                 pred_ent[i][j][:, gold_ent[i][j]].unsqueeze(1)
                                 for i in range(entity_size)
                                 for j in range(2)
                             ] + [pred_rel[:, gold_rel].unsqueeze(1)],
                             dim=1)
            cost = - torch.min(cost, dim=1)[0]
        else:
            raise ValueError("Wrong matcher")
        cost = cost.view(bsz, num_generated_tuples, -1).cpu()
        num_gold_tuples = [len(v["relation"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(cost.split(num_gold_tuples, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


class SetCriterion(nn.Module):
    """ This class computes the loss for Set_RE.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class, subject position and object position)
    """

    def __init__(
            self,
            num_classes: int,
            entity_loss_weight: List[float],
            relation_loss_weight: float = 1.0,
            na_coef: float = 1.0,
            matcher: MatcherType = MatcherType.AVG
    ):
        """ Create the criterion.
        Parameters:
            num_classes: number of relation categories
            matcher: module able to compute a matching between targets and proposals
            relation_loss_weight:
            entity_loss_weight:
            na_coef: list containg the relative classification weight applied to the NA category and positional classification weight applied to the [SEP]
        """
        super().__init__()
        self.num_classes = num_classes
        self.loss_weight = self._format_loss_weight(entity_loss_weight, relation_loss_weight)
        self.matcher = HungarianMatcher(
            entity_loss_weight=entity_loss_weight,
            relation_loss_weight=relation_loss_weight,
            matcher=matcher
        )
        self.losses = ("entity", "relation")
        rel_weight = torch.ones(self.num_classes + 1).float()
        rel_weight[-1] = na_coef
        self.register_buffer('rel_weight', rel_weight)

    @staticmethod
    def _format_loss_weight(
            entity_loss_weight: List[float],
            relation_loss_weight: float = 1.0,
    ):
        """convert to dict[str, float]"""

        weight = {
            "relation": relation_loss_weight
        }
        for k, v in enumerate(entity_loss_weight):
            weight[f"entity_{k}"] = v
        return weight

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
             device:
        """
        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs, targets)
        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            if loss == "entity" and self.empty_targets(targets):
                pass
            else:
                losses.update(self.get_loss(loss, outputs, targets, indices))
        total_loss = sum(losses[k] * self.loss_weight[k] for k in losses.keys() if k in self.loss_weight)
        losses["total"] = total_loss
        return losses

    def relation_loss(self, outputs, targets, indices):
        """Classification loss (NLL)
        targets dicts must contain the key "relation" containing a tensor of dim [bsz]
        """
        src_logits = outputs['pred_rel_logits']  # [bsz, num_generated_tuples, num_rel + 1]
        device = src_logits.device
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["relation"][i] for t, (_, i) in zip(targets, indices)]).to(device)
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=device)
        target_classes[idx] = target_classes_o
        loss = func.cross_entropy(
            src_logits.flatten(0, 1),
            target_classes.flatten(0, 1).to(device),
            weight=self.rel_weight,
        )
        losses = {'relation': loss}
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty tuples
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_rel_logits = outputs['pred_rel_logits']
        device = pred_rel_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_rel_logits.argmax(-1) != pred_rel_logits.shape[-1] - 1).sum(1)
        card_err = func.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    @staticmethod
    def _get_src_permutation_idx(indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    @staticmethod
    def _get_tgt_permutation_idx(indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, **kwargs):
        loss_map = {
            'relation': self.relation_loss,
            'cardinality': self.loss_cardinality,
            'entity': self.entity_loss
        }
        return loss_map[loss](outputs, targets, indices, **kwargs)

    def entity_loss(self, outputs, targets, indices):
        """Compute the losses related to the position of head entity or tail entity
        """
        idx = self._get_src_permutation_idx(indices)
        device = outputs["pred_ent_logits"][0][0][idx].device
        losses = {
            f"entity_{i}": 1 / 2 * (
                    func.cross_entropy(
                        outputs["pred_ent_logits"][i][0][idx],
                        torch.cat([t["entity"][:, i, 0][ii] if len(ii) > 0 else torch.LongTensor(1)
                                   for t, (_, ii) in zip(targets, indices)]).to(device)
                    ) + func.cross_entropy(outputs["pred_ent_logits"][i][1][idx],
                                           torch.cat([t["entity"][:, i, 1][ii] for t, (_, ii) in zip(targets, indices)]).to(device)
                                           ))
            for i in range(len(outputs["pred_ent_logits"]))
        }
        return losses

    @staticmethod
    def empty_targets(targets):
        flag = True
        for target in targets:
            if len(target["relation"]) != 0:
                flag = False
                break
        return flag


class SetDecoder(nn.Module):
    def __init__(
            self,
            config: PretrainedConfig,
            num_classes: int,
            num_entities_in_tuple: int = 2,
            num_generated_tuples: int = 10,
            num_layers: int = 3,
            return_intermediate: bool = False
    ):
        """

        :param config:
        :param num_classes: the num of relation types
        :param num_entities_in_tuple: the entity num of each tuple
        :param num_generated_tuples:
        :param num_layers:
        :param return_intermediate:
        """
        super().__init__()
        self.return_intermediate = return_intermediate
        self.num_generated_tuples = num_generated_tuples
        self.layers = nn.ModuleList([DecoderLayer(config) for _ in range(num_layers)])
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)  # noqa
        self.dropout = nn.Dropout(config.hidden_dropout_prob)  # noqa
        self.query_embed = nn.Embedding(num_generated_tuples, config.hidden_size)  # noqa
        self.decoder2class = nn.Linear(config.hidden_size, num_classes + 1)  # noqa

        self.entity_linears = nn.ModuleList([
            EntityLinears(input_dim=config.hidden_size, output_dim=1, bias=False)  # noqa
            for _ in range(num_entities_in_tuple)
        ])

        nn.init.orthogonal_(self.query_embed.weight, gain=1)

    def forward(self, encoder_hidden_states, encoder_attention_mask):
        bsz = encoder_hidden_states.size()[0]
        hidden_states = self.query_embed.weight.unsqueeze(0).repeat(bsz, 1, 1)
        hidden_states = self.dropout(self.LayerNorm(hidden_states))
        all_hidden_states = ()
        for i, layer_module in enumerate(self.layers):
            if self.return_intermediate:
                all_hidden_states = all_hidden_states + (hidden_states,)
            layer_outputs = layer_module(
                hidden_states, encoder_hidden_states, encoder_attention_mask
            )
            hidden_states = layer_outputs[0]

        class_logits = self.decoder2class(hidden_states)

        ent_logits = ()
        for layer_module in self.entity_linears:
            ent_logits = ent_logits + (layer_module(hidden_states, encoder_hidden_states),)

        return class_logits, ent_logits


class DecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = BertAttention(config)
        self.crossattention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(
            self,
            hidden_states,
            encoder_hidden_states,
            encoder_attention_mask
    ):
        self_attention_outputs = self.attention(hidden_states)
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
        encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
        if encoder_attention_mask.dim() == 3:
            encoder_extended_attention_mask = encoder_attention_mask[:, None, :, :]
        elif encoder_attention_mask.dim() == 2:
            encoder_extended_attention_mask = encoder_attention_mask[:, None, None, :]
        else:
            raise ValueError(
                "Wrong shape for encoder_hidden_shape (shape {}) or encoder_attention_mask (shape {})".format(
                    encoder_hidden_shape, encoder_attention_mask.shape
                )
            )
        encoder_extended_attention_mask = (1.0 - encoder_extended_attention_mask) * -10000.0
        cross_attention_outputs = self.crossattention(
            hidden_states=attention_output, encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask
        )
        attention_output = cross_attention_outputs[0]
        outputs = outputs + cross_attention_outputs[1:]  # add cross attentions if we output attention weights

        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        outputs = (layer_output,) + outputs
        return outputs


class SetPredPlusBert(nn.Module):
    def __init__(
            self,
            encoder: BertModel,
            num_relation_classes: int,
            num_generated_tuples: int,
            num_entities_in_tuple: int,
            entity_loss_weight: List[float],
            relation_loss_weight: float = 1.0,
            num_decoder_layers: int = 3,
            na_rel_coef: float = 1.0,
            matcher: MatcherType = MatcherType.AVG
    ):
        super().__init__()
        self.encoder = encoder
        config = encoder.config
        self.num_relation_classes = num_relation_classes
        self.num_generated_tuples = num_generated_tuples
        self.num_entities_in_tuple = num_entities_in_tuple
        self.decoder = SetDecoder(
            config=config,
            num_entities_in_tuple=num_entities_in_tuple,
            num_generated_tuples=num_generated_tuples,
            num_layers=num_decoder_layers,
            num_classes=num_relation_classes,
            return_intermediate=False
        )
        self.criterion = SetCriterion(
            num_classes=num_relation_classes,
            relation_loss_weight=relation_loss_weight,
            entity_loss_weight=entity_loss_weight,
            na_coef=na_rel_coef,
            matcher=matcher
        )

    def forward(self, input_ids, attention_mask, token_type_ids, targets=None):
        encoder_output = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True
        )
        last_hidden_state = encoder_output.last_hidden_state
        rel_logits, ent_logits = self.decoder(
            encoder_hidden_states=last_hidden_state,
            encoder_attention_mask=attention_mask
        )

        formatted_ent_logits = [
            (
                head.squeeze(-1).masked_fill((1 - attention_mask.unsqueeze(1)).bool(), -10000.0),
                tail.squeeze(-1).masked_fill((1 - attention_mask.unsqueeze(1)).bool(), -10000.0)
                # [bsz, num_generated_tuples, seq_len]
            )
            for head, tail in ent_logits
        ]
        outputs = {
            "pred_rel_logits": rel_logits,
            "pred_ent_logits": formatted_ent_logits
        }
        if targets is not None:
            loss = self.criterion(outputs, targets)
            outputs["loss"] = loss

        return outputs

    @torch.no_grad()
    def gen_tuples(
            self,
            input_ids,
            attention_mask,
            token_type_ids,
            sent_lens,
            sent_idx,
            n_best_size,
            max_span_length,
            allow_null_entities_in_tuple: List[int],
            run_mode: RunMode = RunMode.INFER
    ):
        outputs = self.forward(input_ids, attention_mask, token_type_ids)
        pred_tuple = generate_tuple(
            outputs,
            sent_lens,
            sent_idx,
            self.num_generated_tuples,
            n_best_size,
            max_span_length,
            allow_null_entities_in_tuple,
            self.num_relation_classes,
            run_mode=run_mode
        )
        return pred_tuple
