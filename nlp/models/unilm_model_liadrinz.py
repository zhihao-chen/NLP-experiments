#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: czh
@email:
@date: 2023/2/2 16:20
"""
"""
只用于seq2seq任务
https://github.com/Liadrinz/transformers-unilm/blob/main/unilm/modeling_unilm.py
"""
import torch
import math

from torch import nn
from torch.nn import CrossEntropyLoss
from typing import Optional, Union, Tuple, Callable, List, Dict, Any, OrderedDict
from dataclasses import dataclass
from transformers.models.bert.modeling_bert import (
    apply_chunking_to_forward,
    BertSelfAttention,
    BertAttention,
    BertLayer,
    BertEmbeddings,
    BertEncoder,
    BertModel,
    BertForMaskedLM,
    BaseModelOutputWithPoolingAndCrossAttentions,
    BaseModelOutputWithPastAndCrossAttentions,
)
from transformers.modeling_outputs import MaskedLMOutput
from transformers.utils import logging

from nlp.utils.configuration_unilm import UniLMConfigLiadrinz as UniLMConfig

logger = logging.get_logger(__name__)


def apply_fn_to_past_key_values(fn: Callable[..., torch.FloatTensor], *past_key_values) -> List:
    new_past_key_values = []
    for key_value_list in zip(*past_key_values):
        keys, values = list(zip(*key_value_list))
        new_key = fn(*keys)
        new_value = fn(*values)
        new_past_key_values.append((new_key, new_value))
    return new_past_key_values


@dataclass
class UniLMSeq2SeqOutput(MaskedLMOutput):
    past_key_values: Optional[Tuple[torch.FloatTensor]] = None


class UniLMSelfAttention(BertSelfAttention):

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.FloatTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            past_key_value: Optional[Tuple[torch.FloatTensor]] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        mixed_query_layer = self.query(hidden_states)

        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        if past_key_value is not None:
            key_layer = torch.cat((past_key_value[0], key_layer), dim=2)
            value_layer = torch.cat((past_key_value[1], value_layer), dim=2)

        query_layer = self.transpose_for_scores(mixed_query_layer)

        past_key_value = (key_layer, value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        if use_cache:
            outputs = outputs + (past_key_value,)
        return outputs


class UniLMAttention(BertAttention):

    def __init__(self, config, position_embedding_type=None):
        super().__init__(config, position_embedding_type)
        self.self = UniLMSelfAttention(config, position_embedding_type)

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.FloatTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            past_key_value: Optional[Tuple[torch.FloatTensor]] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            past_key_value,
            use_cache,
            output_attentions,
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


class UniLMLayer(BertLayer):

    def __init__(self, config):
        super().__init__(config)
        self.attention = UniLMAttention(config)

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.FloatTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            past_key_value: Optional[Tuple[torch.FloatTensor]] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            past_key_value,
            use_cache,
            output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0]

        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        outputs = (layer_output,) + outputs

        return outputs


class UniLMEncoder(BertEncoder):

    def __init__(self, config):
        super().__init__(config)
        self.layer = nn.ModuleList([UniLMLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.FloatTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = False,
            output_hidden_states: Optional[bool] = False,
            return_dict: Optional[bool] = True,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPastAndCrossAttentions]:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        next_decoder_cache = () if use_cache else None
        for i, layer_module in enumerate(self.layer):

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:
                logger.warning(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, past_key_value, use_cache, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    past_key_value,
                    use_cache,
                    output_attentions,
                )

            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )


class UniLMEmbedding(BertEmbeddings):

    def __init__(self, config):
        super().__init__(config)
        self.register_buffer(
            "token_type_ids", torch.zeros(self.position_ids.size(), dtype=torch.long).fill_(config.src_type_id),
            persistent=False
        )


class UniLMModel(BertModel):
    config_class = UniLMConfig

    def __init__(self, config: UniLMConfig):
        super().__init__(config)
        self.embeddings = UniLMEmbedding(config)
        self.encoder = UniLMEncoder(config)

    def get_extended_attention_mask(self, attention_mask: torch.Tensor, token_type_ids: torch.Tensor,
                                    input_shape: Tuple[int], past_length: int = 0, device=None,
                                    dtype: torch.float = None) -> torch.Tensor:
        if dtype is None:
            dtype = self.dtype
        if device is None:
            device = attention_mask.device
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(
                f"Wrong shape for input_ids (shape {input_shape}) or attention_mask (shape {attention_mask.shape})"
            )
        if past_length == 0:
            is_src = (token_type_ids == self.config.src_type_id).unsqueeze(-1).to(dtype)
            is_tgt = (token_type_ids == self.config.tgt_type_id).unsqueeze(-1).to(dtype)
            # whether it is source-to-target attention, which should be masked in unilm seq2seq task
            is_s2t = is_src @ is_tgt.transpose(1, 2)
            is_t2t = is_tgt @ is_tgt.transpose(1, 2)  # whether it is target-to-target attention
            is_r2l = 1 - torch.tril(torch.ones(attention_mask.size(0), attention_mask.size(1), attention_mask.size(1),
                                               device=device))  # whether is right-to-left attention
            # whether it is target-to-target attention and also right-to-left attention, which should be masked
            is_tr2tl = is_t2t * is_r2l
            unilm_seq2seq_mask = 1 - (is_s2t + is_tr2tl)
            unilm_seq2seq_mask = unilm_seq2seq_mask[:, None, :, :]
            extended_attention_mask = extended_attention_mask * unilm_seq2seq_mask.to(dtype)
        else:
            extended_attention_mask = torch.ones(attention_mask.size(0), 1, attention_mask.size(1) - past_length,
                                                 attention_mask.size(1), device=device)
            extended_attention_mask[:, :, :, :past_length] = attention_mask[:, None, None, :past_length]
            extended_attention_mask[:, :, :, past_length:].tril_()
        extended_attention_mask = extended_attention_mask.to(dtype=dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(dtype).min
        return extended_attention_mask

    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            token_type_ids: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPoolingAndCrossAttentions]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length), device=device)

        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, token_type_ids,
                                                                                 input_shape, past_key_values_length)

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
        )
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )


class UniLMForConditionalGeneration(BertForMaskedLM):
    _allowed_missing_keys = ["bert.embeddings.position_ids", "cls.predictions.decoder.bias"]
    _allowed_unexpected_keys = ["cls.seq_relationship.weight", "cls.seq_relationship.bias"]

    config_class = UniLMConfig

    def __init__(self, config: UniLMConfig):
        super().__init__(config)
        self.bert = UniLMModel(config)

    def _update_model_kwargs_for_generation(
            self, outputs: UniLMSeq2SeqOutput, model_kwargs: Dict[str, Any], is_encoder_decoder: bool = False
    ) -> Dict[str, Any]:
        model_kwargs = super()._update_model_kwargs_for_generation(outputs, model_kwargs, is_encoder_decoder)
        model_kwargs["past_key_values"] = outputs.past_key_values
        return model_kwargs

    def prepare_inputs_for_generation(self, input_ids, attention_mask, past=None, token_type_ids=None,
                                      position_ids=None, **model_kwargs):
        # torch.cuda.empty_cache()
        batch_size, seq_length = input_ids.size()
        next_input_ids = input_ids.new_empty(batch_size, 1).fill_(self.config.mask_token_id)
        next_token_type_ids = input_ids.new_empty(batch_size, 1).fill_(self.config.tgt_type_id)
        next_attention_mask = input_ids.new_empty(batch_size, 1).fill_(1)
        if past is not None:
            input_ids = torch.cat((input_ids[:, -1:], next_input_ids), dim=-1)
            token_type_ids = input_ids.new_empty(input_ids.size()).fill_(self.config.tgt_type_id)
            next_position_ids = self.prev_position_ids + 1
            position_ids = torch.cat((self.prev_position_ids, next_position_ids), dim=-1)
            attention_mask = torch.cat((attention_mask, next_attention_mask), dim=-1)
            self.prev_position_ids = next_position_ids
        else:
            if token_type_ids is None:
                token_type_ids = input_ids.new_empty(batch_size, seq_length).fill_(self.config.src_type_id)
            if position_ids is None:
                position_ids = self.bert.embeddings.position_ids.repeat(batch_size, 1)[:, :seq_length]
                position_ids[attention_mask == 0] = 0
            next_position_ids = attention_mask.sum(dim=-1, keepdims=True)
            input_ids = torch.cat((input_ids, next_input_ids), dim=-1)
            token_type_ids = torch.cat((token_type_ids, next_token_type_ids), dim=-1)
            position_ids = torch.cat((position_ids, next_position_ids), dim=-1)
            attention_mask = torch.cat((attention_mask, next_attention_mask), dim=-1)
            self.prev_position_ids = next_position_ids
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "position_ids": position_ids,
            "past_key_values": past,
            "use_cache": not self.training,
        }

    @staticmethod
    def _reorder_cache(past, beam_idx):
        reordered_past = ()
        for layer_past in past:
            # cached cross_attention states don't have to be reordered -> they are always the same
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx) for past_state in layer_past[:2]) + layer_past[2:],
            )
        return reordered_past

    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            token_type_ids: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            use_cache: Optional[bool] = None,
            labels: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], UniLMSeq2SeqOutput]:

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()  # -100 index = padding token
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        past_key_values = apply_fn_to_past_key_values(lambda t: t[:, :, :-1, :],
                                                      outputs.past_key_values) if use_cache else None

        if not return_dict:
            output = (prediction_scores,) + outputs[2:] + (past_key_values,)
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return UniLMSeq2SeqOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            past_key_values=past_key_values
        )

    def load_state_dict(self, state_dict: OrderedDict[str, torch.Tensor], strict: bool = True, compat: bool = True):
        if not strict or not compat:
            return super().load_state_dict(state_dict, strict)
        error_msgs: List[str] = []
        load_return = super().load_state_dict(state_dict, strict=False)
        missing_keys, unexpected_keys = load_return
        missing_keys = list(set(missing_keys).difference(self._allowed_missing_keys))
        unexpected_keys = list(set(unexpected_keys).difference(self._allowed_unexpected_keys))
        if len(unexpected_keys) > 0:
            error_msgs.insert(
                0, 'Unexpected key(s) in state_dict: {}. '.format(
                    ', '.join('"{}"'.format(k) for k in unexpected_keys)))
        if len(missing_keys) > 0:
            error_msgs.insert(
                0, 'Missing key(s) in state_dict: {}. '.format(
                    ', '.join('"{}"'.format(k) for k in missing_keys)))
        if len(error_msgs) > 0:
            raise RuntimeError('Error(s) in loading state_dict for {}:\n\t{}'.format(
                self.__class__.__name__, "\n\t".join(error_msgs)))
        return load_return
