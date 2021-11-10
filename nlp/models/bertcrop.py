# -*- coding: utf8 -*-
"""
======================================
    Project Name: NLP
    File Name: bertcrop
    Author: czh
    Create Date: 2021/8/11
--------------------------------------
    Change Activity: 
======================================
"""
import regex
import copy
import logging
from typing import Optional, Union, Dict, Any, List

import torch
import torch.nn as nn
from transformers.models.bert.configuration_bert import BertConfig
from transformers.models.bert.modeling_bert import BertLayer, BertEmbeddings, BertModel

BertLayerNorm = nn.LayerNorm
LOGGER = logging.getLogger(__name__)


class BertCropModel(nn.Module):
    """
    bert模型裁剪，只保留指定数据的encoder_layer
    """

    def __init__(
            self,
            config: BertConfig
    ):
        super().__init__()
        config.output_attentions = True
        config.output_hidden_states = True
        self.embeddings = BertEmbeddings(config)
        num_hidden_layers = config.num_hidden_layers

        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states

        layer = BertLayer(config)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(num_hidden_layers)])
        self.config = config
        self.num_hidden_layers = num_hidden_layers
        self.apply(self.init_bert_weights)

    def init_bert_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(
            self,
            input_ids,
            token_type_ids=None,
            attention_mask=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None
    ):
        """ Forward pass on the Model.
                The model can behave as an encoder (with only self-attention) as well
                as a decoder, in which case a layer of cross-attention is added between
                the self-attention layers, following the architecture described in `Attention is all you need`_ by
                 Ashish Vaswani,
                Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser
                and Illia Polosukhin.
                To behave as an decoder the model needs to be initialized with the
                `is_decoder` argument of the configuration set to `True`; an
                `encoder_hidden_states` is expected as an input to the forward pass.
                .. _`Attention is all you need`:
                    https://arxiv.org/abs/1706.03762
                """
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            # Provided a padding mask of dimensions [batch_size, seq_length]
            # - if the model is a decoder, apply a causal mask in addition to the padding mask
            # - if the model is an encoder, make the mask broadcastable to
            # [batch_size, num_heads, seq_length, seq_length]
            if self.config.is_decoder:
                batch_size, seq_length = input_shape
                seq_ids = torch.arange(seq_length, device=device)
                causal_mask = seq_ids[None, None, :].repeat(batch_size, seq_length, 1) <= seq_ids[None, :, None]
                causal_mask = causal_mask.to(
                    torch.long)  # not converting to long will cause errors with pytorch version < 1.3
                extended_attention_mask = causal_mask[:, None, :, :] * attention_mask[:, None, None, :]
            else:
                extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError("Wrong shape for input_ids (shape {}) or "
                             "attention_mask (shape {})".format(input_shape, attention_mask.shape))

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # If a 2D ou 3D attention mask is provided for the cross-attention
        # we need to make broadcastabe to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)

            if encoder_attention_mask.dim() == 3:
                encoder_extended_attention_mask = encoder_attention_mask[:, None, :, :]
            elif encoder_attention_mask.dim() == 2:
                encoder_extended_attention_mask = encoder_attention_mask[:, None, None, :]
            else:
                raise ValueError(
                    "Wrong shape for encoder_hidden_shape (shape {}) or encoder_attention_mask (shape {})".format(
                        encoder_hidden_shape,
                        encoder_attention_mask.shape))

            encoder_extended_attention_mask = encoder_extended_attention_mask.to(
                dtype=next(self.parameters()).dtype)  # fp16 compatibility
            encoder_extended_attention_mask = (1.0 - encoder_extended_attention_mask) * -10000.0
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(self.config.num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(
                    -1)  # We can specify head_mask for each layer
            head_mask = head_mask.to(
                dtype=next(self.parameters()).dtype)  # switch to fload if need + fp16 compatibility
        else:
            head_mask = [None] * self.config.num_hidden_layers

        embedding_output = self.embeddings(input_ids=input_ids, position_ids=position_ids,
                                           token_type_ids=token_type_ids, inputs_embeds=inputs_embeds)

        hidden_states = embedding_output

        all_hidden_states = ()
        all_attentions = ()

        attention_mask = extended_attention_mask
        encoder_attention_mask = encoder_extended_attention_mask
        for i, layer_module in enumerate(self.layer):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            layer_outputs = layer_module(
                hidden_states,
                attention_mask,
                head_mask[i],
                encoder_hidden_states,
                encoder_attention_mask,
                output_attentions=self.output_attentions
            )
            hidden_states = layer_outputs[0]

            if self.output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        # Add last layer
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)

        # outputs meaning: # last-layer hidden state, (all hidden states), (all attentions)
        return outputs


def save_specify_num_hidden_layers_state(model: BertModel, num_hidden_layers: List, output_model_file):
    """
    裁剪hidden_layer层, 并保存
    :param model:
    :param num_hidden_layers:
    :param output_model_file:
    :return:
    """
    state_dict = model.state_dict()
    unused_key_p = regex.compile(r"encoder\.layer\.(\d{1,2}).")
    # Load from a PyTorch state_dict
    old_keys = []
    new_keys = []

    del_keys = []
    for key in state_dict.keys():
        do_pop = False
        new_key = None
        if "pooler" in key:
            del_keys.append(key)
            do_pop = True

        m = unused_key_p.findall(key)
        for num in m:
            if int(num) not in num_hidden_layers:
                del_keys.append(key)
                do_pop = True
            else:
                # 删除encoder属性，CropBertModel没设置decoder
                new_key = key.replace("encoder.", "")
        if do_pop:
            continue

        if "gamma" in key:
            new_key = key.replace("gamma", "weight")
        if "beta" in key:
            new_key = key.replace("beta", "bias")

        if new_key:
            old_keys.append(key)
            new_keys.append(new_key)

    for key in del_keys:
        state_dict.pop(key)

    for old_key, new_key in zip(old_keys, new_keys):
        state_dict[new_key] = state_dict.pop(old_key)

    # copy state_dict so _load_from_state_dict can modify it
    metadata = getattr(state_dict, "_metadata", None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    torch.save(state_dict, output_model_file)
    LOGGER.info("Model weights saved in {}".format(output_model_file))


def init_from_pretrained(
        model: Union[BertCropModel, BertModel],
        pretrained_state_dict: Optional[Dict[str, Any]] = None,
        verbose: bool = True
):
    state_dict = pretrained_state_dict
    need_crop = isinstance(model, BertCropModel)

    unused_key_p = regex.compile(r"encoder\.layer\.(\d{1,2}).")
    # Load from a PyTorch state_dict
    old_keys = []
    new_keys = []

    del_keys = []
    for key in state_dict.keys():
        do_pop = False
        new_key = None

        if "pooler" in key:
            del_keys.append(key)
            do_pop = True

        m = unused_key_p.findall(key)
        for num in m:
            if int(num) >= model.config.num_hidden_layers:
                del_keys.append(key)
                do_pop = True
            else:
                if need_crop:
                    # 删除encoder属性，CropBertModel没设置decoder
                    new_key = key.replace("encoder.", "")

        if do_pop and need_crop:
            continue

        if "gamma" in key:
            new_key = key.replace("gamma", "weight")
        if "beta" in key:
            new_key = key.replace("beta", "bias")

        if new_key:
            old_keys.append(key)
            new_keys.append(new_key)

    if need_crop:
        for key in del_keys:
            state_dict.pop(key)

    for old_key, new_key in zip(old_keys, new_keys):
        state_dict[new_key] = state_dict.pop(old_key)

    missing_keys = []
    unexpected_keys = []
    error_msgs = []
    # copy state_dict so _load_from_state_dict can modify it
    metadata = getattr(state_dict, "_metadata", None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, prefix=""):
        local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
        #         print(prefix, local_metadata)
        module._load_from_state_dict(
            state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + ".")

    start_prefix = ""
    if not hasattr(model, "bert") and any(s.startswith("bert.") for s in state_dict.keys()):
        start_prefix = "bert."
    load(model, prefix=start_prefix)

    if len(missing_keys) > 0 and verbose:
        LOGGER.info("Weights of {} not initialized from pretrained model: {}".format(
            model.__class__.__name__, missing_keys))
    if len(unexpected_keys) > 0 and verbose:
        LOGGER.info("Weights from pretrained model not used in {}: {}".format(
            model.__class__.__name__, unexpected_keys))
    if len(error_msgs) > 0:
        raise RuntimeError("Error(s) in loading state_dict for {}:\n\t{}".format(
            model.__class__.__name__, "\n\t".join(error_msgs)))
