#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: czh
@email:
@date: 2023/2/2 15:55
"""
from __future__ import absolute_import, division, print_function, unicode_literals

""" 
UniLM model configuration 
"""
import json
import logging
from io import open

from transformers.models.bert.configuration_bert import BertConfig
from transformers.configuration_utils import PretrainedConfig

logger = logging.getLogger(__name__)

UNILM_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    'unilm-large-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/unilm/unilm-large-cased-config.json",
    'unilm-base-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/unilm/unilm-base-cased-config.json"
}


class UnilmConfigYunWen(PretrainedConfig):
    r"""
        https://github.com/YunwenTechnology/Unilm/blob/master/configuration_unilm.py
        :class:`~transformers.UnilmConfig` is the configuration class to store the configuration of a
        `UnilmModel`.


        Arguments:
            vocab_size_or_config_json_file: Vocabulary size of `inputs_ids` in `UnilmModel`.
            hidden_size: Size of the encoder layers and the pooler layer.
            num_hidden_layers: Number of hidden layers in the Transformer encoder.
            num_attention_heads: Number of attention heads for each attention layer in
                the Transformer encoder.
            intermediate_size: The size of the "intermediate" (i.e., feed-forward)
                layer in the Transformer encoder.
            hidden_act: The non-linear activation function (function or string) in the
                encoder and pooler. If string, "gelu", "relu", "swish" and "gelu_new" are supported.
            hidden_dropout_prob: The dropout probabilitiy for all fully connected
                layers in the embeddings, encoder, and pooler.
            attention_probs_dropout_prob: The dropout ratio for the attention
                probabilities.
            max_position_embeddings: The maximum sequence length that this model might
                ever be used with. Typically set this to something large just in case
                (e.g., 512 or 1024 or 2048).
            type_vocab_size: The vocabulary size of the `token_type_ids` passed into
                `UnilmModel`.
            initializer_range: The sttdev of the truncated_normal_initializer for
                initializing all weight matrices.
            layer_norm_eps: The epsilon used by LayerNorm.
    """
    pretrained_config_archive_map = UNILM_PRETRAINED_CONFIG_ARCHIVE_MAP

    def __init__(self,
                 vocab_size=28996,
                 hidden_size=768,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 hidden_act="gelu",
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 max_position_embeddings=512,
                 type_vocab_size=6,
                 initializer_range=0.02,
                 layer_norm_eps=1e-12,
                 **kwargs):
        super(UnilmConfigYunWen, self).__init__(**kwargs)
        if isinstance(vocab_size, str):
            with open(vocab_size, "r", encoding='utf-8') as reader:
                json_config = json.loads(reader.read())
            for key, value in json_config.items():
                self.__dict__[key] = value
        elif isinstance(vocab_size, int):
            self.vocab_size = vocab_size
            self.hidden_size = hidden_size
            self.num_hidden_layers = num_hidden_layers
            self.num_attention_heads = num_attention_heads
            self.hidden_act = hidden_act
            self.intermediate_size = intermediate_size
            self.hidden_dropout_prob = hidden_dropout_prob
            self.attention_probs_dropout_prob = attention_probs_dropout_prob
            self.max_position_embeddings = max_position_embeddings
            self.type_vocab_size = type_vocab_size
            self.initializer_range = initializer_range
            self.layer_norm_eps = layer_norm_eps
        else:
            raise ValueError("First argument must be either a vocabulary size (int)"
                             " or the path to a pretrained model config file (str)")


class UniLMConfigLiadrinz(BertConfig):
    """
    https://github.com/Liadrinz/transformers-unilm/blob/main/unilm/configuration_unilm.py
    """
    def __init__(
            self,
            vocab_size=30522,
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            hidden_act="gelu",
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            max_position_embeddings=512,
            type_vocab_size=6,
            src_type_id=4,
            tgt_type_id=5,
            initializer_range=0.02,
            layer_norm_eps=1e-12,
            pad_token_id=0,
            bos_token_id=101,
            eos_token_id=102,
            mask_token_id=103,
            position_embedding_type="absolute",
            use_cache=True,
            classifier_dropout=None,
            **kwargs
    ):
        super().__init__(
            vocab_size, hidden_size, num_hidden_layers, num_attention_heads,
            intermediate_size, hidden_act, hidden_dropout_prob,
            attention_probs_dropout_prob, max_position_embeddings, type_vocab_size,
            initializer_range, layer_norm_eps, pad_token_id, position_embedding_type,
            use_cache, classifier_dropout, **kwargs)
        self.src_type_id = src_type_id
        self.tgt_type_id = tgt_type_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.mask_token_id = mask_token_id
