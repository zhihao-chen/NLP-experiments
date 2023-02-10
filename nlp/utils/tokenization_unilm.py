#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: czh
@email:
@date: 2023/2/2 15:58
"""
from __future__ import absolute_import, division, print_function, unicode_literals

"""
Tokenization classes for UniLM.
From https://github.com/YunwenTechnology/Unilm/blob/master/tokenization_unilm.py
"""
import logging
from typing import List, Optional

from transformers.models.bert.tokenization_bert import BertTokenizer, whitespace_tokenize

logger = logging.getLogger(__name__)

VOCAB_FILES_NAMES = {'vocab_file': 'vocab.txt'}

PRETRAINED_VOCAB_FILES_MAP = {
    'vocab_file':
    {
        'unilm-large-cased': "",
        'unilm-base-cased': ""
    }
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    'unilm-large-cased': 512,
    'unilm-base-cased': 512
}


class UniLMTokenizerYunWen(BertTokenizer):
    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES


class WhitespaceTokenizer(object):
    @staticmethod
    def tokenize(text):
        return whitespace_tokenize(text)


class UniLMTokenizerLiadrinz(BertTokenizer):
    """
    https://github.com/Liadrinz/transformers-unilm/blob/main/unilm/tokenization_unilm.py
    """
    def __init__(
            self,
            vocab_file,
            do_lower_case=True,
            do_basic_tokenize=True,
            never_split=None,
            unk_token="[UNK]",
            sep_token="[SEP]",
            pad_token="[PAD]",
            cls_token="[CLS]",
            mask_token="[MASK]",
            tokenize_chinese_chars=True,
            strip_accents=None,
            src_type_id=4,
            tgt_type_id=5,
            **kwargs
    ):
        super().__init__(
            vocab_file=vocab_file,
            do_lower_case=do_lower_case,
            do_basic_tokenize=do_basic_tokenize,
            never_split=never_split,
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            tokenize_chinese_chars=tokenize_chinese_chars,
            strip_accents=strip_accents, **kwargs)
        self.src_type_id = src_type_id
        self.tgt_type_id = tgt_type_id

    def create_token_type_ids_from_sequences(
            self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]
        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [self.src_type_id]
        return [self.src_type_id] * len(cls + token_ids_0 + sep) + [self.tgt_type_id] * len(token_ids_1 + sep)

    def get_special_tokens_mask(self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None,
                                already_has_special_tokens: bool = False) -> List[int]:
        # UniLM should learn to restore the [SEP] at the end of the sentence
        if already_has_special_tokens:
            if token_ids_1 is not None:
                raise ValueError(
                    "You should not supply a second sequence if the provided sequence of "
                    "ids is already formatted with special tokens for the model."
                )
            all_special_ids = self.all_special_ids
            special_tokens_mask = [1 if token in all_special_ids else 0 for token in token_ids_0]
            for i, tkid in reversed(list(enumerate(token_ids_0))):
                if tkid == self.sep_token_id:
                    special_tokens_mask[i] = 0
                    break
            return special_tokens_mask

        if token_ids_1 is not None:
            return [1] + ([0] * len(token_ids_0)) + [1] + ([0] * len(token_ids_1)) + [0]
        return [1] + ([0] * len(token_ids_0)) + [0]
