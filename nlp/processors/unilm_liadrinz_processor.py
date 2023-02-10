#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: czh
@email:
@date: 2023/2/2 18:41
"""
"""
https://github.com/Liadrinz/transformers-unilm/blob/main/unilm/collator.py
https://github.com/Liadrinz/transformers-unilm/blob/main/unilm/data_utils.py
"""
from collections.abc import Mapping
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from torch.utils.data.dataset import Dataset
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.data.data_collator import DataCollatorForLanguageModeling
from transformers.data.data_collator import _tf_collate_batch, _torch_collate_batch, _numpy_collate_batch
from dataclasses import dataclass


class Seq2SeqDataset(Dataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, src_file, tgt_file, max_src_len=448, max_tgt_len=64,
                 inference=False) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        with open(src_file) as fsrc, open(tgt_file) as ftgt:
            self.srcs = [line.strip() for line in fsrc]
            self.tgts = [line.strip() for line in ftgt]
        assert len(self.srcs) == len(self.tgts)
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len
        self.inference = inference

    def __getitem__(self, idx):
        src = self.srcs[idx]
        tgt = self.tgts[idx]
        src_ids = self.tokenizer.encode(
            src, add_special_tokens=False, truncation=True, max_length=self.max_src_len - 2)
        tgt_ids = self.tokenizer.encode(
            tgt, add_special_tokens=False, truncation=True, max_length=self.max_tgt_len - 1)
        if self.inference:
            input_ids = [self.tokenizer.cls_token_id] + src_ids + [self.tokenizer.sep_token_id]
            attention_mask = [1] * len(input_ids)
            token_type_ids = [self.tokenizer.src_type_id] * len(input_ids)
        else:
            input_ids = [self.tokenizer.cls_token_id] + src_ids + [self.tokenizer.sep_token_id] + tgt_ids + [
                self.tokenizer.sep_token_id]
            attention_mask = [1] * len(input_ids)
            token_type_ids = [self.tokenizer.src_type_id] * (len(src_ids) + 2) + [self.tokenizer.tgt_type_id] * (
                        len(tgt_ids) + 1)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }

    def __len__(self):
        return len(self.srcs)


class CorpusDataset(Dataset):

    def __init__(self, tokenizer: PreTrainedTokenizer, corpus_file, max_seq_len=512) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        with open(corpus_file) as fin:
            self.corpus = [line.strip() for line in fin]
        self.max_seq_len = max_seq_len

    def __getitem__(self, idx):
        sentence = self.corpus[idx]
        sentence_ids = self.tokenizer.encode(sentence, add_special_tokens=False, truncation=True,
                                             max_length=self.max_seq_len)
        sep_indices = [i for i, tk in enumerate(sentence_ids) if tk == self.tokenizer.eos_token_id]
        while len(sentence_ids) > self.max_seq_len - 1:
            if len(sep_indices) == 0:
                break
            sentence_ids = sentence_ids[:sep_indices.pop()]
        sentence_ids = sentence_ids[:self.max_seq_len - 1]
        input_ids = sentence_ids + [self.tokenizer.sep_token_id]
        attention_mask = [1] * len(input_ids)
        token_type_ids = [self.tokenizer.tgt_type_id] * (len(input_ids))
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }

    def __len__(self):
        return len(self.corpus)


@dataclass
class DataCollatorForUniLMSeq2Seq(DataCollatorForLanguageModeling):

    @property
    def tgt_type_id(self):
        if hasattr(self.tokenizer, "tgt_type_id"):
            return self.tokenizer.tgt_type_id
        return 1

    @staticmethod
    def tf_bernoulli(shape, probability):
        import tensorflow as tf

        prob_matrix = tf.fill(shape, probability)
        return tf.cast(prob_matrix - tf.random.uniform(shape, 0, 1) >= 0, tf.bool)

    def tf_mask_tokens(self, inputs: Any, token_type_ids: Any, vocab_size,
                       mask_token_id, special_tokens_mask: Optional[Any] = None) -> Tuple[Any, Any]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        import tensorflow as tf

        input_shape = tf.shape(inputs)
        # 1 for a special token, 0 for a normal token in the special tokens mask
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        masked_indices = self.tf_bernoulli(input_shape, self.mlm_probability) & ~special_tokens_mask
        masked_indices = tf.where(token_type_ids == self.tgt_type_id, masked_indices, False)
        # Replace unmasked indices with -100 in the labels since we only compute loss on masked tokens
        labels = tf.where(masked_indices, inputs, -100)

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = self.tf_bernoulli(input_shape, 0.8) & masked_indices

        inputs = tf.where(indices_replaced, mask_token_id, inputs)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = self.tf_bernoulli(input_shape, 0.1) & masked_indices & ~indices_replaced
        random_words = tf.random.uniform(input_shape, maxval=vocab_size, dtype=tf.int64)
        inputs = tf.where(indices_random, random_words, inputs)

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels

    def tf_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        import tensorflow as tf

        # Handle dict or lists with proper padding and conversion to tensor.
        if isinstance(examples[0], Mapping):
            batch = self.tokenizer.pad(examples, return_tensors="tf", pad_to_multiple_of=self.pad_to_multiple_of)
        else:
            batch = {
                "input_ids": _tf_collate_batch(examples, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of)
            }

        # If special token mask has been preprocessed, pop it from the dict.
        special_tokens_mask = batch.pop("special_tokens_mask", None)
        if self.mlm:
            if special_tokens_mask is None:
                special_tokens_mask = [
                    self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True)
                    for val in batch["input_ids"].numpy().tolist()
                ]
                # Cannot directly create as bool
                special_tokens_mask = tf.cast(tf.convert_to_tensor(special_tokens_mask, dtype=tf.int64), tf.bool)
            else:
                special_tokens_mask = tf.cast(special_tokens_mask, tf.bool)
            batch["input_ids"], batch["labels"] = self.tf_mask_tokens(
                tf.cast(batch["input_ids"], tf.int64),
                tf.cast(batch["token_type_ids"], tf.int64),
                special_tokens_mask=special_tokens_mask,
                mask_token_id=self.tokenizer.mask_token_id,
                vocab_size=len(self.tokenizer),
            )
        else:
            labels = batch["input_ids"]
            if self.tokenizer.pad_token_id is not None:
                # Replace self.tokenizer.pad_token_id with -100
                labels = tf.where(labels == self.tokenizer.pad_token_id, -100, labels)
            else:
                labels = tf.identity(labels)  # Makes a copy, just in case
            batch["labels"] = labels
        return batch

    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        # Handle dict or lists with proper padding and conversion to tensor.
        if isinstance(examples[0], Mapping):
            batch = self.tokenizer.pad(examples, return_tensors="pt", pad_to_multiple_of=self.pad_to_multiple_of)
        else:
            batch = {
                "input_ids": _torch_collate_batch(examples, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of)
            }

        # If special token mask has been preprocessed, pop it from the dict.
        special_tokens_mask = batch.pop("special_tokens_mask", None)
        if self.mlm:
            batch["input_ids"], batch["labels"] = self.torch_mask_tokens(
                batch["input_ids"], batch["token_type_ids"], special_tokens_mask=special_tokens_mask
            )
        else:
            labels = batch["input_ids"].clone()
            if self.tokenizer.pad_token_id is not None:
                labels[labels == self.tokenizer.pad_token_id] = -100
            batch["labels"] = labels
        return batch

    def torch_mask_tokens(self, inputs: Any, token_type_ids: Any,
                          special_tokens_mask: Optional[Any] = None) -> Tuple[Any, Any]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        import torch

        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        probability_matrix[token_type_ids != self.tgt_type_id] = 0.0
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels

    def numpy_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        # Handle dict or lists with proper padding and conversion to tensor.
        if isinstance(examples[0], Mapping):
            batch = self.tokenizer.pad(examples, return_tensors="np", pad_to_multiple_of=self.pad_to_multiple_of)
        else:
            batch = {
                "input_ids": _numpy_collate_batch(examples, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of)
            }

        # If special token mask has been preprocessed, pop it from the dict.
        special_tokens_mask = batch.pop("special_tokens_mask", None)
        if self.mlm:
            batch["input_ids"], batch["labels"] = self.numpy_mask_tokens(
                batch["input_ids"], batch["token_type_ids"], special_tokens_mask=special_tokens_mask
            )
        else:
            labels = np.copy(batch["input_ids"])
            if self.tokenizer.pad_token_id is not None:
                labels[labels == self.tokenizer.pad_token_id] = -100
            batch["labels"] = labels
        return batch

    def numpy_mask_tokens(self, inputs: Any, token_type_ids: Any,
                          special_tokens_mask: Optional[Any] = None) -> Tuple[Any, Any]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        labels = np.copy(inputs)
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = np.full(labels.shape, self.mlm_probability)
        probability_matrix = np.where(token_type_ids == self.tgt_type_id, probability_matrix, 0.0)
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            special_tokens_mask = np.array(special_tokens_mask, dtype=bool)
        else:
            special_tokens_mask = special_tokens_mask.astype(bool)

        probability_matrix[special_tokens_mask] = 0
        # Numpy doesn't have bernoulli, so we use a binomial with 1 trial
        masked_indices = np.random.binomial(1, probability_matrix, size=probability_matrix.shape).astype(bool)
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = np.random.binomial(1, 0.8, size=labels.shape).astype(bool) & masked_indices
        inputs[indices_replaced] = self.tokenizer.mask_token_id

        # 10% of the time, we replace masked input tokens with random word
        # indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        indices_random = (
                np.random.binomial(1, 0.5, size=labels.shape).astype(bool) & masked_indices & ~indices_replaced
        )
        random_words = np.random.randint(
            low=0, high=len(self.tokenizer), size=np.count_nonzero(indices_random), dtype=np.int64
        )
        inputs[indices_random] = random_words

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels
