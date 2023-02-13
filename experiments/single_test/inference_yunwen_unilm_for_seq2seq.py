#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: czh
@email:
@date: 2023/2/10 17:38
"""
import json
import os
import sys
import logging
import glob
import math
import argparse
from tqdm import tqdm

import pickle
import torch
from transformers import set_seed

dirname = os.path.dirname(os.path.abspath(__file__))
print(dirname)
sys.path.append(os.path.join('/'.join(dirname.split('/')[:-2])))
from nlp.utils.tokenization_unilm import UniLMTokenizerYunWen, WhitespaceTokenizer
from nlp.models.unilm_model_yunwen import UnilmForSeq2SeqDecode, UnilmConfig
from nlp.processors.unlim_yunwen_preprocessor import Preprocess4Seq2seqDecode, batch_list_to_batch_tensors

ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys())
                  for conf in (UnilmConfig,)), ())
MODEL_CLASSES = {
    'unilm': (UnilmConfig, UnilmForSeq2SeqDecode, UniLMTokenizerYunWen)
}

project_path = "/root/work2/work2/chenzhihao/NLP"
model_name_or_path = "/root/work2/work2/chenzhihao/pretrained_models/torch_unilm_model"
model_recover_path = project_path + "/datas/output_dir/unilm/yunwen_unilm/seq2seq_on_natural_conv/model.5.bin"
DEVICE = "cuda"
BEAM_SIZE = 2
FP16 = True
fp16_opt_level = 'O1'
length_penalty = 0
forbid_duplicate_ngrams = False
forbid_ignore_word = ""
ngram_size = 3
max_tgt_length = 128
max_seq_length = 512
prefix = "用户："
postfix = " 机器人："


def detokenize(tk_list):
    r_list = []
    for tk in tk_list:
        if tk.startswith('##') and len(r_list) > 0:
            r_list[-1] = r_list[-1] + tk[2:]
        else:
            r_list.append(tk)
    return r_list


def preprocess(text, tokenizer: UniLMTokenizerYunWen):
    _tril_matrix = torch.tril(torch.ones((max_seq_length, max_seq_length), dtype=torch.long))
    tokens = tokenizer.tokenize(text)
    padded_tokens_a = ['[CLS]'] + tokens + ['[SEP]']
    a_len = len(tokens)
    max_len_in_batch = min(max_tgt_length + a_len + 2, max_seq_length)
    input_ids = tokenizer.convert_tokens_to_ids(padded_tokens_a)
    segment_ids = [4] * (len(padded_tokens_a)) + [5] * (max_len_in_batch - len(padded_tokens_a))

    position_ids = []
    for i in range(a_len + 2):
        position_ids.append(i)
    for i in range(a_len + 2, max_len_in_batch):
        position_ids.append(i)
    input_mask = torch.zeros(
        max_len_in_batch, max_len_in_batch, dtype=torch.long)
    input_mask[:, :len(tokens) + 2].fill_(1)
    second_st, second_end = len(padded_tokens_a), max_len_in_batch

    input_mask[second_st:second_end, second_st:second_end].copy_(
        _tril_matrix[:second_end - second_st, :second_end - second_st])
    input_ids = torch.LongTensor([input_ids])
    segment_ids = torch.LongTensor([segment_ids])
    position_ids = torch.LongTensor([position_ids])
    input_mask = torch.LongTensor([input_mask])
    return input_ids, segment_ids, position_ids, input_mask


def generate(model, tokenizer, instances):
    with torch.no_grad():
        batch = batch_list_to_batch_tensors(instances)
        batch = [t.to(DEVICE) if t is not None else None for t in batch]
        input_ids, token_type_ids, position_ids, input_mask = batch
        traces = model(input_ids, token_type_ids, position_ids, input_mask)
        if BEAM_SIZE > 1:
            traces = {k: v.tolist() for k, v in traces.items()}
            output_ids = traces['pred_seq']
        else:
            output_ids = traces.tolist()
    print(output_ids)
    output_buf = tokenizer.convert_ids_to_tokens(output_ids[0])
    output_tokens = []
    for t in output_buf:
        if t in ("[SEP]", "[PAD]"):
            break
        output_tokens.append(t)
    output_sequence = ''.join(detokenize(output_tokens))
    return output_sequence


def interact(model, tokenizer, processor):
    max_src_length = max_seq_length - 2 - max_tgt_length
    torch.cuda.empty_cache()
    model.eval()
    print("*************** Please input question or 'stop' to quit ***********************")
    history = ""
    while True:
        raw_text = input("\n输入：")
        while not raw_text:
            print('Prompt should not be empty!')
            raw_text = input("\n输入：")
        raw_text = raw_text.strip()
        if raw_text == "stop":
            break
        history += prefix + raw_text + postfix
        instance = processor((tokenizer.tokenize(history)[:max_src_length], len(history)))
        output_lines = generate(model, tokenizer, [instance])
        print(output_lines)
        history += output_lines
        if len(history) > max_src_length:
            break


def main():
    config = UnilmConfig.from_pretrained(model_name_or_path, max_position_embeddings=max_seq_length)
    tokenizer = UniLMTokenizerYunWen.from_pretrained(model_name_or_path, do_lower_case=True)
    model_recover = torch.load(model_recover_path)

    mask_word_id, eos_word_ids, sos_word_id = tokenizer.convert_tokens_to_ids(
        ["[MASK]", "[SEP]", "[S2S_SOS]"])
    forbid_ignore_set = None
    if forbid_ignore_word:
        w_list = []
        for w in forbid_ignore_word.split('|'):
            if w.startswith('[') and w.endswith(']'):
                w_list.append(w.upper())
            else:
                w_list.append(w)
        forbid_ignore_set = set(tokenizer.convert_tokens_to_ids(w_list))

    model = UnilmForSeq2SeqDecode.from_pretrained(model_name_or_path, state_dict=model_recover, config=config,
                                                  mask_word_id=mask_word_id, search_beam_size=BEAM_SIZE,
                                                  length_penalty=length_penalty,
                                                  eos_id=eos_word_ids, sos_id=sos_word_id,
                                                  forbid_duplicate_ngrams=forbid_duplicate_ngrams,
                                                  forbid_ignore_set=forbid_ignore_set, ngram_size=ngram_size,
                                                  min_len=1)
    del model_recover

    model.to(DEVICE)

    if FP16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model = amp.initialize(model, opt_level=fp16_opt_level)

    bi_uni_pipeline = Preprocess4Seq2seqDecode(list(tokenizer.vocab.keys()), tokenizer.convert_tokens_to_ids,
                                               max_seq_length, max_tgt_length=max_tgt_length)

    interact(model, tokenizer, bi_uni_pipeline)


if __name__ == '__main__':
    main()
