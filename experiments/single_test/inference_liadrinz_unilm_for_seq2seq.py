#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: czh
@email:
@date: 2023/2/7 18:29
"""
import os
import sys

import torch
dirname = os.path.dirname(os.path.abspath(__file__))
print(dirname)
sys.path.append(os.path.join('/'.join(dirname.split('/')[:-2])))
from nlp.models.unilm_model_liadrinz import UniLMForConditionalGeneration
from nlp.processors.unilm_liadrinz_processor import DataCollatorForUniLMSeq2Seq, CorpusDataset
from nlp.utils.tokenization_unilm import UniLMTokenizerLiadrinz

project_path = "/root/work2/work2/chenzhihao/NLP/"
model_name_or_path = project_path + "datas/output_dir/unilm/liadrinz_unilm/seq2seq_on_natural_conv/checkpoint-1500"
device = "cuda:7"
TOP_K = 0
TOP_P = 0.9
TEMPERATURE = 0.7
DO_SAMPLE = True
OUTPUT_MAX_LENGTH = 32
OUTPUT_MIN_LENGTH = 1
PREFIX = "用户："
POSTFIX = " 机器人："


def interact(tokenizer: UniLMTokenizerLiadrinz, model: UniLMForConditionalGeneration):
    history = ""
    while True:
        raw_text = input("\n输入：")
        while not raw_text:
            print('Prompt should not be empty!')
            raw_text = input("\n输入：")
        raw_text = raw_text.strip()
        if raw_text == "stop":
            break
        history += PREFIX + raw_text + "。" + POSTFIX
        inputs = tokenizer(history, return_tensors='pt')
        for k in inputs:
            inputs[k] = inputs[k].to(device)
        with torch.no_grad():
            if DO_SAMPLE:
                output_ids = model.generate(**inputs,
                                            max_new_tokens=OUTPUT_MAX_LENGTH,
                                            min_length=OUTPUT_MIN_LENGTH,
                                            top_k=TOP_K,
                                            top_p=TOP_P,
                                            temperature=TEMPERATURE,
                                            do_sample=True,
                                            no_repeat_ngram_size=3)
            else:
                output_ids = model.generate(**inputs, max_new_tokens=OUTPUT_MAX_LENGTH, num_beams=1, length_penalty=0.6)
        output_text = tokenizer.decode(output_ids[0])
        result = output_text.split("[SEP]")[1].strip()
        print(result)
        result = "".join(result.split())
        result = result.split(PREFIX.replace("：", ":"))[0]
        print("\n回复：", result)
        history += result
        print(history)
        if len(history) > 512:
            history = ""


def main():
    tokenizer = UniLMTokenizerLiadrinz.from_pretrained(model_name_or_path)
    model = UniLMForConditionalGeneration.from_pretrained(model_name_or_path)
    model.to(device)

    interact(tokenizer, model)


if __name__ == "__main__":
    main()
