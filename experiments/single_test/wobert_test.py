#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: czh
@email:
@date: 2022/11/11 15:06
"""
import regex
import torch
from transformers import BertForMaskedLM, BertTokenizer
from roformer import RoFormerForMaskedLM

from nlp.utils.wobert_tokenization import WoBertTokenizer


texts = [
    "今天[MASK]很好，我[MASK]去公园玩。"
    ]

pretrained_model_or_path_list = [
    "/root/work2/work2/chenzhihao/pretrained_models/chinese_wobert_plus",
    "/root/work2/work2/chenzhihao/pretrained_models/chinese_wobert_base"
]
for path in pretrained_model_or_path_list:
    tokenizer = WoBertTokenizer.from_pretrained(path)
    model = BertForMaskedLM.from_pretrained(path)
    for text in texts:
        # inputs = tokenizer(text, return_tensors="pt")
        # with torch.no_grad():
        #     outputs = model(**inputs).logits[0]
        # outputs_sentence = ""
        # for i, id in enumerate(tokenizer.encode(text)):
        #     if id == tokenizer.mask_token_id:
        #         tokens = tokenizer.convert_ids_to_tokens(outputs[i].topk(k=5)[1])
        #         outputs_sentence += "[" + "||".join(tokens) + "]"
        #     else:
        #         outputs_sentence += "".join(
        #             tokenizer.convert_ids_to_tokens([id],
        #                                             skip_special_tokens=True))
        # print(outputs_sentence)
        output_sent = text
        while len(regex.findall(r"MASK", output_sent)) > 0:
            inputs = tokenizer(output_sent, return_tensors='pt')
            with torch.no_grad():
                outputs = model(**inputs).logits[0]
            text_list = output_sent.split("[MASK]")
            outputs_sentence = text_list[0]
            for i, id in enumerate(tokenizer.encode(output_sent)):
                if id == tokenizer.mask_token_id:
                    tokens = tokenizer.convert_ids_to_tokens(outputs[i].topk(k=5)[1])
                    # print(tokens)
                    outputs_sentence += "".join(tokens[0])
                    break
            output_sent = outputs_sentence + "[MASK]".join(text_list[1:])
        print(output_sent)


print("****"*10)

roberta_model_list = [
    "/root/work2/work2/chenzhihao/pretrained_models/chinese-roberta-wwm-ext",
    "/root/work2/work2/chenzhihao/pretrained_models/macbert-chinese-base",
    ]

for path in roberta_model_list:
    tokenizer = BertTokenizer.from_pretrained(path)
    model = BertForMaskedLM.from_pretrained(path)
    for text in texts:
    #     inputs = tokenizer(text, return_tensors="pt")
    #     with torch.no_grad():
    #         outputs = model(**inputs).logits[0]
    #     outputs_sentence = ""
    #     for i, id in enumerate(tokenizer.encode(text)):
    #         if id == tokenizer.mask_token_id:
    #             tokens = tokenizer.convert_ids_to_tokens(outputs[i].topk(k=5)[1])
    #             outputs_sentence += "[" + "||".join(tokens) + "]"
    #         else:
    #             outputs_sentence += "".join(
    #                 tokenizer.convert_ids_to_tokens([id],
    #                                                 skip_special_tokens=True))
    #     print(outputs_sentence)
        output_sent = text
        while len(regex.findall(r"MASK", output_sent)) > 0:
            inputs = tokenizer(output_sent, return_tensors='pt')
            with torch.no_grad():
                outputs = model(**inputs).logits[0]
            text_list = output_sent.split("[MASK]")
            outputs_sentence = text_list[0]
            for i, id in enumerate(tokenizer.encode(output_sent)):
                if id == tokenizer.mask_token_id:
                    tokens = tokenizer.convert_ids_to_tokens(outputs[i].topk(k=5)[1])
                    # print(tokens)
                    outputs_sentence += "".join(tokens[0])
                    break
            output_sent = outputs_sentence + "[MASK]".join(text_list[1:])
        print(output_sent)


roformer_model_name = "junnyu/roformer_v2_chinese_char_base"
tokenizer = BertTokenizer.from_pretrained(roformer_model_name)
model = RoFormerForMaskedLM.from_pretrained(roformer_model_name)
for text in texts:
    #     inputs = tokenizer(text, return_tensors="pt")
    #     with torch.no_grad():
    #         outputs = model(**inputs).logits[0]
    #     outputs_sentence = ""
    #     for i, id in enumerate(tokenizer.encode(text)):
    #         if id == tokenizer.mask_token_id:
    #             tokens = tokenizer.convert_ids_to_tokens(outputs[i].topk(k=5)[1])
    #             outputs_sentence += "[" + "||".join(tokens) + "]"
    #         else:
    #             outputs_sentence += "".join(
    #                 tokenizer.convert_ids_to_tokens([id],
    #                                                 skip_special_tokens=True))
    #     print(outputs_sentence)
    output_sent = text
    while len(regex.findall(r"MASK", output_sent)) > 0:
        inputs = tokenizer(output_sent, return_tensors='pt')
        with torch.no_grad():
            outputs = model(**inputs).logits[0]
        text_list = output_sent.split("[MASK]")
        outputs_sentence = text_list[0]
        for i, id in enumerate(tokenizer.encode(output_sent)):
            if id == tokenizer.mask_token_id:
                tokens = tokenizer.convert_ids_to_tokens(outputs[i].topk(k=5)[1])
                # print(tokens)
                outputs_sentence += "".join(tokens[0])
                break
        output_sent = outputs_sentence + "[MASK]".join(text_list[1:])
    print(output_sent)

