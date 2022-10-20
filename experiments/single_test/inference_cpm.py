#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: czh
@email:
@date: 2022/7/22 09:55
"""
import os
import sys
from argparse import ArgumentParser

import torch
import torch.nn.functional as nnf
from transformers import GPT2LMHeadModel, CpmTokenizer, GPT2Config, TextGenerationPipeline

dirname = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join('/'.join(dirname.split('/')[:-2]), 'nlp'))

from nlp.utils.generate_util import top_filtering

PREFIX = "问题:"
POSTFIX = " 答案:"
ROOT_PATH = "/data2/work2/chenzhihao/NLP-experiments/"
MODEL_CHECKPOINT_PATH = ROOT_PATH + "datas/output_dir/CPM-large2/"
TOKENIZER_PATH = "/data2/work2/chenzhihao/pretrained_models/CPM-generate"
TOP_K = 1
TOP_P = 0.0
TEMPERATURE = 0.9
DO_SAMPLE = False
OUTPUT_MAX_LENGTH = 50
OUTPUT_MIN_LENGTH = 5
DEVICE = 1  # cpu:-1


def build_input(args, context, reply):
    token_ids = context + reply
    length = len(token_ids)
    position_ids = list(range(length))

    token_ids = torch.LongTensor([token_ids]).to(args.device)
    position_ids = torch.LongTensor([position_ids]).to(args.device)
    return {
        'input_ids': token_ids,
        'position_ids': position_ids
    }


def generate_sample(args, model, context, current_output=None):
    if current_output is None:
        current_output = []
    for i in range(args.output_max_length):
        inputs = build_input(args, context, current_output)

        outputs = model(**inputs)
        logits = outputs.logits
        logits = logits[0, -1, :] / args.temperature
        logits = top_filtering(logits, top_k=args.top_k, top_p=args.top_p)
        probs = nnf.softmax(logits, dim=-1)

        prev = torch.topk(probs, 1)[1] if not args.do_sample else torch.multinomial(probs, 1)
        if i < args.output_min_length and prev.item() == args.eod_token_id:
            while prev.item() == args.eod_token_id:
                prev = torch.multinomial(probs, num_samples=1)

        if prev.item() == args.eod_token_id:
            break
        current_output.append(prev.item())

    return current_output


def interact(args, model, tokenizer: CpmTokenizer):
    print("*************** Please input question or q to quit ***********************")
    while True:
        raw_text = input("\nContext prompt (stop to exit) >>> ")
        while not raw_text:
            print('Prompt should not be empty!')
            raw_text = input("\nContext prompt (stop to exit) >>> ")
        raw_text = raw_text.strip()
        if raw_text == "stop":
            break
        text = PREFIX + raw_text + POSTFIX
        input_ids = tokenizer.encode_plus(text, add_special_tokens=False)['input_ids']
        with torch.no_grad():
            output_ids = generate_sample(args, model, input_ids)
        output_text = tokenizer.decode(output_ids, skip_special_tokens=True)
        print("".join(output_text.split(" ")))


def interact_by_pipeline(args, model, tokenizer):
    print("*************** Please input question or q to quit ***********************")
    while True:
        raw_text = input("\nContext prompt (stop to exit) >>> ")
        while not raw_text:
            print('Prompt should not be empty!')
            raw_text = input("\nContext prompt (stop to exit) >>> ")
        raw_text = raw_text.strip()
        if raw_text == "stop":
            break
        text = PREFIX + raw_text + POSTFIX
        input_ids = tokenizer.encode_plus(text, return_tensors="pt")['input_ids'].to(args.device)
        outputs = model.generate(input_ids,
                                 do_sample=args.do_sample,
                                 min_length=args.output_min_length,
                                 max_length=args.output_max_length,
                                 temperature=args.temperature,
                                 top_p=args.top_p,
                                 top_k=args.top_k)
        result = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        print(result)
        result = result.replace(text, "")
        print(PREFIX, raw_text)
        print(POSTFIX.strip(), result)


def run(args, model, tokenizer, raw_text):
    text = PREFIX + raw_text + POSTFIX
    input_ids = tokenizer.encode_plus(text, add_special_tokens=False)['input_ids']
    with torch.no_grad():
        output_ids = generate_sample(args, model, input_ids)
    output_text = tokenizer.decode(output_ids, skip_special_tokens=True)
    result = "".join(output_text.split(" "))
    return result


def mian():
    parser = ArgumentParser()
    parser.add_argument("--model_checkpoint_path", type=str, default=MODEL_CHECKPOINT_PATH,
                        help="Path, url or short name of the model")
    parser.add_argument("--tokenizer_path", type=str, default=TOKENIZER_PATH)
    parser.add_argument("--config_path", type=str, default=MODEL_CHECKPOINT_PATH)
    parser.add_argument("--device", type=str,
                        default=f"cuda:{DEVICE}" if torch.cuda.is_available() and DEVICE != -1 else "cpu",
                        help="Device (cuda or cpu)")

    parser.add_argument("--do_sample", type=bool, default=DO_SAMPLE,
                        help="Set to use greedy decoding instead of sampling")
    parser.add_argument("--output_max_length", type=int, default=256, help="Maximum length of the output utterances")
    parser.add_argument("--output_min_length", type=int, default=5, help="Minimum length of the output utterances")
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    parser.add_argument("--temperature", type=float, default=TEMPERATURE, help="Sampling softmax temperature")
    parser.add_argument("--top_k", type=int, default=TOP_K,
                        help="Filter top-k tokens before sampling (<=0: no filtering)")
    parser.add_argument("--top_p", type=float, default=TOP_P,
                        help="Nucleus filtering (top-p) before sampling (<=0.0: no filtering)")
    args = parser.parse_args()

    device = torch.device(args.device)
    args.device = device

    tokenizer = CpmTokenizer.from_pretrained(args.tokenizer_path)
    args.pad_id, args.eod_token_id = tokenizer.convert_tokens_to_ids(["<pad>", "<eod>"])

    model_config = GPT2Config.from_pretrained(args.config_path)
    model = GPT2LMHeadModel.from_pretrained(args.model_checkpoint_path, config=model_config)
    model.to(device)
    model.eval()

    # interact(args, model, tokenizer)
    interact_by_pipeline(args, model, tokenizer)


if __name__ == "__main__":
    mian()
