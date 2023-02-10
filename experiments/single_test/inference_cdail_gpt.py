#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: czh
@email:
@date: 2022/7/11 18:46
"""
import logging
import random
from pprint import pformat
from itertools import chain
from argparse import ArgumentParser

import torch
from torch.nn import functional as nnf
from transformers import OpenAIGPTLMHeadModel, GPT2LMHeadModel, BertTokenizer
# import gradio as gr

SPECIAL_TOKENS = ["[CLS]", "[SEP]", "[PAD]", "[speaker1]", "[speaker2]"]


def top_filtering(logits, top_k_=0, top_p_=0.0, threshold=-float('Inf'), filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k, top-p (nucleus) and/or threshold filtering
        Args:
            :param logits: logits distribution shape (vocabulary size)
            :param top_k_: <=0: no filtering, >0: keep only top k tokens with highest probability.
            :param top_p_: <=0.0: no filtering, >0.0: keep only a subset S of candidates, where S is the smallest subset
                whose total probability mass is greater than or equal to the threshold top_p.
                In practice, we select the highest probability tokens whose cumulative probability mass exceeds
                the threshold top_p.
            :param threshold: a minimal threshold to keep logits
            :param filter_value:
    """
    assert logits.dim() == 1  # Only work for batch size 1 for now - could update but it would obfuscate a bit the code
    top_k_ = min(top_k_, logits.size(-1))
    if top_k_ > 0:
        # Remove all tokens with a probability less than the last token in the top-k tokens
        indices_to_remove = logits < torch.topk(logits, top_k_)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p_ > 0.0:
        # Compute cumulative probabilities of sorted tokens
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probabilities = torch.cumsum(nnf.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probabilities > top_p_
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # Back to unsorted indices and set them to -infinity
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value

    indices_to_remove = logits < threshold
    logits[indices_to_remove] = filter_value

    return logits


def build_input_from_segments(history, reply, tokenizer_, with_eos=True):
    """ Build a sequence of input from 3 segments: persona, history and last reply """
    bos, eos, pad, speaker1, speaker2 = tokenizer_.convert_tokens_to_ids(SPECIAL_TOKENS)
    sequence = [[bos]] + history + [reply + ([eos] if with_eos else [])]
    sequence = [sequence[0]] + [[speaker2 if i % 2 else speaker1] + s for i, s in enumerate(sequence[1:])]
    instance = {"input_ids": list(chain(*sequence)),
                "token_type_ids": [bos] + [speaker2 if i % 2 else speaker1 for i, s in enumerate(sequence[1:])
                                           for _ in s]}
    return instance, sequence


def sample_sequence(history, tokenizer_, model_, args_, current_output=None):
    special_tokens_ids = tokenizer_.convert_tokens_to_ids(SPECIAL_TOKENS)
    if current_output is None:
        current_output = []

    for i in range(args_.max_length):
        instance, sequence = build_input_from_segments(history, current_output, tokenizer_, with_eos=False)
        input_ids = torch.tensor(instance["input_ids"], dtype=torch.long, device=args_.device).unsqueeze(0)
        token_type_ids = torch.tensor(instance["token_type_ids"], dtype=torch.long, device=args_.device).unsqueeze(0)

        outputs = model_(input_ids, token_type_ids=token_type_ids)
        logits = outputs.logits
        logits = logits[0, -1, :] / args_.temperature
        logits = top_filtering(logits, top_k_=args_.top_k, top_p_=args_.top_p)
        probs = nnf.softmax(logits, dim=-1)

        prev = torch.topk(probs, 1)[1] if args_.no_sample else torch.multinomial(probs, 1)
        if i < args_.min_length and prev.item() in special_tokens_ids:
            while prev.item() in special_tokens_ids:
                prev = torch.multinomial(probs, num_samples=1)

        if prev.item() in special_tokens_ids:
            break
        current_output.append(prev.item())

    return current_output


def tokenize(obj):
    if isinstance(obj, str):
        return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(obj))
    if isinstance(obj, dict):
        return dict((n, tokenize(o)) for n, o in obj.items())
    return list(tokenize(o) for o in obj)


def interact(multi_turn=True):

    history = []
    print("*************** Please input question or stop to quit ***********************")
    while True:
        raw_text = input(">>> ")
        while not raw_text:
            print('Prompt should not be empty!')
            raw_text = input(">>> ")
        if raw_text == 'stop':
            break
        raw_text = " ".join(list(raw_text.replace(" ", "")))
        if multi_turn:
            history.append(tokenize(raw_text))
        else:
            history = [tokenize(raw_text)]
        with torch.no_grad():
            out_ids = sample_sequence(history, tokenizer, model, args)
        if multi_turn:
            history.append(out_ids)
            history = history[-(2 * args.max_history + 1):]
        out_text = tokenizer.decode(out_ids, skip_special_tokens=True)
        print("".join(out_text.split(" ")))


def run(raw_text):
    raw_text = " ".join(list(raw_text.replace(" ", "")))
    history = [tokenize(raw_text)]
    with torch.no_grad():
        out_ids = sample_sequence(history, tokenizer, model, args)
    out_text = tokenizer.decode(out_ids, skip_special_tokens=True)
    result = "".join(out_text.split(" "))
    return result


if __name__ == "__main__":
    root_path = "/root/work2/work2/chenzhihao/NLP"
    model_path = f"{root_path}/datas/output_dir/CDail-GPT"

    max_history = 5
    max_length = 64
    min_length = 2
    top_k = 6
    top_p = 0.9
    temperature = 0.9
    no_sample = False
    is_multi_turn = True

    parser = ArgumentParser()
    parser.add_argument('--gpt2', action='store_true', help="use gpt2")
    parser.add_argument("--model_checkpoint", type=str, default=model_path, help="Path, url or short name of the model")
    parser.add_argument("--max_history", type=int, default=max_history,
                        help="Number of previous utterances to keep in history")
    parser.add_argument("--device", type=str, default="cuda:2" if torch.cuda.is_available() else "cpu",
                        help="Device (cuda or cpu)")

    parser.add_argument("--no_sample", type=bool, default=no_sample,
                        help="Set to use greedy decoding instead of sampling")
    parser.add_argument("--max_length", type=int, default=max_length, help="Maximum length of the output utterances")
    parser.add_argument("--min_length", type=int, default=min_length, help="Minimum length of the output utterances")
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    parser.add_argument("--temperature", type=float, default=temperature, help="Sampling softmax temperature")
    parser.add_argument("--top_k", type=int, default=top_k,
                        help="Filter top-k tokens before sampling (<=0: no filtering)")
    parser.add_argument("--top_p", type=float, default=top_p,
                        help="Nucleus filtering (top-p) before sampling (<=0.0: no filtering)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__file__)
    logger.info(pformat(args))

    if args.model_checkpoint == "":
        logging.error("Checkpoint needed!")

    random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    logger.info("Get pretrained model and tokenizer")
    tokenizer_class = BertTokenizer
    model_class = OpenAIGPTLMHeadModel if not args.gpt2 else GPT2LMHeadModel
    tokenizer = tokenizer_class.from_pretrained(args.model_checkpoint,
                                                do_lower_case=True,
                                                never_split=["[speaker1]", "[speaker2]"])
    model = model_class.from_pretrained(args.model_checkpoint)
    device = torch.device(args.device)
    args.device = device
    model.to(args.device)
    model.eval()
    text = ""
    interact(is_multi_turn)

    # demo = gr.Interface(fn=run,
    #                     inputs=gr.Textbox(lines=3, placeholder="Question Here..."),
    #                     outputs="text")
    # demo.launch(share=True)
