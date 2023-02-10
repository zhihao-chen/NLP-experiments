#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: czh
@email:
@date: 2023/2/7 17:47
"""
# https://github.com/Liadrinz/transformers-unilm/blob/main/examples/dialog/run_dialog.py
import os
import sys
from pathlib import Path
from argparse import ArgumentParser

import torch
from transformers.trainer_seq2seq import Seq2SeqTrainer
from transformers.trainer_utils import IntervalStrategy
from transformers.training_args import TrainingArguments

dirname = os.path.dirname(os.path.abspath(__file__))
print(dirname)
sys.path.append(os.path.join('/'.join(dirname.split('/')[:-2])))
from nlp.models.unilm_model_liadrinz import UniLMForConditionalGeneration
from nlp.processors.unilm_liadrinz_processor import DataCollatorForUniLMSeq2Seq, CorpusDataset
from nlp.utils.tokenization_unilm import UniLMTokenizerLiadrinz


MODELS = {
    "unilm": (UniLMTokenizerLiadrinz, UniLMForConditionalGeneration),
}


def train(args):
    tokenizer_cls, model_cls = MODELS[args.model_type]
    tokenizer = tokenizer_cls.from_pretrained(args.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = model_cls.from_pretrained(args.model_name_or_path)
    collator = DataCollatorForUniLMSeq2Seq(tokenizer, mlm=True, mlm_probability=args.mask_prob)
    if args.model_recover_path:
        state_dict = torch.load(args.model_recover_path)
        model.load_state_dict(state_dict)
    n_gpus = torch.cuda.device_count()
    if n_gpus > 0:
        args.batch_size //= n_gpus
    dataset = CorpusDataset(tokenizer, args.corpus_file, args.max_seq_len)
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        do_train=True,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.lr,
        num_train_epochs=args.num_train_epochs,
        no_cuda=args.no_cuda,
        fp16=args.fp16,
        save_strategy=IntervalStrategy.STEPS,
        save_steps=500,
        logging_strategy=IntervalStrategy.STEPS,
        logging_steps=1,
        local_rank=args.local_rank,
        remove_unused_columns=False,
        seed=args.seed
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        data_collator=collator,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )
    resume = False
    if list(Path(args.output_dir).glob("checkpoint-*")):
        resume = True
    trainer.train(resume_from_checkpoint=resume)


def main():
    parser = ArgumentParser()
    parser.add_argument("task", type=str, default="train")
    parser.add_argument("--model_type", type=str, default="bart")
    parser.add_argument("--model_name_or_path", type=str, default="bart-base")
    parser.add_argument("--model_recover_path", type=str, default=None)
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--corpus_file", type=str, default="data/train.txt")
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--mask_prob", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", action="store_true")
    args, _ = parser.parse_known_args()
    if args.task == "train":
        parser.add_argument("--local_rank", type=int, default=-1)
        parser.add_argument("--output_dir", type=str, default="output_dir")
        parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
        parser.add_argument("--lr", type=float, default=3e-5)
        parser.add_argument("--num_train_epochs", type=int, default=10)
        args = parser.parse_args()
        train(args)


if __name__ == "__main__":
    main()
