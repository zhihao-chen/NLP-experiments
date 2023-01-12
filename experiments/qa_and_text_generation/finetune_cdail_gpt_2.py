#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: czh
@email:
@date: 2023/1/11 19:31
"""
# 对话任务
import os
import sys
import codecs
import logging
import math
import json
from tqdm import tqdm
from pprint import pformat
from argparse import ArgumentParser

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import Dataset, DataLoader
from transformers import AdamW, Trainer, Pipeline, get_scheduler
from transformers import (OpenAIGPTLMHeadModel, OpenAIGPTConfig, GPT2LMHeadModel, BertTokenizer, set_seed,
                          GPT2Config, CONFIG_NAME, WEIGHTS_NAME)
dirname = os.path.dirname(os.path.abspath(__file__))
print(dirname)
sys.path.append(os.path.join('/'.join(dirname.split('/')[:-2])))

from nlp.processors.dataset import CdailDataset

logger = logging.getLogger(__file__)


def load_lccc_datas(input_file):
    """
    读取lccc对话数据，数据格式[[s1, s2, s3, ...], [s1, s2, s3, ...]]
    :param input_file:
    :return:
    """
    with codecs.open(input_file, encoding="utf8") as fr:
        datasets = json.load(fr)
    return datasets


def load_natural_conv_dataset(dialogue_release, train_path, valid_path, test_path):
    """
    读取腾讯natural_conv 对话数据集。
    :param dialogue_release: dialog_release.json 格式：{'dialog_id': '0_3', 'document_id': 0, 'content': []}
    :param train_path:train.txt  每行表示"dialog_id"
    :param valid_path:dev.txt 每行表示"dialog_id"
    :param test_path:test.txt 每行表示"dialog_id"
    :return:
    """
    id2content = {}
    with codecs.open(dialogue_release, encoding='utf8') as fr:
        dialogue_list = json.load(fr)
        for item in dialogue_list:
            dialog_id = item['dialog_id']
            content = item['content']
            id2content[dialog_id] = content

    def load_samples(input_file):
        samples = []
        with codecs.open(input_file, encoding='utf8') as fr_train:
            for line in fr_train:
                line = line.strip()
                if not line:
                    continue
                if line in id2content:
                    sample = id2content[line]
                    samples.append(sample)
        return samples
    train_samples = load_samples(train_path)
    print("total train data: ", len(train_samples))
    valid_samples = load_samples(valid_path)
    print("total train data: ", len(valid_samples))
    test_samples = load_samples(test_path)
    print("total train data: ", len(test_samples))
    return train_samples, valid_samples, test_samples


def average_distributed_scalar(scalar, args):
    """ Average a scalar over the nodes if we are in distributed training. We use this for distributed evaluation. """
    if args.local_rank == -1:
        return scalar
    scalar_t = torch.tensor(scalar, dtype=torch.float, device=args.device) / torch.distributed.get_world_size()
    torch.distributed.all_reduce(scalar_t, op=torch.distributed.ReduceOp.SUM)
    return scalar_t.item()


def get_parse():
    parser = ArgumentParser()
    parser.add_argument('--gpt2', action='store_true', help="use gpt2")
    parser.add_argument("--model_checkpoint", type=str, default="config/cgpt/", help="Path or URL of the model")
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--from_step", type=int, default=-1, help="Init learning rate from this step")
    parser.add_argument('--pretrained', action='store_true', help="If False train from scratch")
    parser.add_argument("--data_path", type=str, default="",
                        help="Path or url of the dataset. ")
    parser.add_argument('--data_type', type=str, default="lccc", choices=["lccc", "natural_conv"])
    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--do_valid", action="store_true")
    parser.add_argument("--do_test", action="store_true")
    parser.add_argument("--train_path", type=str, default=None,
                        help="Path of the train dataset for dist dataset. ")
    parser.add_argument("--valid_path", type=str, default=None,
                        help="Path of the valid dataset for dist dataset. ")
    parser.add_argument("--test_path", type=str, default=None,
                        help="Path of the test dataset for dist dataset. ")
    parser.add_argument("--dataset_cache", type=str, default="dataset_cache",
                        help="Path or url of the dataset cache")
    parser.add_argument('--log_file', '-log_file', type=str, default="", help="Output logs to a file under this path")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of subprocesses for data loading")
    parser.add_argument("--n_epochs", type=int, default=70, help="Number of training epochs")
    parser.add_argument("--train_batch_size", type=int, default=2, help="Batch size for training")
    parser.add_argument("--valid_batch_size", type=int, default=2, help="Batch size for validation")
    parser.add_argument("--max_history", type=int, default=15, help="Number of previous exchanges to keep in history")
    parser.add_argument("--scheduler", type=str, default="noam", choices=['noam', 'linear'], help="method of optim")
    parser.add_argument("--n_emd", type=int, default=768, help="Number of n_emd in config file (for noam)")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--eval_before_start", action='store_true',
                        help="If true start with a first evaluation before training")
    parser.add_argument("--warmup_steps", type=int, default=5000, help="Warm up steps")
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--valid_steps", type=int, default=5000, help="Perfom validation every X steps")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=64,
                        help="Accumulate gradients on several steps")
    parser.add_argument("--max_norm", type=float, default=1.0, help="Clipping gradient norm")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device (cuda or cpu)")
    parser.add_argument("--fp16", type=str, default="",
                        help="Set to O0, O1, O2 or O3 for fp16 training (see apex documentation)")
    parser.add_argument("--fp16_backend", type=str, default="amp")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="Local rank for distributed training (-1: not distributed)")
    parser.add_argument("--seed", type=int, default=2333)
    args = parser.parse_args()
    return args


def init_model(args):
    model_class = OpenAIGPTLMHeadModel if not args.gpt2 else GPT2LMHeadModel
    config_class = OpenAIGPTConfig if not args.gpt2 else GPT2Config
    tokenizer_class = BertTokenizer
    if args.pretrained:
        tokenizer = tokenizer_class.from_pretrained(args.model_checkpoint,
                                                    do_lower_case=True,
                                                    never_split=["[speaker1]", "[speaker2]"])
        model = model_class.from_pretrained(args.model_checkpoint, ignore_mismatched_sizes=True)
    else:
        tokenizer = tokenizer_class(os.path.join(args.model_checkpoint, "vocab.txt"),
                                    do_lower_case=True,
                                    never_split=["[speaker1]", "[speaker2]"])
        config = config_class.from_json_file(os.path.join(args.model_checkpoint, CONFIG_NAME))
        model = model_class(config)

    return tokenizer, model


def evaluate(valid_loader, model, args):
    model.eval()

    loss_func = nn.CrossEntropyLoss(ignore_index=-1)
    total_loss = 0.0
    global_steps = 0
    for step, batch in enumerate(tqdm(valid_loader, desc="Evaluating on QA dataset")):
        input_ids = batch['input_ids'].to(args.device)
        token_type_ids = batch['token_type_ids'].to(args.device)
        lm_labels = batch['labels'].to(args.device)

        with torch.no_grad():
            outputs = model(input_ids, token_type_ids=token_type_ids)
            lm_logits = outputs.logits
            lm_logits_flat_shifted = lm_logits[..., :-1, :].contiguous().view(-1, lm_logits.size(-1))
            lm_labels_flat_shifted = lm_labels[..., 1:].contiguous().view(-1)

            loss = loss_func(lm_logits_flat_shifted, lm_labels_flat_shifted)
            total_loss += loss.item()
            global_steps += 1

    avg_loss = total_loss / global_steps
    average_ppl = math.exp(avg_loss)

    return {'avg_loss': avg_loss, 'avg_ppl': average_ppl}


def train(train_samples, valid_sample, model, tokenizer, args):
    train_dataset = CdailDataset(train_samples, tokenizer)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if args.distributed else None
    train_loader = DataLoader(train_dataset,
                              sampler=train_sampler,
                              collate_fn=train_dataset.collate,
                              num_workers=args.num_workers,
                              batch_size=args.train_batch_size,
                              shuffle=(not args.distributed))

    valid_dataset = CdailDataset(valid_sample, tokenizer)
    valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_dataset) if args.distributed else None
    valid_loader = DataLoader(valid_dataset, sampler=valid_sampler,
                              collate_fn=valid_dataset.collate,
                              num_workers=args.num_workers,
                              batch_size=args.valid_batch_size,
                              shuffle=False)
    t_total = len(train_samples) / args.gradient_accumulation_steps * args.n_epochs
    optimizer = AdamW([{'params': model.parameters(), 'initial_lr': args.lr}], lr=args.lr, correct_bias=True)
    args.warmup_steps = int(t_total * args.warmup_ratio)
    scheduler = get_scheduler(args.scheduler, optimizer=optimizer, num_warmup_steps=args.warmup_steps,
                              num_training_steps=t_total)
    # Prepare model for FP16 and distributed training if needed (order is important, distributed should be the last)
    # if args.fp16:
    #     from apex import amp  # Apex is only required if we use fp16 training
    #     model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16)
    if args.fp16 and args.fp16_backend == 'amp':
        from torch.cuda.amp import autocast, GradScaler
        scaler = GradScaler()
    if args.distributed:
        model = DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num train examples = %d", len(train_samples))
    logger.info("  Num Epochs = %d", args.n_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    model.zero_grad()
    best_score = float('inf')
    best_epoch = 0
    global_steps = 0
    total_loss = 0.0
    loss_func = nn.CrossEntropyLoss(ignore_index=-1)

    for epoch in range(args.n_epochs):
        model.train()
        print(f"******Current epoch {epoch}/{args.n_epochs}******")
        for step, batch in enumerate(tqdm(train_loader, desc="Training CDail-GPT_LCCC-larger in QA dataset")):
            input_ids = batch['input_ids'].to(args.device)
            token_type_ids = batch['token_type_ids'].to(args.device)
            lm_labels = batch['labels'].to(args.device)

            outputs = model(input_ids=input_ids, token_type_ids=token_type_ids)
            lm_logits = outputs.logits
            lm_logits_flat_shifted = lm_logits[..., :-1, :].contiguous().view(-1, lm_logits.size(-1))
            lm_labels_flat_shifted = lm_labels[..., 1:].contiguous().view(-1)

            loss = loss_func(lm_logits_flat_shifted, lm_labels_flat_shifted)
            # loss = outputs.loss
            loss = loss / args.gradient_accumulation_steps

            if args.fp16 and args.fp16_backend == 'amp':
                scaler.scale(loss).backward()
            else:
                loss.backward()
            total_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16 and args.fp16_backend == 'amp':
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_steps += 1

            if (step + 1) % args.valid_steps == 0:
                results = evaluate(valid_loader, model, args)
                valid_loss = results['avg_loss']
                valid_ppl = results['avg_ppl']

                if valid_ppl < best_score:
                    best_epoch = epoch
                    best_score = valid_ppl
                    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                    if not os.path.exists(args.output_dir):
                        os.makedirs(args.output_dir)

                    model_to_save.save_pretrained(args.output_dir)
                    tokenizer.save_vocabulary(args.output_dir)

                logger.info(json.dumps({'epoch': epoch, 'global_steps': global_steps,
                                        'train_loss': total_loss / global_steps,
                                        'valid_loss': valid_loss, 'valid_ppl': valid_ppl,
                                        'best_score': best_score, 'best_epoch': best_epoch},
                                       ensure_ascii=False, indent=2))


def main():
    args = get_parse()

    set_seed(args.seed)
    # logging is set to INFO (resp. WARN) for main (resp. auxiliary) process.
    # logger.info => log main process only, logger.warning => log all processes
    logging.basicConfig(level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Running process %d", args.local_rank)
    logger.info("Arguments: %s", pformat(args))

    # Initialize distributed training if needed
    args.distributed = (args.local_rank != -1)
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        args.device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
    else:
        args.device = torch.device(args.device)

    logger.info("Prepare tokenizer, pretrained model and optimizer - add special tokens for fine-tuning")
    tokenizer, model = init_model(args)
    model.to(args.device)

    logger.info("Prepare train samples and valid samples")
    assert args.train_path
    assert args.valid_path
    assert args.test_path

    train_path = os.path.join(args.data_path, args.train_path)
    valid_path = os.path.join(args.data_path, args.valid_path)
    test_path = os.path.join(args.data_path, args.test_path)
    if args.data_type == "lccc":
        train_samples = load_lccc_datas(train_path)
        valid_samples = load_lccc_datas(valid_path)
        test_samples = load_lccc_datas(test_path)
    elif args.data_type == "natural_conv":
        dialog_release = os.path.join(args.data_path, "dialog_release.json")
        train_samples, valid_samples, test_samples = load_natural_conv_dataset(dialog_release, train_path,
                                                                               valid_path, test_path)
    else:
        raise ValueError("data_type must be one of [lccc, natural_conv]")

    if args.eval_before_start:
        valid_dataset = CdailDataset(train_samples, tokenizer)
        valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_dataset) if args.distributed else None
        valid_loader = DataLoader(valid_dataset, sampler=valid_sampler,
                                  collate_fn=valid_dataset.collate,
                                  num_workers=args.num_workers,
                                  batch_size=args.valid_batch_size,
                                  shuffle=False)
        valid_loss, valid_ppl = evaluate(valid_loader, model, args)
        print(f"valid loss: {valid_loss}\tvalid ppl: {valid_ppl}")

    if args.do_train:
        train(train_samples, valid_samples, model, tokenizer, args)
    if args.do_valid:
        args.model_checkpoint = args.output_dir
        tokenizer, model = init_model(args)
        model.to(args.device)

        valid_dataset = CdailDataset(valid_samples, tokenizer)
        valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_dataset) if args.distributed else None
        valid_loader = DataLoader(valid_dataset, sampler=valid_sampler,
                                  collate_fn=valid_dataset.collate,
                                  num_workers=args.num_workers,
                                  batch_size=args.valid_batch_size,
                                  shuffle=False)
        valid_loss, valid_ppl = evaluate(valid_loader, model, args)
        print(f"valid loss: {valid_loss}\tvalid ppl: {valid_ppl}")

    if args.do_test:
        args.model_checkpoint = args.output_dir
        tokenizer, model = init_model(args)
        model.to(args.device)

        test_dataset = CdailDataset(test_samples, tokenizer)
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset) if args.distributed else None
        test_loader = DataLoader(test_dataset, sampler=test_sampler,
                                 collate_fn=test_dataset.collate,
                                 num_workers=args.num_workers,
                                 batch_size=args.valid_batch_size,
                                 shuffle=False)
        test_loss, test_ppl = evaluate(test_loader, model, args)
        print(f"valid loss: {test_loss}\tvalid ppl: {test_ppl}")


if __name__ == "__main__":
    main()

