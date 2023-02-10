#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: czh
@email:
@date: 2023/1/13 15:13
"""
import math
import os
import sys
import json
import codecs
import random
import logging
from argparse import ArgumentParser

from tqdm.auto import tqdm
from pprint import pformat

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from transformers import get_scheduler, AdamW, GPT2LMHeadModel, CpmTokenizer, GPT2Config, set_seed
from accelerate import Accelerator, DistributedType
from accelerate.tracking import WandBTracker
from rouge import Rouge

dirname = os.path.dirname(os.path.abspath(__file__))
print(dirname)
sys.path.append(os.path.join('/'.join(dirname.split('/')[:-2])))
from nlp.processors.dataset import CPMDataset
from nlp.tools.accelerate_tracker import CustomWandbTracker

logger = logging.getLogger(__file__)

MAX_GPU_BATCH_SIZE = 16


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def load_lccc_datas(args, input_file):
    """
    读取lccc对话数据，数据格式[[s1, s2, s3, ...], [s1, s2, s3, ...]]
    :param args:
    :param input_file:
    :return:
    """
    all_samples = []
    with codecs.open(input_file, encoding="utf8") as fr:
        datasets = json.load(fr)
    for data in datasets:
        sample = []
        for i, d in enumerate(data[-2 * args.max_history:]):
            if i == 0:
                sp = args.speaker1
            else:
                sp = args.speaker2
            text = sp + d
            sample_len = sum([len(s) for s in sample])
            if sample_len + len(text) <= args.max_seq_length:
                sample.append(text)
            else:
                if sp == args.speaker2 and args.speaker1 in sample[-1]:
                    sample = sample[:-1]
            if sample:
                all_samples.append("".join(sample))
    print("total session: ", len(all_samples))
    return datasets


def load_natural_conv_dataset(args, dialogue_release, train_path, valid_path, test_path):
    """
    读取腾讯natural_conv 对话数据集。
    :param args:
    :param dialogue_release: dialog_release.json 格式：{'dialog_id': '0_3', 'document_id': 0, 'content': []}
    :param train_path:train.txt  每行表示"dialog_id"
    :param valid_path:dev.txt 每行表示"dialog_id"
    :param test_path:test.txt 每行表示"dialog_id"
    :return:
    """
    id2content = {}
    max_len = 0
    with codecs.open(dialogue_release, encoding='utf8') as fr:
        dialogue_list = json.load(fr)
        for item in dialogue_list:
            dialog_id = item['dialog_id']
            content = item['content']
            id2content[dialog_id] = content
            ll = sum([len(c) for c in content])
            max_len = max(ll, max_len)

    def load_samples(input_file):
        all_samples = []
        with codecs.open(input_file, encoding='utf8') as fr_train:
            for line in fr_train:
                line = line.strip()
                if not line:
                    continue
                if line in id2content:
                    contents = id2content[line]
                    sample = []
                    for i, d in enumerate(contents[-2 * args.max_history:]):
                        if i % 2 == 0:
                            sp = args.speaker1
                        else:
                            sp = args.speaker2
                        text = sp + d
                        sample_len = sum([len(s) for s in sample])
                        if sample_len + len(text) <= args.max_seq_length:
                            sample.append(text)
                        else:
                            if sp == args.speaker2 and args.speaker1 in sample[-1]:
                                sample = sample[:-1]
                    if len(sample) >= 2:
                        # print(sample)
                        all_samples.append("".join(sample))
                # break
        return all_samples
    print("max len: ", max_len)
    train_samples = load_samples(train_path)
    print("total train data: ", len(train_samples))
    valid_samples = load_samples(valid_path)
    print("total train data: ", len(valid_samples))
    test_samples = load_samples(test_path)
    print("total train data: ", len(test_samples))
    return train_samples, valid_samples, test_samples


def init_model(args):
    model_class = GPT2LMHeadModel
    if args.pretrained:
        tokenizer = CpmTokenizer.from_pretrained(args.model_checkpoint, do_lower_case=True)
        model = model_class.from_pretrained(args.model_checkpoint)
    else:
        tokenizer = CpmTokenizer(os.path.join(args.model_checkpoint, "vocab.txt"), do_lower_case=True)
        if not args.config_path:
            config_path = os.path.join(args.model_checkpoint, 'config.json')
        else:
            config_path = args.config_path
        model_config = GPT2Config.from_json_file(config_path)
        model = model_class(config=model_config)
    return tokenizer, model


def get_parser():
    parser = ArgumentParser()
    parser.add_argument('--model_checkpoint', type=str, default="TsinghuaAI/CPM-Generate", help="预训练模型的路径")
    parser.add_argument('--pretrained', action='store_true', help="If False train from scratch")
    parser.add_argument('--config_path', type=str, default="TsinghuaAI/CPM-Generate", help="预训练模型的配置文件")
    parser.add_argument('--tokenizer_path', type=str, default="TsinghuaAI/CPM-Generate", help="tokenizer的词典文件")
    parser.add_argument("--output_dir", type=str)

    parser.add_argument("--project_name", type=str, default=None)
    parser.add_argument("--group_name", type=str, default=None)
    parser.add_argument("--experiment_name", type=str, default=None)
    parser.add_argument("--speaker1", type=str, default=None)
    parser.add_argument("--speaker2", type=str, default=None)

    parser.add_argument("--data_path", type=str, default=None, help="Path or url of the dataset. ")
    parser.add_argument('--data_type', type=str, default="lccc", choices=["lccc", "natural_conv"])
    parser.add_argument("--train_path", type=str, default=None,
                        help="Path of the train dataset for dist dataset. ")
    parser.add_argument("--valid_path", type=str, default=None,
                        help="Path of the valid dataset for dist dataset. ")
    parser.add_argument("--test_path", type=str, default=None,
                        help="Path of the test dataset for dist dataset. ")
    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--do_valid", action="store_true")
    parser.add_argument("--do_test", action="store_true")

    parser.add_argument('--max_seq_length', type=int, default=512)
    parser.add_argument("--max_history", type=int, default=15, help="Number of previous exchanges to keep in history")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of subprocesses for data loading")
    parser.add_argument("--num_epochs", type=int, default=70, help="Number of training epochs")
    parser.add_argument("--train_batch_size", type=int, default=2, help="Batch size for training")
    parser.add_argument("--valid_batch_size", type=int, default=2, help="Batch size for validation")
    parser.add_argument("--scheduler", type=str, default="linear", help="method of optim")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--eval_before_start", action='store_true',
                        help="If true start with a first evaluation before training")
    parser.add_argument("--warmup_steps", type=int, default=5000, help="Warm up steps")
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument('--adam_epsilon', type=float, default=1e-8)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument("--valid_steps", type=int, default=5000, help="Perfom validation every X steps")
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default=None,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument("--gradient_accumulation_steps", type=int, default=64,
                        help="Accumulate gradients on several steps")
    parser.add_argument("--max_norm", type=float, default=1.0, help="Clipping gradient norm")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device (cuda or cpu)")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--fp16_backend", type=str, default="amp")
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help="Whether to use mixed precision. Choose"
             "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
             "and an Nvidia Ampere GPU.",
    )
    parser.add_argument("--cpu", action="store_true", help="If passed, will train on the CPU.")
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to load in all available experiment trackers from the environment and use them for logging.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help="Location on where to store experiment tracking logs`",
    )
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="Local rank for distributed training (-1: not distributed)")
    parser.add_argument('--seed', type=int, default=2333)
    parser.add_argument("--do_sample", action="store_true",
                        help="Set to use greedy decoding instead of sampling")
    parser.add_argument("--output_max_length", type=int, default=256, help="Maximum length of the output utterances")
    parser.add_argument("--output_min_length", type=int, default=5, help="Minimum length of the output utterances")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling softmax temperature")
    parser.add_argument("--top_k", type=int, default=0,
                        help="Filter top-k tokens before sampling (<=0: no filtering)")
    parser.add_argument("--top_p", type=float, default=0.0,
                        help="Nucleus filtering (top-p) before sampling (<=0.0: no filtering)")
    args = parser.parse_args()
    return args


def prepare_datas(args):
    train_path = os.path.join(args.data_path, args.train_path)
    valid_path = os.path.join(args.data_path, args.valid_path)
    test_path = os.path.join(args.data_path, args.test_path)
    if args.data_type == "lccc":
        train_samples = load_lccc_datas(args, train_path)
        valid_samples = load_lccc_datas(args, valid_path)
        test_samples = load_lccc_datas(args, test_path)
    elif args.data_type == "natural_conv":
        dialog_release = os.path.join(args.data_path, "dialog_release.json")
        train_samples, valid_samples, test_samples = load_natural_conv_dataset(args, dialog_release, train_path,
                                                                               valid_path, test_path)
    else:
        raise ValueError("data_type must be one of [lccc, natural_conv]")

    return train_samples, valid_samples, test_samples


def checkpoint_model(checkpoint_folder, ckpt_id, model, epoch, last_global_step, **kwargs):
    """Utility function for checkpointing model + optimizer dictionaries
    The main purpose for this is to be able to resume training from that instant again
    """
    checkpoint_state_dict = {
        "epoch": epoch,
        "last_global_step": last_global_step,
    }
    # Add extra kwargs too
    checkpoint_state_dict.update(kwargs)

    success = model.save_checkpoint(checkpoint_folder, ckpt_id, checkpoint_state_dict)
    status_msg = f"checkpointing: checkpoint_folder={checkpoint_folder}, ckpt_id={ckpt_id}"
    if success:
        logging.info(f"Success {status_msg}")
    else:
        logging.warning(f"Failure {status_msg}")
    return


# New Code #
def load_training_checkpoint(model, load_dir, tag=None, **kwargs):
    """Utility function for checkpointing model + optimizer dictionaries
    The main purpose for this is to be able to resume training from that instant again
    """
    _, checkpoint_state_dict = model.load_checkpoint(load_dir, tag=tag, **kwargs)
    epoch = checkpoint_state_dict["epoch"]
    last_global_step = checkpoint_state_dict["last_global_step"]
    del checkpoint_state_dict
    return epoch, last_global_step


def evaluate_on_ppl(args, valid_dataloader, model, accelerator):
    model.eval()
    all_losses = []
    loss_func = nn.CrossEntropyLoss(ignore_index=-1)
    with torch.no_grad():
        for step, (batch, no_model_batch) in enumerate(tqdm(valid_dataloader, desc="Evaluating",
                                                            disable=not accelerator.is_local_main_process)):
            for k in batch:
                batch[k] = batch[k].to(args.device)
            for k in no_model_batch:
                no_model_batch[k] = no_model_batch[k].to(args.device)
            lm_labels = no_model_batch['labels']
            loss_mask = no_model_batch["loss_mask"]

            outputs = model(**batch)
            # cross_entropy loss
            lm_logits = outputs.logits
            lm_logits_flat_shifted = lm_logits[..., :-1, :].contiguous().view(-1, lm_logits.size(-1))
            lm_labels_flat_shifted = lm_labels[..., 1:].contiguous().view(-1)

            losses = loss_func(lm_logits_flat_shifted, lm_labels_flat_shifted)
            losses = losses * loss_mask
            loss = torch.sum(losses, dim=-1) / loss_mask.sum(dim=-1)

            # if args.n_gpu > 1:
            #     loss = loss.mean()
            # all_losses.extend(loss.cpu().numpy())
            all_losses.append(accelerator.gather(loss.repeat(args.valid_batch_size)))
    all_losses = torch.cat(all_losses)
    all_losses = all_losses[:len(valid_dataloader.dataset)]
    return torch.mean(all_losses)


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


def evaluate_on_rouge(args, valid_samples, model, tokenizer: CpmTokenizer, accelerator):
    rouge_handle = Rouge()
    model.eval()
    labels = []
    predicts = []
    with torch.no_grad():
        batch_inputs = []
        for step, sample in enumerate(tqdm(valid_samples, desc="Evaluating",
                                           disable=not accelerator.is_local_main_process)):
            assert args.speaker1 in sample
            assert args.speaker2 in sample
            lst = sample.split(args.speaker2)
            labels.append(sample)

            prompt = f"{args.speaker2}".join(lst[:-1]) + args.speaker2
            batch_inputs.append(prompt)
            if len(batch_inputs) == 8:
                input_ids = tokenizer.batch_encode_plus(batch_inputs, return_tensors='pt', padding=True,
                                                        truncation=True,
                                                        max_length=args.max_seq_length)['input_ids'].to(args.device)
                outputs = model.generate(input_ids,
                                         do_sample=args.do_sample,
                                         min_length=args.output_min_length,
                                         max_length=args.output_max_length,
                                         temperature=args.temperature,
                                         top_p=args.top_p,
                                         top_k=args.top_k)
                preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                for i, pred in enumerate(preds):
                    predicts.append(pred)
                batch_inputs = []
        if batch_inputs:
            input_ids = tokenizer.batch_encode_plus(batch_inputs, return_tensors='pt', padding=True,
                                                    truncation=True,
                                                    max_length=args.max_seq_length)['input_ids'].to(args.device)
            outputs = model.generate(input_ids,
                                     do_sample=args.do_sample,
                                     min_length=args.output_min_length,
                                     max_length=args.output_max_length,
                                     temperature=args.temperature,
                                     top_p=args.top_p,
                                     top_k=args.top_k)
            preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            for i, pred in enumerate(preds):
                predicts.append(pred)
            batch_inputs = []

    assert len(labels) == len(predicts)
    scores = rouge_handle.get_scores(predicts, labels, avg=True)
    for key in scores:
        scores[key] = scores[key]['f'] * 100
    results = {k: round(v, 4) for k, v in scores.items()}
    return results


def train(train_samples, valid_samples, model, tokenizer, args, accelerator: Accelerator):
    # If the batch size is too big we use gradient accumulation
    gradient_accumulation_steps = 1
    batch_size = args.train_batch_size
    if batch_size > MAX_GPU_BATCH_SIZE and accelerator.distributed_type != DistributedType.TPU:
        gradient_accumulation_steps = batch_size // MAX_GPU_BATCH_SIZE
        batch_size = MAX_GPU_BATCH_SIZE

    train_dataset = CPMDataset(args, train_samples, tokenizer, args.max_seq_length)
    train_loader = DataLoader(train_dataset,
                              shuffle=True,
                              collate_fn=train_dataset.collate_fn,
                              num_workers=args.num_workers,
                              batch_size=batch_size)
    if valid_samples:
        valid_dataset = CPMDataset(args, valid_samples, tokenizer, args.max_seq_length)
        valid_loader = DataLoader(valid_dataset, shuffle=False,
                                  collate_fn=valid_dataset.collate_fn,
                                  num_workers=args.num_workers,
                                  batch_size=args.valid_batch_size)

    t_total = len(train_samples) * args.num_epochs // gradient_accumulation_steps
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, correct_bias=True, eps=args.adam_epsilon)
    args.warmup_steps = int(t_total * args.warmup_ratio)
    scheduler = get_scheduler(args.scheduler, optimizer=optimizer, num_warmup_steps=args.warmup_steps,
                              num_training_steps=t_total)

    model, optimizer, train_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, scheduler
    )
    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num train examples = %d", len(train_samples))
    logger.info("  Num Epochs = %d", args.num_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", batch_size)
    logger.info("  Gradient Accumulation steps = %d", gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    starting_epoch = 0
    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        # New Code #
        # Loads the DeepSpeed checkpoint from the specified path
        _, last_global_step = load_training_checkpoint(
            model,
            args.resume_from_checkpoint,
            **{"load_optimizer_states": True, "load_lr_scheduler_states": True},
        )
        accelerator.print(f"Resumed from checkpoint: {args.resume_from_checkpoint}")
        resume_step = last_global_step
        starting_epoch = resume_step // len(train_loader)
        resume_step -= starting_epoch * len(train_loader)

    # Figure out how many steps we should save the Accelerator states
    if hasattr(args.checkpointing_steps, "isdigit"):
        checkpointing_steps = args.checkpointing_steps
        if args.checkpointing_steps.isdigit():
            checkpointing_steps = int(args.checkpointing_steps)
    else:
        checkpointing_steps = None

    train_config = {
        'lr': args.lr,
        'train_batch_size': batch_size,
        'max_seq_length': args.max_seq_length,
        'scheduler': args.scheduler,
        'warmup_steps': args.warmup_steps,
        'warmup_ratio': args.warmup_ratio,
        'max_norm': args.max_norm,
        'gradient_accumulation_steps': gradient_accumulation_steps
    }
    if accelerator.is_main_process:
        accelerator.init_trackers(args.project_name, train_config)
    # writer, run = init_wandb_writer(
    #     project_name="gpt-qa",
    #     train_args=train_config,
    #     group_name="chatbot",
    #     experiment_name="finetune_cpm_large_accelerate"
    # )

    # writer.watch(model, log='all')
    model.zero_grad()
    min_valid_loss = float('inf')
    min_ppl = float('inf')
    best_epoch = 0
    best_steps = 0
    best_score = 0.0
    patience = 0
    global_steps = 0
    loss_func = nn.CrossEntropyLoss(ignore_index=-1)
    set_seed(args.seed)
    for epoch in range(starting_epoch, args.num_epochs):
        model.train()
        total_loss = 0.0
        for step, (batch, no_model_batch) in enumerate(tqdm(train_loader,
                                                            desc=f"Training epoch {epoch}/{args.num_epochs}",
                                                            disable=not accelerator.is_local_main_process)):
            labels = no_model_batch["labels"]
            outputs = model(**batch)
            # loss = outputs.loss

            lm_logits = outputs.logits
            lm_logits_flat_shifted = lm_logits[..., :-1, :].contiguous().view(-1, lm_logits.size(-1))
            lm_labels_flat_shifted = labels[..., 1:].contiguous().view(-1)

            loss = loss_func(lm_logits_flat_shifted, lm_labels_flat_shifted)
            loss_mask = no_model_batch["loss_mask"].view(-1)
            loss = torch.sum(loss.view(-1) * loss_mask) / loss_mask.sum()

            if gradient_accumulation_steps > 1:
                loss = loss / gradient_accumulation_steps

            accelerator.backward(loss)
            total_loss += loss.detach().float()

            if (step + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_steps += 1

            accelerator.log({'Train/loss': total_loss / global_steps}, step=global_steps)
            if isinstance(checkpointing_steps, int):
                if global_steps % checkpointing_steps == 0:
                    output_dir = f"step_{global_steps}"
                    if args.output_dir is not None:
                        output_dir = os.path.join(args.output_dir, output_dir)
                    accelerator.save_state(output_dir)

            if args.do_valid:
                if (step + 1) % args.valid_steps == 0 or step == len(train_loader) - 1:
                    valid_loss = evaluate_on_ppl(args, valid_loader, model, accelerator)
                    ppl = math.exp(valid_loss)

                    accelerator.log({'Evaluate': {'loss': valid_loss.item(), 'ppl': ppl}}, step=global_steps)
                    # results = evaluate_on_rouge(args, valid_samples, model, tokenizer, accelerator)
                    # f1 = results['rouge-l']
                    # log_dict = {'Eval/rouge-1': results['rouge-1'], 'Eval/rouge-2': results['rouge-2'],
                    #             'Eval/rouge-l': f1}
                    # accelerator.log(log_dict, step=global_steps)
                    if valid_loss < min_valid_loss:
                    # if f1 > best_score:
                        best_epoch = epoch
                        best_steps = global_steps
                        min_valid_loss = valid_loss.item()
                        min_ppl = ppl
                        # best_score = f1
                        patience = 0

                        if not os.path.exists(args.output_dir):
                            os.makedirs(args.output_dir)
                        if accelerator.is_main_process:
                            tokenizer.save_vocabulary(args.output_dir)

                        accelerator.wait_for_everyone()
                        unwrapped_model = accelerator.unwrap_model(model)
                        unwrapped_model.save_pretrained(
                            args.output_dir, is_main_process=accelerator.is_main_process,
                            save_function=accelerator.save,
                            state_dict=accelerator.get_state_dict(model)
                        )
                    else:
                        patience += 1
                    log_dict = {'best_epoch': best_epoch, 'best_steps': best_steps,
                                'min_val_loss': min_valid_loss, 'min_ppl': min_ppl}
                    # log_dict = {'best_epoch': best_epoch, 'best_steps': best_steps, 'best_f1': best_score}
                    accelerator.print(json.dumps(log_dict, ensure_ascii=False, indent=2))
            else:
                if not os.path.exists(args.output_dir):
                    os.makedirs(args.output_dir)
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    tokenizer.save_vocabulary(args.output_dir)

                accelerator.wait_for_everyone()
                unwrapped_model = accelerator.unwrap_model(model)
                unwrapped_model.save_pretrained(
                    args.output_dir, is_main_process=accelerator.is_main_process,
                    save_function=accelerator.save,
                    state_dict=accelerator.get_state_dict(model)
                )
        if patience > 5:
            break
    accelerator.end_training()


def main():
    args = get_parser()
    # logging is set to INFO (resp. WARN) for main (resp. auxiliary) process.
    # logger.info => log main process only, logger.warning => log all processes
    logging.basicConfig(level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Running process %d", args.local_rank)
    logger.info("Arguments: %s", pformat(args))

    logger.info("Prepare train samples and valid samples")
    train_samples, valid_samples, test_samples = prepare_datas(args)

    logger.info("Prepare tokenizer, pretrained model and optimizer - add special tokens for fine-tuning")
    tokenizer, model = init_model(args)
    args.pad_id, args.eod_token_id = tokenizer.convert_tokens_to_ids(["<pad>", "<eod>"])

    # tracker = CustomWandbTracker(run_name=args.experiment_name,
    #                              project=args.project_name,
    #                              group=args.group_name)
    tracker = WandBTracker(run_name=args.project_name, name=args.experiment_name, group=args.group_name)
    accelerator = Accelerator(cpu=args.device == 'cpu' or not torch.cuda.is_available(),
                              mixed_precision=args.mixed_precision,
                              log_with=tracker,
                              logging_dir=args.logging_dir)

    if args.do_train:
        train(train_samples, valid_samples, model, tokenizer, args, accelerator)
    if args.do_test:
        args.model_checkpoint = args.output_dir
        tokenizer, model = init_model(args)

        test_dataset = CPMDataset(args, test_samples, tokenizer, args.max_seq_length)
        test_loader = DataLoader(test_dataset,
                                 collate_fn=test_dataset.collate_fn,
                                 num_workers=args.num_workers,
                                 batch_size=args.valid_batch_size,
                                 shuffle=False)
        valid_loss = evaluate_on_ppl(args, test_loader, model, accelerator)
        print(f"valid loss: {valid_loss.item()}\tvalid ppl: {math.exp(valid_loss)}")
        # args.device = "cuda"
        # tokenizer = CpmTokenizer.from_pretrained(args.model_checkpoint_path)
        #
        # model_config = GPT2Config.from_pretrained(args.model_checkpoint_path)
        # model = GPT2LMHeadModel.from_pretrained(args.output_dir, config=model_config)
        # model.to(args.device)
        #
        # results = evaluate_on_rouge(args, test_samples, model, tokenizer, accelerator)
        # log_dict = {'rouge-1': results['rouge-1'], 'rouge-2': results['rouge-2'], 'rouge-l': results['rouge-l']}
        # print(json.dumps(log_dict, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

