#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: czh
@email:
@date: 2022/11/11 17:36
"""
# 实验PairSupCon方法
# https://github.com/amazon-science/sentence-representations/tree/main/PairSupCon
import os
import sys
import logging
from tqdm import tqdm
dirname = os.path.dirname(os.path.abspath(__file__))
print(dirname)
sys.path.append(os.path.join('/'.join(dirname.split('/')[:-2])))
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import argparse

import numpy as np
import torch
from torch.optim import Adam, AdamW
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import BertConfig, AutoTokenizer, set_seed, get_scheduler
from sklearn.metrics.pairwise import paired_cosine_distances

from nlp.tools.common import init_wandb_writer
from nlp.models.sentence_embedding_models import PairSupConBert
from nlp.processors.semantic_match_preprocessor import load_data, SentDataSet, collate_fn
from nlp.losses.loss import HardConLoss
from nlp.metrics.sematic_match_metric import compute_corrcoef, compute_pearsonr, l2_normalize

logger = logging.getLogger(__name__)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', default='bert', type=str)
    parser.add_argument('--model_name_or_path', default=None, type=str)
    parser.add_argument('--config_name', default=None, type=str)
    parser.add_argument('--tokenizer_name', default=None, type=str)
    parser.add_argument('--output_dir', type=str, default=None)

    parser.add_argument('--gpuid', type=int, default=0, help="ex:--gpuid 0.")
    parser.add_argument('--seed', type=int, default=0, help="")
    parser.add_argument('--num_worker', type=int, default=0)
    parser.add_argument('--logging_step', type=float, default=100, help="")

    parser.add_argument("--project_name", type=str, default=None)
    parser.add_argument("--group_name", type=str, default=None)
    parser.add_argument("--experiment_name", type=str, default=None)

    # Dataset
    parser.add_argument('--num_labels', type=int, default=2)
    parser.add_argument('--data_dir', default=None, type=str)
    parser.add_argument('--data_type', default='STS-B', type=str)
    parser.add_argument('--object_type', default='classifier', type=str,
                        choices=["classifier", "regression", "triplet", "multi_neg_rank"])
    parser.add_argument('--train_dataset', default=None, type=str)
    parser.add_argument('--valid_dataset', default=None, type=str)
    parser.add_argument('--test_dataset', default=None, type=str)
    parser.add_argument('--max_seq_length', type=int, default=32)
    parser.add_argument('--pad_to_max_length', action='store_true', help="")
    # Training parameters
    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--do_valid", action="store_true")
    parser.add_argument("--do_test", action="store_true")
    parser.add_argument('--lr_rate', type=float, default=5e-6, help="")
    parser.add_argument('--lr_scale', type=int, default=100, help="")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--warmup_ratio', type=float, default=0.1)
    parser.add_argument('--adam_epsilon', type=float, default=1e-8)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--scheduler_type', type=str, default='linear')
    parser.add_argument('--train_batch_size', type=int, default=32)
    parser.add_argument('--valid_batch_size', type=int, default=32)
    parser.add_argument('--num_train_epochs', type=int, default=5)
    parser.add_argument('--valid_steps', type=int, default=1000)
    parser.add_argument('--max_iter', type=int, default=100000000)
    # Contrastive learning
    parser.add_argument('--task_type', default=None, type=str,
                        choices=["classification", "contrastive", "pairsupcon"])
    parser.add_argument('--temperature', type=float, default=0.05, help="temperature required by contrastive loss")
    parser.add_argument('--contrast_type', type=str, default="HardNeg")
    parser.add_argument('--feat_dim', type=int, default=128,
                        help="dimension of the projected features for instance discrimination loss")
    parser.add_argument('--beta', type=float, default=1, help=" ")
    parser.add_argument('--fp16', action='store_true')

    args = parser.parse_args()
    args.use_gpu = args.gpuid >= 0
    return args


def prepare_datasets(data_dir, object_type="classification", seg_tag='\t'):
    dataset = load_data(data_dir, seg_tag=seg_tag)
    data_samples = []
    for data in dataset:
        label = data[2] if object_type == "classification" else float(data[2])
        data_samples.append([data[0], data[1], label])
    return data_samples


def init_model(model_path, num_labels, args, flag='train'):
    bert_config = BertConfig.from_pretrained(args.config_name if args.config_name else model_path)
    if flag == 'train':
        bert_config.save_pretrained(args.model_save_path)
        model = PairSupConBert(bert_name_or_path=model_path, num_labels=num_labels)
    else:
        model = PairSupConBert(config=bert_config, num_labels=num_labels)
        model.load_state_dict(torch.load(os.path.join(args.output_dir, 'pytorch_model.bin'), map_location=args.device))
    return model


def init_optimizer(total, model: PairSupConBert, args):
    if args.task_type == "contrastive":
        optimizer = Adam([
            {'params': model.bert.parameters()},
            {'params': model.contrast_head.parameters(), 'lr': args.lr_rate * args.lr_scale}], lr=args.lr_rate)
    elif args.task_type == "classification":
        optimizer = Adam([
            {'params': model.bert.parameters()},
            {'params': model.classify_head.parameters(), 'lr': args.lr_rate * args.lr_scale}], lr=args.lr_rate)
    elif args.task_type == "pairsupcon":
        optimizer = Adam([
            {'params': model.bert.parameters()},
            {'params': model.classify_head.parameters(), 'lr': args.lr_rate * args.lr_scale},
            {'params': model.contrast_head.parameters(), 'lr': args.lr_rate * args.lr_scale}], lr=args.lr_rate)
    # optimizer = Adam(model.parameters(), lr=args.lr_rate, eps=args.adam_epsilon)
    scheduler = get_scheduler(args.scheduler_type, optimizer=optimizer,
                              num_warmup_steps=args.warmup_steps,
                              num_training_steps=total)

    return optimizer, scheduler


def evaluate(dataloader, model, args):
    model.eval()
    all_pos_vectors = []
    all_neg_vectors = []
    all_labels = []

    for step, batch in enumerate(tqdm(dataloader, desc="Evaluation")):
        anchor_input_ids = batch['anchor_input_ids'].to(args.device)
        pos_input_ids = batch['pos_input_ids'].to(args.device)
        label_id = batch['label'].detach().cpu().numpy()

        with torch.no_grad():
            pos_embedding = model.encode(anchor_input_ids)
            neg_embedding = model.encode(pos_input_ids)

            pos_embedding = pos_embedding.detach().cpu().numpy()
            neg_embedding = neg_embedding.detach().cpu().numpy()

        all_pos_vectors.extend(pos_embedding)
        all_neg_vectors.extend(neg_embedding)
        all_labels.extend(label_id)
    all_pos_vectors = np.array(all_pos_vectors)
    all_neg_vectors = np.array(all_neg_vectors)
    all_labels = np.array(all_labels)
    a_vecs = l2_normalize(all_pos_vectors)
    b_vecs = l2_normalize(all_neg_vectors)

    cosine_scores = (a_vecs * b_vecs).sum(axis=1)
    # cosine_scores = 1 - (paired_cosine_distances(all_pos_vectors, all_neg_vectors))
    corrcoef = compute_corrcoef(all_labels, cosine_scores)
    pearsonr = compute_pearsonr(all_labels, cosine_scores)
    return corrcoef, pearsonr


def train(args, train_samples, valid_samples, model, tokenizer):
    train_batch_size = args.train_batch_size
    train_dataset = SentDataSet(dataset=train_samples, tokenizer=tokenizer, max_seq_length=args.max_seq_length,
                                task_type='match')
    train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True,
                                  num_workers=args.num_worker, collate_fn=collate_fn)

    valid_dataset = SentDataSet(dataset=valid_samples, tokenizer=tokenizer, max_seq_length=args.max_seq_length,
                                task_type='match')
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.valid_batch_size, shuffle=False,
                                  num_workers=args.num_worker, collate_fn=collate_fn)

    t_total = len(train_samples) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    warmup_steps = int(t_total * args.warmup_ratio)
    args.warmup_steps = warmup_steps
    optimizer, scheduler = init_optimizer(t_total, model, args)

    if args.fp16:
        from torch.cuda.amp import autocast, GradScaler
        scaler = GradScaler()

    train_config = {
        'lr_rate': args.lr_rate,
        'gradient_accumulation_steps': args.gradient_accumulation_steps,
        'warmup_ratio': args.warmup_ratio,
        'adam_epsilon': args.adam_epsilon,
        'weight_decay': args.weight_decay,
        'scheduler_type': args.scheduler_type,
        'fp16': args.fp16
    }

    wandb_tacker, _ = init_wandb_writer(
        project_name=args.project_name,
        group_name=args.group_name,
        experiment_name=args.experiment_name,
        train_args=train_config
    )
    wandb_tacker.watch(model, 'all')
    # Pairwise classificaiton loss
    mle_loss_func = nn.CrossEntropyLoss().to(args.device)
    # Hard negative sampling based instance-discrimination loss
    inst_disc_loss_func = HardConLoss(temperature=args.temperature, contrast_type=args.contrast_type).to(args.device)

    global_steps = 0
    best_score = -float('inf')
    best_epoch = 0
    patience = 0
    for epoch in range(args.num_train_epochs):
        model.train()
        total_loss = 0.0
        epoch_steps = 0
        for step, batch in enumerate(tqdm(train_dataloader, desc=f"training on epoch {epoch}/{args.num_train_epochs}")):
            inputs = {
                'premise_input_ids': batch['anchor_input_ids'].to(args.device),
                'hypothesis_input_ids': batch['pos_input_ids'].to(args.device),
                'task_type': args.task_type
            }
            label = batch['label'].to(args.device)
            if args.task_type == 'classification':
                if args.fp16:
                    with autocast():
                        classify_pred = model(**inputs)
                        loss = mle_loss_func(classify_pred, label)
                else:
                    classify_pred = model(**inputs)
                    loss = mle_loss_func(classify_pred, label)
                losses = {"classification_loss": loss}
            elif args.task_type == 'contrastive':
                if args.fp16:
                    with autocast():
                        feat1, feat2 = model(**inputs)
                        losses = inst_disc_loss_func(feat1, feat2, label)
                else:
                    feat1, feat2 = model(**inputs)
                    losses = inst_disc_loss_func(feat1, feat2, label)
                loss = losses['instdisc_loss']
            elif args.task_type == "pairsupcon":
                if args.fp16:
                    with autocast():
                        classify_pred, feat1, feat2 = model(**inputs)
                        classify_loss = mle_loss_func(classify_pred, label)
                        losses = inst_disc_loss_func(feat1, feat2, label)
                else:
                    classify_pred, feat1, feat2 = model(**inputs)
                    classify_loss = mle_loss_func(classify_pred, label)
                    losses = inst_disc_loss_func(feat1, feat2, label)
                loss = args.beta * losses["instdisc_loss"]

                loss += classify_loss
                losses["classification_loss"] = classify_loss
                losses["loss"] = loss
            else:
                raise Exception("Please specify the loss type!")
            if args.fp16:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            total_loss += loss.item()
            epoch_steps += 1

            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_steps += 1

            wandb_tacker.log({'Train/loss': losses,
                              'Train/total_loss': total_loss/epoch_steps}, step=global_steps)
        corrcoef, pearsonr = evaluate(valid_dataloader, model, args)
        wandb_tacker.log({'Eval/spearman': corrcoef, 'Eval/pearsonr': pearsonr, 'epoch': epoch}, step=global_steps)
        logger.info(f"evaluate results at epoch {epoch}/{args.num_train_epochs}:"
                    f" spearman: {corrcoef}\tpearsonr: {pearsonr}")
        if corrcoef > best_score:
            best_score = corrcoef
            best_epoch = best_epoch
            patience = 0

            model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
            output_file = args.model_save_path
            torch.save(model_to_save.state_dict(), os.path.join(output_file, 'pytorch_model.bin'))
            tokenizer.save_pretrained(output_file)
        else:
            patience += 1
            if patience > 5:
                break
    return best_score, best_epoch


def main():
    args = get_args()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    set_seed(args.seed)

    data_dir = args.data_dir
    if not os.path.exists(data_dir):
        raise ValueError(f"The path of '{data_dir}' not exist")
    model_save_path = f"{args.output_dir}/{args.data_type}-pairsupcon-{args.model_type}"
    args.model_save_path = model_save_path
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)

    device = torch.device("cuda:{}".format(args.gpuid) if torch.cuda.is_available() else "cpu")
    args.device = device

    logger.info("******** load datasets *********")
    if args.train_dataset is not None:
        train_samples = prepare_datasets(os.path.join(data_dir, args.train_dataset))
    else:
        train_samples = prepare_datasets(os.path.join(data_dir, args.data_type + '.train.data'))
    if args.valid_dataset is not None:
        valid_samples = prepare_datasets(os.path.join(data_dir, args.valid_dataset))
    else:
        valid_samples = prepare_datasets(os.path.join(data_dir, args.data_type + '.test.data'))

    logger.info("********* load model ***********")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path)
    model = init_model(args.model_name_or_path, args.num_labels, args)
    model.to(device)

    if args.do_train:
        best_score, best_epoch = train(args, train_samples, valid_samples, model, tokenizer)
        logger.info(f"best score: {best_score}\tbest epoch: {best_epoch}")
    if args.do_test:
        if args.test_dataset is not None:
            test_samples = prepare_datasets(os.path.join(data_dir, args.test_dataset))
        else:
            test_samples = prepare_datasets(os.path.join(data_dir, args.data_type + '.test.data'))
        test_dataset = SentDataSet(dataset=test_samples, tokenizer=tokenizer, max_seq_length=args.max_seq_length,
                                   task_type='match')
        test_dataloader = DataLoader(test_dataset, batch_size=args.valid_batch_size,
                                     shuffle=False, num_workers=args.num_worker, collate_fn=collate_fn)
        model = init_model(model_save_path, num_labels=args.num_labels, args=args)
        model.to(device)
        corrcoef, pearsonr = evaluate(test_dataloader, model, args)
        print(f"Result on test dataset. spearman: {corrcoef}\tpearsonr: {pearsonr}")


if __name__ == '__main__':
    main()
