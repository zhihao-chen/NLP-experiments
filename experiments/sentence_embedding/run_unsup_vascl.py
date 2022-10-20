#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: czh
@email:
@date: 2022/10/19 10:38
"""
# 基于VaSCL模型的无监督语义向量训练
# https://github.com/amazon-research/sentence-representations/blob/main/VaSCL/main.py

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
from torch.optim import Adam
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoTokenizer, set_seed, get_scheduler
from sklearn.metrics.pairwise import paired_cosine_distances

from nlp.tools.common import init_wandb_writer
from nlp.models.sentence_embedding_models import VaSCLBERT, VaSCLRoBERTa
from nlp.processors.semantic_match_preprocessor import load_data, VaSCLDataset
from nlp.losses.loss import VaSCLContrastiveLoss, VaSCLNBiDir, VaSCLNUniDir
from nlp.utils.vat_utils import VaSCLPturb
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
    parser.add_argument('--data_dir', default=None, type=str)
    parser.add_argument('--data_type', default='STS-B', type=str,
                        choices=["ATEC", "BQ", "LCQMC", "PAWSX", "STS-B", "SNLI"])
    parser.add_argument('--object_type', default='classifier', type=str,
                        choices=["classifier", "regression", "triplet", "multi_neg_rank"])
    parser.add_argument('--train_dataset', default="train.data", type=str)
    parser.add_argument('--valid_dataset', default="valid.data", type=str)
    parser.add_argument('--test_dataset', default="test.data", type=str)
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
    # VaSCL loss
    parser.add_argument('--temperature', type=float, default=0.05, help="temperature required by contrastive loss")
    parser.add_argument('--topk', type=int, default=16, help=" ")
    parser.add_argument('--eps', type=float, default=15, help=" ")

    args = parser.parse_args()
    args.use_gpu = args.gpuid >= 0
    return args


def prepare_datas(args, data_dir, need_label=True, seg_tag='\t'):
    datas = load_data(data_dir, seg_tag=seg_tag)
    samples = []
    for data in datas:
        if args.data_type == "STS-B":
            label = data[2] / 5.0
        else:
            label = data[2] if args.object_type == "classification" else float(data[2])
        if need_label:
            samples.append([data[0].strip(), data[1].strip(), label])
        else:
            samples.append([data[0].strip()])
            samples.append([data[1].strip()])

    np.random.shuffle(samples)
    return samples


def get_bert_config_tokenizer(args):
    config = AutoConfig.from_pretrained(args.config_name if args.config_name else args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path)
    return config, tokenizer


def init_optimizer(total, parameters, args):
    optimizer = Adam(parameters, lr=args.lr_rate)
    scheduler = get_scheduler(args.scheduler_type, optimizer=optimizer,
                              num_warmup_steps=args.warmup_steps,
                              num_training_steps=total)
    # Check if saved optimizer or scheduler states exist
    if os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt")) and os.path.isfile(
            os.path.join(args.model_name_or_path, "scheduler.pt")):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))

    return optimizer, scheduler


def get_parameters(args, model):
    optimizer_grouped_parameters = [
        {'params': model.bert.parameters()},
        {'params': model.contrast_head.parameters(), 'lr': args.lr_rate * args.lr_scale}
    ]
    return optimizer_grouped_parameters


def evaluate(dataloader, model):
    model.eval()
    all_pos_vectors = []
    all_neg_vectors = []
    all_labels = []

    for step, batch in enumerate(tqdm(dataloader, desc="Evaluation")):
        pos_inputs = batch['pos_inputs']
        neg_inputs = batch['neg_inputs']
        label_id = batch['label']

        with torch.no_grad():
            pos_embedding = model.get_mean_embeddings(**pos_inputs)
            neg_embedding = model.get_mean_embeddings(**neg_inputs)

            pos_embedding = pos_embedding.detach().cpu().numpy()
            neg_embedding = neg_embedding.detach().cpu().numpy()

        all_pos_vectors.extend(pos_embedding)
        all_neg_vectors.extend(neg_embedding)
        all_labels.extend(label_id)
    all_pos_vectors = np.array(all_pos_vectors)
    all_neg_vectors = np.array(all_neg_vectors)
    all_labels = np.array(all_labels)
    # a_vecs = l2_normalize(all_pos_vectors)
    # b_vecs = l2_normalize(all_neg_vectors)

    # cosine_scores = (a_vecs * b_vecs).sum(axis=1)
    cosine_scores = 1 - (paired_cosine_distances(all_pos_vectors, all_neg_vectors))

    corrcoef = compute_corrcoef(all_labels, cosine_scores)
    pearsonr = compute_pearsonr(all_labels, cosine_scores)
    return corrcoef, pearsonr


def train(args, train_samples, valid_samples, model, tokenizer):
    train_batch_size = args.train_batch_size
    train_dataset = VaSCLDataset(dataset=train_samples, tokenizer=tokenizer, args=args)
    train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True,
                                  num_workers=args.num_worker)

    valid_dataset = VaSCLDataset(dataset=valid_samples, tokenizer=tokenizer, args=args)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.valid_batch_size, shuffle=False,
                                  num_workers=args.num_worker)

    t_total = len(train_samples) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    optimizer_grouped_parameters = get_parameters(args, model)
    warmup_steps = int(t_total * args.warmup_ratio)
    args.warmup_steps = warmup_steps
    optimizer, scheduler = init_optimizer(t_total, optimizer_grouped_parameters, args)

    train_config = {
        'lr_rate': args.lr_rate,
        'gradient_accumulation_steps': args.gradient_accumulation_steps,
        'warmup_ratio': args.warmup_ratio,
        'adam_epsilon': args.adam_epsilon,
        'weight_decay': args.weight_decay,
        'scheduler_type': args.scheduler_type
    }

    wandb_tacker, _ = init_wandb_writer(
        project_name=args.project_name,
        group_name=args.group_name,
        experiment_name=args.experiment_name,
        train_args=train_config
    )
    wandb_tacker.watch(model, 'all')

    paircon_loss = VaSCLContrastiveLoss(temperature=args.temperature, topk=args.topk).to(args.device)
    uni_criterion = VaSCLNUniDir(temperature=args.temperature).to(args.device)
    bi_criterion = VaSCLNBiDir(temperature=args.temperature).to(args.device)
    perturb_embed = VaSCLPturb(xi=args.eps, eps=args.eps, uni_criterion=uni_criterion,
                               bi_criterion=bi_criterion).to(args.device)

    global_steps = 0
    best_score = -float('inf')
    best_epoch = 0
    for epoch in range(args.num_train_epochs):
        model.train()
        total_loss = 0.0
        for step, batch in enumerate(tqdm(train_dataloader, desc=f"training on epoch {epoch}/{args.num_train_epochs}")):
            embeddings, hard_indices, feat1, feat2 = model(**batch, topk=args.topk)
            losses = paircon_loss(feat1, feat2)
            loss = losses['loss']
            losses['vcl_loss'] = loss.item()
            if args.eps > 0:
                lds_losses = perturb_embed(model, embeddings.detach(), hard_indices)
                losses.update(lds_losses)
                loss += lds_losses["lds_loss"]
                losses['optimized_loss'] = loss
            loss.backward()

            wandb_tacker.log({'Train/loss': losses}, step=global_steps)
            total_loss += loss.item()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                # scheduler.step()
                optimizer.zero_grad()
                global_steps += 1

        corrcoef, pearsonr = evaluate(valid_dataloader, model)
        wandb_tacker.log({'Eval/corrcoef': corrcoef, 'Eval/pearsonr': pearsonr}, step=global_steps)
        logger.info(f"evaluate results at epoch {epoch}/{args.num_train_epochs}:"
                    f" corrcoef: {corrcoef}\tpearsonr: {pearsonr}")
        if corrcoef > best_score:
            best_score = corrcoef
            best_epoch = best_epoch

            model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
            output_file = args.model_save_path
            model_to_save.save_pretrained(output_file)
            tokenizer.save_pretrained(output_file)
    return best_score, best_epoch


def main():
    args = get_args()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    set_seed(args.seed)

    data_dir = os.path.join(args.data_dir, args.data_type)
    if not os.path.exists(data_dir):
        raise ValueError(f"The path of '{data_dir}' not exist")
    model_save_path = f"{args.output_dir}/{args.data_type}-unsup_vascl-{args.model_type}"
    args.model_save_path = model_save_path
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)

    device = torch.device("cuda:{}".format(args.gpuid) if torch.cuda.is_available() else "cpu")
    args.device = device

    logger.info("******** load datasets *********")
    train_samples = prepare_datas(args,
                                  os.path.join(data_dir, args.data_type + '.' + args.train_dataset),
                                  need_label=False)
    valid_samples = prepare_datas(args,
                                  os.path.join(data_dir, args.data_type + '.' + args.valid_dataset),
                                  need_label=True)
    test_samples = prepare_datas(args,
                                 os.path.join(data_dir, args.data_type + '.' + args.test_dataset),
                                 need_label=True)

    logger.info("********* load model ***********")
    config, tokenizer = get_bert_config_tokenizer(args)
    if args.model_type == 'roberta':
        model = VaSCLRoBERTa.from_pretrained(args.model_name_or_path)
    else:
        model = VaSCLBERT.from_pretrained(args.model_name_or_path)
    model.to(device)

    if args.do_train:
        best_score, best_epoch = train(args, train_samples, valid_samples, model, tokenizer)
        logger.info(f"best score: {best_score}\tbest epoch: {best_epoch}")
    if args.do_test:
        test_dataset = VaSCLDataset(dataset=test_samples, tokenizer=tokenizer, args=args)
        test_dataloader = DataLoader(test_dataset, batch_size=args.valid_batch_size,
                                     shuffle=False, num_workers=args.num_worker)
        if args.model_type == 'roberta':
            model = VaSCLRoBERTa.from_pretrained(model_save_path)
        else:
            model = VaSCLBERT.from_pretrained(model_save_path)
        model.to(device)
        corrcoef, pearsonr = evaluate(test_dataloader, model)
        print(f"Result on test dataset. spearman: {corrcoef}\tpearsonr: {pearsonr}")


if __name__ == "__main__":
    main()
