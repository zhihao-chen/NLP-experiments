#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: czh
@email: zhihao.chen@kuwo.cn
@date: 2022/9/21 11:39
"""
import codecs
# 记录下训练过程中的参数，用于画图做数据分析
# 参考自：https://github.com/beyondguo/TrainingDynamics
import os
import sys
import json
import logging
from argparse import ArgumentParser

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertConfig, get_scheduler, AdamW, set_seed
from sklearn.metrics.pairwise import paired_cosine_distances

dirname = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join('/'.join(dirname.split('/')[:-3]), 'NLP'))

from nlp.tools.common import init_wandb_writer
from nlp.models.sentence_embedding_models import SBERTModel
from nlp.processors.semantic_match_preprocessor import load_data, SentDataSet, collate_fn
from nlp.metrics.sematic_match_metric import compute_corrcoef, compute_pearsonr, l2_normalize

logger = logging.getLogger(__name__)


def get_args():
    parser = ArgumentParser()
    parser.add_argument('--model_type', type=str, default=None)
    parser.add_argument('--model_name_or_path', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--dy_log_path', type=str, default=None, help="Log path of training dynamic records")
    parser.add_argument('--config_name_or_path', type=str, default=None)
    parser.add_argument('--tokenizer_name_or_path', type=str, default=None)
    parser.add_argument('--data_dir', type=str, default=None)
    parser.add_argument('--data_type', type=str, default='STS-B', choices=['ATEC', 'BQ', 'LCQMC', 'PAWSX', 'STS-B'])
    parser.add_argument('--train_dataset', type=str, default='train.data')
    parser.add_argument('--valid_dataset', type=str, default='valid.data')
    parser.add_argument('--test_dataset', type=str, default='test.data')

    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--do_valid', action='store_true')
    parser.add_argument('--do_test', action='store_true')

    parser.add_argument("--project_name", type=str, default='sup-sbert')
    parser.add_argument("--group_name", type=str, default='semantic_match')
    parser.add_argument("--experiment_name", type=str, default='sbert-training-dynamics')

    parser.add_argument('--object_type', type=str, default='classification',
                        choices=['classification', 'regression', 'triplet'])
    parser.add_argument('--task_type', type=str, default='match', choices=['match', 'nli'])
    parser.add_argument('--pooling_strategy', type=str, default='pooler',
                        choices=['first-last-avg', 'last-avg', 'cls', 'pooler'])
    parser.add_argument('--max_seq_length', type=int, default=128)
    parser.add_argument('--num_train_epochs', type=int, default=30)
    parser.add_argument('--train_batch_size', type=int, default=32)
    parser.add_argument('--valid_batch_size', type=int, default=32)
    parser.add_argument('--test_batch_size', type=int, default=32)
    parser.add_argument('--valid_steps', type=int, default=500)
    parser.add_argument('--num_labels', type=int, default=2)

    parser.add_argument('--lr_rate', type=float, default=2e-5, help="Learning rate")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--warmup_ratio', type=float, default=0.1)
    parser.add_argument('--adam_epsilon', type=float, default=1e-8)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--scheduler_type', type=str, default='linear')
    parser.add_argument('--distance_type', type=str, default=None)
    parser.add_argument('--triplet_margin', type=str, default=0.5)

    parser.add_argument('--do_recording', action='store_true', help="Whether to record the training dynamics.")
    parser.add_argument("--with_data_selection", action="store_true",
                        help="Use only a selected subset of the training data for model training.")
    parser.add_argument("--data_selection_region", default=None, choices=("easy", "hard", "ambiguous"),
                        help="Three regions from the dataset cartography: easy, hard and ambiguous")
    parser.add_argument("--selected_indices_filename", type=str)

    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--seed', type=int, default=2333)
    parser.add_argument('--cuda_number', type=int, default=0)
    args = parser.parse_args()

    return args


def init_model(model_path, args, flag="train"):
    bert_config = BertConfig.from_pretrained(args.config_name_or_path if args.config_name_or_path else model_path)
    if flag == "train":
        model = SBERTModel(bert_model_path=model_path, bert_config=bert_config,
                           num_labels=args.num_labels, object_type=args.object_type)
    else:
        model = SBERTModel(bert_config=bert_config, num_labels=args.num_labels, object_type=args.object_type)
        model.load_state_dict(torch.load(model_path+"/best_model.bin", map_location=args.device))
    return model


def init_optimizer(total, parameters, args):
    optimizer = AdamW(parameters, lr=args.lr_rate, eps=args.adam_epsilon)
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


def get_parameters(model, args):
    no_decay = ["bias", "LayerNorm.weight", "LayerNorm.bias"]
    bert_param_optimizer = list(model.bert.named_parameters())
    linear_param_optimizer = list(model.classifier.named_parameters())
    optimizer_grouped_parameters = [
        {'params': [p for n, p in bert_param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay, 'lr': args.lr_rate},
        {'params': [p for n, p in bert_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
         'lr': args.lr_rate},

        {'params': [p for n, p in linear_param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay, 'lr': args.lr_rate},
        {'params': [p for n, p in linear_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
         'lr': args.lr_rate}
    ]
    return optimizer_grouped_parameters


def evaluate(data_loader, model, args):

    model.eval()
    all_anchor_vectors = []
    all_pos_vectors = []
    all_labels = []

    for step, batch in enumerate(tqdm(data_loader, desc="Evaluation")):
        anchor_input_ids = batch['anchor_input_ids'].to(args.device)
        pos_input_ids = batch['pos_input_ids'].to(args.device)
        label_id = batch['label']
        if args.object_type == "regression":
            label_id = label_id.to(torch.float)
        label_id = label_id.detach().cpu().numpy()
        with torch.no_grad():
            anchor_embedding = model.encode(anchor_input_ids, pooling_strategy=args.pooling_strategy)
            pos_embedding = model.encode(pos_input_ids, pooling_strategy=args.pooling_strategy)

            anchor_embedding = anchor_embedding.detach().cpu().numpy()
            pos_embedding = pos_embedding.detach().cpu().numpy()

        all_anchor_vectors.extend(anchor_embedding)
        all_pos_vectors.extend(pos_embedding)
        all_labels.extend(label_id)
    all_anchor_vectors = np.array(all_anchor_vectors)
    all_pos_vectors = np.array(all_pos_vectors)
    all_labels = np.array(all_labels)

    # a_vecs = l2_normalize(all_a_vecs)
    # b_vecs = l2_normalize(all_b_vecs)

    # sims = (a_vecs * b_vecs).sum(axis=1)
    cosine_scores = 1 - (paired_cosine_distances(all_anchor_vectors, all_pos_vectors))
    corrcoef = compute_corrcoef(all_labels, cosine_scores)
    pearsonr = compute_pearsonr(all_labels, cosine_scores)
    return corrcoef, pearsonr


def train(train_samples, valid_samples, model, tokenizer, args):
    train_batch_size = args.train_batch_size
    train_dataset = SentDataSet(dataset=train_samples, tokenizer=tokenizer,
                                max_seq_length=args.max_seq_length, task_type=args.task_type)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=train_batch_size,
                                  collate_fn=collate_fn, num_workers=args.num_workers)
    # data selection is only applied on train set
    if args.with_data_selection:
        assert args.data_selection_region is not None, "You much specify `data_selection_region` " \
                                                       "when using `with_data_selection`"
        file_name = os.path.join(args.output_dir, 'three_regions_data_indices.json')
        if not os.path.exists(file_name):
            raise FileExistsError('Selection indices file not found!')
        with codecs.open(file_name, encoding='utf8') as f:
            three_regions_data_indices = json.loads(f.read())
            selected_indices = three_regions_data_indices[args.data_selection_region]
            pass
    valid_batch_size = args.valid_batch_size
    valid_dataset = SentDataSet(dataset=valid_samples, tokenizer=tokenizer,
                                max_seq_length=args.max_seq_length, task_type=args.task_type)
    valid_dataloader = DataLoader(valid_dataset, batch_size=valid_batch_size, collate_fn=collate_fn,
                                  num_workers=args.num_workers)

    t_total = len(train_samples) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    optimizer_grouped_parameters = get_parameters(model, args)
    warmup_steps = int(t_total * args.warmup_ratio)
    args.warmup_steps = warmup_steps
    optimizer, scheduler = init_optimizer(t_total, optimizer_grouped_parameters, args)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", train_batch_size)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    if args.object_type == "classification":
        loss_func = nn.CrossEntropyLoss()
    elif args.object_type == "regression":
        loss_func = nn.CosineEmbeddingLoss()
    else:
        loss_func = nn.MarginRankingLoss(margin=args.triplet_margin)
    train_config = {
        'lr_rate': args.lr_rate,
        'gradient_accumulation_steps': args.gradient_accumulation_steps,
        'warmup_ratio': args.warmup_ratio,
        'adam_epsilon': args.adam_epsilon,
        'weight_decay': args.weight_decay,
        'scheduler_type': args.scheduler_type
    }
    wandb, run = init_wandb_writer(project_name=args.project_name,
                                   train_args=train_config,
                                   group_name=args.group_name,
                                   experiment_name=args.experiment_name)
    wandb.watch(model, log='all')

    global_steps = 0
    model.to(args.device)
    model.zero_grad()
    best_score = 0.0
    best_epoch = 0
    set_seed(args.seed)
    for epoch in range(args.num_train_epochs):
        model.train()
        total_loss = 0.0
        for step, batch in enumerate(tqdm(train_dataloader, desc="Training")):
            inputs = {
                'anchor_input_ids': batch['anchor_input_ids'].to(args.device),
                'pos_input_ids': batch['pos_input_ids'].to(args.device),
            }
            if args.task_type == 'nli':
                inputs['neg_input_ids'] = batch['neg_input_ids'].to(args.device)

            logits = model(**inputs, pooling_strategy=args.pooling_strategy)
            label = batch['label'].to(args.device)
            if args.object_type == "regression":
                label = label.to(torch.float)
            loss = loss_func(logits, label)
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            loss.backward()

            wandb.log({'Train/loss': loss.item()}, step=global_steps)
            total_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_steps += 1

            if (step + 1) % args.valid_steps == 0:
                corrcoef, pearsonr = evaluate(valid_dataloader, model, args)
                wandb.log({'Evaluation/corrcoef': corrcoef, 'Evaluation/pearsonr': pearsonr}, step=global_steps)
                logger.info(f"evaluate results: corrcoef: {corrcoef}\tpearsonr: {pearsonr}")
                if pearsonr > best_score:
                    best_score = pearsonr
                    best_epoch = best_epoch

                    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                    output_file = os.path.join(args.model_save_path, 'best_model.bin')
                    torch.save(model_to_save.state_dict(), output_file)
                    tokenizer.save_pretrained(args.model_save_path)
        # ------------------ Recording Training Dynamics --------------------
        # 在每一个epoch之后，对train set所有样本再过一遍，记录dynamics
        # 每个epoch单独一个文件
        if args.do_recording:
            dy_log_path = os.path.join(args.dy_log_path, args.task_name, args.model_name, "training_dynamics")
            if not os.path.exists(dy_log_path):
                os.makedirs(dy_log_path)
            writer = codecs.open(os.path.join(args.dy_log_path, f'dynamics_epoch_{epoch}.json'), 'w', encoding='utf8')
            logger.info('---------- Recording Training Dynamics (Epoch %s) -----------' % epoch)
            all_ids = []
            for step, batch in enumerate(tqdm(train_dataloader, desc=f"Recording Training Dynamics, epoch {epoch}")):
                id_lst = batch['id_lst']
                label_lst = batch['label'].detach().cpu().tolist()
                inputs = {
                    'anchor_input_ids': batch['anchor_input_ids'].to(args.device),
                    'pos_input_ids': batch['pos_input_ids'].to(args.device),
                }
                if args.task_type == 'nli':
                    inputs['neg_input_ids'] = batch['neg_input_ids'].to(args.device)

                logits_list = model(**inputs, pooling_strategy=args.pooling_strategy)
                logits_list = logits_list.detach().cpu().tolist()

                for idx, label, logits in zip(id_lst, label_lst, logits_list):
                    if idx in all_ids:
                        continue
                    all_ids.append(idx)
                    record = {'guid': idx, f'logits_epoch_{epoch}': logits, 'gold': label}
                    writer.write(json.dumps(record, ensure_ascii=False)+'\n')
            logger.info(f'Epoch {epoch}, Saved to {dy_log_path}')
            writer.close()


def main():
    args = get_args()

    data_dir = os.path.join(args.data_dir, args.data_type)
    if not os.path.exists(data_dir):
        raise ValueError(f"The path of '{data_dir}' not exist")
    model_save_path = os.path.join(args.output_dir, 'sup_sbert_training_dynamics', args.model_type+'_'+args.data_type)
    args.model_save_path = model_save_path
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)

    device = torch.device(f"cuda:{args.cuda_number}") if torch.cuda.is_available() else torch.device('cpu')
    args.device = device

    train_samples = load_data(os.path.join(data_dir, args.data_type+'.'+args.train_dataset))
    valid_samples = load_data(os.path.join(data_dir, args.data_type+'.'+args.valid_dataset))

    tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path
                                              if not args.tokenizer_name_or_path else args.tokenizer_name_or_path)
    model = init_model(model_path=args.model_name_or_path, args=args)
    model.to(device)

    if args.do_train:

        train(train_samples, valid_samples, model, tokenizer, args)
    if args.do_test:
        test_samples = load_data(os.path.join(data_dir, args.data_type + '.' + args.test_dataset))
        test_dataset = SentDataSet(test_samples, tokenizer, args.max_seq_length, task_type=args.task_type)
        test_dataloader = DataLoader(test_dataset, batch_size=args.test_batch_size, collate_fn=collate_fn)

        model = init_model(model_path=model_save_path, args=args, flag='test')
        model.to(device)
        corrcoef, pearsonr = evaluate(test_dataloader, model, args)
        print("test results: ", json.dumps({'corrcoef': corrcoef, 'pearsonr': pearsonr}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
