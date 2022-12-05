#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: czh
@email:
@date: 2022/11/3 19:19
"""
# 测试加入交叉注意力的sentence bert, 该方法目前效果不如原始SBERT。改造的不好，等分析完问题再优化
import os
import sys
import codecs
import logging

import numpy as np
from tqdm import tqdm
from datetime import datetime

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertConfig, get_scheduler, set_seed
from sklearn.metrics.pairwise import paired_cosine_distances

dirname = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join('/'.join(dirname.split('/')[:-2])))

from nlp.tools.common import init_wandb_writer
from nlp.models.sentence_embedding_models import PairSupConForSBERT
from nlp.processors.semantic_match_preprocessor import load_data, SentDataSet, collate_fn
from nlp.metrics.sematic_match_metric import compute_corrcoef, compute_pearsonr, l2_normalize
from nlp.losses.loss import SupConLoss

logger = logging.getLogger(__file__)


def prepare_datasets(data_dir, data_type="STS-B", object_type="classification", seg_tag='\t'):
    dataset = load_data(data_dir, seg_tag=seg_tag)
    data_samples = []
    for data in dataset:
        # if data_type == "STS-B":
        #     label = data[2] / 5.0
        # else:
        label = data[2] if object_type == "classification" else float(data[2])
        data_samples.append([data[0], data[1], label])
    return data_samples


def init_model(model_path, num_labels, args, flag="train"):
    bert_config = BertConfig.from_pretrained(args['config_path'] if args['config_path'] else model_path)
    bert_config.save_pretrained(args['model_save_path'])
    if flag == "train":
        model = PairSupConForSBERT(bert_model_path=model_path, bert_config=bert_config, num_labels=num_labels)
    else:
        model = PairSupConForSBERT(bert_config=bert_config, num_labels=num_labels)
        model.load_state_dict(torch.load(model_path+"/best_model.bin", map_location=args['device']))
    return model


def init_optimizer(total, model: PairSupConForSBERT, args):
    optimizer = Adam(model.parameters(), lr=args['lr_rate'], eps=args['adam_epsilon'])
    scheduler = get_scheduler(args['scheduler_type'], optimizer=optimizer,
                              num_warmup_steps=args['warmup_steps'],
                              num_training_steps=total)
    # Check if saved optimizer or scheduler states exist
    if os.path.isfile(os.path.join(args['model_name_or_path'], "optimizer.pt")) and os.path.isfile(
            os.path.join(args["model_name_or_path"], "scheduler.pt")):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(args["model_name_or_path"], "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args["model_name_or_path"], "scheduler.pt")))

    return optimizer, scheduler


def evaluate(data_loader, model, args):

    model.eval()
    all_anchor_vectors = []
    all_pos_vectors = []
    all_labels = []

    for step, batch in enumerate(tqdm(data_loader, desc="Evaluation")):
        anchor_input_ids = batch['anchor_input_ids'].to(args['device'])
        pos_input_ids = batch['pos_input_ids'].to(args['device'])
        label_id = batch['label']
        if args['object_type'] == "regression":
            label_id = label_id.to(torch.float)
        label_id = label_id.detach().cpu().numpy()
        with torch.no_grad():
            anchor_embedding = model.encode(anchor_input_ids, pooling_strategy=args['pooling_strategy'])
            pos_embedding = model.encode(pos_input_ids, pooling_strategy=args['pooling_strategy'])

            anchor_embedding = anchor_embedding.detach().cpu().numpy()
            pos_embedding = pos_embedding.detach().cpu().numpy()

        all_anchor_vectors.extend(anchor_embedding)
        all_pos_vectors.extend(pos_embedding)
        all_labels.extend(label_id)
    all_anchor_vectors = np.array(all_anchor_vectors)
    all_pos_vectors = np.array(all_pos_vectors)
    all_labels = np.array(all_labels)

    a_vecs = l2_normalize(all_anchor_vectors)
    b_vecs = l2_normalize(all_pos_vectors)
    cosine_scores = (a_vecs * b_vecs).sum(axis=1)
    # cosine_scores = 1 - (paired_cosine_distances(all_anchor_vectors, all_pos_vectors))
    corrcoef = compute_corrcoef(all_labels, cosine_scores)
    pearsonr = compute_pearsonr(all_labels, cosine_scores)
    return corrcoef, pearsonr


def train(train_samples, valid_samples, model, tokenizer, args):
    train_batch_size = args['train_batch_size']
    train_dataset = SentDataSet(dataset=train_samples, tokenizer=tokenizer,
                                max_seq_length=args['max_seq_length'], task_type=args['task_type'])
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=train_batch_size,
                                  collate_fn=collate_fn, num_workers=args['num_worker'])

    valid_batch_size = args['valid_batch_size']
    valid_dataset = SentDataSet(dataset=valid_samples, tokenizer=tokenizer,
                                max_seq_length=args['max_seq_length'], task_type=args['task_type'])
    valid_dataloader = DataLoader(valid_dataset, batch_size=valid_batch_size, collate_fn=collate_fn,
                                  num_workers=args['num_worker'])

    t_total = len(train_samples) // args['gradient_accumulation_steps'] * args['num_epochs']

    # Prepare optimizer and schedule (linear warmup and decay)
    warmup_steps = int(t_total * args['warmup_ratio'])
    args['warmup_steps'] = warmup_steps
    optimizer, scheduler = init_optimizer(t_total, model, args)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args['num_epochs'])
    logger.info("  Instantaneous batch size per GPU = %d", train_batch_size)
    logger.info("  Gradient Accumulation steps = %d", args['gradient_accumulation_steps'])
    logger.info("  Total optimization steps = %d", t_total)

    train_config = {
        'lr_rate': args['lr_rate'],
        'gradient_accumulation_steps': args['gradient_accumulation_steps'],
        'warmup_ratio': args['warmup_ratio'],
        'adam_epsilon': args['adam_epsilon'],
        'weight_decay': args['weight_decay'],
        'scheduler_type': args['scheduler_type']
    }
    wandb_tacker, _ = init_wandb_writer(project_name=args['project_name'],
                                        train_args=train_config,
                                        group_name=args['group_name'],
                                        experiment_name=args['experiment_name'])
    wandb_tacker.watch(model, 'all')

    # 损失函数
    ce_loss_func = nn.CrossEntropyLoss().to(args['device'])
    scl_loss_func = SupConLoss(temperature=args['temp']).to(args['device'])

    global_steps = 0
    model.to(args['device'])
    model.zero_grad()
    best_score = 0.0
    best_epoch = 0
    patience = 0
    set_seed(args['seed'])
    for epoch in range(args['num_epochs']):
        model.train()
        total_loss = 0.0
        for step, batch in enumerate(tqdm(train_dataloader, desc=f"Training {epoch}/{args['num_epochs']}")):
            inputs = {
                'anchor_input_ids': batch['anchor_input_ids'].to(args['device']),
                'pos_input_ids': batch['pos_input_ids'].to(args['device']),
            }
            ce_feature, scl_feature = model(**inputs)
            label = batch['label'].to(args['device'])
            loss_ce = ce_loss_func(ce_feature.to(args['device']), label)
            loss_scl = scl_loss_func(scl_feature.to(args['device']), label)
            loss = loss_ce + loss_scl * args['alpha']

            if args['gradient_accumulation_steps'] > 1:
                loss = loss / args['gradient_accumulation_steps']
            loss.backward()

            wandb_tacker.log({'Train/loss': loss.item()}, step=global_steps)
            total_loss += loss.item()
            if (step + 1) % args['gradient_accumulation_steps'] == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_steps += 1

        corrcoef, pearsonr = evaluate(valid_dataloader, model, args)
        wandb_tacker.log({'Evaluation': {'spearman': corrcoef, 'pearsonr': pearsonr}}, step=global_steps)
        logger.info(f"evaluate results. Spearman: {corrcoef}\tPearsonr: {pearsonr}")
        if corrcoef > best_score:
            best_score = corrcoef
            best_epoch = best_epoch
            patience = 0

            model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
            output_file = os.path.join(args['model_save_path'], 'best_model.bin')
            torch.save(model_to_save.state_dict(), output_file)
            tokenizer.save_pretrained(args['model_save_path'])

            loginfo = "best_epoch: {}\tspearman: {}\tpearsonr: {}".format(best_epoch, best_score, pearsonr)
            logger.info(loginfo)
            with codecs.open(os.path.join(args['model_save_path'], "eval_result.txt"), "w",
                             encoding="utf8") as fw:
                fw.write("best_epoch: {}\nspearman: {}\npearsonr: {}".format(best_epoch, best_score, pearsonr))
        else:
            patience += 1
            if patience >= 10:
                break


def main():
    root_path = "/root/work2/work2/chenzhihao/NLP"
    config = {
        'model_type': "roberta-wwm-ext",
        'model_name_or_path': "/root/work2/work2/chenzhihao/pretrained_models/chinese-roberta-wwm-ext",
        'output_dir': root_path + "/experiments/output_file_dir/semantic_match",
        'config_path': "/root/work2/work2/chenzhihao/pretrained_models/chinese-roberta-wwm-ext",
        'tokenizer_path': "/root/work2/work2/chenzhihao/pretrained_models/chinese-roberta-wwm-ext",
        'do_train': True,
        'do_test': True,
        'num_labels': 2,  # 注意STS-B的label是6个
        'train_batch_size': 64,
        'valid_batch_size': 64,
        'test_batch_size': 64,
        'num_epochs': 100,
        'max_seq_length': 128,
        'scheduler_type': "linear",
        'object_type': "classification",  # classification, regression, triplet
        'task_type': "match",  # "match" or "nli"
        'pooling_strategy': "last-avg",  # first-last-avg, last-avg, cls, pooler
        'distance_type': "",
        'triplet_margin': 0.5,
        'temp': 0.05,  # temperature rate
        'alpha': 1.0,
        'lr_rate': 2e-6,
        'gradient_accumulation_steps': 1,
        'warmup_ratio': 0.1,
        'adam_epsilon': 1e-8,
        'weight_decay': 0.01,
        'data_type': "PAWSX",  # ATEC, BQ, LCQMC, PAWSX, STS-B
        'train_dataset': "train.data",  # train.data
        'valid_dataset': "valid.data",  # valid.data
        'test_dataset': "test.data",  # test.data
        'project_name': 'semantic_match',
        'group_name': "nlp",
        'experiment_name': "PAWSX_pairscl_sbert-roberta-wwm-ext",
        'cuda_number': "3",
        'num_worker': 4,
        'seed': 2333
    }

    data_dir = "/root/work2/work2/chenzhihao/datasets/chinese-semantics-match-dataset/" + config['data_type']
    if not os.path.exists(data_dir):
        raise ValueError(f"The path of '{data_dir}' not exist")
    output_dir = root_path + "/experiments/output_file_dir/semantic_match"
    model_save_path = output_dir + f"/{config['data_type']}-pairscl_sbert-{config['model_type']}-{config['pooling_strategy']}"
    config['model_save_path'] = model_save_path
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)

    device = torch.device(f"cuda:{config['cuda_number']}") if torch.cuda.is_available() else torch.device('cpu')
    config['device'] = device

    train_samples = prepare_datasets(os.path.join(data_dir, config['data_type'] + '.' + config['train_dataset']),
                                     config['data_type'], config['object_type'])
    valid_samples = prepare_datasets(os.path.join(data_dir, config['data_type'] + '.' + config['valid_dataset']),
                                     config['data_type'], config['object_type'])
    # train_samples = prepare_datasets(os.path.join(data_dir, config['train_dataset']),
    #                                  config['data_type'], config['object_type'])
    # valid_samples = prepare_datasets(os.path.join(data_dir, config['valid_dataset']),
    #                                  config['data_type'], config['object_type'])
    tokenizer = BertTokenizer.from_pretrained(config['model_name_or_path']
                                              if not config['tokenizer_path'] else config['tokenizer_path'])
    model = init_model(model_path=config['model_name_or_path'], num_labels=config['num_labels'], args=config)
    model.to(device)
    if config['do_train']:
        train(train_samples, valid_samples, model, tokenizer, config)
    if config['do_test']:
        test_samples = prepare_datasets(os.path.join(data_dir, config['data_type'] + '.' + config['test_dataset']),
                                        config['data_type'], config['object_type'])
        # test_samples = prepare_datasets(os.path.join(data_dir, config['test_dataset']),
        #                                 config['data_type'], config['object_type'])
        test_dataset = SentDataSet(test_samples, tokenizer, config['max_seq_length'], task_type=config['task_type'])
        test_dataloader = DataLoader(test_dataset, batch_size=config['test_batch_size'], collate_fn=collate_fn)

        model = init_model(model_path=config['model_save_path'], num_labels=config['num_labels'],
                           args=config, flag='test')
        model.to(device)
        corrcoef, pearsonr = evaluate(test_dataloader, model, config)
        loginfo = "Test result. pearsonr: {}\tspearman: {}".format(pearsonr, corrcoef)
        logger.info(loginfo)
        with codecs.open(os.path.join(model_save_path, 'test_result.txt'), "w", encoding="utf8") as fw:
            fw.write("Test result. pearsonr: {}\nspearman: {}".format(pearsonr, corrcoef))


if __name__ == "__main__":
    main()
