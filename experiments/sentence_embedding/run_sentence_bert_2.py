#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: czh
@email:
@date: 2022/6/28 20:04
"""
import os
import sys
import json

import numpy as np
from tqdm import tqdm
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertConfig, get_scheduler, AdamW, set_seed
from sklearn.metrics.pairwise import paired_cosine_distances

sys.path.append('/data2/work2/chenzhihao/NLP')

from nlp.tools.common import init_wandb_writer
from nlp.models.sentence_embedding_models import SBERTModel
from nlp.processors.semantic_match_preprocessor import load_data, SentDataSet, collate_fn
from nlp.metrics.sematic_match_metric import compute_corrcoef, compute_pearsonr, l2_normalize


TRAIN_CONFIG = {
    'lr_rate': 2e-5,
    'gradient_accumulation_steps': 1,
    'warmup_ratio': 0.1,
    'adam_epsilon': 1e-8,
    'weight_decay': 0.01,
    'scheduler_type': 'linear'
}
WANDB = None


def init_model(model_path, num_labels, args, flag="train"):
    bert_config = BertConfig.from_pretrained(args['config_path'] if args['config_path'] else model_path)
    if flag == "train":
        model = SBERTModel(bert_model_path=model_path, bert_config=bert_config,
                           num_labels=num_labels, object_type=args['object_type'])
    else:
        model = SBERTModel(bert_config=bert_config, num_labels=num_labels, object_type=args['object_type'])
        model.load_state_dict(torch.load(model_path+"/best_model.bin", map_location=args['device']))
    return model


def init_optimizer(total, parameters, args):
    optimizer = AdamW(parameters, lr=TRAIN_CONFIG['lr_rate'], eps=TRAIN_CONFIG['adam_epsilon'])
    scheduler = get_scheduler(TRAIN_CONFIG['scheduler_type'], optimizer=optimizer,
                              num_warmup_steps=args['warmup_steps'],
                              num_training_steps=total)
    # Check if saved optimizer or scheduler states exist
    if os.path.isfile(os.path.join(args['model_name_or_path'], "optimizer.pt")) and os.path.isfile(
            os.path.join(args["model_name_or_path"], "scheduler.pt")):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(args["model_name_or_path"], "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args["model_name_or_path"], "scheduler.pt")))

    return optimizer, scheduler


def get_parameters(model):
    no_decay = ["bias", "LayerNorm.weight", "LayerNorm.bias"]
    bert_param_optimizer = list(model.bert.named_parameters())
    linear_param_optimizer = list(model.classifier.named_parameters())
    optimizer_grouped_parameters = [
        {'params': [p for n, p in bert_param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': TRAIN_CONFIG["weight_decay"], 'lr': TRAIN_CONFIG['lr_rate']},
        {'params': [p for n, p in bert_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
         'lr': TRAIN_CONFIG['lr_rate']},

        {'params': [p for n, p in linear_param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': TRAIN_CONFIG["weight_decay"], 'lr': TRAIN_CONFIG['lr_rate']},
        {'params': [p for n, p in linear_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
         'lr': TRAIN_CONFIG['lr_rate']}
    ]
    return optimizer_grouped_parameters


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

    # a_vecs = l2_normalize(all_a_vecs)
    # b_vecs = l2_normalize(all_b_vecs)

    # sims = (a_vecs * b_vecs).sum(axis=1)
    cosine_scores = 1 - (paired_cosine_distances(all_anchor_vectors, all_pos_vectors))
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

    t_total = len(train_samples) // TRAIN_CONFIG['gradient_accumulation_steps'] * args['num_epochs']

    # Prepare optimizer and schedule (linear warmup and decay)
    optimizer_grouped_parameters = get_parameters(model)
    warmup_steps = int(t_total * TRAIN_CONFIG['warmup_ratio'])
    args['warmup_steps'] = warmup_steps
    optimizer, scheduler = init_optimizer(t_total, optimizer_grouped_parameters, args)

    # Train!
    print("***** Running training *****")
    print("  Num examples = %d", len(train_dataset))
    print("  Num Epochs = %d", args['num_epochs'])
    print("  Instantaneous batch size per GPU = %d", train_batch_size)
    print("  Gradient Accumulation steps = %d", TRAIN_CONFIG['gradient_accumulation_steps'])
    print("  Total optimization steps = %d", t_total)

    if args['object_type'] == "classification":
        loss_func = nn.CrossEntropyLoss()
    elif args['object_type'] == "regression":
        loss_func = nn.CosineEmbeddingLoss()
    else:
        loss_func = nn.MarginRankingLoss(margin=args['triplet_margin'])

    WANDB.watch(model, log='all')
    global_steps = 0
    model.to(args['device'])
    model.zero_grad()
    best_score = 0.0
    best_epoch = 0
    set_seed(args['seed'])
    for epoch in range(args['num_epochs']):
        model.train()
        total_loss = 0.0
        for step, batch in enumerate(tqdm(train_dataloader, desc="Training")):
            inputs = {
                'anchor_input_ids': batch['anchor_input_ids'].to(args['device']),
                'pos_input_ids': batch['pos_input_ids'].to(args['device']),
            }
            if args['task_type'] == 'nli':
                inputs['neg_input_ids'] = batch['neg_input_ids'].to(args['device'])

            logits = model(**inputs, pooling_strategy=args['pooling_strategy'])
            label = batch['label'].to(args['device'])
            if args['object_type'] == "regression":
                label = label.to(torch.float)
            loss = loss_func(logits, label)
            if TRAIN_CONFIG['gradient_accumulation_steps'] > 1:
                loss = loss / TRAIN_CONFIG['gradient_accumulation_steps']
            loss.backward()

            WANDB.log({'Train/loss': loss.item()}, step=global_steps)
            total_loss += loss.item()
            if (step + 1) % TRAIN_CONFIG['gradient_accumulation_steps'] == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_steps += 1

            if (step + 1) % args['eval_steps'] == 0:
                corrcoef, pearsonr = evaluate(valid_dataloader, model, args)
                WANDB.log({'Evaluation/corrcoef': corrcoef, 'Evaluation/pearsonr': pearsonr}, step=global_steps)
                print(f"evaluate results: corrcoef: {corrcoef}\tpearsonr: {pearsonr}")
                if pearsonr > best_score:
                    best_score = pearsonr
                    best_epoch = best_epoch

                    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                    output_file = os.path.join(args['model_save_path'], 'best_model.bin')
                    torch.save(model_to_save.state_dict(), output_file)


def main():
    root_path = "/data2/work2/chenzhihao/NLP"
    config = {
        'model_type': "roberta",
        'model_name_or_path': "/data2/work2/chenzhihao/NLP/pretrained_models/chinese-roberta-wwm-ext",
        'output_dir': root_path + "/experiments/output_file_dir/semantic_match",
        'config_path': "/data2/work2/chenzhihao/NLP/pretrained_models/chinese-roberta-wwm-ext",
        'tokenizer_path': "/data2/work2/chenzhihao/NLP/pretrained_models/chinese-roberta-wwm-ext",
        'do_train': True,
        'do_test': True,
        'num_labels': 2,
        'train_batch_size': 64,
        'valid_batch_size': 64,
        'test_batch_size': 64,
        'num_epochs': 30,
        'max_seq_length': 128,
        'eval_steps': 500,
        'object_type': "classification",  # classification, regression, triplet
        'task_type': "match",  # "match" or "nli"
        'scheduler_type': "linear",
        'pooling_strategy': "first-last-avg",  # first-last-avg, last-avg, cls, pooler
        'distance_type': "",
        'triplet_margin': 0.5,
        'data_type': "BQ",  # ATEC, BQ, LCQMC, PAWSX, STS-B
        'train_dataset': "train.data",
        'valid_dataset': "valid.data",
        'test_dataset': "test.data",
        'cuda_number': "1",
        'num_worker': 4,
        'seed': 2333
    }

    data_dir = "/data2/work2/chenzhihao/datasets/chinese-semantics-match-dataset/" + config['data_type']
    if not os.path.exists(data_dir):
        raise ValueError(f"The path of '{data_dir}' not exist")
    output_dir = root_path + "/experiments/output_file_dir/semantic_match"
    date = datetime.now().strftime("%Y-%m-%d_%H")  # noqa
    model_save_path = output_dir + f"/{config['data_type']}-sbert2-{config['model_type']}-{date}"
    config['model_save_path'] = model_save_path
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)

    device = torch.device(f"cuda:{config['cuda_number']}") if torch.cuda.is_available() else torch.device('cpu')
    config['device'] = device

    train_samples = load_data(os.path.join(data_dir, config['data_type']+'.'+config['train_dataset']))
    valid_samples = load_data(os.path.join(data_dir, config['data_type']+'.'+config['valid_dataset']))

    tokenizer = BertTokenizer.from_pretrained(config['model_name_or_path']
                                              if not config['tokenizer_path'] else config['tokenizer_path'])
    model = init_model(model_path=config['model_name_or_path'], num_labels=config['num_labels'], args=config)
    model.to(device)

    if config['do_train']:
        global WANDB
        WANDB, run = init_wandb_writer(project_name='sentence_bert',
                                       train_args=TRAIN_CONFIG,
                                       group_name="semantic_match",
                                       experiment_name="roberta")
        train(train_samples, valid_samples, model, tokenizer, config)
    if config['do_test']:
        test_samples = load_data(os.path.join(data_dir, config['data_type'] + '.' + config['test_dataset']))
        test_dataset = SentDataSet(test_samples, tokenizer, config['max_seq_length'], task_type=config['task_type'])
        test_dataloader = DataLoader(test_dataset, batch_size=config['test_batch_size'], collate_fn=collate_fn)

        model = init_model(model_path=config['model_save_path'], num_labels=config['num_labels'],
                           args=config, flag='test')
        model.to(device)
        corrcoef, pearsonr = evaluate(test_dataloader, model, config)
        print("test results: ", json.dumps({'corrcoef': corrcoef, 'pearsonr': pearsonr}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
