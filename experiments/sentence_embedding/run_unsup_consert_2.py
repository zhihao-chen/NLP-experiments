#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: czh
@email:
@date: 2022/7/5 14:15
"""
# 无监督对比学习ConSERT
# 参考https://github.com/shawroad/Semantic-Textual-Similarity-Pytorch/blob/main/ConSERT/run_unsup_consert.py
import os
import sys

import numpy as np
from tqdm import tqdm
from datetime import datetime

import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertConfig, get_scheduler, AdamW, set_seed
from sklearn.metrics.pairwise import paired_cosine_distances

sys.path.append('/data2/work2/chenzhihao/NLP')

from nlp.tools.common import init_wandb_writer
from nlp.models.sentence_embedding_models import ConSERTV1
from nlp.processors.semantic_match_preprocessor import load_data, SentDataSet, pad_to_maxlen
from nlp.metrics.sematic_match_metric import compute_corrcoef, compute_pearsonr, l2_normalize

WANDB = None


def collate_fn(batch_data):
    max_len = 0
    for d in batch_data:
        max_len = max(max_len, len(d['anchor_input_ids']))
        max_len = max(max_len, len(d['pos_input_ids']))

    all_input_ids = []
    all_attention_masks = []
    for item in batch_data:
        anchor_input_ids = pad_to_maxlen(item['anchor_input_ids'], max_len=max_len)
        anchor_attention_mask = pad_to_maxlen(item['anchor_attention_mask'], max_len=max_len)

        pos_input_ids = pad_to_maxlen(item['pos_input_ids'], max_len=max_len)
        pos_attention_mask = pad_to_maxlen(item['pos_attention_mask'], max_len=max_len)

        all_input_ids.append(anchor_input_ids)
        all_attention_masks.append(anchor_attention_mask)

        all_input_ids.append(pos_input_ids)
        all_attention_masks.append(pos_attention_mask)
    assert len(all_input_ids) == len(all_attention_masks)

    input_ids = torch.LongTensor(all_input_ids)
    attention_mask = torch.LongTensor(all_attention_masks)

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask
    }


def init_model(model_name_or_path, args, flag='train'):
    """
    当训练时，model_name_or_path就是语言模型的地址，当测试时就是保存的模型地址
    :param model_name_or_path:
    :param args:
    :param flag:
    :return:
    """
    bert_config = BertConfig.from_pretrained(args['config_path'] if args['config_path'] else model_name_or_path)
    if flag == 'train':
        model = ConSERTV1(args=args, bert_config=bert_config,
                          bert_model_path=model_name_or_path,
                          temperature=args['temperature'],
                          cutoff_rate=args['cutoff_rate'],
                          close_dropout=args['close_dropout'])
    else:
        model = ConSERTV1(
            args=args, bert_config=bert_config,
            temperature=args['temperature'],
            cutoff_rate=args['cutoff_rate'],
            close_dropout=args['close_dropout']
        )
        model.load_state_dict(torch.load(model_name_or_path+"/best_model.bin", map_location=args['device']))
    return model


def init_optimizer(total, parameters, args):
    optimizer = AdamW(parameters, lr=args['lr_rate'], eps=args['adam_epsilon'])
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


def get_parameters(model, args):
    no_decay = ["bias", "LayerNorm.weight", "LayerNorm.bias"]
    bert_param_optimizer = list(model.named_parameters())
    optimizer_grouped_parameters = [
        {'params': [p for n, p in bert_param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': args["weight_decay"], 'lr': args['lr_rate']},
        {'params': [p for n, p in bert_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
         'lr': args['lr_rate']},
    ]
    return optimizer_grouped_parameters


def evaluate(valid_samples, model, tokenizer, args):
    model.eval()
    all_a_vecs = []
    all_b_vecs = []
    all_labels = []
    for step, batch in enumerate(tqdm(valid_samples,
                                      desc=f"Evaluating unsup-ConSERT on {args['data_type']}")):
        s1, s2, label = batch
        s1_inputs = tokenizer(s1, return_tensors='pt', truncation=True, max_length=args['max_seq_length'])
        s1_input_ids = s1_inputs['input_ids'].to(args['device'])
        s1_attention_mask = s1_inputs['attention_mask'].to(args['device'])

        s2_inputs = tokenizer(s2, return_tensors='pt', truncation=True, max_length=args['max_seq_length'])
        s2_input_ids = s2_inputs['input_ids'].to(args['device'])
        s2_attention_mask = s2_inputs['attention_mask'].to(args['device'])

        with torch.no_grad():
            s1_embed = model.encode(s1_input_ids, s1_attention_mask)[0]
            s2_embed = model.encode(s2_input_ids, s2_attention_mask)[0]

        all_a_vecs.append(s1_embed.cpu().numpy())
        all_b_vecs.append(s2_embed.cpu().numpy())
        all_labels.extend(np.array([label]))

    all_a_vecs = np.array(all_a_vecs)
    all_b_vecs = np.array(all_b_vecs)
    all_labels = np.array(all_labels)

    # a_vecs = l2_normalize(all_a_vecs)
    # b_vecs = l2_normalize(all_b_vecs)

    # sims = (a_vecs * b_vecs).sum(axis=1)
    cosine_scores = 1 - (paired_cosine_distances(all_a_vecs, all_b_vecs))
    corrcoef = compute_corrcoef(all_labels, cosine_scores)
    pearsonr = compute_pearsonr(all_labels, cosine_scores)
    return corrcoef, pearsonr


def train(train_samples, valid_samples, model, tokenizer, args, train_config):
    train_data = SentDataSet(train_samples, tokenizer, args['max_seq_length'], args['task_type'])
    train_dataloader = DataLoader(train_data, shuffle=True, batch_size=args['train_batch_size'],
                                  collate_fn=collate_fn, num_workers=args['num_workers'])

    t_total = len(train_samples) / args['gradient_accumulation_steps'] * args['num_train_epochs']

    # Prepare optimizer and schedule (linear warmup and decay)
    optimizer_grouped_parameters = get_parameters(model, args)
    warmup_steps = int(t_total * args['warmup_ratio'])
    args['warmup_steps'] = warmup_steps

    args['warmup_steps'] = warmup_steps
    optimizer, scheduler = init_optimizer(t_total, optimizer_grouped_parameters, args)

    global WANDB
    WANDB, run = init_wandb_writer(project_name='unsup_consert',
                                   train_args=train_config,
                                   group_name="semantic_match",
                                   experiment_name="roberta")

    # Train!
    print("***** Running training *****")
    print("  Num train examples = %d", len(train_samples))
    print("  Num Epochs = %d", args['num_train_epochs'])
    print("  Instantaneous batch size per GPU = %d", args['train_batch_size'])
    print("  Gradient Accumulation steps = %d", args['gradient_accumulation_steps'])
    print("  Total optimization steps = %d", t_total)

    WANDB.watch(model, log='all')
    global_steps = 0
    model.to(args['device'])
    model.zero_grad()
    best_score = 0.0
    best_epoch = 0
    set_seed(args['seed'])

    for epoch in range(args['num_train_epochs']):
        model.train()
        total_loss = 0.0
        for step, batch in enumerate(tqdm(train_dataloader,
                                          desc=f"Training unsup ConSERT on {args['data_type']}")):
            inputs = {
                'input_ids': batch['input_ids'].to(args['device']),
                'attention_mask': batch['attention_mask'].to(args['device'])
            }
            outputs = model(**inputs)
            loss = outputs['loss']
            if args['gradient_accumulation_steps'] > 1:
                loss = loss / args['gradient_accumulation_steps']
            loss.backward()
            WANDB.log({'Train/loss': loss.item()})

            if (step + 1) % args['gradient_accumulation_steps'] == 0:
                total_loss += loss.item()
                optimizer.step()
                scheduler.step()
                model.zero_grad()
                global_steps += 1

            if (step + 1) % args['eval_steps'] == 0:
                corrcoef, pearsonr = evaluate(valid_samples, model, tokenizer, args)
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
        'lr_rate': 2e-5,
        'gradient_accumulation_steps': 1,
        'warmup_ratio': 0.1,
        'adam_epsilon': 1e-8,
        'weight_decay': 0.01,
        'scheduler_type': 'linear',
        'model_type': "roberta",
        'model_name_or_path': "/data2/work2/chenzhihao/NLP/pretrained_models/chinese-roberta-wwm-ext",
        'output_dir': root_path + "/experiments/output_file_dir/semantic_match",
        'config_path': "/data2/work2/chenzhihao/NLP/pretrained_models/chinese-roberta-wwm-ext",
        'tokenizer_path': "/data2/work2/chenzhihao/NLP/pretrained_models/chinese-roberta-wwm-ext",
        'do_train': True,
        'do_test': True,
        'train_batch_size': 64,
        'valid_batch_size': 32,
        'test_batch_size': 32,
        'num_train_epochs': 30,
        'max_seq_length': 64,
        'temperature': 0.05,
        'cutoff_rate': 0.15,
        'close_dropout': True,
        'eval_steps': 40,
        'task_type': "match",  # match, nli
        'data_type': "STS-B",  # ATEC, BQ, LCQMC, PAWSX, STS-B
        'object_type': "regression",  # classifier, regression, triplet
        'train_dataset': "train.data",
        'valid_dataset': "valid.data",
        'test_dataset': "test.data",
        'cuda_number': '2',
        'num_workers': 4,
        'seed': 2333
    }
    train_config = {
        'lr_rate': config['lr_rate'],
        'gradient_accumulation_steps': config['gradient_accumulation_steps'],
        'warmup_ratio': config['warmup_ratio'],
        'adam_epsilon': config['adam_epsilon'],
        'weight_decay': config['weight_decay'],
        'scheduler_type': config['scheduler_type'],
        'temperature': config['temperature'],
    }
    data_dir = "/data2/work2/chenzhihao/datasets/chinese-semantics-match-dataset/" + config['data_type']
    if not os.path.exists(data_dir):
        raise ValueError(f"The path of '{data_dir}' not exist")
    model_save_path = f"{config['output_dir']}/{config['data_type']}-unsup-consert2-{config['model_type']}"
    config['model_save_path'] = model_save_path
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)

    device = torch.device(f"cuda:{config['cuda_number']}") if torch.cuda.is_available() else torch.device('cpu')
    config['device'] = device

    torch.cuda.set_device(device)

    print("****** Loading datasets ******")
    train_samples = load_data(os.path.join(data_dir, config['data_type'] + '.' + config['train_dataset']))
    valid_samples = load_data(os.path.join(data_dir, config['data_type'] + '.' + config['valid_dataset']))
    test_samples = load_data(os.path.join(data_dir, config['data_type'] + '.' + config['test_dataset']))

    tokenizer = BertTokenizer.from_pretrained(config['model_name_or_path']
                                              if not config['tokenizer_path'] else config['tokenizer_path'])
    model = init_model(config['model_name_or_path'], config)
    model.to(device)

    if config['do_train']:
        train(train_samples, valid_samples, model, tokenizer, config, train_config)
    if config['do_test']:
        model = init_model(config['model_save_path'], args=config, flag="test")
        model.to(device)

        corrcoef, pearsonr = evaluate(test_samples, model, tokenizer, config)
        print(f"test results: corrcoef: {corrcoef}\tpearsonr: {pearsonr}")


if __name__ == "__main__":
    main()

