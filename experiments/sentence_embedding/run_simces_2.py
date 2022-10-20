#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: czh
@email:
@date: 2022/7/4 11:52
"""
import os
import sys
import random

import numpy as np
from tqdm import tqdm
from datetime import datetime

import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertConfig, get_scheduler, AdamW, set_seed
from sklearn.metrics.pairwise import paired_cosine_distances

sys.path.append('/data2/work2/chenzhihao/NLP')

from nlp.tools.common import init_wandb_writer
from nlp.models.sentence_embedding_models import SimCSEModel
from nlp.processors.semantic_match_preprocessor import load_data, SentDataSet, load_data_for_snli, pad_to_maxlen
from nlp.metrics.sematic_match_metric import compute_corrcoef, compute_pearsonr, l2_normalize

WANDB = None


def collate_fn(batch_data):
    max_len = 0
    for d in batch_data:
        max_len = max(max_len, len(d['anchor_input_ids']))
        max_len = max(max_len, len(d['pos_input_ids']))
        if 'neg_input_ids' in d:
            max_len = max(max_len, len(d['neg_input_ids']))

    input_ids_lst = []
    attention_mask_lst = []
    label_lst = []
    for item in batch_data:

        anchor_input_ids = pad_to_maxlen(item['anchor_input_ids'], max_len=max_len)
        pos_input_ids = pad_to_maxlen(item['pos_input_ids'], max_len=max_len)
        anchor_attention_mask = pad_to_maxlen(item['anchor_attention_mask'], max_len=max_len)
        pos_attention_mask = pad_to_maxlen(item['pos_attention_mask'], max_len=max_len)
        if 'neg_input_ids' in item:
            neg_input_ids = pad_to_maxlen(item['neg_input_ids'], max_len=max_len)
            neg_attention_mask = pad_to_maxlen(item['neg_attention_mask'], max_len=max_len)
            input_ids_lst.append([anchor_input_ids, pos_input_ids, neg_input_ids])
            attention_mask_lst.append([anchor_attention_mask, pos_attention_mask, neg_attention_mask])
        else:
            input_ids_lst.append([anchor_input_ids, pos_input_ids])
            attention_mask_lst.append([anchor_attention_mask, pos_attention_mask])
        if item['label'] is not None:
            label_lst.append(item['label'])

    input_ids = torch.LongTensor(input_ids_lst)
    attention_masks = torch.LongTensor(attention_mask_lst)
    if label_lst:
        labels = torch.FloatTensor(label_lst)  # 分类这里为torch.long  回归这里为torch.float
    else:
        labels = None

    return {
        'input_ids': input_ids,
        'attention_mask': attention_masks,
        'labels': labels
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
        model = SimCSEModel(model_args=args, bert_config=bert_config, bert_model_path=model_name_or_path,
                            pooling_strategy=args['pooling_strategy'])
    else:
        model = SimCSEModel(model_args=args, bert_config=bert_config, pooling_strategy=args['pooling_strategy'])
        model.load_state_dict(torch.load(model_name_or_path + "/best_model.bin", map_location=args['device']))
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


def prepare_datasets(data_dir, data_type="STS-B", object_type="classification",
                     seg_tag='\t', train_mode='sup', is_train=False):
    dataset = load_data(data_dir, seg_tag=seg_tag)  # noqa
    data_samples = []
    for data in dataset:
        if data_type == "STS-B":
            label = data[2] / 5.0
        else:
            label = data[2] if object_type == "classification" else float(data[2])
        if train_mode == 'sup' or not is_train:
            data_samples.append([data[0], data[1], label])
        if train_mode == 'unsup':
            data_samples.append([data[0], data[0]])
            data_samples.append([data[1], data[1]])
    return data_samples


def prepare_snli_datasets(data_dir, prefix='cnsd_snli_v1.0'):
    def add_to_samples(sent1, sent2, label):
        if sent1 not in train_data:
            train_data[sent1] = {'contradiction': set(), 'entailment': set(), 'neutral': set()}
        train_data[sent1][label].add(sent2)

    train_data_path = os.path.join(data_dir, prefix + ".train.json")
    dev_data_path = os.path.join(data_dir, prefix + ".dev.json")
    test_data_path = os.path.join(data_dir, prefix + ".test.json")

    train_data_lst = load_data_for_snli(train_data_path, return_list=True)
    dev_data_lst = load_data_for_snli(dev_data_path, return_list=True)
    test_data_lst = load_data_for_snli(test_data_path, return_list=True)

    all_datas = train_data_lst + dev_data_lst + test_data_lst

    train_data = {}
    for item in all_datas:
        s1 = item['sentence1'].strip()
        s2 = item['sentence2'].strip()
        tag = item['gold_label'].strip()

        add_to_samples(s1, s2, tag)
        add_to_samples(s2, s1, tag)

    train_samples = []
    for s1, others in train_data.items():
        if len(others['entailment']) > 0 and len(others['contradiction']) > 0:
            train_samples.append([s1, random.choice(list(others['entailment'])),
                                  random.choice(list(others['contradiction']))])
            train_samples.append([random.choice(list(others['entailment'])), s1,
                                  random.choice(list(others['contradiction']))])
    np.random.shuffle(train_samples)
    return train_samples


def evaluate(valid_samples, model, tokenizer, args):
    model.eval()
    all_a_vecs = []
    all_b_vecs = []
    all_labels = []
    for step, batch in enumerate(tqdm(valid_samples,
                                      desc=f"Evaluating {args['train_mode']}SimCSE on {args['data_type']}")):
        s1, s2, label = batch
        s1_inputs = tokenizer(s1, return_tensors='pt', truncation=True, max_length=args['max_seq_length'])
        s1_input_ids = s1_inputs['input_ids'].to(args['device'])
        s1_attention_mask = s1_inputs['attention_mask'].to(args['device'])

        s2_inputs = tokenizer(s2, return_tensors='pt', truncation=True, max_length=args['max_seq_length'])
        s2_input_ids = s2_inputs['input_ids'].to(args['device'])
        s2_attention_mask = s2_inputs['attention_mask'].to(args['device'])

        with torch.no_grad():
            pooler_output_s1 = model.get_sent_embed(s1_input_ids, s1_attention_mask)[0]
            pooler_output_s2 = model.get_sent_embed(s2_input_ids, s2_attention_mask)[0]

        all_a_vecs.append(pooler_output_s1.cpu().numpy())
        all_b_vecs.append(pooler_output_s2.cpu().numpy())
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
    train_batch_size = args['train_batch_size']
    train_data = SentDataSet(train_samples, tokenizer, args['max_seq_length'], task_type=args['task_type'])
    train_dataloader = DataLoader(train_data, batch_size=train_batch_size,
                                  collate_fn=collate_fn, num_workers=args['num_worker'])

    t_total = len(train_samples) / args['gradient_accumulation_steps'] * args['num_train_epochs']

    # Prepare optimizer and schedule (linear warmup and decay)
    optimizer_grouped_parameters = get_parameters(model, args)
    warmup_steps = int(t_total * args['warmup_ratio'])
    args['warmup_steps'] = warmup_steps

    args['warmup_steps'] = warmup_steps
    optimizer, scheduler = init_optimizer(t_total, optimizer_grouped_parameters, args)

    global WANDB
    WANDB, run = init_wandb_writer(project_name=args['train_mode'] + 'simces',
                                   train_args=train_config,
                                   group_name="semantic_match",
                                   experiment_name="roberta")

    # Train!
    print("***** Running training *****")
    print("  Num train examples = %d", len(train_samples))
    print("  Num Epochs = %d", args['num_train_epochs'])
    print("  Instantaneous batch size per GPU = %d", train_batch_size)
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
                                          desc=f"Training {args['train_mode']}SimCSE on {args['train_data_type']}")):
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
                scheduler.step()
                optimizer.step()
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
        'lr_rate': 5e-5,
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
        'do_mlm': False,  # "Whether to use MLM auxiliary objective."
        'train_batch_size': 90,
        'valid_batch_size': 90,
        'test_batch_size': 90,
        'num_train_epochs': 30,
        'max_seq_length': 128,
        'eval_steps': 50,
        'object_type': "classification",  # classification, regression, triplet
        'task_type': "match",  # "match" or "nli"
        'pooling_strategy': "first-last-avg",  # first-last-avg, last-avg, cls, pooler, last2avg
        'distance_type': "",
        'triplet_margin': 0.5,
        'temp': 0.05,  # "Temperature for softmax."
        'hard_negative_weight': 0,  # The logit of weight for hard negatives(only effective if hard negatives are used)
        'mlm_weight': 0.1,  # "Weight for MLM auxiliary objective (only effective if --do_mlm)."
        'mlp_only_train': False,  # "Use MLP only during training"
        'train_mode': 'unsup',  # sup, unsup
        'train_data_type': 'STS-B',
        'data_type': "STS-B",  # ATEC, BQ, LCQMC, PAWSX, STS-B, SNLI
        'train_dataset': "train.data",
        'valid_dataset': "valid.data",
        'test_dataset': "test.data",
        'cuda_number': "1",
        'num_worker': 0,
        'seed': 2333
    }
    train_config = {
        'train_mode': config['train_mode'],
        'lr_rate': config['lr_rate'],
        'gradient_accumulation_steps': config['gradient_accumulation_steps'],
        'warmup_ratio': config['warmup_ratio'],
        'adam_epsilon': config['adam_epsilon'],
        'weight_decay': config['weight_decay'],
        'scheduler_type': config['scheduler_type'],
        'temp': config['temp'],
        'hard_negative_weight': config['hard_negative_weight']
    }

    data_dir = "/data2/work2/chenzhihao/datasets/chinese-semantics-match-dataset/"
    if not os.path.exists(data_dir):
        raise ValueError(f"The path of '{data_dir}' not exist")
    date = datetime.now().strftime("%Y-%m-%d_%H")  # noqa
    model_save_path = f"{config['output_dir']}/{config['train_data_type']}-" \
                      f"{config['train_mode']}simcse2-{config['model_type']}-{date}"
    config['model_save_path'] = model_save_path
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)

    device = torch.device(f"cuda:{config['cuda_number']}") if torch.cuda.is_available() else torch.device('cpu')
    config['device'] = device

    # 有监督训练集是SNLI数据集
    if config['train_mode'] == 'sup':
        train_samples = prepare_snli_datasets(os.path.join(data_dir, config['train_data_type']))
    else:
        train_samples = prepare_datasets(os.path.join(data_dir, config['data_type'],
                                                      config['data_type'] + '.' + config['train_dataset']),
                                         data_type=config['data_type'],
                                         object_type=config['object_type'],
                                         train_mode=config['train_mode'])
    # 验证集和测试集是STS-B数据集
    valid_samples = prepare_datasets(os.path.join(data_dir, config['data_type'],
                                                  config['data_type'] + '.' + config['valid_dataset']),
                                     data_type=config['data_type'],
                                     object_type=config['object_type'])
    test_samples = prepare_datasets(os.path.join(data_dir, config['data_type'],
                                                 config['data_type'] + '.' + config['test_dataset']),
                                    data_type=config['data_type'],
                                    object_type=config['object_type'])

    tokenizer = BertTokenizer.from_pretrained(config['model_name_or_path']
                                              if not config['tokenizer_path'] else config['tokenizer_path'])
    model = init_model(config['model_name_or_path'], config)
    model.to(device)

    if config['do_train']:
        train(train_samples, valid_samples, model, tokenizer, config, train_config)
    if config['do_test']:
        model = init_model(config['model_save_path'], args=config, flag="test")
        model.to(device)

        # test_data = SentDataSet(test_samples, tokenizer, config['max_seq_length'], task_type="match")
        # test_dataloader = DataLoader(test_data, batch_size=config['test_batch_size'], collate_fn=collate_fn)
        corrcoef, pearsonr = evaluate(test_samples, model, tokenizer, config)
        print(f"test results: corrcoef: {corrcoef}\tpearsonr: {pearsonr}")


if __name__ == "__main__":
    main()
