#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: czh
@email:
@date: 2022/7/6 11:20
"""
import codecs
# 有监督监督CoSENT
# https://github.com/shawroad/Semantic-Textual-Similarity-Pytorch/blob/main/CoSENT/run_cosent.py
import os
import sys

import numpy as np
from tqdm import tqdm
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertConfig, get_scheduler, AdamW, set_seed

dirname = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join('/'.join(dirname.split('/')[:-2])))

from nlp.tools.common import init_wandb_writer
from nlp.models.sentence_embedding_models import CoSENT
from nlp.processors.semantic_match_preprocessor import pad_to_maxlen
from nlp.metrics.sematic_match_metric import compute_corrcoef, compute_pearsonr, l2_normalize


def load_data(path):
    sentence, label = [], []
    with open(path, 'r', encoding='utf8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split('\t')
            try:
                sentence.extend([line[0], line[1]])
                lab = int(line[2])
                label.extend([lab, lab])
            except:
                continue
    return sentence, label


def load_test_data(path):
    sent1, sent2, label = [], [], []
    with open(path, 'r', encoding='utf8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split('\t')
            sent1.append(line[0])
            sent2.append(line[1])
            label.append(int(line[2]))
    return sent1, sent2, label


class CustomDataset(Dataset):
    def __init__(self, sentence, label, tokenizer, max_seq_length=512):
        self.sentence = sentence
        self.label = label
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    def __len__(self):
        return len(self.sentence)

    def __getitem__(self, index):
        inputs = self.tokenizer.encode_plus(
            text=self.sentence[index],
            text_pair=None,
            truncation=True,
            max_length=self.max_seq_length,
            add_special_tokens=True,
            return_token_type_ids=True
        )
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids,
            'label': self.label[index]
        }


def collate_fn(batch):
    # 按batch进行padding获取当前batch中最大长度
    max_len = max([len(d['input_ids']) for d in batch])

    # 定一个全局的max_len
    # max_len = 128

    input_ids, attention_mask, token_type_ids, labels = [], [], [], []

    for item in batch:
        input_ids.append(pad_to_maxlen(item['input_ids'], max_len=max_len))
        attention_mask.append(pad_to_maxlen(item['attention_mask'], max_len=max_len))
        token_type_ids.append(pad_to_maxlen(item['token_type_ids'], max_len=max_len))
        labels.append(item['label'])

    all_input_ids = torch.tensor(input_ids, dtype=torch.long)
    all_input_mask = torch.tensor(attention_mask, dtype=torch.long)
    all_segment_ids = torch.tensor(token_type_ids, dtype=torch.long)
    all_label_ids = torch.tensor(labels, dtype=torch.float)
    return {
        'input_ids': all_input_ids,
        'attention_mask': all_input_mask,
        'segment_ids': all_segment_ids,
        'labels': all_label_ids
    }


def calc_loss(y_true, y_pred, args):
    # 1. 取出真实的标签
    y_true = y_true[::2]  # tensor([1, 0, 1]) 真实的标签

    # 2. 对输出的句子向量进行l2归一化   后面只需要对应为相乘  就可以得到cos值了
    norms = (y_pred ** 2).sum(axis=1, keepdims=True) ** 0.5
    y_pred = y_pred / torch.clip(norms, 1e-8, torch.inf)
    # y_pred = y_pred / norms

    # 3. 奇偶向量相乘
    y_pred = torch.sum(y_pred[::2] * y_pred[1::2], dim=1) * 20

    # 4. 取出负例-正例的差值
    y_pred = y_pred[:, None] - y_pred[None, :]  # 这里是算出所有位置 两两之间余弦的差值
    # 矩阵中的第i行j列  表示的是第i个余弦值-第j个余弦值
    y_true = y_true[:, None] < y_true[None, :]  # 取出负例-正例的差值
    y_true = y_true.float()
    y_pred = y_pred - (1 - y_true) * 1e12
    y_pred = y_pred.view(-1)
    if torch.cuda.is_available():
        y_pred = torch.cat((torch.tensor([0]).float().to(args['device']), y_pred), dim=0)  # 这里加0是因为e^0 = 1相当于在log中加了1
    else:
        y_pred = torch.cat((torch.tensor([0]).float(), y_pred), dim=0)  # 这里加0是因为e^0 = 1相当于在log中加了1

    return torch.logsumexp(y_pred, dim=0)


def get_sent_id_tensor(s_list, tokenizer, max_seq_length=512):
    input_ids, attention_mask, token_type_ids, labels = [], [], [], []
    max_len = max([len(_)+2 for _ in s_list])   # 这样写不太合适 后期想办法改一下
    max_len = min(max_len, max_seq_length)
    for s in s_list:
        inputs = tokenizer(text=s, add_special_tokens=True, return_token_type_ids=True,
                           padding='max_length', max_length=max_len, truncation=True)
        input_ids.append(inputs['input_ids'])
        attention_mask.append(inputs['attention_mask'])
        token_type_ids.append(inputs['token_type_ids'])
    all_input_ids = torch.tensor(input_ids, dtype=torch.long)
    all_input_mask = torch.tensor(attention_mask, dtype=torch.long)
    all_segment_ids = torch.tensor(token_type_ids, dtype=torch.long)
    return all_input_ids, all_input_mask, all_segment_ids


def evaluate(test_dataset, model, tokenizer, args):
    sent1, sent2, label = test_dataset
    all_a_vecs = []
    all_b_vecs = []
    all_labels = []
    model.eval()
    for step, (s1, s2, lab) in enumerate(tqdm(zip(sent1, sent2, label),
                                         desc=f"Evaluating supCoSENT on {args['data_type']}")):
        input_ids, input_mask, segment_ids = get_sent_id_tensor([s1, s2], tokenizer,
                                                                max_seq_length=args['max_seq_length'])
        lab = torch.tensor([lab], dtype=torch.float)

        input_ids = input_ids.to(args['device'])
        input_mask = input_mask.to(args['device'])
        segment_ids = segment_ids.to(args['device'])
        lab = lab.to(args['device'])

        with torch.no_grad():
            output = model(input_ids=input_ids, attention_mask=input_mask, token_type_ids=segment_ids)

        all_a_vecs.append(output[0].cpu().numpy())
        all_b_vecs.append(output[1].cpu().numpy())
        all_labels.extend(lab.cpu().numpy())

    all_a_vecs = np.array(all_a_vecs)
    all_b_vecs = np.array(all_b_vecs)
    all_labels = np.array(all_labels)

    a_vecs = l2_normalize(all_a_vecs)
    b_vecs = l2_normalize(all_b_vecs)
    sims = (a_vecs * b_vecs).sum(axis=1)
    corrcoef = compute_corrcoef(all_labels, sims)
    pearsonr = compute_pearsonr(all_labels, sims)
    return corrcoef, pearsonr


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
        bert_config.save_pretrained(args['model_save_path'])
        model = CoSENT(bert_config=bert_config, model_name_or_path=model_name_or_path,
                       pooler_type=args['pooling_strategy'])
    else:
        model = CoSENT(bert_config=bert_config, pooler_type=args['pooling_strategy'])
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


def train(train_samples, valid_samples, model, tokenizer, args, train_config):
    train_sentence, train_label = train_samples
    train_dataset = CustomDataset(sentence=train_sentence, label=train_label, tokenizer=tokenizer,
                                  max_seq_length=args['max_seq_length'])
    train_dataloader = DataLoader(dataset=train_dataset, shuffle=False, batch_size=args['train_batch_size'],
                                  collate_fn=collate_fn)

    t_total = len(train_sentence) * args['num_train_epochs']
    num_train_optimization_steps = int(
        len(train_dataset) / args['train_batch_size'] / args['gradient_accumulation_steps']) * args['num_train_epochs']

    # Prepare optimizer and schedule (linear warmup and decay)
    optimizer_grouped_parameters = get_parameters(model, args)
    warmup_steps = int(t_total * args['warmup_ratio'])
    args['warmup_steps'] = warmup_steps

    args['warmup_steps'] = warmup_steps
    optimizer, scheduler = init_optimizer(t_total, optimizer_grouped_parameters, args)
    if args['fp16']:
        from torch.cuda.amp import autocast, GradScaler
        scaler = GradScaler()

    wandb_logger, run = init_wandb_writer(project_name=args['project_name'],
                                          train_args=train_config,
                                          group_name=args['group_name'],
                                          experiment_name=args['experiment_name'])

    # Train!
    print("***** Running training *****")
    print("  Num train examples = %d", len(train_samples))
    print("  Num Epochs = %d", args['num_train_epochs'])
    print("  Instantaneous batch size per GPU = %d", args['train_batch_size'])
    print("  Gradient Accumulation steps = %d", args['gradient_accumulation_steps'])
    print("  Total optimization steps = %d", num_train_optimization_steps)

    wandb_logger.watch(model, log='all')
    model.to(args['device'])
    model.zero_grad()

    torch.cuda.empty_cache()
    global_steps = 0
    best_score = 0.0
    best_epoch = 0
    set_seed(args['seed'])

    for epoch in range(args['num_train_epochs']):
        model.train()
        total_loss = 0.0
        epoch_steps = 0
        for step, batch in enumerate(tqdm(train_dataloader,
                                          desc=f"Training on epoch {epoch}/{args['num_train_epochs']}")):
            inputs = {
                'input_ids': batch['input_ids'].to(args['device']),
                'attention_mask': batch['attention_mask'].to(args['device']),
                'token_type_ids': batch['segment_ids'].to(args['device'])
            }
            label_ids = batch['labels'].to(args['device'])
            if args['fp16']:
                logits = model(**inputs)
                loss = calc_loss(label_ids, logits, args)
            else:
                logits = model(**inputs)
                loss = calc_loss(label_ids, logits, args)

            if args['gradient_accumulation_steps'] > 1:
                loss = loss / args['gradient_accumulation_steps']
            if args['fp16']:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (step + 1) % args['gradient_accumulation_steps'] == 0:
                total_loss += loss.item()
                if args['fp16']:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                model.zero_grad()
                epoch_steps += 1
                global_steps += 1
                wandb_logger.log({'Train/loss': total_loss/epoch_steps})

            if (step + 1) % args['eval_steps'] == 0 or step == len(train_dataloader) - 1:
                corrcoef, pearsonr = evaluate(valid_samples, model, tokenizer, args)
                wandb_logger.log({'Evaluation/spearman': corrcoef, 'Evaluation/pearsonr': pearsonr}, step=global_steps)
                print(f"evaluate results: spearman: {corrcoef}\tpearsonr: {pearsonr}")
                if corrcoef > best_score:
                    best_score = corrcoef
                    best_score_pearsonr = pearsonr
                    best_epoch = best_epoch

                    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                    output_file = os.path.join(args['model_save_path'], 'pytorch_model.bin')
                    torch.save(model_to_save.state_dict(), output_file)
                    tokenizer.save_pretrained(args['model_save_path'])
                    with codecs.open(os.path.join(args['model_save_path'], "eval_result.txt"),
                                     "w", encoding="utf8") as fw:
                        fw.write("best_epoch: {}\tpearsonr: {}\tspearman: {}".format(best_epoch,
                                                                                     best_score_pearsonr,
                                                                                     best_score))
                torch.cuda.empty_cache()


def main():
    root_path = "/root/work2/work2/chenzhihao/NLP"
    config = {
        'model_type': "roberta-wwm-ext",
        'model_name_or_path': "/root/work2/work2/chenzhihao/pretrained_models/chinese-roberta-wwm-ext",
        'output_dir': root_path + "/experiments/output_file_dir/semantic_match",
        'data_dir': "/root/work2/work2/chenzhihao/datasets/chinese-semantics-match-dataset/",
        'config_path': None,
        'tokenizer_path': None,
        'project_name': 'semantic_match',
        'group_name': 'nlp',
        'experiment_name': 'feedback-sup_cosent-roberta-wwm-ext',
        'do_train': True,
        'do_test': True,
        'lr_rate': 2e-5,
        'gradient_accumulation_steps': 1,
        'warmup_ratio': 0.1,
        'adam_epsilon': 1e-8,
        'weight_decay': 0.01,
        'scheduler_type': 'linear',
        'train_batch_size': 128,  # 必须是2的倍数
        'valid_batch_size': 128,
        'test_batch_size': 128,
        'num_train_epochs': 200,
        'max_seq_length': 128,
        'eval_steps': 3000,
        'object_type': "classification",  # classification, regression, triplet
        'task_type': "match",  # "match" or "nli"
        'pooling_strategy': "last-avg",  # first-last-avg, last-avg, cls, pooler
        'data_type': "ATEC",  # ATEC, BQ, LCQMC, PAWSX, STS-B
        'train_dataset': ".train.data",
        'valid_dataset': ".valid.data",
        'test_dataset': ".test.data",
        'cuda_number': "4",
        'num_worker': 4,
        'fp16': True,
        'seed': 2333
    }
    train_config = {
        'lr_rate': config['lr_rate'],
        'gradient_accumulation_steps': config['gradient_accumulation_steps'],
        'warmup_ratio': config['warmup_ratio'],
        'adam_epsilon': config['adam_epsilon'],
        'weight_decay': config['weight_decay'],
        'scheduler_type': config['scheduler_type']
    }

    data_dir = config['data_dir'] + config['data_type']
    if not os.path.exists(data_dir):
        raise ValueError(f"The path of '{data_dir}' not exist")
    model_save_path = f"{config['output_dir']}/{config['data_type']}-supcosent-{config['model_type']}"
    config['model_save_path'] = model_save_path
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)

    device = torch.device(f"cuda:{config['cuda_number']}") if torch.cuda.is_available() else torch.device('cpu')
    config['device'] = device

    train_file_path = os.path.join(data_dir, config['data_type'] + config['train_dataset'])
    valid_file_path = os.path.join(data_dir, config['data_type'] + config['valid_dataset'])
    test_file_path = os.path.join(data_dir, config['data_type'] + config['test_dataset'])

    train_samples = load_data(train_file_path)
    valid_samples = load_test_data(valid_file_path)
    test_samples = load_test_data(test_file_path)

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
        print(f"test results: spearman: {corrcoef}\tpearsonr: {pearsonr}")
        with codecs.open(os.path.join(model_save_path, 'test_result.txt'), "w", encoding="utf8") as fw:
            fw.write("pearsonr: {}\tspearman: {}".format(pearsonr, corrcoef))


if __name__ == "__main__":
    main()
