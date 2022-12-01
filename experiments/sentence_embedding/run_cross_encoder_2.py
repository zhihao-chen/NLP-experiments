#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: czh
@email: zhihao.chen@kuwo.cn
@date: 2022/11/30 15:31
"""
import os
import sys
import codecs
import logging
import json

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertConfig, get_scheduler, set_seed
from sklearn.metrics import precision_recall_fscore_support

dirname = os.path.dirname(os.path.abspath(__file__))
print(dirname)
sys.path.append(os.path.join('/'.join(dirname.split('/')[:-1])))
from nlp.models.sentence_embedding_models import CrossEncoder
from nlp.tools.common import init_wandb_writer

logger = logging.getLogger(__file__)


def load_data(path, data_type):
    datas = []
    with codecs.open(path, 'r', encoding='utf8') as f:
        lines = f.readlines()
        for line in tqdm(lines, desc=f"Loading {data_type} datas"):
            line = line.strip()
            if not line:
                continue
            lst = line.split('\t')
            assert len(lst) == 3, f"{len(lst)}\t{line}"
            label = int(lst[2])
            datas.append([lst[0], lst[1], label])

    return datas


class CustomDataset(Dataset):
    def __init__(self, dataset, tokenizer: BertTokenizer, max_seq_length):
        super(CustomDataset, self).__init__()
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.pad_token_id = tokenizer.pad_token_id

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        assert len(data) == 3
        text1 = data[0].strip()
        text2 = data[1].strip()
        label = int(data[2])

        inputs = self.tokenizer(text=text1, text_pair=text2, max_length=self.max_seq_length, truncation=True,
                                return_token_type_ids=True)
        return {
            'input_ids': inputs['input_ids'],
            'attention_mask': inputs['attention_mask'],
            'token_type_ids': inputs['token_type_ids'],
            'label': label
        }

    def pad_to_maxlen(self, input_ids, max_len):
        if len(input_ids) >= max_len:
            input_ids = input_ids[:max_len]
        else:
            input_ids = input_ids + [self.pad_token_id] * (max_len - len(input_ids))
        return input_ids

    def collate_fn(self, batch):
        max_len = max([len(item['input_ids']) for item in batch])

        batch_input_ids = []
        batch_attention_masks = []
        batch_token_type_ids = []
        batch_labels = []
        for item in batch:
            batch_input_ids.append(self.pad_to_maxlen(item['input_ids'], max_len))
            batch_attention_masks.append(self.pad_to_maxlen(item['attention_mask'], max_len))
            batch_token_type_ids.append(self.pad_to_maxlen(item['token_type_ids'], max_len))
            batch_labels.append(item['label'])

        input_ids = torch.LongTensor(batch_input_ids)
        attention_mask = torch.LongTensor(batch_attention_masks)
        token_type_ids = torch.LongTensor(batch_token_type_ids)
        labels = torch.LongTensor(batch_labels)
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids,
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
    bert_config = BertConfig.from_pretrained(args['config_path'] if args['config_path'] else model_name_or_path,
                                             num_labels=args['num_labels'])
    if flag == 'train':
        bert_config.save_pretrained(args['model_save_path'])
        model = CrossEncoder(config=bert_config, model_name_or_path=model_name_or_path, num_labels=args['num_labels'])
    else:
        model = CrossEncoder(config=bert_config, num_labels=args['num_labels'])
        model.load_state_dict(torch.load(os.path.join(model_name_or_path, "pytorch_model.bin"),
                                         map_location=args['device']))
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
    param_optimizer = list(model.named_parameters())
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': args["weight_decay"], 'lr': args['lr_rate']},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
         'lr': args['lr_rate']},
    ]
    return optimizer_grouped_parameters


def evaluate(args, valid_dataloader, model):
    model.eval()
    loss_func = nn.CrossEntropyLoss()

    all_labels = []
    all_preds = []
    total_loss = 0.0
    for step, batch in enumerate(tqdm(valid_dataloader, desc="Evaluating")):
        inputs = {
            'input_ids': batch['input_ids'].to(args['device']),
            'attention_mask': batch['attention_mask'].to(args['device']),
            'token_type_ids': batch['token_type_ids'].to(args['device'])
        }
        labels = batch['labels'].detach().cpu().numpy()
        all_labels.extend(labels)

        with torch.no_grad():
            logits = model(**inputs)
            loss = loss_func(logits, batch['labels'].to(args['device']))
            total_loss += loss.item()

        pred = np.argmax(logits.detach().cpu().numpy(), axis=-1)
        all_preds.extend(pred)
    results = precision_recall_fscore_support(all_labels, all_preds, average='macro')
    return {
        'precision': float('%.5f' % results[0]),
        'recall': float('%.5f' % results[1]),
        'f1': float("%.5f" % results[2]),
        'loss': float("%.10f" % (total_loss/len(valid_dataloader)))
    }


def train(args, train_samples, valid_samples, model, tokenizer):
    train_dataset = CustomDataset(train_samples, tokenizer, args['max_seq_length'])
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args['train_batch_size'],
                                  num_workers=args['num_worker'], collate_fn=train_dataset.collate_fn)

    valid_dataset = CustomDataset(valid_samples, tokenizer, args['max_seq_length'])
    valid_dataloader = DataLoader(valid_dataset, shuffle=False, batch_size=args['valid_batch_size'],
                                  num_workers=args['num_worker'], collate_fn=valid_dataset.collate_fn)

    t_total = len(train_samples) * args['num_train_epochs']
    num_train_optimization_steps = int(
        len(train_dataset) / args['train_batch_size'] / args['gradient_accumulation_steps']) * args['num_train_epochs']

    # Prepare optimizer and schedule (linear warmup and decay)
    optimizer_grouped_parameters = get_parameters(model, args)
    warmup_steps = int(t_total * args['warmup_ratio'])
    args['warmup_steps'] = warmup_steps

    args['warmup_steps'] = warmup_steps
    optimizer, scheduler = init_optimizer(t_total, optimizer_grouped_parameters, args)

    train_config = {
        'lr_rate': args['lr_rate'],
        'gradient_accumulation_steps': args['gradient_accumulation_steps'],
        'warmup_ratio': args['warmup_ratio'],
        'adam_epsilon': args['adam_epsilon'],
        'weight_decay': args['weight_decay'],
        'scheduler_type': args['scheduler_type']
    }
    wandb_logger, run = init_wandb_writer(project_name=args['project_name'],
                                          train_args=train_config,
                                          group_name=args['group_name'],
                                          experiment_name=args['experiment_name'])
    wandb_logger.watch(model, log='all')

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num train examples = %d", len(train_samples))
    logger.info("  Num Epochs = %d", args['num_train_epochs'])
    logger.info("  Instantaneous batch size per GPU = %d", args['train_batch_size'])
    logger.info("  Gradient Accumulation steps = %d", args['gradient_accumulation_steps'])
    logger.info("  Total optimization steps = %d", num_train_optimization_steps)

    model.to(args['device'])
    model.zero_grad()
    loss_func = nn.CrossEntropyLoss()
    global_steps = 0
    best_score = 0.0
    best_epoch = 0
    patience = 0
    set_seed(args['seed'])
    for epoch in range(args['num_train_epochs']):
        model.train()
        total_loss = 0.0
        epoch_steps = 0
        for step, batch in enumerate(tqdm(train_dataloader,
                                          desc=f"Training cross_encoder, {epoch}/{args['num_train_epochs']}")):
            inputs = {
                'input_ids': batch['input_ids'].to(args['device']),
                'attention_mask': batch['attention_mask'].to(args['device']),
                'token_type_ids': batch['token_type_ids'].to(args['device'])
            }
            label_ids = batch['labels'].to(args['device'])

            logits = model(**inputs)
            loss = loss_func(logits, label_ids)

            loss = loss / args['gradient_accumulation_steps']
            loss.backward()

            total_loss += loss.item()
            epoch_steps += 1

            if (step + 1) % args['gradient_accumulation_steps'] == 0:
                optimizer.step()
                scheduler.step()
                model.zero_grad()
                global_steps += 1

                wandb_logger.log({'Train/loss': loss.item(), 'Train/total_loss': total_loss/epoch_steps},
                                 step=global_steps)
        results = evaluate(args, valid_dataloader, model)
        wandb_logger.log({'Eval': results}, step=global_steps)
        if results['f1'] > best_score:
            best_score = results['f1']
            best_epoch = epoch
            patience = 0

            model_to_save = model.module if hasattr(model, 'module') else model
            torch.save(model_to_save.state_dict(), os.path.join(args['model_save_path'], 'pytorch_model.bin'))
            tokenizer.save_pretrained(args['model_save_path'])

            log_dict = {'best_score': best_epoch, 'best_epoch': best_epoch, 'eval_score': results}
            logger.info(json.dumps(log_dict, ensure_ascii=False, indent=2))

            with codecs.open(os.path.join(args['model_save_path'], 'eval_result.json'), "w", encoding="utf8") as fw:
                json.dump(log_dict, fw, ensure_ascii=False, indent=4)
        else:
            patience += 1
            if patience >= 10:
                break
    return best_score, best_epoch


def main():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    root_path = "/root/work2/work2/chenzhihao/NLP"
    config = {
        'model_type': "roberta-wwm-ext",
        'model_name_or_path': "/root/work2/work2/chenzhihao/pretrained_models/chinese-roberta-wwm-ext",
        'output_dir': root_path + "/experiments/output_file_dir/semantic_match",
        'config_path': None,
        'tokenizer_path': None,
        'data_dir': "/root/work2/work2/chenzhihao/datasets/chinese-semantics-match-dataset/",
        'data_type': "BQ",  # ATEC, BQ, LCQMC, PAWSX
        'train_dataset': "train.data",
        'valid_dataset': "valid.data",
        'test_dataset': "test.data",
        'project_name': 'semantic_match',
        'group_name': 'nlp',
        'experiment_name': 'BQ-cross_encoder-roberta-wwm-ext',
        'do_train': True,
        'do_test': True,
        'lr_rate': 5e-5,
        'gradient_accumulation_steps': 1,
        'warmup_ratio': 0.1,
        'adam_epsilon': 1e-8,
        'weight_decay': 0.01,
        'scheduler_type': 'linear',
        'num_labels': 2,
        'train_batch_size': 64,  # 必须是2的倍数
        'valid_batch_size': 64,
        'test_batch_size': 64,
        'num_train_epochs': 200,
        'max_seq_length': 512,
        'cuda_number': "4",
        'num_worker': 4,
        'seed': 2333
    }

    data_dir = config['data_dir'] + config['data_type']
    if not os.path.exists(data_dir):
        raise ValueError(f"The path of '{data_dir}' not exist")
    model_save_path = f"{config['output_dir']}/{config['data_type']}-cross_encoder2-{config['model_type']}"
    config['model_save_path'] = model_save_path
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)

    device = torch.device(f"cuda:{config['cuda_number']}") if torch.cuda.is_available() else torch.device('cpu')
    config['device'] = device

    logger.info("****** Loading datasets ******")
    train_samples = load_data(os.path.join(data_dir, config['train_dataset']), 'train')
    valid_samples = load_data(os.path.join(data_dir, config['valid_dataset']), 'valid')

    logger.info("****** init model ******")
    tokenizer = BertTokenizer.from_pretrained(config['model_name_or_path']
                                              if not config['tokenizer_path'] else config['tokenizer_path'])
    model = init_model(config['model_name_or_path'], config)
    model.to(device)

    if config['do_train']:
        best_score, best_epoch = train(config, train_samples, valid_samples, model, tokenizer)
        print("best score: {}\tbest epoch: {}".format(best_score, best_epoch))

    if config['do_test']:
        test_samples = load_data(os.path.join(data_dir, config['test_dataset']), 'test')
        test_dataset = CustomDataset(test_samples, tokenizer, config['max_seq_length'])
        test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=config['test_batch_size'],
                                     num_workers=config['num_worker'], collate_fn=test_dataset.collate_fn)
        model = init_model(model_save_path, config, 'test')
        model.to(device)

        results = evaluate(config, test_dataloader, model)
        print("test result: ", json.dumps(results, ensure_ascii=False))
        with codecs.open(os.path.join(model_save_path, "test_result.json"), "w", encoding="utf8") as fw:
            json.dump(results, fw, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    main()

