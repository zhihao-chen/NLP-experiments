#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: czh
@email: zhihao.chen@kuwo.cn
@date: 2022/11/30 15:29
"""
import os
import codecs
import math
import logging

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sentence_transformers import LoggingHandler, util
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CESoftmaxAccuracyEvaluator, CEBinaryClassificationEvaluator
from sentence_transformers.readers import InputExample


logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
logger = logging.getLogger(__file__)


def load_data(input_file, seg_tag):
    all_datas = []
    with codecs.open(input_file, encoding='utf8') as fr:
        for line in fr:
            line = line.strip()
            if not line:
                continue
            sent_lst = line.split(seg_tag)
            assert len(sent_lst) == 3, f"{len(sent_lst)} {line}"
            all_datas.append([sent_lst[0], sent_lst[1], int(sent_lst[2])])
    return all_datas


def prepare_datasets(data_dir, seg_tag='\t'):
    dataset = load_data(data_dir, seg_tag=seg_tag)
    data_samples = []
    for data in dataset:
        label = int(data[2])
        data_samples.append(InputExample(texts=[data[0], data[1]], label=label))
    return data_samples


def train(train_samples, model, dev_evaluator, args):
    train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=args['batch_size'])
    warmup_steps = math.ceil(len(train_dataloader) * args['num_epochs'] * args['warmup_ratio'])
    evaluation_steps = int(len(train_dataloader) * 1)  # Evaluate every 10% of the data
    logging.info("Training sentences: {}".format(len(train_samples)))
    logging.info("Warmup-steps: {}".format(warmup_steps))
    logging.info("Performance before training")
    dev_evaluator(model)

    # 模型训练
    logging.info("***** Training model *****")
    model.fit(train_dataloader=train_dataloader,
              evaluator=dev_evaluator,
              epochs=args['num_epochs'],
              evaluation_steps=evaluation_steps,
              warmup_steps=warmup_steps,
              show_progress_bar=True,
              output_path=args['model_save_path'],
              optimizer_params={'lr': args['lr_rate']},
              use_amp=args['use_amp']  # Set to True, if your GPU supports FP16 cores
              )


def test_model(model_save_path, test_evaluator, config):
    logging.info("***** test model *****")
    model = CrossEncoder(model_name=model_save_path, num_labels=1, device=config['device'],
                         max_length=config['max_seq_length'])
    model.max_seq_length = config['max_seq_length']
    test_evaluator(model, output_path=model_save_path)


def main():

    root_path = "/root/work2/work2/chenzhihao/NLP"
    config = {
        'model_type': "structbert-large-zh",
        # 'model_name': root_path + "pretrained_models/chinese-roberta-wwm-ext",
        'model_name': "/root/work2/work2/chenzhihao/pretrained_models/structbert-large-zh",
        'data_dir': '/root/work2/work2/chenzhihao/datasets/chinese-semantics-match-dataset/',
        'output_dir': root_path + "/experiments/output_file_dir/semantic_match",
        'data_type': 'ATEC',  # ATEC, BQ, LCQMC, PAWSX
        'train_dataset': "train.data",
        'valid_dataset': "valid.data",
        'test_dataset': "test.data",
        'batch_size': 20,
        'num_epochs': 30,
        'max_seq_length': 512,
        'lr_rate': 5e-06,
        'gradient_accumulation_steps': 1,
        'warmup_ratio': 0.1,
        'use_amp': True,
        'cuda_number': 2
    }
    data_dir = config['data_dir'] + config['data_type']
    if not os.path.exists(data_dir):
        raise ValueError(f"The path of '{config['data_dir']}' not exist")
    output_dir = config['output_dir']
    model_save_path = output_dir + f"/{config['data_type']}-cross_encoder-{config['model_type']}/"
    config['model_save_path'] = model_save_path
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)

    device = torch.device(f"cuda:{config['cuda_number']}") if torch.cuda.is_available() else torch.device('cpu')
    config['device'] = device

    logger.info("****** Loading model ******")
    model = CrossEncoder(config['model_name'], num_labels=1, device=config['device'],
                         max_length=config['max_seq_length'])

    logger.info("****** prepare datas ******")
    train_data_dir = config['data_dir'] + config['data_type']
    valid_data_dir = config['data_dir'] + config['data_type']
    test_data_dir = config['data_dir'] + config['data_type']
    # 准备数据
    train_samples = prepare_datasets(os.path.join(train_data_dir, config['data_type']+'.'+config['train_dataset']))
    valid_samples = prepare_datasets(os.path.join(valid_data_dir, config['data_type']+'.'+config['valid_dataset']))
    test_samples = prepare_datasets(os.path.join(test_data_dir, config['data_type']+'.'+config['test_dataset']))

    logger.info("****** init evaluator *******")
    # dev_evaluator = CESoftmaxAccuracyEvaluator.from_input_examples(valid_samples, name=f"{config['data_type']}-valid")
    # test_evaluator = CESoftmaxAccuracyEvaluator.from_input_examples(test_samples, name=f"{config['data_type']}-test")
    dev_evaluator = CEBinaryClassificationEvaluator.from_input_examples(valid_samples, name=f"{config['data_type']}-valid")
    test_evaluator = CEBinaryClassificationEvaluator.from_input_examples(test_samples, name=f"{config['data_type']}-test")

    train(train_samples, model, dev_evaluator, config)

    test_model(model_save_path, test_evaluator, config)


if __name__ == '__main__':
    main()

