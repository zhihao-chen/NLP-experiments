#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: czh
@email:
@date: 2022/7/5 10:36
"""
# 实验无监督ConSERT
# 参考https://github.com/zhoujx4/NLP-Series-sentence-embeddings/blob/main/run_unsup_consert.py
import os
import sys
import math
import logging
from datetime import datetime

import torch
from sentence_transformers import InputExample, SentenceTransformer, LoggingHandler
from sentence_transformers import losses
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator, SimilarityFunction
from torch.utils.data import DataLoader
sys.path.append('/data2/work2/chenzhihao/NLP')
from nlp.processors.semantic_match_preprocessor import load_data
from nlp.models.sentence_embedding_models import ConSERTSentenceTransformer


logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])


def init_model(model_name, args):

    model = ConSERTSentenceTransformer(model_name, cutoff_rate=args['cutoff_rate'],
                                       close_dropout=args['close_dropout'], device=args['device'])
    model.max_seq_length = args['max_seq_length']
    return model


def prepare_datasets(data_dir, data_type="STS-B", object_type="classification", seg_tag='\t', is_train=False):
    dataset = load_data(data_dir, seg_tag=seg_tag)
    data_samples = []
    for data in dataset:
        if data_type == "STS-B":
            label = data[2] / 5.0
        else:
            label = data[2] if object_type == "classification" else float(data[2])
        if is_train:
            data_samples.append(InputExample(texts=[data[0], data[0]]))
        else:
            data_samples.append(InputExample(texts=[data[0], data[1]], label=label))
    return data_samples


def train(train_samples, model, dev_evaluator, args):
    train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=args['batch_size'])
    loss_func = losses.MultipleNegativesRankingLoss(model)
    # 10% of train data for warm-up
    warmup_steps = math.ceil(len(train_dataloader) * args['num_epochs'] * args['warmup_ratio'])
    evaluation_steps = int(len(train_dataloader) * 0.1)  # Evaluate every 10% of the data
    logging.info("Training sentences: {}".format(len(train_samples)))
    logging.info("Warmup-steps: {}".format(warmup_steps))
    logging.info("Performance before training")
    dev_evaluator(model)

    # 模型训练
    logging.info("***** Training model *****")
    model.fit(train_objectives=[(train_dataloader, loss_func)],
              evaluator=dev_evaluator,
              epochs=args['num_epochs'],
              evaluation_steps=evaluation_steps,
              warmup_steps=warmup_steps,
              show_progress_bar=False,
              output_path=args['model_save_path'],
              optimizer_params={'lr': args['lr_rate']},
              use_amp=args['use_amp']  # Set to True, if your GPU supports FP16 cores
              )


def test_model(model_save_path, test_evaluator):
    logging.info("***** test model *****")
    model = SentenceTransformer(model_save_path)
    test_evaluator(model, output_path=model_save_path)


def main():
    root_path = "/data2/work2/chenzhihao/NLP"
    config = {
        'lr_rate': 2e-5,
        'gradient_accumulation_steps': 1,
        'warmup_ratio': 0.1,
        'use_amp': False,
        'model_type': "roberta",
        'model_name_or_path': "/data2/work2/chenzhihao/NLP/pretrained_models/chinese-roberta-wwm-ext",
        'batch_size': 64,
        'num_epochs': 30,
        'max_seq_length': 64,
        'cutoff_rate': 0.15,
        'close_dropout': True,
        'data_type': "STS-B",  # ATEC, BQ, LCQMC, PAWSX, STS-B
        'object_type': "regression",  # classifier, regression, triplet
        'train_dataset': "train.data",
        'valid_dataset': "valid.data",
        'test_dataset': "test.data",
        'cuda_number': '2'
    }
    data_dir = "/data2/work2/chenzhihao/datasets/chinese-semantics-match-dataset/" + config['data_type']
    if not os.path.exists(data_dir):
        raise ValueError(f"The path of '{data_dir}' not exist")
    output_dir = root_path + "/experiments/output_file_dir/semantic_match"
    date = datetime.now().strftime("%Y-%m-%d_%H")
    model_save_path = output_dir + f"/{config['data_type']}-unsup-consert-{config['model_type']}-{date}"
    config['model_save_path'] = model_save_path
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)

    device = torch.device(f"cuda:{config['cuda_number']}") if torch.cuda.is_available() else torch.device('cpu')
    config['device'] = device

    print("****** Loading datasets ******")
    train_samples = prepare_datasets(os.path.join(data_dir, config['data_type'] + '.' + config['train_dataset']),
                                     data_type=config['data_type'],
                                     object_type=config['object_type'],
                                     is_train=True)
    valid_samples = prepare_datasets(os.path.join(data_dir, config['data_type'] + '.' + config['valid_dataset']),
                                     data_type=config['data_type'],
                                     object_type=config['object_type'])
    test_samples = prepare_datasets(os.path.join(data_dir, config['data_type'] + '.' + config['test_dataset']),
                                    data_type=config['data_type'],
                                    object_type=config['object_type'])

    model = init_model(config['model_name_or_path'], config)

    # 初始化评估器
    valid_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(
        valid_samples, batch_size=config['batch_size'], name=f"{config['data_type']}-valid",
        main_similarity=SimilarityFunction.COSINE
    )
    test_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(
        test_samples, batch_size=config['batch_size'], name=f"{config['data_type']}-test",
        main_similarity=SimilarityFunction.COSINE
    )

    # 训练模型
    train(train_samples, model, valid_evaluator, config)

    # 测试模型
    test_model(model_save_path, test_evaluator)


if __name__ == "__main__":
    main()
