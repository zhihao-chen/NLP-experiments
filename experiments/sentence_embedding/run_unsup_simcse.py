#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: czh
@email:
@date: 2022/7/1 16:08
"""
# 无监督simcse实验
import os
import sys
import math
import logging
from datetime import datetime

import torch
import numpy as np
from sentence_transformers import InputExample, SentenceTransformer, LoggingHandler, models, datasets
from sentence_transformers import losses
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator, SimilarityFunction
sys.path.append('/data2/work2/chenzhihao/NLP')
from nlp.processors.semantic_match_preprocessor import load_data  # noqa

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])


def init_model(model_name, device, max_seq_length):
    # Use Huggingface/transformers model (like BERT, RoBERTa, XLNet, XLM-R) for mapping tokens to embeddings
    word_embedding_model = models.Transformer(model_name)

    # Apply mean pooling to get one fixed sized sentence vector
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                                   pooling_mode='cls',
                                   pooling_mode_mean_tokens=False,
                                   pooling_mode_cls_token=True,
                                   pooling_mode_max_tokens=False)

    model = SentenceTransformer(modules=[word_embedding_model, pooling_model], device=device)

    model.max_seq_length = max_seq_length
    return model


def prepare_datasets(data_dir, data_type="STS-B", object_type="classification", seg_tag='\t', is_train=False):
    dataset = load_data(data_dir, seg_tag=seg_tag)  # noqa
    data_samples = []
    for data in dataset:
        if data_type == "STS-B":
            label = data[2] / 5.0
        else:
            label = data[2] if object_type == "classification" else float(data[2])
        if is_train:
            data_samples.append(InputExample(texts=[data[0], data[0]]))
            data_samples.append(InputExample(texts=[data[1], data[1]]))
        else:
            data_samples.append(InputExample(texts=[data[0], data[1]], label=label))
    return data_samples


def train(train_samples, model, dev_evaluator, args):  # noqa
    np.random.shuffle(train_samples)
    # train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=args['batch_size'])
    train_dataloader = datasets.NoDuplicatesDataLoader(train_samples, batch_size=args['batch_size'])

    if args['object_type'] == "regression":
        loss_func = losses.CosineSimilarityLoss(model)
    elif args['object_type'] == "classification":
        loss_func = losses.SoftmaxLoss(model=model,
                                       sentence_embedding_dimension=model.get_sentence_embedding_dimension(),
                                       num_labels=2)
    elif args['object_type'] == "multi_neg_rank":
        loss_func = losses.MultipleNegativesRankingLoss(model)
    elif args['object_type'] == "triplet":
        loss_func = losses.TripletLoss(model, distance_metric=losses.TripletDistanceMetric.COSINE,
                                       triplet_margin=args['triplet_margin'])
    elif args['object_type'] == "contrastive":
        loss_func = losses.ContrastiveLoss(model)
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
    config = {
        'model_type': "roberta",
        'model_name': "/data2/work2/chenzhihao/NLP/pretrained_models/chinese-roberta-wwm-ext",
        'lr_rate': 2e-5,
        'gradient_accumulation_steps': 1,
        'warmup_ratio': 0.1,
        'use_amp': False,
        'batch_size': 64,
        'num_epochs': 10,
        'max_seq_length': 128,
        'triplet_margin': 0.5,
        'data_type': "STS-B",  # ATEC, BQ, LCQMC, PAWSX, STS-B, SNLI, MNLI
        'object_type': "multi_neg_rank",  # classifier, regression, triplet, multi_neg_rank, contrastive
        'train_dataset': "train.data",
        'valid_dataset': "valid.data",
        'test_dataset': "test.data",
        'cuda_number': 2
    }

    root_path = "/data2/work2/chenzhihao/NLP"
    data_dir = "/data2/work2/chenzhihao/datasets/chinese-semantics-match-dataset/" + config['data_type']
    if not os.path.exists(data_dir):
        raise ValueError(f"The path of '{data_dir}' not exist")
    output_dir = root_path + "/experiments/output_file_dir/semantic_match"
    date = datetime.now().strftime("%Y-%m-%d_%H")
    model_save_path = output_dir + f"/{config['data_type']}-unsup_simcse-{config['model_type']}-{date}"
    config['model_save_path'] = model_save_path
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)

    device = torch.device(f"cuda:{config['cuda_number']}") if torch.cuda.is_available() else torch.device('cpu')
    # 初始化模型
    model = init_model(config['model_name'], device, config['max_seq_length'])

    # 准备数据
    train_samples = prepare_datasets(os.path.join(data_dir,
                                                  config['data_type']+'.'+config['train_dataset']),
                                     data_type=config['data_type'],
                                     object_type=config['object_type'], is_train=True)
    valid_samples = prepare_datasets(os.path.join(data_dir,
                                                  config['data_type']+'.'+config['valid_dataset']),
                                     data_type=config['data_type'],
                                     object_type=config['object_type'])
    test_samples = prepare_datasets(os.path.join(data_dir,
                                                 config['data_type']+'.'+config['test_dataset']),
                                    data_type=config['data_type'],
                                    object_type=config['object_type'])

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

