#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: czh
@email: zhihao.chen@kuwo.cn
@date: 2022/7/5 16:19
"""
# 无监督对比学习ConSERT
# 参考https://github.com/yym6472/ConSERT/blob/master/main.py

import os
import sys
import math
import logging

import torch
import numpy as np
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

sys.path.append('/data2/work2/chenzhihao/NLP')

from nlp.sentence_transformers import models, losses
from nlp.sentence_transformers import SentenceTransformer, LoggingHandler, SentencesDataset, InputExample
from nlp.sentence_transformers.evaluation import EmbeddingSimilarityEvaluator, SimilarityFunction
from nlp.processors.semantic_match_preprocessor import load_data, load_data_for_snli

logging.basicConfig(format='%(asctime)s - %(filename)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])


def init_model(model_name, args):
    # Use Huggingface/transformers model (like BERT, RoBERTa, XLNet, XLM-R) for mapping tokens to embeddings
    if args['no_dropout']:
        word_embedding_model = models.Transformer(model_name, attention_probs_dropout_prob=0.0, hidden_dropout_prob=0.0)
    else:
        word_embedding_model = models.Transformer(model_name)

    # Apply mean pooling to get one fixed sized sentence vector
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                                   pooling_mode_mean_tokens=True,
                                   pooling_mode_cls_token=False,
                                   pooling_mode_max_tokens=False)
    if args['use_simsiam']:
        projection_model = models.MLP3(hidden_dim=args['projection_hidden_dim'], norm=args['projection_norm_type'])
        model = SentenceTransformer(modules=[word_embedding_model, pooling_model, projection_model],
                                    device=args['device'])
    else:
        model = SentenceTransformer(modules=[word_embedding_model, pooling_model], device=args['device'])
    model.max_seq_length = args['max_seq_length']
    # Tensorboard writer
    tensorboard_writer = SummaryWriter(args['tensorboard_log_dir'] or os.path.join(args['model_save_path'], "logs"))
    model.tensorboard_writer = tensorboard_writer
    return model


def prepare_datasets(data_dir, args, need_label=True, seg_tag='\t'):
    dataset = load_data(data_dir, seg_tag=seg_tag)  # noqa
    data_samples = []
    for data in dataset:
        if args['data_type'] == "STS-B":
            label = data[2] / 5.0
        else:
            label = data[2] if args['object_type'] == "classification" else float(data[2])
        if need_label:
            data_samples.append(InputExample(texts=[data[0], data[1]], label=label))
        else:
            if args['no_pair']:
                data_samples.append(InputExample(texts=[data[0]]))
                data_samples.append(InputExample(texts=[data[1]]))
            else:
                data_samples.append(InputExample(texts=[data[0], data[1]]))
    return data_samples


def prepare_snli_datasets(data_dir, args, label2int, prefix='cnsd_snli_v1.0'):
    train_data_path = os.path.join(data_dir, prefix+".train.json")
    dev_data_path = os.path.join(data_dir, prefix+".dev.json")
    test_data_path = os.path.join(data_dir, prefix+".test.json")

    train_data_lst = load_data_for_snli(train_data_path, return_list=True)
    dev_data_lst = load_data_for_snli(dev_data_path, return_list=True)
    test_data_lst = load_data_for_snli(test_data_path, return_list=True)

    all_datas = train_data_lst + dev_data_lst + test_data_lst

    train_samples = []
    for item in all_datas:
        s1 = item['sentence1'].strip()
        s2 = item['sentence2'].strip()
        label = item['gold_label'].strip()
        label_id = label2int[label]

        if args['no_pair']:
            assert args['cl_loss_only'], "no pair texts only used when contrastive loss only"
            train_samples.append(InputExample(texts=[s1]))
            train_samples.append(InputExample(texts=[s2]))
        else:
            train_samples.append(InputExample(texts=[s1, s2], label=label_id))
    np.random.shuffle(train_samples)
    return train_samples


def train(train_samples, model, dev_evaluator, args: dict):  # noqa
    train_dataset = SentencesDataset(train_samples, model=model)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args['train_batch_size'])

    if args['adv_training'] and args['add_cl']:
        train_loss = losses.AdvCLSoftmaxLoss(model=model,
                                             sentence_embedding_dimension=model.get_sentence_embedding_dimension(),
                                             num_labels=args['num_labels'],
                                             concatenation_sent_max_square=args['concatenation_sent_max_square'],
                                             use_adversarial_training=args['adv_training'],
                                             noise_norm=args['noise_norm'],
                                             adv_loss_stop_grad=args['adv_loss_stop_grad'],
                                             adversarial_loss_rate=args['adv_loss_rate'],
                                             use_contrastive_loss=args['add_cl'],
                                             contrastive_loss_type=args['cl_type'],
                                             contrastive_loss_rate=args['cl_rate'],
                                             temperature=args['temperature'],
                                             contrastive_loss_stop_grad=args['contrastive_loss_stop_grad'],
                                             mapping_to_small_space=args['mapping_to_small_space'],
                                             add_contrastive_predictor=args['add_contrastive_predictor'],
                                             projection_hidden_dim=args['projection_hidden_dim'],
                                             projection_use_batch_norm=args['projection_use_batch_norm'],
                                             add_projection=args['add_projection'],
                                             projection_norm_type=args['projection_norm_type'],
                                             contrastive_loss_only=args['cl_loss_only'],
                                             data_augmentation_strategy=args['data_augmentation_strategy'],
                                             cutoff_direction=args['cutoff_direction'],
                                             cutoff_rate=args['cutoff_rate'],
                                             regularization_term_rate=args['regularization_term_rate'],
                                             loss_rate_scheduler=args['loss_rate_scheduler'])
    elif args['adv_training']:
        train_loss = losses.AdvCLSoftmaxLoss(model=model,
                                             sentence_embedding_dimension=model.get_sentence_embedding_dimension(),
                                             num_labels=args['num_labels'],
                                             concatenation_sent_max_square=args['concatenation_sent_max_square'],
                                             use_adversarial_training=args['adv_training'],
                                             noise_norm=args['noise_norm'],
                                             adv_loss_stop_grad=args['adv_loss_stop_grad'],
                                             adversarial_loss_rate=args['adv_loss_rate'])
    elif args['add_cl']:
        train_loss = losses.AdvCLSoftmaxLoss(model=model,
                                             sentence_embedding_dimension=model.get_sentence_embedding_dimension(),
                                             num_labels=args['num_labels'],
                                             concatenation_sent_max_square=args['concatenation_sent_max_square'],
                                             use_contrastive_loss=args['add_cl'],
                                             contrastive_loss_type=args['cl_type'],
                                             contrastive_loss_rate=args['cl_rate'],
                                             temperature=args['temperature'],
                                             contrastive_loss_stop_grad=args['contrastive_loss_stop_grad'],
                                             mapping_to_small_space=args['mapping_to_small_space'],
                                             add_contrastive_predictor=args['add_contrastive_predictor'],
                                             projection_hidden_dim=args['projection_hidden_dim'],
                                             projection_use_batch_norm=args['projection_use_batch_norm'],
                                             add_projection=args['add_projection'],
                                             projection_norm_type=args['projection_norm_type'],
                                             contrastive_loss_only=args['cl_loss_only'],
                                             data_augmentation_strategy=args['data_augmentation_strategy'],
                                             cutoff_direction=args['cutoff_direction'],
                                             cutoff_rate=args['cutoff_rate'],
                                             no_pair=args['no_pair'],
                                             regularization_term_rate=args['regularization_term_rate'],
                                             loss_rate_scheduler=args['loss_rate_scheduler'],
                                             data_augmentation_strategy_final_1=args['da_final_1'],
                                             data_augmentation_strategy_final_2=args['da_final_2'],
                                             cutoff_rate_final_1=args['cutoff_rate_final_1'],
                                             cutoff_rate_final_2=args['cutoff_rate_final_2'])
    elif args['use_simclr']:
        train_loss = losses.SimCLRLoss(model=model,
                                       sentence_embedding_dimension=model.get_sentence_embedding_dimension(),
                                       num_labels=args['num_labels'],
                                       concatenation_sent_max_square=args['concatenation_sent_max_square'],
                                       data_augmentation_strategy=args['data_augmentation_strategy'],
                                       temperature=args['temperature'])
    elif args['use_simsiam']:
        train_loss = losses.SimSiamLoss(model=model,
                                        sentence_embedding_dimension=model.get_sentence_embedding_dimension(),
                                        num_labels=args['num_labels'],
                                        concatenation_sent_max_square=args['concatenation_sent_max_square'],
                                        data_augmentation_strategy=args['data_augmentation_strategy'],
                                        temperature=args['temperature'])
    else:
        train_loss = losses.AdvCLSoftmaxLoss(model=model,
                                             sentence_embedding_dimension=model.get_sentence_embedding_dimension(),
                                             num_labels=args['num_labels'],
                                             concatenation_sent_max_square=args['concatenation_sent_max_square'],
                                             normal_loss_stop_grad=args['normal_loss_stop_grad'])
    # 10% of train data for warm-up
    model.num_steps_total = math.ceil(len(train_dataset) * args['num_train_epochs'] / args['train_batch_size'])
    warmup_steps = math.ceil(len(train_dataloader) * args['num_train_epochs'] * args['warmup_ratio'])
    evaluation_steps = int(len(train_dataloader) * 0.1)  # Evaluate every 10% of the data
    logging.info("Training sentences: {}".format(len(train_samples)))
    logging.info("Warmup-steps: {}".format(warmup_steps))
    logging.info("Performance before training")
    # dev_evaluator(model)

    # 模型训练
    logging.info("***** Training model *****")
    model.fit(train_objectives=[(train_dataloader, train_loss)],
              evaluator=dev_evaluator,
              epochs=args['num_train_epochs'],
              evaluation_steps=args['evaluation_steps'],
              warmup_steps=warmup_steps,
              output_path=args['model_save_path'],
              optimizer_params={'lr': args['lr_rate'], 'eps': 1e-6, 'correct_bias': False},
              use_apex_amp=args['use_apex_amp'],  # Set to True, if your GPU supports FP16 cores
              apex_amp_opt_level=args['apex_amp_opt_level'],
              early_stop_patience=args['patience']
              )


def test_model(model_save_path, test_evaluator):
    logging.info("***** test model *****")
    model = SentenceTransformer(model_save_path)
    test_evaluator(model, output_path=model_save_path)


def main():
    root_path = "/data2/work2/chenzhihao/NLP"

    config = {
        'model_type': "roberta",
        'model_name_or_path': "/data2/work2/chenzhihao/NLP/pretrained_models/chinese-roberta-wwm-ext",
        'output_dir': root_path + "/experiments/output_file_dir/semantic_match",
        'config_path': "/data2/work2/chenzhihao/NLP/pretrained_models/chinese-roberta-wwm-ext",
        'tokenizer_path': "/data2/work2/chenzhihao/NLP/pretrained_models/chinese-roberta-wwm-ext",
        'tensorboard_log_dir': None,
        'do_train': True,
        'do_test': True,
        'use_apex_amp': False,
        'apex_amp_opt_level': "01",
        'lr_rate': 5e-5,
        'gradient_accumulation_steps': 1,
        'warmup_ratio': 0.1,
        'adam_epsilon': 1e-8,
        'weight_decay': 0.01,
        'scheduler_type': 'linear',
        'train_batch_size': 64,
        'valid_batch_size': 64,
        'test_batch_size': 64,
        'num_train_epochs': 100,
        'max_seq_length': 64,
        'evaluation_steps': 500,
        'train_data_type': "SNLI",  # # ATEC, BQ, LCQMC, PAWSX, STS-B, SNLI
        'data_type': "STS-B",  # ATEC, BQ, LCQMC, PAWSX, STS-B, SNLI
        'train_dataset': "train.data",
        'valid_dataset': "valid.data",
        'test_dataset': "test.data",
        'cuda_number': "1",
        'num_worker': 4,
        'seed': 2333,
        'no_pair': True,  # If provided, do not pair two training texts
        'no_dropout': False,  # Add no dropout when training
        'use_simsiam': False,  # Use simsiam training or not
        'use_simclr': False,  # Use simclr training or not
        'add_contrastive_predictor': None,  # Whether to use a predictor on one side (similar to SimSiam) and give the projection added to which side (normal or adv)
        'add_projection': False,  # Add projection layer before predictor, only be considered when add_contrastive_predictor is not None
        'projection_hidden_dim': 768,  # The hidden dimension of the projection or predictor MLP
        'projection_norm_type': None,  # The norm type used in the projection layer beforn predictor
        'projection_use_batch_norm': False,  # Whether to use batch normalization in the hidden layer of MLP
        'adv_training': False,  # Use adversarial training or not
        'adv_loss_rate': 1.0,  # The adversarial loss rate
        'noise_norm': 1.0,  # The perturbation norm
        'adv_loss_stop_grad': False,  # Use stop gradient to adversarial loss or not
        'loss_rate_scheduler': 0,  # The loss rate scheduler, default strategy 0 (i.e. do nothing, see AdvCLSoftmaxLoss for more details)
        'add_cl': True,  # Use contrastive loss or not
        'data_augmentation_strategy': 'adv',  # The data augmentation strategy in contrastive learning.[adv,none,meanmax,shuffle,cutoff,shuffle-cutoff,shuffle+cutoff,shuffle_embeddings]
        'cutoff_direction': "random",  # The direction of cutoff strategy, row, column or random
        'cutoff_rate': 0.0,  # The rate of cutoff strategy, in (0.0, 1.0)
        'cl_loss_only': True,  # Ignore the main task loss (e.g. the CrossEntropy loss) and use the contrastive loss only
        'cl_rate': 0.15,  # The contrastive loss rate
        'temperature': 0.1,
        'regularization_term_rate': 0.0,  # The loss rate of regularization term for contrastive learning
        'cl_type': "nt_xent",  # The contrastive loss type, nt_xent or cosine
        'mapping_to_small_space': None,  # Whether to mapping sentence representations to a low dimension space (similar to SimCLR) and give the dimension
        'da_final_1': "feature_cutoff",  # The final 5 data augmentation strategies for view1 (none, shuffle, token_cutoff, feature_cutoff, dropout, span)
        'da_final_2': "shuffle",  # The final 5 data augmentation strategies for view2 (none, shuffle, token_cutoff, feature_cutoff, dropout, span)
        'cutoff_rate_final_1': 0.2,  # The final cutoff/dropout rate for view1
        'cutoff_rate_final_2': 0,  # The final cutoff/dropout rate for view2
        'patience': 10,  # The patience for early stop
        'concatenation_sent_max_square': False,  # Concat max-square features of two text representations when training classification
        'normal_loss_stop_grad': False,  # Use stop gradient to normal loss or not
        'contrastive_loss_stop_grad': None,  # Use stop gradient to contrastive loss (and which mode to apply) or not
    }

    data_dir = "/data2/work2/chenzhihao/datasets/chinese-semantics-match-dataset/"
    if not os.path.exists(data_dir):
        raise ValueError(f"The path of '{data_dir}' not exist")
    model_save_path = f"{config['output_dir']}/{config['train_data_type']}-consert3-{config['model_type']}"
    config['model_save_path'] = model_save_path
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)

    label2int = {"contradiction": 0, "entailment": 1, "neutral": 2}
    config['num_labels'] = len(label2int)

    device = torch.device(f"cuda:{config['cuda_number']}") if torch.cuda.is_available() else torch.device('cpu')
    config['device'] = device

    # 初始化模型
    model = init_model(config['model_name_or_path'], config)

    # 训练集是SNLI数据集
    if config['train_data_type'] == "snli":
        train_samples = prepare_snli_datasets(os.path.join(data_dir, config['train_data_type']),
                                              args=config, label2int=label2int)
    else:
        train_samples = prepare_datasets(os.path.join(data_dir, config['data_type'],
                                                      config['data_type'] + '.' + config['train_dataset']),
                                         args=config, need_label=False)
    # 验证集和测试集是STS-B数据集
    valid_samples = prepare_datasets(os.path.join(data_dir, config['data_type'],
                                                  config['data_type'] + '.' + config['valid_dataset']),
                                     args=config)
    test_samples = prepare_datasets(os.path.join(data_dir, config['data_type'],
                                                 config['data_type'] + '.' + config['test_dataset']),
                                    args=config)

    # 初始化评估器
    valid_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(
        valid_samples, batch_size=config['valid_batch_size'], name=f"{config['data_type']}-valid",
        main_similarity=SimilarityFunction.COSINE
    )
    test_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(
        test_samples, batch_size=config['test_batch_size'], name=f"{config['data_type']}-test",
        main_similarity=SimilarityFunction.COSINE
    )

    if config['do_train']:
        # 训练模型
        train(train_samples, model, valid_evaluator, config)

    # 测试模型
    if config['do_test']:
        test_model(model_save_path, test_evaluator)


if __name__ == "__main__":
    main()
