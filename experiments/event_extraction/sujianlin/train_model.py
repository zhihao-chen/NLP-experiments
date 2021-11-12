# -*- coding: utf8 -*-
"""
======================================
    Project Name: EventExtraction
    File Name: train_model
    Author: czh
    Create Date: 2021/9/15
--------------------------------------
    Change Activity: 
======================================
"""
from nlp.event_extractor.event_extractor import EventExtractor
from nlp.utils.ee_arguments import DataAndTrainArguments

config = {
    'task_name': 'ner',  # ner
    'data_dir': '../data/normal_data/ner',
    'model_type': 'bert',  # bert, nezha
    'model_name_or_path': 'hfl/chinese-roberta-wwm-ext',  # '/data/chenzhihao/nezha-base-www'
    'model_sate_dict_path': '../data/output/bert/best_model',   # 保存的checkpoint文件地址用于继续训练
    'output_dir': '../data/output/',  # 模型训练中保存的中间结果，模型，日志等文件的主目录False
    'do_lower_case': False,  # 主要是tokenize时是否将大写转为小写
    'cache_dir': '',   # 指定下载的预训练模型保存地址
    'evaluate_during_training': True,  # 是否在训练过程中验证模型, 默认为True
    'use_lstm': False,  # 默认为False, 表示模型结构为bert_crf
    'from_scratch': True,  # 是否从头开始训练，默认为True
    'from_last_checkpoint': False,  # 是否从最新的checkpoint模型继续训练，默认为False
    'early_stop': False,
    'overwrite_output_dir': True,
    'overwrite_cache': True,  # 是否重写特征，默认为True，若为False表示从特征文件中加载特征
    'no_cuda': False,  # 是否使用GPU。默认为False, 表示只使用CPU
    'fp16': True,
    'train_max_seq_length': 32,  # 默认为512
    'eval_max_seq_length': 32,  # 默认为512
    'per_gpu_train_batch_size': 16,
    'per_gpu_eval_batch_size': 16,
    'gradient_accumulation_steps': 1,
    'learning_rate': 5e-05,  # bert和lstm的学习率
    'crf_learning_rate': 5e-05,
    'weight_decay': 0.01,
    'adam_epsilon': 1e-08,
    'warmup_proportion': 0.1,
    'num_train_epochs': 3.0,
    'max_steps': -1,  # 当指定了该字段值后，'num_train_epochs'就不起作用了
    'tolerance': 5,   # 指定early stop容忍的epoch数量
    'logging_steps': 500,  # 指定tensorboard日志在哪个阶段记录
    'save_steps': 500,  # 指定哪些步骤保存中间训练结果
    # ["linear","cosine","cosine_with_restarts","polynomial","constant","constant_with_warmup"]
    'scheduler_type': 'linear',
    'cuda_number': '3',   # '0,1,2,3' 使用GPU时需指定GPU卡号
    'seed': 2333,
    'dropout_rate': 0.3
}

args = DataAndTrainArguments(**config)  # noqa
extractor = EventExtractor(args)

# training from scratch, set config['from_scratch'] = True
extractor.train_and_valid()

# continue train from 'model_sate_dict_path', set config['from_scratch'] = False
# extractor.train_and_valid()

# continue train from last checkpoint file, set config['from_scratch'] = False, config['from_last_checkpoint']=True.
# And should rise the 'num_train_epochs'
# extractor.train_and_valid()
