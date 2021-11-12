# -*- coding: utf8 -*-
"""
======================================
    Project Name: EventExtraction
    File Name: eval_model
    Author: czh
    Create Date: 2021/9/16
--------------------------------------
    Change Activity: 
======================================
"""
from nlp.event_extractor.event_extractor import EventExtractor
from nlp.utils.ee_arguments import DataAndTrainArguments

config = {
    'task_name': 'ee',  # ee
    'data_dir': '../data/normal_data/news2',
    'model_type': 'bert',  # bert, nezha
    'model_name_or_path': 'hfl/chinese-roberta-wwm-ext',  # nezha-base-wwm
    'output_dir': '../data/output/',  # 模型训练中保存的中间结果，模型，日志等文件的主目录
    'do_lower_case': False,  # 主要是tokenize时是否将大写转为小写
    'use_lstm': False,  # 默认为False, 表示模型结构为bert_crf
    'no_cuda': False,  # 是否使用GPU。默认为False, 表示只使用CPU
    'eval_max_seq_length': 128,  # 默认为512
    'per_gpu_eval_batch_size': 8,
    'cuda_number': '0',   # '0,1,2,3' 使用GPU时需指定GPU卡号
}

args = DataAndTrainArguments(**config) # noqa
extractor = EventExtractor(args)

# evaluate all checkpoints file for the dev datasets
# extractor.evaluate(eval_all_checkpoints=True)

# only evaluate best model for the dev datasets
# extractor.evaluate()

# evaluate all checkpoints file for the test datasets, and the test datasets sample must labeled
# extractor.evaluate(data_type='test', eval_all_checkpoints=True)

# only evaluate best model for the test datasets, and the test datasets sample must labeled
extractor.evaluate(data_type='test')
