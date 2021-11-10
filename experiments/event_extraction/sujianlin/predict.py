# -*- coding: utf8 -*-
"""
======================================
    Project Name: EventExtraction
    File Name: predict_raw_text
    Author: czh
    Create Date: 2021/9/16
--------------------------------------
    Change Activity: 
======================================
"""
import json

from nlp.event_extractor.event_extractor import EventExtractor
from nlp.utils.arguments import DataAndTrainArguments

config = {
    'task_name': 'ee',
    'model_type': 'bert',
    'use_lstm': True,  # 默认是False
    'eval_max_seq_length': 512,
}

args = DataAndTrainArguments(**config)  # noqa
extractor = EventExtractor(args, state='pred', model_path='../data/model')

# data_type: 只能是'test'，或者None。若为test则表示在测试数据集上预测
# input_texts: 若不为空，则表示是预测新的数据
# pred_output_dir: 若不为空，则表示将预测结果写入指定位置保存，可以是目录，也可以是文件

# 表示在测试数据集上预测, 不保存预测结果
# for res in extractor.predict(data_type='test'):
#     print(res)

# 表示在测试数据集上预测, 保存预测结果
# for res in extractor.predict(data_type='test', pred_output_dir="../data/output/bert"):
#     print(res)

# 表示预测raw text, raw text可以是str, List[str]
# texts = "博盛医疗完成Pre-A轮融资澳银资本重点参与"
texts = ["博盛医疗完成Pre-A轮融资澳银资本重点参与",
         "百炼智能完成A轮一亿元融资，由今日头条领投"]
for res in extractor.predict(input_texts=texts):
    print(json.dumps(res, ensure_ascii=False, indent=2))
