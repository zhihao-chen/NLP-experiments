#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: czh
@email: zhihao.chen@kuwo.cn
@date: 2022/9/21 19:22
"""
# 根据metric来选择不同的子训练集
# 参考自：https://github.com/beyondguo/TrainingDynamics/blob/master/data_selection.py
# Only applied to training set
# python data_selection.py --task_name qnli --model_name bert-base-cased --proportion 0.5 --burn_out 4
import json
import random

random.seed(1)
import argparse

from train_dynamics_filtering import read_training_dynamics, compute_train_dy_metrics


class Config:
    task_name = 'BQ'
    model_name = "roberta-wwm"
    proportion = 0.33
    burn_out = 5


parser = argparse.ArgumentParser()
parser.add_argument("--task_name", type=str)
parser.add_argument("--model_name", type=str)
parser.add_argument("--proportion", type=float, default=0.33)
parser.add_argument("--burn_out", type=int)
# args = parser.parse_args()
args = Config()

TASK_NAME = args.task_name
MODEL = args.model_name
PROPORTION = args.proportion
LOG_PATH = '/root/work2/work2/chenzhihao/NLP/output_file_dir/semantic_match'

# 读取并合并到一个文件
td = read_training_dynamics(LOG_PATH + f'dy_logs/{TASK_NAME}/{MODEL}/')
# 计算 metrics，转化成一个 dataframe
td_df, _ = compute_train_dy_metrics(td, burn_out=args.burn_out)


def consider_ascending_order(filtering_metric: str) -> bool:
    """
    Determine if the metric values' sorting order to get the most `valuable` examples for training.
    """
    if filtering_metric == "variability":
        return False
    elif filtering_metric == "confidence":
        return True
    elif filtering_metric == "threshold_closeness":
        return False
    elif filtering_metric == "forgetfulness":
        return False
    elif filtering_metric == "correctness":
        return True
    else:
        raise NotImplementedError(f"Filtering based on {filtering_metric} not implemented!")


def data_selection(metric, select_worst, proportion, shuffle=True):
    ascending = consider_ascending_order(metric)
    if select_worst:
        ascending = not consider_ascending_order(metric)
    sorted_df = td_df.sort_values(by=metric, ascending=ascending)
    selected_df = sorted_df.head(n=int(proportion * len(sorted_df)))
    indices = list(selected_df['guid'])
    if shuffle:
        random.shuffle(indices)
    return {'indices': indices, 'df': selected_df}


"""
选择hard-to-learn的数据，设置METRIC = 'confidence'
选择easy-to-learn的数据，设置METRIC = 'confidence', SELECT_WORST = True
选择ambiguoug的数据，设置METRIC = 'variability'
"""

three_regions_data_indices = {'hard': data_selection('confidence', False, PROPORTION)['indices'],
                              'easy': data_selection('confidence', True, PROPORTION)['indices'],
                              'ambiguous': data_selection('variability', False, PROPORTION)['indices']}

with open(f'dy_log/{TASK_NAME}/{MODEL}/three_regions_data_indices.json', 'w', encoding='utf8') as f:
    f.write(json.dumps(three_regions_data_indices, ensure_ascii=False))

# 然后可以直接跑glue任务，在选择训练集的时候，使用select函数来指定对应样本即可：
""" e.g.
from datasets import load_dataset
raw_datasets = load_dataset('glue','sst2')
easy_train_set = raw_datasets['train'].select(three_regions_data_indices['easy'])
"""
