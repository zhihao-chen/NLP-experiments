# -*- coding: utf8 -*-
"""
======================================
    Project Name: NLP
    File Name: casel_train
    Author: czh
    Create Date: 2021/8/16
--------------------------------------
    Change Activity: 
======================================
"""
from framework import Framework
import argparse
from nlp.models.bert_for_relation_extraction import Casrel
import os
import torch
import numpy as np
import random

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


class Config(object):
    def __init__(self, args):
        self.args = args

        # train hyper parameter
        self.multi_gpu = args.multi_gpu
        self.learning_rate = args.lr
        self.batch_size = args.batch_size
        self.max_epoch = args.max_epoch
        self.max_len = args.max_len
        self.rel_num = args.rel_num

        # dataset
        self.dataset = args.dataset

        # path and name
        self.root = '/data/chenzhihao/news_relation'
        self.data_path = self.root + '/data/' + self.dataset
        self.checkpoint_dir = self.root + '/checkpoint/' + self.dataset
        self.log_dir = self.root + '/log/' + self.dataset
        self.result_dir = self.root + '/result/' + self.dataset
        self.train_prefix = args.train_prefix
        self.dev_prefix = args.dev_prefix
        self.test_prefix = args.test_prefix
        self.model_save_name = args.model_name + '_DATASET_' + self.dataset + "_LR_" + \
                               str(self.learning_rate) + "_BS_" + str(self.batch_size)
        self.log_save_name = 'LOG_' + args.model_name + '_DATASET_' + self.dataset + "_LR_" + \
                             str(self.learning_rate) + "_BS_" + str(self.batch_size)
        self.result_save_name = 'RESULT_' + args.model_name + '_DATASET_' + self.dataset + "_LR_" + \
                                str(self.learning_rate) + "_BS_" + str(self.batch_size) + ".json"

        # log setting
        self.period = args.period
        self.test_epoch = args.test_epoch

        # debug
        self.debug = args.debug
        if self.debug:
            self.dev_prefix = self.train_prefix
            self.test_prefix = self.train_prefix


seed = 1234
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='Casrel', help='name of the model')
parser.add_argument('--lr', type=float, default=1e-5)
parser.add_argument('--multi_gpu', type=bool, default=False)
parser.add_argument('--dataset', type=str, default='CMED')
parser.add_argument('--batch_size', type=int, default=6)
parser.add_argument('--max_epoch', type=int, default=30)
parser.add_argument('--test_epoch', type=int, default=5)
parser.add_argument('--train_prefix', type=str, default='train_triples')
parser.add_argument('--dev_prefix', type=str, default='dev_triples')
parser.add_argument('--test_prefix', type=str, default='test_triples')
parser.add_argument('--max_len', type=int, default=512)
parser.add_argument('--rel_num', type=int, default=44)
parser.add_argument('--period', type=int, default=50)
parser.add_argument('--debug', type=bool, default=False)
args = parser.parse_args()

con = Config(args)

fw = Framework(con)

model = {
    'Casrel': Casrel
}

fw.train(model[args.model_name])
