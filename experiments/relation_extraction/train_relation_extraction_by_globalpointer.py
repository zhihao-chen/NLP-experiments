# -*- coding: utf8 -*-
"""
======================================
    Project Name: NLP
    File Name: train_relation_extraction_by_globalpointer
    Author: czh
    Create Date: 2022/2/9
--------------------------------------
    Change Activity: 
======================================
"""
import os
import regex
import glob
import json
import time
from tqdm import tqdm
import typing
import codecs
from collections import defaultdict

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import BertConfig, BertTokenizerFast, HfArgumentParser, get_scheduler

from nlp.models.bert_for_relation_extraction import GlobalPointerForRel

