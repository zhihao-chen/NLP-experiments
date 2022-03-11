# -*- coding: utf8 -*-
"""
======================================
    Project Name: NLP
    File Name: lear_for_ner
    Author: czh
    Create Date: 2022/3/11
--------------------------------------
    Change Activity: 
======================================
"""
import os
import sys
import json
from tqdm import tqdm
import codecs
from typing import List, Tuple
sys.path.append("/data/chenzhihao/NLP")

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import BertConfig, BertTokenizerFast

from nlp.models.bert_for_ner import LearForNer
from nlp.tools.path import project_root_path
