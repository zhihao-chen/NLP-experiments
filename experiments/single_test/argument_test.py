# -*- coding: utf8 -*-
"""
======================================
    Project Name: NLP
    File Name: argument_test
    Author: czh
    Create Date: 2021/11/11
--------------------------------------
    Change Activity: 
======================================
"""
from dataclasses import dataclass, field
from transformers import HfArgumentParser

from nlp.arguments import TrainingArguments, ModelArguments, DataArguments


@dataclass
class MyArgument:
    early_stop: bool = field(default=False)
    patience: int = field(default=5, metadata={"help": "早停的轮数"})


parser = HfArgumentParser((DataArguments, ModelArguments, TrainingArguments, MyArgument))
parser.print_help()
args = parser.parse_args()
# print(args.patience)
args.device = 'cpu'
print(args.device)
