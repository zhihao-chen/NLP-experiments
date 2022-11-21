#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: czh
@email:
@date: 2022/11/11 15:12
"""
from transformers.models.bert.convert_bert_original_tf_checkpoint_to_pytorch import convert_tf_checkpoint_to_pytorch

# chinese_wobert_plus
path = "/Users/chenzhihao/Downloads/chinese_wobert_plus_L-12_H-768_A-12"
tf_checkpoint_path = path + "/bert_model.ckpt"
bert_config_file = path + "/bert_config.json"
pytorch_dump_path = "/Users/chenzhihao/Downloads/chinese_wobert_plus/pytorch_model.bin"

convert_tf_checkpoint_to_pytorch(tf_checkpoint_path, bert_config_file,
                                 pytorch_dump_path)

# chinese_wobert
path = "/Users/chenzhihao/Downloads/chinese_wobert_L-12_H-768_A-12"
tf_checkpoint_path = path + "/bert_model.ckpt"
bert_config_file = path + "/bert_config.json"
pytorch_dump_path = "/Users/chenzhihao/Downloads/chinese_wobert_base/pytorch_model.bin"

convert_tf_checkpoint_to_pytorch(tf_checkpoint_path, bert_config_file,
                                 pytorch_dump_path)
