# -*- coding: utf8 -*-
"""
======================================
    Project Name: NLP
    File Name: run_ner_tplinker_v2
    Author: czh
    Create Date: 2021/8/24
--------------------------------------
    Change Activity: 
======================================
"""
import time

from nlp.utils.tplinker_plus_utils import DataAndTrainArguments
from nlp.models.tplinker_plus_for_ner import TPLinkerPlusForNER

root_dir = "/data/chenzhihao/NLP/experiments"
config = {
    "bert_name_or_path": "hfl/chinese-roberta-wwm-ext",
    "data_dir": root_dir + "/datas/tplinker",
    "task_name": "ner",
    "model_type": "BERT",
    "train_data_name": "train_data.json",
    "valid_data_name": "valid_data.json",
    "ent2id": "ent2id.json",
    "output_dir": root_dir + "/output_file_dir/tplinker_plus_ner_bert/train_results",
    "log_dir": root_dir + "/logs/tplinker_plus_ner.log",
    "tensorboard_log_dir": root_dir + "/tensorboard/tplinker_plus_ner/",
    "path_to_save_model": root_dir + "/output_file_dir/tplinker_plus_ner_bert/train_results/best_model",
    "model_state_dict_path": root_dir + "/output_file_dir/tplinker_plus_ner_bert/train_results/best_model",
    "save_res_dir": root_dir + "/output_file_dir/tplinker_plus_ner_bert/eval_results",
    "score": True,  # set true only if test set is tagged
    "n_gpu": "0",
    "num_workers": 4,
    "logger": "default",
    "train_batch_size": 16,
    "eval_batch_size": 16,
    "epochs": 4,
    "fp16": True,
    "gradient_accumulation_steps": 1,
    "shaking_type": "cln_plus",
    "match_pattern": "whole_text",
    "inner_enc_type": "lstm",
    "f1_2_save": 0,
    "fr_scratch": True,
    "fr_last_checkpoint": False,
    "note": "start from scratch",
    "log_interval": 10,
    "max_seq_len": 512,
    "sliding_len": 20,
    "last_k_model": 1,
    "scheduler": "CAWR",  # Step
    "ghm": False,
    "tok_pair_sample_rate": 1,
    "force_split": False,
    "lr": 5e-5,
    "T_mult": 1,
    "rewarm_epoch_num": 2,
    "save_steps": 500,
    "logging_steps": 500
}

start = time.time()
args = DataAndTrainArguments(**config)
# print(args.__dict__)
trainer = TPLinkerPlusForNER(args)
trainer.init_env()

# training
trainer.train_and_valid()
print(time.time()-start)

start = time.time()
# evaluating
trainer.evaluate()
print(time.time()-start)

# predicting
start = time.time()
text = "百炼智能是一家人工智能科技公司，公司CEO是冯是聪"
trainer.init_others(len(text)+2)
model = trainer.init_model(16)
trainer.restore(model)
test_data, ori_test_data, max_seq_len = trainer.process_predict_data(text, max_seq_len=len(text)+2)
result = trainer.predict(test_data=test_data,
                         ori_test_data=ori_test_data,
                         model=model,
                         max_seq_len=max_seq_len,
                         batch_size=1)
print(result)
print(time.time()-start)
