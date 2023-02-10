#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: czh
@email:
@date: 2023/2/2 18:58
"""

# https://github.com/Liadrinz/transformers-unilm
from tqdm import tqdm

from transformers.trainer_seq2seq import Seq2SeqTrainer
from transformers.training_args import TrainingArguments

from nlp.utils.tokenization_unilm import UniLMTokenizerLiadrinz as UniLMTokenizer
from nlp.models.unilm_model_liadrinz import UniLMForConditionalGeneration
from nlp.processors.unilm_liadrinz_processor import DataCollatorForUniLMSeq2Seq, Seq2SeqDataset


# 中文摘要任务生成
news_article = (
    "12月23日，河北石家庄。8岁哥哥轻车熟路哄睡弟弟，姿势标准动作熟练。"
    "妈妈杨女士表示：哥哥很喜欢弟弟，因为心思比较细，自己平时带孩子的习惯他都会跟着学习，"
    "哄睡孩子也都会争着来，技巧很娴熟，两人在一块很有爱，自己感到很幸福，平时帮了自己很大的忙，感恩有这么乖的宝宝。"
)

tokenizer = UniLMTokenizer.from_pretrained("Yuang/unilm-base-chinese-news-sum")
model = UniLMForConditionalGeneration.from_pretrained("Yuang/unilm-base-chinese-news-sum") # 在微博新闻摘要数据上fine-tune过的模型

inputs = tokenizer(news_article, return_tensors="pt")
output_ids = model.generate(**inputs, max_new_tokens=16)
output_text = tokenizer.decode(output_ids[0])
print(output_text)  # "[CLS] <news_article> [SEP] <news_summary> [SEP]"
news_summary = output_text.split("[SEP]")[1].strip()
print(news_summary)

# 训练
tokenizer = UniLMTokenizer.from_pretrained("microsoft/unilm-base-cased")
dataset = Seq2SeqDataset(tokenizer, "train.src", "train.tgt", max_src_len=448, max_tgt_len=64)
collator = DataCollatorForUniLMSeq2Seq(tokenizer, mlm=True, mlm_probability=0.7)
model = UniLMForConditionalGeneration.from_pretrained("microsoft/unilm-base-cased")
training_args = TrainingArguments(
    output_dir="output_dir",
    do_train=True,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    learning_rate=1e-4,
    num_train_epochs=3,
)
trainer = Seq2SeqTrainer(
    model,
    args=training_args,
    data_collator=collator,
    train_dataset=dataset,
    tokenizer=tokenizer,
)
trainer.train()
