# -*- coding: utf8 -*-
"""
======================================
    Project Name: NLP
    File Name: cluener_globalpointer
    Author: czh
    Create Date: 2021/8/6
--------------------------------------
    Change Activity: 
======================================
"""
import os
import sys
import json
import codecs
import logging
from tqdm import tqdm

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import BertConfig, BertTokenizerFast, BertModel
sys.path.append("/data/chenzhihao/NLP")
from nlp.layers.global_pointer import GlobalPointer, EfficientGlobalPointer
from nlp.metrics.metric import MetricsCalculator
from nlp.processors.dataset import MyDataset, DataMaker
from nlp.losses.loss import global_pointer_crossentropy

LOGGER = logging.getLogger(__name__)
root = os.path.abspath(os.path.dirname(__file__))
bert_model_name_or_path = "/data/chenzhihao/chinese-roberta-ext"
data_dir = "/data/chenzhihao/NLP/datas/cluener"
train_data_dir = os.path.join(data_dir, 'train.json')
dev_data_dir = os.path.join(data_dir, 'dev.json')
output_dir = "../output_file_dir/cluener"
batch_size = 128
max_seq_length = 256
max_epochs = 50
lr_rate = 2e-5
gradient_accumulation_steps = 1
logging_steps = 500
num_worker = 2

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

label_list = ["address", "book", "company", 'game', 'government', 'movie', 'name', 'organization',
              'position', 'scene', 'O']
label2id = {label: i for i, label in enumerate(label_list)}
id2label = {i: label for i, label in enumerate(label_list)}

device = torch.device('cuda:0')

tokenizer = BertTokenizerFast.from_pretrained(bert_model_name_or_path, do_lower_case=False, add_special_tokens=True)
bert_config = BertConfig.from_pretrained(bert_model_name_or_path)
encoder = BertModel.from_pretrained(bert_model_name_or_path)
# mymodel = GlobalPointer(encoder, len(label_list), 64)  # 9个实体类型
mymodel = EfficientGlobalPointer(encoder, len(label_list))
mymodel.to(device)

optimizer = AdamW(params=mymodel.parameters(), lr=lr_rate)


def load_data(data_path, data_type="train"):
    """读取数据集
    Args:
        data_path (str): 数据存放路径
        data_type (str, optional): 数据类型. Defaults to "train".
    Returns:
        (json): train和valid中一条数据格式：{"text":"","entity_list":[(start, end, label), (start, end, label)...]}
    """
    if data_type == "train" or data_type == "valid":
        datas = []
        with codecs.open(data_path, encoding="utf-8") as f:
            for line in f:
                line = json.loads(line)
                item = {"text": line["text"], "entity_list": []}
                for k, v in line['label'].items():
                    for spans in v.values():
                        for start, end in spans:
                            item["entity_list"].append((start, end, k))
                datas.append(item)
        return datas
    else:
        return json.load(open(data_path, encoding="utf-8"))


def data_generator(data_type="train"):
    """
    读取数据，生成DataLoader。
    """

    if data_type == "train":
        train_data_path = os.path.join(data_dir, "train.json")
        train_data = load_data(train_data_path, "train")
        valid_data_path = os.path.join(data_dir, "dev.json")
        valid_data = load_data(valid_data_path, "valid")
    elif data_type == "valid":
        valid_data_path = os.path.join(data_dir, "dev.json")
        valid_data = load_data(valid_data_path, "valid")
        train_data = []

    all_data = train_data + valid_data

    # TODO:句子截取
    max_tok_num = 0
    for sample in all_data:
        tokens = tokenizer(sample["text"])["input_ids"]
        max_tok_num = max(max_tok_num, len(tokens))
    assert max_tok_num <= max_seq_length, f'数据文本最大token数量{max_tok_num}超过预设{max_seq_length}'
    max_seq_len = min(max_tok_num, max_seq_length)

    data_maker = DataMaker(tokenizer)

    if data_type == "train":
        train_inputs = data_maker.generate_inputs(train_data, max_seq_len, label2id)
        valid_inputs = data_maker.generate_inputs(valid_data, max_seq_len, label2id)
        train_dataloader = DataLoader(MyDataset(train_inputs),
                                      batch_size=batch_size,
                                      shuffle=True,
                                      num_workers=num_worker,
                                      drop_last=False,
                                      collate_fn=data_maker.generate_batch
                                      )
        valid_dataloader = DataLoader(MyDataset(valid_inputs),
                                      batch_size=batch_size,
                                      shuffle=False,
                                      num_workers=num_worker,
                                      drop_last=False,
                                      collate_fn=data_maker.generate_batch
                                      )
        return train_dataloader, valid_dataloader
    elif data_type == "valid":
        valid_inputs = data_maker.generate_inputs(valid_data, max_seq_len, label2id)
        valid_dataloader = DataLoader(MyDataset(valid_inputs),
                                      batch_size=batch_size,
                                      shuffle=False,
                                      num_workers=num_worker,
                                      drop_last=False,
                                      )
        return valid_dataloader


def collate_fn(batch):
    batch_samples, batch_input_ids, batch_attention_mask, batch_token_type_ids, batch_labels = batch
    input_ids, attention_masks, token_type_ids, labels = (batch_input_ids.to(device),
                                                          batch_attention_mask.to(device),
                                                          batch_token_type_ids.to(device),
                                                          batch_labels.to(device)
                                                        )

    return {"input_ids": input_ids, "token_type_ids": token_type_ids, "attention_mask": attention_masks}, labels


metric = MetricsCalculator()


def evaluate(model, eval_dataloader):
    model.eval()
    f1, precision, recall = 0.0, 0.0, 0.0
    total_num = len(eval_dataloader)

    for i, batch in tqdm(enumerate(eval_dataloader), desc="Evaluating"):
        inputs, label_ids = collate_fn(batch)
        with torch.no_grad():
            logits = model(**inputs)
            sample_f1, sample_p, sample_r = metric.get_evaluate_fpr(label_ids, logits)
            f1 += sample_f1
            precision += sample_p
            recall += sample_r
    avg_f1 = f1 / total_num
    avg_precision = precision / total_num
    avg_recall = recall / total_num
    return avg_f1, avg_precision, avg_recall


def train(model, train_dataloader, eval_dataloader):
    model.train()
    model.zero_grad()
    best_f1 = 0.0
    best_epoch = 0
    print("Training")
    for epoch in range(max_epochs):
        tr_loss = 0.0
        global_steps = 0.0
        print("**********************training**********************")
        for step, batch in enumerate(train_dataloader):
            inputs, label_ids = collate_fn(batch)
            logits = model(**inputs)
            loss = global_pointer_crossentropy(logits, label_ids)

            loss.backward()
            tr_loss += loss.item()
            global_steps += 1
            if (step+1) % gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            sample_f1 = metric.get_sample_f1(logits, label_ids)
            print(f"Training epoch: {epoch}\tsteps: {step+1}\tloss: {loss.item()}\tf1: {sample_f1}")
        print("***********************evaluating***********************")
        f1, precision, recall = evaluate(model, eval_dataloader)
        print(f"Evaluating epoch: {epoch}\ttotal loss: {tr_loss/global_steps}\tf1: {f1}\tprecision: {precision}\trecall: {recall}")
        if f1 > best_f1:
            best_epoch = epoch
            best_f1 = f1
            model_to_save = (
                model.module if hasattr(model, "module") else model
            )  # Take care of distributed/parallel training
            # model_to_save.save_pretrained(output_dir)
            torch.save(model_to_save.state_dict(), output_dir+'/pytorch.bin')
            tokenizer.save_vocabulary(output_dir)
            print(f"save model to {output_dir}")
    return best_epoch, best_f1


def main():
    train_dataloader, eval_dataloader = data_generator()

    best_epoch, best_f1 = train(mymodel, train_dataloader, eval_dataloader)
    print(f"best epoch: {best_epoch}\tbest f1: {best_f1}")
    print("Evaluating")
    encoder_ = BertModel.from_pretrained(bert_model_name_or_path)
    # eval_model = GlobalPointer(encoder_, len(label_list), 64)
    eval_model = EfficientGlobalPointer(encoder_, len(label_list), 64)
    eval_model.load_state_dict(torch.load(output_dir+'/pytorch.bin', map_location='cpu'))
    eval_model.to(device)
    f1, precision, recall = evaluate(eval_model, eval_dataloader)
    print(f"evaluate result: f1: {f1}\tprecision: {precision}\trecall: {recall}")


if __name__ == "__main__":
    main()
