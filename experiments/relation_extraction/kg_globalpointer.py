# -*- coding: utf8 -*-
"""
======================================
    Project Name: NLP
    File Name: kg_globalpointer
    Author: czh
    Create Date: 2022/2/10
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

import numpy as np
import torch
from torch.optim import AdamW, Adam, SGD
from torch.utils.data import DataLoader
from transformers import BertConfig, BertTokenizerFast, get_scheduler
sys.path.append("/data/chenzhihao/NLP")

from nlp.models.bert_for_relation_extraction import GlobalPointerForRel
from nlp.callback.optimizers.ema import EMA
from nlp.processors.dataset import MyDataset
from nlp.losses.loss import global_pointer_crossentropy
from nlp.utils.tokenizers import tokenize
from nlp.tools.common import init_wandb_writer

LOGGER = logging.getLogger(__name__)
root = os.path.abspath(os.path.dirname(__file__))

config = {
    'batch_size': 10,
    'max_seq_length': 300,
    'max_epochs': 100,
    'lr_rate': 1e-5,
    'gradient_accumulation_steps': 1,
    'logging_steps': 500,
    'warmup_ratio': 0.1
}
# bert_model_name_or_path = "/Users/czh/Downloads/chinese-roberta-ext"
# data_dir = "/Users/czh/PycharmProjects/NLP/datas/kg"
bert_model_name_or_path = "/data/chenzhihao/chinese-roberta-ext"
data_dir = "/data/chenzhihao/NLP/datas/kg"
train_data_dir = os.path.join(data_dir, 'train_data.json')
dev_data_dir = os.path.join(data_dir, 'dev_data.json')
scheme_data_dir = os.path.join(data_dir, 'all_50_schemes.json')
output_dir = "../output_file_dir/kg"
num_worker = 0
device = torch.device('cuda:3') if torch.cuda.is_available() else torch.device('cpu')

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

WANDB, run = init_wandb_writer(project_name='kg_globalpointer',
                               train_args=config,
                               group_name="NLP",
                               experiment_name="test")


def load_data(filename):
    """加载数据
    单条格式：{'text': text, 'spo_list': [(s, p, o)]}
    """
    samples = []
    with codecs.open(filename, encoding='utf-8') as f:
        for line in tqdm(f, desc=f"load datas from {filename}"):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except Exception as e:
                print("error line: ", line)
                raise e
            samples.append({
                'text': item['text'],
                'spo_list': [(spo['subject'], spo['predicate'], spo['object'])
                             for spo in item['spo_list']]
            })
    return samples


def load_schemes(filename):

    with codecs.open(filename) as f:
        alist = json.load(f)

    id2predicate, predicate2id = alist
    id2predicate = {int(k): v for k, v in id2predicate.items()}
    predicate2id = {k: int(v) for k, v in predicate2id.items()}
    return predicate2id, id2predicate


rel2id, id2rel = load_schemes(scheme_data_dir)
print("number of relations: ", len(rel2id))
print("rel2id: ", rel2id)
print("id2rel: ", id2rel)

tokenizer = BertTokenizerFast.from_pretrained(bert_model_name_or_path, do_lower_case=False, add_special_tokens=True)
bert_config = BertConfig.from_pretrained(bert_model_name_or_path)
my_model = GlobalPointerForRel(config=bert_config, entity_types_num=2,
                               encoder_model_path=bert_model_name_or_path,
                               relation_num=len(rel2id), efficient=False)
my_model.to(device)
no_decay = ["bias", "LayerNorm.weight", "LayerNorm.bias"]
bert_param_optimizer = list(my_model.encoder.named_parameters())
optimizer_grouped_parameters = [
    {'params': [p for n, p in bert_param_optimizer if not any(nd in n for nd in no_decay)],
     'weight_decay': 0.01, 'lr': config['lr_rate']},
    {'params': [p for n, p in bert_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
     'lr': config['lr_rate']}
]

optimizer = Adam(params=optimizer_grouped_parameters, lr=config['lr_rate'])
ema = EMA(my_model, 0.999)
ema.register()


class SPO(tuple):
    """用来存三元组的类
    表现跟tuple基本一致，只是重写了 __hash__ 和 __eq__ 方法，
    使得在判断两个三元组是否等价时容错性更好。
    """
    def __init__(self, spo):
        self.spox = (
            tuple(tokenizer.tokenize(spo[0], add_special_tokens=False)),
            spo[1],
            tuple(tokenizer.tokenize(spo[2], add_special_tokens=False)),
        )

    def __hash__(self):
        return self.spox.__hash__()

    def __eq__(self, spo):
        return self.spox == spo.spox


def search(pattern, sequence):
    """从sequence中寻找子串pattern
    如果找到，返回第一个下标；否则返回-1。
    """
    n = len(pattern)
    for i in range(len(sequence)):
        if sequence[i:i + n] == pattern:
            return i
    return -1


def generate_inputs(datasets, max_seq_len):
    all_inputs = []
    error = 0
    for sample_ in tqdm(datasets):
        # tokens = tokenizer.tokenize(sample_['text'])
        # if len(tokens) > max_seq_len-2:
        #     tokens = tokens[:max_seq_len]
        # tokens = ["[CLS]"] + tokens + ["[SEP]"]
        # input_ids = tokenizer.convert_tokens_to_ids(tokens)
        # attention_mask = [1] * len(input_ids)
        # token_type_ids = [0] * len(input_ids)
        outputs = tokenizer.encode_plus(text=sample_['text'], add_special_tokens=True,
                                        max_length=max_seq_len, return_offsets_mapping=True)
        input_ids = outputs['input_ids']
        attention_mask = outputs['attention_mask']
        token_type_ids = outputs['token_type_ids']
        offset_mapping = outputs['offset_mapping']

        spoes = set()
        for s, p, o in sample_['spo_list']:
            s_ids = tokenizer.encode_plus(s, add_special_tokens=False)['input_ids']
            p = rel2id[p]
            o_ids = tokenizer.encode_plus(o, add_special_tokens=False)['input_ids']
            sh = search(s_ids, input_ids)
            oh = search(o_ids, input_ids)
            if sh != -1 and oh != -1:
                spoes.add((sh, sh + len(s_ids) - 1, p, oh, oh + len(o_ids) - 1))
            else:
                print(sample_)
                error += 1
        item = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids,
            'labels': list(spoes),
            'text': sample_['text'],
            'spo_list': sample_['spo_list'],
            'offset_mapping': offset_mapping
        }
        all_inputs.append(item)
    print(f"总计 {error} 样本无法对齐")
    return all_inputs


def collate_fn(batch):
    bs = len(batch)
    seq_lens = [len(sample['input_ids']) for sample in batch]
    max_len = max(seq_lens)
    label_lens = [len(sample['labels']) for sample in batch]
    max_spo_num = max(label_lens)

    batch_input_ids = torch.zeros(bs, max_len, dtype=torch.long)
    batch_attention_masks = torch.zeros(bs, max_len, dtype=torch.long)
    batch_token_type_ids = torch.zeros(bs, max_len, dtype=torch.long)
    batch_entity_labels = torch.zeros(bs, 2, max_spo_num, 2, dtype=torch.long)
    batch_head_labels = torch.zeros(
        bs, len(rel2id), max_spo_num, 2, dtype=torch.long
    )
    batch_tail_labels = torch.zeros(
        bs, len(rel2id), max_spo_num, 2, dtype=torch.long
    )
    batch_texts = []
    batch_spo_list = []
    batch_offset_mapping = []

    for i, sample in enumerate(batch):
        batch_input_ids[i, :seq_lens[i]] = torch.tensor(sample['input_ids'])
        batch_attention_masks[i, :seq_lens[i]] = torch.tensor(sample['attention_mask'])
        batch_token_type_ids[i, :seq_lens[i]] = torch.tensor(sample['token_type_ids'])
        for j, (sh, st, p, oh, ot) in enumerate(sample['labels']):
            batch_entity_labels[i, 0, j, :] = torch.tensor([sh, st])
            batch_entity_labels[i, 1, j, :] = torch.tensor([oh, ot])
            batch_head_labels[i, p, j, :] = torch.tensor([sh, oh])
            batch_tail_labels[i, p, j, :] = torch.tensor([st, ot])
        batch_texts.append(sample['text'])
        batch_offset_mapping.append(sample['offset_mapping'])
        batch_spo_list.append(sample['spo_list'])

    item = {'input_ids': batch_input_ids.to(device),
            'attention_mask': batch_attention_masks.to(device),
            'token_type_ids': batch_token_type_ids.to(device),
            'entity_labels': batch_entity_labels.to(device),
            'head_labels': batch_head_labels.to(device),
            'tail_labels': batch_tail_labels.to(device),
            'texts': batch_texts,
            'spo_list': batch_spo_list,
            'offset_mapping': batch_offset_mapping
            }
    return item


# 加载数据集
def data_generator(data_type="train"):
    """
    读取数据，生成DataLoader。
    """
    if data_type == "train":
        train_data = load_data(train_data_dir)
        valid_data = load_data(dev_data_dir)
    elif data_type == "valid":
        valid_data = load_data(dev_data_dir)
        train_data = []

    all_data = train_data + valid_data

    # TODO:句子截取
    max_tok_num = 0
    for sample in all_data:
        tokens = tokenizer.tokenize(sample["text"], add_special_tokens=True)
        max_tok_num = max(max_tok_num, len(tokens))
    assert max_tok_num <= config['max_seq_length'], f"数据文本最大token数量{max_tok_num}超过预设{config['max_seq_length']}"
    max_seq_len = min(max_tok_num, config['max_seq_length'])

    if data_type == "train":
        train_inputs = generate_inputs(train_data, max_seq_len)
        valid_inputs = generate_inputs(valid_data, max_seq_len)
        train_dataloader = DataLoader(MyDataset(train_inputs),
                                      batch_size=config['batch_size'],
                                      shuffle=True,
                                      num_workers=num_worker,
                                      drop_last=False,
                                      collate_fn=collate_fn
                                      )
        valid_dataloader = DataLoader(MyDataset(valid_inputs),
                                      batch_size=config['batch_size'],
                                      shuffle=False,
                                      num_workers=num_worker,
                                      drop_last=False,
                                      collate_fn=collate_fn
                                      )
        return train_dataloader, valid_dataloader
    elif data_type == "valid":
        valid_inputs = generate_inputs(valid_data, max_seq_len)
        valid_dataloader = DataLoader(MyDataset(valid_inputs),
                                      batch_size=config['batch_size'],
                                      shuffle=False,
                                      num_workers=num_worker,
                                      drop_last=False,
                                      )
        return valid_dataloader


def extract_spoes(model, eval_dataloader, threshold=0):
    """抽取输入text所包含的三元组
    """
    # output = tokenizer.encode_plus(text, return_offsets_mapping=True, add_special_tokens=True)
    # token_ids = torch.LongTensor([output['input_ids']]).to(device)
    # segment_ids = torch.LongTensor([output['token_type_ids']]).to(device)
    # attention_mask = torch.LongTensor([output['attention_mask']]).to(device)
    # offset_mapping = output['offset_mapping']

    all_predicts = []
    all_texts = []
    all_spo_list = []

    model.eval()
    for batch in tqdm(eval_dataloader, desc="Evaluating"):

        with torch.no_grad():
            outputs = model(batch['input_ids'], batch['attention_mask'], batch['token_type_ids'])
            outputs = [o.detach().cpu().numpy() for _, o in outputs.items()]
        offset_mappings = batch['offset_mapping']
        texts = batch['texts']
        all_texts.extend(texts)
        all_spo_list.extend(batch['spo_list'])
        assert len(outputs[0]) == len(texts)

        for entity_output, head_output, tai_output, offset_mapping, text in zip(
            outputs[0], outputs[1], outputs[2], offset_mappings, texts
        ):

            # 抽取subject和object
            subjects, objects = set(), set()
            outputs[0][:, [0, -1]] -= np.inf
            outputs[0][:, :, [0, -1]] -= np.inf
            for l_, h, t in zip(*np.where(outputs[0] > threshold)):
                if l_ == 0:
                    subjects.add((h, t))
                else:
                    objects.add((h, t))
            # 识别对应的predicate
            spoes = set()
            for sh, st in subjects:
                for oh, ot in objects:
                    p1s = np.where(outputs[1][:, sh, oh] > threshold)[0]
                    p2s = np.where(outputs[2][:, st, ot] > threshold)[0]
                    ps = set(p1s) & set(p2s)
                    for p in ps:
                        try:
                            spoes.add((
                                text[offset_mapping[sh][0]:offset_mapping[st][-1]],
                                id2rel[p],
                                text[offset_mapping[oh][0]:offset_mapping[ot][-1]]
                            ))
                        except Exception as e:
                            print(len(id2rel), len(offset_mapping), offset_mapping)
                            print((sh, st), p, (oh, ot))
                            raise e
            all_predicts.append(list(spoes))
    assert len(all_predicts) == len(all_texts), f"{len(all_predicts)}\t{len(all_texts)}"
    assert len(all_predicts) == len(all_spo_list), f"{len(all_predicts)}\t{len(all_spo_list)}"
    return all_predicts, all_texts, all_spo_list


def evaluate(model, eval_dataloader):
    """评估函数，计算f1、precision、recall
    """
    all_predicts, all_texts, all_spo_list = extract_spoes(model, eval_dataloader)

    x, y, z = 0, 0, 0
    f = open('dev_pred.json', 'w', encoding='utf-8')
    pbar = tqdm()
    f1, precision, recall = 1e-10, 1e-10, 1e-10
    for pred, text, golds in zip(all_predicts, all_texts, all_spo_list):
        r = set(pred)
        t = set(golds)
        x += len(r & t)
        y += len(r)
        z += len(t)
        # pbar.update()
        # pbar.set_description(
        #     'f1: %.5f, precision: %.5f, recall: %.5f' % (f1, precision, recall)
        # )
        s = json.dumps({
            'text': text,
            'spo_list': list(t),
            'spo_list_pred': list(r),
            'new': list(r - t),
            'lack': list(t - r),
        },
            ensure_ascii=False,
            indent=4)
        f.write(s + '\n')
    f1 = 2 * x / (y + z) if y and z else 0
    precision = x / y if y else 0
    recall = x / z if z else 0
    # pbar.close()
    f.close()
    return f1, precision, recall


def train(model, train_dataloader, eval_data, scheduler):
    WANDB.watch(model, log="all")

    model.train()
    model.zero_grad()
    best_f1 = 0.0
    best_epoch = 0
    global_steps = 0
    print("Training")
    for epoch in range(config['max_epochs']):
        tr_loss = 0.0
        epoch_steps = 0
        print("**********************training**********************\n")
        pbar = tqdm()
        for step, batch in enumerate(train_dataloader):
            outputs = model(batch['input_ids'], batch['attention_mask'], batch['token_type_ids'])
            entity_logits = outputs['entity_logits']
            head_logits = outputs['head_logits']
            tail_logits = outputs['tail_logits']

            entity_loss = global_pointer_crossentropy(entity_logits, batch['entity_labels'], sparse=True, mask_zero=True)
            head_loss = global_pointer_crossentropy(head_logits, batch['head_labels'], sparse=True, mask_zero=True)
            tail_loss = global_pointer_crossentropy(tail_logits, batch['tail_labels'], sparse=True, mask_zero=True)

            loss = (entity_loss+head_loss+tail_loss) / 3 / config['gradient_accumulation_steps']

            pbar.update()
            pbar.set_description(
                f"Training at epoch {epoch}\ttotal loss: {loss.item()}\tentity loss: {entity_loss.item()}\t"
                f"head loss: {head_loss.item()}\ttail loss: {tail_loss.item()}"
            )

            loss.backward()
            WANDB.log({'Train/total_loss': loss.item(),
                       'Train/entity_loss': entity_loss.item(),
                       'Train/head_loss': head_loss.item(),
                       'Train/tail_loss': tail_loss.item()}, step=global_steps)
            tr_loss += loss.item()
            epoch_steps += 1
            if (step + 1) % config['gradient_accumulation_steps'] == 0:
                optimizer.step()
                # ema.update()
                # scheduler.step()
                optimizer.zero_grad()
                global_steps += 1

        print("***********************evaluating***********************\n")
        f1, precision, recall = evaluate(model, eval_data)
        print(
            f"Evaluating epoch: {epoch}\ttotal loss: {tr_loss / global_steps}\tf1: {f1}\tprecision: "
            f"{precision}\trecall: {recall}\n")
        WANDB.log({'Eval/f1': f1,
                   'Eval/precision': precision, 'Eval/recall': recall}, step=global_steps)
        if f1 > best_f1:
            best_epoch = epoch
            best_f1 = f1
            model_to_save = (
                model.module if hasattr(model, "module") else model
            )  # Take care of distributed/parallel training
            # model_to_save.save_pretrained(output_dir)
            torch.save(model_to_save.state_dict(), output_dir + '/pytorch.bin')
            tokenizer.save_vocabulary(output_dir)
            print(f"save model to {output_dir}")
        pbar.close()
    return best_epoch, best_f1


def main():
    train_dataloader, eval_dataloader = data_generator()
    t_total = len(train_dataloader) // config['gradient_accumulation_steps'] * config['max_epochs']
    scheduler = get_scheduler(name='linear', optimizer=optimizer, num_warmup_steps=0, num_training_steps=t_total)

    best_epoch, best_f1 = train(my_model, train_dataloader, eval_dataloader, scheduler)
    print(f"best epoch: {best_epoch}\tbest f1: {best_f1}")
    print("Evaluating")
    eval_model = GlobalPointerForRel(config=bert_config,
                                     entity_types_num=2, relation_num=len(rel2id))
    eval_model.load_state_dict(torch.load(output_dir+'/pytorch.bin', map_location='cpu'))
    eval_model.to(device)
    f1, precision, recall = evaluate(eval_model, eval_dataloader)
    print(f"evaluate result: f1: {f1}\tprecision: {precision}\trecall: {recall}")


if __name__ == "__main__":
    main()
