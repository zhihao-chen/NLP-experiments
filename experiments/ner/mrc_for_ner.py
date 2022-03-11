# -*- coding: utf8 -*-
"""
======================================
    Project Name: NLP
    File Name: mrc_for_ner
    Author: czh
    Create Date: 2022/2/23
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

from nlp.models.bert_for_ner import BertQueryNER
from nlp.tools.path import project_root_path


root_path = project_root_path()
# bert_model_name_or_path = "/Users/czh/Downloads/chinese-roberta-ext"
bert_model_name_or_path = "/data/chenzhihao/chinese-roberta-ext"
data_dir = "datas/cluener"
output_dir = "output_file_dir/mrc"
max_sequence_length = 512
batch_size = 10
epochs = 30
lr_rate = 2e-5
gradient_accumulation_steps = 1
logging_steps = 500
num_worker = 0
warmup_ratio = 0.1
weight_start = 1.0
weight_end = 1.0
weight_span = 1.0
span_loss_candidates = "all"  # ["all", "pred_and_gold","pred_gold_random","gold"],Candidates used to compute span loss
device = torch.device('cuda:1') if torch.cuda.is_available() else torch.device('cpu')

tokenizer = BertTokenizerFast.from_pretrained(bert_model_name_or_path, do_lower_case=False, add_special_tokens=True)
bert_config = BertConfig.from_pretrained(bert_model_name_or_path)

bce_loss = nn.BCEWithLogitsLoss(reduction='none')


query_map = {
               'address': "找出省、市、区、街道乡村等抽象或具体的地点",
               'book': "找出小说、杂志、习题集、教科书、教辅、地图册、食谱等具体的书名",
               'company': "找出公司、集团、银行（央行，中国人民银行除外，二者属于政府机构）等具体的公司名",
               'game': "找出常见的游戏名",
               'government': "找出中央行政机关和地方行政机关的名字",
               'movie': "找出电影、纪录片等放映或上线的影片名字",
               'name': "找出真实和虚构的人名",
               'organization': "找出包括篮球队、足球队、乐团、社团、小说里面的帮派等真实或虚构的组织机构名",
               'position': "找出现代和古时候的职称名",
               'scene': "找出常见的旅游景点"
}


def trans_data_to_mrc(input_file):
    mrc_samples = []

    with codecs.open(input_file, encoding="utf8") as fr:
        for line in fr:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            text = item['text']
            label_dict = item['label']
            for label, query in query_map.items():
                start_positions = []
                end_positions = []
                entity_dict = label_dict.get(label, None)
                if not entity_dict:
                    continue
                for k, offsets in entity_dict.items():
                    for s, e in offsets:
                        start_positions.append(s)
                        end_positions.append(e+1)
                mrc_samples.append(
                    {
                        "context": text,
                        "start_position": start_positions,
                        "end_position": end_positions,
                        "query": query
                    }
                )

    return mrc_samples


class MRCNERDataset(Dataset):
    """
    MRC NER Dataset
    Args:
        datasets:
        max_length: int, max length of query+context
        possible_only: if True, only use possible samples that contain answer for the query/context
        is_chinese: is chinese dataset
    """
    def __init__(self, datasets, max_length: int = 512, possible_only=False,
                 is_chinese=False, pad_to_maxlen=False):
        self.all_data = datasets
        self.max_length = max_length
        self.possible_only = possible_only
        if self.possible_only:
            self.all_data = [
                x for x in self.all_data if x["start_position"]
            ]
        self.is_chinese = is_chinese
        self.pad_to_maxlen = pad_to_maxlen

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, item):
        """
        Args:
            item: int, idx
        Returns:
            tokens: tokens of query + context, [seq_len]
            token_type_ids: token type ids, 0 for query, 1 for context, [seq_len]
            start_labels: start labels of NER in tokens, [seq_len]
            end_labels: end labelsof NER in tokens, [seq_len]
            label_mask: label mask, 1 for counting into loss, 0 for ignoring. [seq_len]
            match_labels: match labels, [seq_len, seq_len]
            sample_idx: sample id
            label_idx: label id
        """
        data = self.all_data[item]

        qas_id = data.get("qas_id", "0.0")
        sample_idx, label_idx = qas_id.split(".")
        sample_idx = torch.LongTensor([int(sample_idx)])
        label_idx = torch.LongTensor([int(label_idx)])

        query = data["query"]
        context = data["context"]
        start_positions = data["start_position"]
        end_positions = data["end_position"]
        # TODO: 修改tokenizer
        query_context_tokens = tokenizer.encode_plus(query, context, add_special_tokens=True, return_offsets_mapping=True)
        tokens = query_context_tokens['input_ids']
        attention_masks = query_context_tokens['attention_mask']
        type_ids = query_context_tokens['token_type_ids']
        offsets = query_context_tokens['offset_mapping']

        # find new start_positions/end_positions, considering
        # 1. we add query tokens at the beginning
        # 2. word-piece tokenize
        origin_offset2token_idx_start = {}
        origin_offset2token_idx_end = {}
        for token_idx in range(len(tokens)):
            # skip query tokens
            if type_ids[token_idx] == 0:
                continue
            token_start, token_end = offsets[token_idx][0], offsets[token_idx][1]
            # skip [CLS] or [SEP]
            if token_start == token_end == 0:
                continue
            origin_offset2token_idx_start[token_start] = token_idx
            origin_offset2token_idx_end[token_end] = token_idx

        new_start_positions = []
        new_end_positions = []
        for s, e in zip(start_positions, end_positions):
            try:
                new_s = origin_offset2token_idx_start[s]
                new_e = origin_offset2token_idx_end[e]
            except Exception as exc:
                print((s, e), offsets)
                print(origin_offset2token_idx_start)
                print(origin_offset2token_idx_end)
                print(tokenizer.tokenize(query, context, add_special_tokens=True))
                print(exc)
                continue

            if 0 < new_s <= new_e < len(offsets):
                new_start_positions.append(new_s)
                new_end_positions.append(new_e)
            else:
                print((s, e), (new_s, new_e), query, context)

        label_mask = [
            (0 if type_ids[token_idx] == 0 or offsets[token_idx] == (0, 0) else 1)
            for token_idx in range(len(tokens))
        ]
        start_label_mask = label_mask.copy()
        end_label_mask = label_mask.copy()

        assert all(start_label_mask[p] != 0 for p in new_start_positions)
        assert all(end_label_mask[p] != 0 for p in new_end_positions)

        assert len(label_mask) == len(tokens)
        start_labels = [(1 if idx in new_start_positions else 0)
                        for idx in range(len(tokens))]
        end_labels = [(1 if idx in new_end_positions else 0)
                      for idx in range(len(tokens))]

        # truncate
        token_ids = tokens[: self.max_length]
        token_type_ids = type_ids[: self.max_length]
        attention_masks = attention_masks[: self.max_length]
        start_labels = start_labels[: self.max_length]
        end_labels = end_labels[: self.max_length]
        start_label_mask = start_label_mask[: self.max_length]
        end_label_mask = end_label_mask[: self.max_length]

        # make sure last token is [SEP]
        sep_token = tokenizer.sep_token_id
        if token_ids[-1] != sep_token:
            assert len(token_ids) == self.max_length
            token_ids = token_ids[: -1] + [sep_token]
            start_labels[-1] = 0
            end_labels[-1] = 0
            start_label_mask[-1] = 0
            end_label_mask[-1] = 0

        if self.pad_to_maxlen:
            token_ids = self.pad(token_ids, 0)
            attention_masks = self.pad(attention_masks, 0)
            token_type_ids = self.pad(token_type_ids, 1)
            start_labels = self.pad(start_labels)
            end_labels = self.pad(end_labels)
            start_label_mask = self.pad(start_label_mask)
            end_label_mask = self.pad(end_label_mask)

        seq_len = len(token_ids)
        match_labels = torch.zeros([seq_len, seq_len], dtype=torch.long)
        for start, end in zip(new_start_positions, new_end_positions):
            if start >= seq_len or end >= seq_len:
                continue
            match_labels[start, end] = 1

        return [
            torch.LongTensor(token_ids),
            torch.LongTensor(attention_masks),
            torch.LongTensor(token_type_ids),
            torch.LongTensor(start_labels),
            torch.LongTensor(end_labels),
            torch.LongTensor(start_label_mask),
            torch.LongTensor(end_label_mask),
            match_labels,
            sample_idx,
            label_idx
        ]

    def pad(self, lst, value=0, max_length=None):
        max_length = max_length or self.max_length
        while len(lst) < max_length:
            lst.append(value)
        return lst


def collate_fn(batch):
    batch_size_ = len(batch)
    max_length = max(x[0].shape[0] for x in batch)

    outputs = []

    for field_idx in range(7):
        pad_output = torch.full([batch_size_, max_length], 0, dtype=batch[0][field_idx].dtype)
        for sample_idx in range(batch_size_):
            data = batch[sample_idx][field_idx]
            pad_output[sample_idx][: data.shape[0]] = data
        outputs.append(pad_output)

    pad_match_labels = torch.zeros([batch_size_, max_length, max_length], dtype=torch.long)
    for sample_idx in range(batch_size_):
        data = batch[sample_idx][7]
        pad_match_labels[sample_idx, : data.shape[1], : data.shape[1]] = data

    outputs.append(pad_match_labels)

    outputs.append(torch.stack([x[-2] for x in batch]))
    outputs.append(torch.stack([x[-1] for x in batch]))

    return outputs


def load_features(data_type):
    data_file = os.path.join(root_path, data_dir, f"{data_type}.json")
    samples = trans_data_to_mrc(data_file)

    if data_type != "test":
        datasets = MRCNERDataset(datasets=samples, max_length=max_sequence_length)
        if data_type == 'train':
            data_loader = DataLoader(datasets, shuffle=True, batch_size=batch_size, collate_fn=collate_fn)
        else:
            data_loader = DataLoader(datasets, shuffle=False, batch_size=batch_size, collate_fn=collate_fn)
        return data_loader
    else:
        return samples


def compute_loss(start_logits, end_logits, span_logits,
                 start_labels, end_labels, match_labels, start_label_mask, end_label_mask):
    batch_size_, seq_len = start_logits.size()

    start_float_label_mask = start_label_mask.view(-1).float()
    end_float_label_mask = end_label_mask.view(-1).float()
    match_label_row_mask = start_label_mask.bool().unsqueeze(-1).expand(-1, -1, seq_len)
    match_label_col_mask = end_label_mask.bool().unsqueeze(-2).expand(-1, seq_len, -1)
    match_label_mask = match_label_row_mask & match_label_col_mask
    match_label_mask = torch.triu(match_label_mask, 0)  # start should be less equal to end

    if span_loss_candidates == "all":
        # naive mask
        float_match_label_mask = match_label_mask.view(batch_size_, -1).float()
    else:
        # use only pred or golden start/end to compute match loss
        start_preds = start_logits > 0
        end_preds = end_logits > 0
        if span_loss_candidates == "gold":
            match_candidates = ((start_labels.unsqueeze(-1).expand(-1, -1, seq_len) > 0)
                                & (end_labels.unsqueeze(-2).expand(-1, seq_len, -1) > 0))
        elif span_loss_candidates == "pred_gold_random":
            gold_and_pred = torch.logical_or(
                (start_preds.unsqueeze(-1).expand(-1, -1, seq_len)
                 & end_preds.unsqueeze(-2).expand(-1, seq_len, -1)),
                (start_labels.unsqueeze(-1).expand(-1, -1, seq_len)
                 & end_labels.unsqueeze(-2).expand(-1, seq_len, -1))
            )
            data_generator = torch.Generator()
            data_generator.manual_seed(0)
            random_matrix = torch.empty(batch_size_, seq_len, seq_len).uniform_(0, 1)
            random_matrix = torch.bernoulli(random_matrix, generator=data_generator).long()
            random_matrix = random_matrix.to(device)
            match_candidates = torch.logical_or(
                gold_and_pred, random_matrix
            )
        else:
            match_candidates = torch.logical_or(
                (start_preds.unsqueeze(-1).expand(-1, -1, seq_len)
                 & end_preds.unsqueeze(-2).expand(-1, seq_len, -1)),
                (start_labels.unsqueeze(-1).expand(-1, -1, seq_len)
                 & end_labels.unsqueeze(-2).expand(-1, seq_len, -1))
            )
        match_label_mask = match_label_mask & match_candidates
        float_match_label_mask = match_label_mask.view(batch_size_, -1).float()

    start_loss = bce_loss(start_logits.view(-1), start_labels.view(-1).float())
    start_loss = (start_loss * start_float_label_mask).sum() / start_float_label_mask.sum()
    end_loss = bce_loss(end_logits.view(-1), end_labels.view(-1).float())
    end_loss = (end_loss * end_float_label_mask).sum() / end_float_label_mask.sum()
    match_loss = bce_loss(span_logits.view(batch_size_, -1), match_labels.view(batch_size_, -1).float())
    match_loss = match_loss * float_match_label_mask
    match_loss = match_loss.sum() / (float_match_label_mask.sum() + 1e-10)

    return start_loss, end_loss, match_loss


def query_span_f1(start_preds, end_preds, match_logits, start_label_mask, end_label_mask, match_labels, flat=False):
    """
    Compute span f1 according to query-based model output
    Args:
        start_preds: [bsz, seq_len]
        end_preds: [bsz, seq_len]
        match_logits: [bsz, seq_len, seq_len]
        start_label_mask: [bsz, seq_len]
        end_label_mask: [bsz, seq_len]
        match_labels: [bsz, seq_len, seq_len]
        flat: if True, decode as flat-ner
    Returns:
        span-f1 counts, tensor of shape [3]: tp, fp, fn
    """
    start_label_mask = start_label_mask.bool()
    end_label_mask = end_label_mask.bool()
    match_labels = match_labels.bool()
    bsz, seq_len = start_label_mask.size()
    # [bsz, seq_len, seq_len]
    match_preds = match_logits > 0
    # [bsz, seq_len]
    start_preds = start_preds.bool()
    # [bsz, seq_len]
    end_preds = end_preds.bool()

    match_preds = (match_preds
                   & start_preds.unsqueeze(-1).expand(-1, -1, seq_len)
                   & end_preds.unsqueeze(1).expand(-1, seq_len, -1))
    match_label_mask = (start_label_mask.unsqueeze(-1).expand(-1, -1, seq_len)
                        & end_label_mask.unsqueeze(1).expand(-1, seq_len, -1))
    match_label_mask = torch.triu(match_label_mask, 0)  # start should be less or equal to end
    match_preds = match_label_mask & match_preds

    tp = (match_labels & match_preds).long().sum()
    fp = (~match_labels & match_preds).long().sum()
    fn = (match_labels & ~match_preds).long().sum()

    precision = tp / (tp+fp)
    recall = tp / (tp+fn)
    f1 = 2*precision*recall / (precision+recall)
    res = {
        'precision': precision.item(),
        'recall': recall.item(),
        'f1': f1.item()
    }
    return res


def train(model, optimizer, train_dataloader, eval_dataloader):
    model.zero_grad()

    best_f1 = 0.0
    best_epoch = 0
    for epoch in range(epochs):
        model.train()
        tr_loss = 0.0
        global_step = 0.0
        pbar = tqdm()
        for step, batch in enumerate(train_dataloader):
            (input_ids, attention_masks, token_type_ids, start_labels, end_labels, start_label_mask,
             end_label_mask, match_labels, sample_idx, label_idx) = batch

            inputs = {
                'input_ids': input_ids.to(device),
                'attention_mask': attention_masks.to(device),
                'token_type_ids': token_type_ids.to(device)
            }
            outputs = model(**inputs)
            start_logits = outputs['start_logits']
            end_logits = outputs['end_logits']
            span_logits = outputs['span_logits']

            start_loss, end_loss, match_loss = compute_loss(
                start_logits=start_logits,
                end_logits=end_logits,
                span_logits=span_logits,
                start_labels=start_labels.to(device),
                end_labels=end_labels.to(device),
                match_labels=match_labels.to(device),
                start_label_mask=start_label_mask.to(device),
                end_label_mask=end_label_mask.to(device)
            )
            weight_sum = weight_start + weight_end + weight_span
            ws = weight_start / weight_sum
            we = weight_end / weight_sum
            wsp = weight_span / weight_sum

            total_loss = ws * start_loss + we * end_loss + wsp * match_loss
            total_loss.backward()
            train_res = {
                'train_total_loss': total_loss.item(),
                'start_loss': start_loss.item(),
                'end_loss': end_loss.item(),
                'match_loss': match_loss.item()
            }
            pbar.update()
            pbar.set_description(
                f"Training epoch {epoch}, result: {json.dumps(train_res, ensure_ascii=False)}"
            )

            tr_loss += total_loss.item()
            global_step += 1
            if (step + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                model.zero_grad()

        print("***********************evaluating***********************")
        eval_res = evaluate(model, eval_dataloader)
        print(json.dumps(eval_res, ensure_ascii=False, indent=2))
        if eval_res['span_f1_stats']['f1'] > best_f1:
            best_f1 = eval_res["span_f1_stats"]['f1']
            best_epoch = epoch

            save_dir = os.path.join(output_dir, 'best_model')
            model_to_save = (
                model.module if hasattr(model, "module") else model
            )  # Take care of distributed/parallel training
            model_to_save.save_pretrained(save_dir)

    return best_epoch, best_f1


def evaluate(model, eval_dataloader):
    model.eval()

    for step, batch in tqdm(enumerate(eval_dataloader), desc="Evaluating"):
        (input_ids, attention_masks, token_type_ids, start_labels, end_labels, start_label_mask,
         end_label_mask, match_labels, sample_idx, label_idx) = batch
        inputs = {
            'input_ids': input_ids.to(device),
            'attention_mask': attention_masks.to(device),
            'token_type_ids': token_type_ids.to(device)
        }

        with torch.no_grad():
            outputs = model(**inputs)

        start_logits = outputs['start_logits']
        end_logits = outputs['end_logits']
        span_logits = outputs['span_logits']

        start_loss, end_loss, match_loss = compute_loss(
            start_logits=start_logits,
            end_logits=end_logits,
            span_logits=span_logits,
            start_labels=start_labels.to(device),
            end_labels=end_labels.to(device),
            match_labels=match_labels.to(device),
            start_label_mask=start_label_mask.to(device),
            end_label_mask=end_label_mask.to(device)
        )
        weight_sum = weight_start + weight_end + weight_span
        ws = weight_start / weight_sum
        we = weight_end / weight_sum
        wsp = weight_span / weight_sum

        total_loss = ws * start_loss + we * end_loss + wsp * match_loss
        start_preds, end_preds = start_logits > 0, end_logits > 0

        span_f1_stats = query_span_f1(
            start_preds=start_preds, end_preds=end_preds, match_logits=span_logits,
            start_label_mask=start_label_mask.to(device), end_label_mask=end_label_mask.to(device),
            match_labels=match_labels.to(device)
        )
        results = {
            'val_loss': total_loss.item(),
            'val_start_loss': start_loss.item(),
            'val_end_loss': end_loss.item(),
            'val_match_loss': match_loss.item(),
            'span_f1_stats': span_f1_stats
        }
        return results


def extract_nested_spans(start_preds, end_preds, match_preds, start_label_mask, end_label_mask, pseudo_tag="TAG"):
    start_label_mask = start_label_mask.bool()
    end_label_mask = end_label_mask.bool()
    bsz, seq_len = start_label_mask.size()
    start_preds = start_preds.bool()
    end_preds = end_preds.bool()

    match_preds = (match_preds & start_preds.unsqueeze(-1).expand(-1, -1, seq_len) &
                   end_preds.unsqueeze(1).expand(-1, seq_len, -1))
    match_label_mask = (start_label_mask.unsqueeze(-1).expand(-1, -1, seq_len) &
                        end_label_mask.unsqueeze(1).expand(-1, seq_len, -1))
    match_label_mask = torch.triu(match_label_mask, 0)  # start should be less or equal to end
    match_preds = match_label_mask & match_preds
    match_pos_pairs = np.transpose(np.nonzero(match_preds.numpy())).tolist()
    return [(pos[0], pos[1], pseudo_tag) for pos in match_pos_pairs]


class Tag(object):
    def __init__(self, term, tag, begin, end):
        self.term = term
        self.tag = tag
        self.begin = begin
        self.end = end

    def to_tuple(self):
        return tuple([self.term, self.begin, self.end])

    def __str__(self):
        return str({key: value for key, value in self.__dict__.items()})

    def __repr__(self):
        return str({key: value for key, value in self.__dict__.items()})


def bmes_decode(char_label_list: List[Tuple[str, str]]) -> List[Tag]:
    """
    decode inputs to tags
    Args:
        char_label_list: list of tuple (word, bmes-tag)
    Returns:
        tags
    Examples:
        >>> x = [("Hi", "O"), ("Beijing", "S-LOC")]
        >>> bmes_decode(x)
        [{'term': 'Beijing', 'tag': 'LOC', 'begin': 1, 'end': 2}]
    """
    idx = 0
    length = len(char_label_list)
    tags = []
    while idx < length:
        term, label = char_label_list[idx]
        current_label = label[0]

        # correct labels
        if idx + 1 == length and current_label == "B":
            current_label = "S"

        # merge chars
        if current_label == "O":
            idx += 1
            continue
        if current_label == "S":
            tags.append(Tag(term, label[2:], idx, idx + 1))
            idx += 1
            continue
        if current_label == "B":
            end = idx + 1
            while end + 1 < length and char_label_list[end][1][0] == "M":
                end += 1
            if char_label_list[end][1][0] == "E":  # end with E
                entity = "".join(char_label_list[i][0] for i in range(idx, end + 1))
                tags.append(Tag(entity, label[2:], idx, end + 1))
                idx = end + 1
            else:  # end with M/B
                entity = "".join(char_label_list[i][0] for i in range(idx, end))
                tags.append(Tag(entity, label[2:], idx, end))
                idx = end
            continue
        else:
            raise Exception("Invalid Inputs")
    return tags


def extract_flat_spans(start_pred, end_pred, match_pred, label_mask, pseudo_tag = "TAG"):
    """
    Extract flat-ner spans from start/end/match logits
    Args:
        start_pred: [seq_len], 1/True for start, 0/False for non-start
        end_pred: [seq_len, 2], 1/True for end, 0/False for non-end
        match_pred: [seq_len, seq_len], 1/True for match, 0/False for non-match
        label_mask: [seq_len], 1 for valid boundary.
    Returns:
        tags: list of tuple (start, end)
    Examples:
        >>> start_pred = [0, 1]
        >>> end_pred = [0, 1]
        >>> match_pred = [[0, 0], [0, 1]]
        >>> label_mask = [1, 1]
        >>> extract_flat_spans(start_pred, end_pred, match_pred, label_mask)
        [(1, 2)]
    """
    pseudo_input = "a"

    bmes_labels = ["O"] * len(start_pred)
    start_positions = [idx for idx, tmp in enumerate(start_pred) if tmp and label_mask[idx]]
    end_positions = [idx for idx, tmp in enumerate(end_pred) if tmp and label_mask[idx]]

    for start_item in start_positions:
        bmes_labels[start_item] = f"B-{pseudo_tag}"
    for end_item in end_positions:
        bmes_labels[end_item] = f"E-{pseudo_tag}"

    for tmp_start in start_positions:
        tmp_end = [tmp for tmp in end_positions if tmp >= tmp_start]
        if len(tmp_end) == 0:
            continue
        else:
            tmp_end = min(tmp_end)
        if match_pred[tmp_start][tmp_end]:
            if tmp_start != tmp_end:
                for i in range(tmp_start+1, tmp_end):
                    bmes_labels[i] = f"M-{pseudo_tag}"
            else:
                bmes_labels[tmp_end] = f"S-{pseudo_tag}"

    tags = bmes_decode([(pseudo_input, label) for label in bmes_labels])

    return [(entity.begin, entity.end, entity.tag) for entity in tags]


def remove_overlap(spans):
    """
    remove overlapped spans greedily for flat-ner
    Args:
        spans: list of tuple (start, end), which means [start, end] is a ner-span
    Returns:
        spans without overlap
    """
    output = []
    occupied = set()
    for start, end in spans:
        if any(x for x in range(start, end+1)) in occupied:
            continue
        output.append((start, end))
        for x in range(start, end + 1):
            occupied.add(x)
    return output


def main():
    mrc_model = BertQueryNER.from_pretrained(bert_model_name_or_path, config=bert_config)
    mrc_model.to(device)
    optimizer = AdamW(mrc_model.parameters(), lr=lr_rate)

    train_dataloader = load_features('train')
    eval_dataloader = load_features('dev')
    best_epoch, best_f1 = train(mrc_model, optimizer, train_dataloader, eval_dataloader)
    print(f"best f1: {best_f1}\tbest epoch: {best_epoch}")


if __name__ == "__main__":
    main()
