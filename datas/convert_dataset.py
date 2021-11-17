# -*- coding: utf8 -*-
"""
======================================
    Project Name: NLP
    File Name: convert_dataset
    Author: czh
    Create Date: 2021/8/19
--------------------------------------
    Change Activity: 
======================================
"""
# 将其它数据转换成tplinker plus要求的格式
import sys

sys.path.append("../experiments/")
import os
import json
import re
import copy
from transformers import BertTokenizerFast
from sklearn.model_selection import train_test_split
from tqdm import tqdm

config = {
    "encoder": "BERT",
    "bert_path": "/Users/czh/Downloads/nezha-base-www",
}


class Preprocessor:
    """
    1. transform the dataset to normal format, which can fit in our codes
    2. add token level span to all entities in the entities, which will be used in tagging phase
    """

    def __init__(self, tokenize_func, get_tok2char_span_map_func):
        self._tokenize = tokenize_func
        self._get_tok2char_span_map = get_tok2char_span_map_func

    def transform_data(self, data, ori_format, dataset_type, add_id=True):
        """
        NOTE:
        data格式要求：
        [
            {
                "text":"原始语句",
                "entity_list":[{"text":"实体","type":"实体类型","char_span":"实体char级别的span","token_span":"实体token级别的span"}]
            }
        ]
        原始的tplinker在此处做了一个根据不同数据集格式读取相应的subject,object等信息。但在此精简了这个功能，私以为此功能应在数据预处理（整理数据集）阶段完成。
        所有此处的ori_format无效
        """
        normal_sample_list = []
        # NewFeature:
        # 增加entity_list，引入实体类别信息
        normal_entity_list = None
        for ind, sample in tqdm(enumerate(data), desc="Transforming data format"):

            text = sample["text"]
            normal_entity_list = sample["entity_list"]

            normal_sample = {
                "text": text,
            }
            if add_id:
                normal_sample["id"] = "{}_{}".format(dataset_type, ind)
            # NewFeature: 新增获取实体列表和类型信息
            if normal_entity_list:
                normal_sample["entity_list"] = normal_entity_list
            normal_sample_list.append(normal_sample)

        return self._clean_sp_char(normal_sample_list)

    def split_into_short_samples(self, sample_list, max_seq_len, sliding_len=50, encoder="BERT", data_type="train"):
        new_sample_list = []
        for sample in tqdm(sample_list, desc="Splitting into subtexts"):
            text_id = sample["id"]
            text = sample["text"]
            tokens = self._tokenize(text)
            tok2char_span = self._get_tok2char_span_map(text)

            # sliding at token level
            split_sample_list = []
            for start_ind in range(0, len(tokens), sliding_len):
                if encoder == "BERT":  # if use bert, do not split a word into two samples
                    while "##" in tokens[start_ind]:
                        start_ind -= 1
                end_ind = start_ind + max_seq_len  # start_ind和end_ind是token的起止index

                char_span_list = tok2char_span[start_ind:end_ind]
                char_level_span = [char_span_list[0][0], char_span_list[-1][1]]  # char_level_span是char的起止
                sub_text = text[char_level_span[0]:char_level_span[1]]

                new_sample = {
                    "id": text_id,
                    "text": sub_text,
                    "tok_offset": start_ind,
                    "char_offset": char_level_span[0],
                }
                if data_type == "test":  # test set
                    if len(sub_text) > 0:
                        split_sample_list.append(new_sample)
                else:  # train or valid dataset, only save entities in the subtext
                    # entity
                    sub_ent_list = []
                    for ent in sample["entity_list"]:
                        tok_span = ent["tok_span"]
                        # if entity in this subtext, add the entity to new sample
                        if tok_span[0] >= start_ind and tok_span[1] <= end_ind:
                            new_ent = copy.deepcopy(ent)
                            new_ent["tok_span"] = [tok_span[0] - start_ind, tok_span[1] - start_ind]

                            new_ent["char_span"][0] -= char_level_span[0]
                            new_ent["char_span"][1] -= char_level_span[0]

                            sub_ent_list.append(new_ent)

                    new_sample["entity_list"] = sub_ent_list  # maybe empty
                    split_sample_list.append(new_sample)

                # all segments covered, no need to continue
                if end_ind > len(tokens):
                    break

            new_sample_list.extend(split_sample_list)
        return new_sample_list

    def _clean_sp_char(self, dataset):
        def clean_text(text):
            text = re.sub("�", "", text)
            #             text = re.sub("([A-Za-z]+)", r" \1 ", text)
            #             text = re.sub("(\d+)", r" \1 ", text)
            #             text = re.sub("\s+", " ", text).strip()
            return text

        for sample in tqdm(dataset, desc="Clean"):
            sample["text"] = clean_text(sample["text"])
            for ent in sample["entity_list"]:
                ent["text"] = clean_text(ent["text"])
        return dataset

    def clean_data_wo_span(self, ori_data, separate=False, data_type="train"):
        """
        rm duplicate whitespaces
        and add whitespaces around tokens to keep special characters from them
        """

        def clean_text(text):
            text = re.sub("\s+", " ", text).strip()
            if separate:
                text = re.sub("([^A-Za-z0-9])", r" \1 ", text)
                text = re.sub("\s+", " ", text).strip()
            return text

        for sample in tqdm(ori_data, desc="clean data"):
            sample["text"] = clean_text(sample["text"])
            if data_type == "test":
                continue
            for ent in sample["entity_list"]:
                ent["text"] = clean_text(ent["text"])
        return ori_data

    def clean_data_w_span(self, ori_data):
        """
        stripe whitespaces and change spans
        add a stake to bad samples(char span error) and remove them from the clean data
        """
        bad_samples, clean_data = [], []

        def strip_white(entity, entity_char_span):
            p = 0
            while entity[p] == " ":
                entity_char_span[0] += 1
                p += 1

            p = len(entity) - 1
            while entity[p] == " ":
                entity_char_span[1] -= 1
                p -= 1
            return entity.strip(), entity_char_span

        for sample in tqdm(ori_data, desc="clean data w char spans"):
            text = sample["text"]

            bad = False
            for ent in sample["entity_list"]:
                # rm whitespaces
                ent["text"], ent["char_span"] = strip_white(ent["text"], ent["char_span"])

                char_span = ent["char_span"]
                if ent["text"] not in text or ent["text"] != text[char_span[0]:char_span[1]]:
                    ent["stake"] = 0
                    bad = True

            if bad:
                bad_samples.append(copy.deepcopy(sample))

            new_ent_list = [ent for ent in sample["entity_list"] if "stake" not in ent]
            if len(new_ent_list) > 0:
                sample["entity_list"] = new_ent_list
                clean_data.append(sample)
        return clean_data, bad_samples

    def _get_char2tok_span(self, text):
        """
        map character index to token level span
        """
        tok2char_span = self._get_tok2char_span_map(text)
        char_num = None
        for tok_ind in range(len(tok2char_span) - 1, -1, -1):
            if tok2char_span[tok_ind][1] != 0:
                char_num = tok2char_span[tok_ind][1]
                break
        char2tok_span = [[-1, -1] for _ in range(char_num)]  # [-1, -1] is whitespace
        for tok_ind, char_sp in enumerate(tok2char_span):
            for char_ind in range(char_sp[0], char_sp[1]):
                tok_sp = char2tok_span[char_ind]
                # 因为char to tok 也可能出现1对多的情况，比如韩文。所以char_span的pos1以第一个tok_ind为准，pos2以最后一个tok_ind为准
                if tok_sp[0] == -1:
                    tok_sp[0] = tok_ind
                tok_sp[1] = tok_ind + 1
        return char2tok_span

    def _get_ent2char_spans(self, text, entities, ignore_subword_match=True):
        """
        if ignore_subword_match is true, find entities with whitespace around, e.g. "entity" -> " entity "
        """
        entities = sorted(entities, key=lambda x: len(x), reverse=True)
        text_cp = " {} ".format(text) if ignore_subword_match else text
        ent2char_spans = {}
        for ent in entities:
            spans = []
            target_ent = " {} ".format(ent) if ignore_subword_match else ent
            for m in re.finditer(re.escape(target_ent), text_cp):
                if not ignore_subword_match and re.match("\d+",
                                                         target_ent):  # avoid matching a inner number of a number
                    if (m.span()[0] - 1 >= 0 and re.match("\d", text_cp[m.span()[0] - 1])) or (
                            m.span()[1] < len(text_cp) and re.match("\d", text_cp[m.span()[1]])):
                        continue
                span = [m.span()[0], m.span()[1] - 2] if ignore_subword_match else m.span()
                spans.append(span)
            #             if len(spans) == 0:
            #                 set_trace()
            ent2char_spans[ent] = spans
        return ent2char_spans

    def add_char_span(self, dataset, ignore_subword_match=True):
        # print(ignore_subword_match)
        # return
        miss_sample_list = []
        for sample in tqdm(dataset, desc="adding char level spans"):
            entities = [ent["text"] for ent in sample["entity_list"]]
            ent2char_spans = self._get_ent2char_spans(sample["text"], entities,
                                                      ignore_subword_match=ignore_subword_match)

            new_entity_list = []
            for ent in sample["entity_list"]:
                char_spans = ent2char_spans[ent["text"]]
                for char_sp in char_spans:
                    new_entity_list.append({
                        "text": ent["text"],
                        "char_span": char_sp,
                        "type": ent["type"],
                    })

            if len(sample["entity_list"]) > len(new_entity_list):
                miss_sample_list.append(sample)
            sample["entity_list"] = new_entity_list

        return dataset, miss_sample_list

    def add_tok_span(self, dataset):
        """
        dataset must has char level span
        """

        def char_span2tok_span(char_span, char2tok_span):
            tok_span_list = char2tok_span[char_span[0]:char_span[1]]
            tok_span = [tok_span_list[0][0], tok_span_list[-1][1]]
            return tok_span

        for sample in tqdm(dataset, desc="adding token level spans"):
            text = sample["text"]
            char2tok_span = self._get_char2tok_span(sample["text"])
            for ent in sample["entity_list"]:
                char_span = ent["char_span"]
                ent["tok_span"] = char_span2tok_span(char_span, char2tok_span)
        return dataset


def handle_normal_dataset(dataset, ignore_subword_match=False):
    """
    if ignore_subword_match is true, find entities with whitespace around, e.g. "entity" -> " entity "
    """
    # 加载preprocessor
    if config["encoder"] == "BERT":
        tokenizer = BertTokenizerFast.from_pretrained(
            config["bert_path"], add_special_tokens=False, do_lower_case=False
        )
        tokenize = tokenizer.tokenize
        get_tok2char_span_map = lambda text: tokenizer.encode_plus(
            text, return_offsets_mapping=True, add_special_tokens=False
        )["offset_mapping"]
    elif config["encoder"] == "BiLSTM":
        tokenize = lambda text: text.split(" ")

        def get_tok2char_span_map(text):
            tokens = tokenize(text)
            tok2char_span = []
            char_num = 0
            for tok in tokens:
                tok2char_span.append((char_num, char_num + len(tok)))
                char_num += len(tok) + 1  # +1: whitespace
            return tok2char_span

    preprocessor = Preprocessor(
        tokenize_func=tokenize, get_tok2char_span_map_func=get_tok2char_span_map
    )
    # add char span
    dataset, miss_sample_list = preprocessor.add_char_span(
        dataset, ignore_subword_match=False
    )

    if len(miss_sample_list) > 0:
        print("=========存在不匹配实体，请检查===========")
        print(miss_sample_list)
        print("========================================")

    # add token span
    dataset = preprocessor.add_tok_span(dataset)

    return dataset


def main(data_path, data_type, out_dir, split_train_valid=False, split_radio=0.8, shuffle=True, test_label=False):
    """
    先将数据集先整理为预处理数据集的指定格式：
    [
        {
            "text":"2008年奥运会在北京举行。",
            "entity_list":[
                {"text":"2008","type":"year"},
                {"text":"北京","type":"loc"}
                ]
        }
    ]
    """

    cnt = 0
    ent_type_set = set()
    new_datas = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            item = {}
            data = json.loads(line)
            text = data["text"]

            item["id"] = cnt
            item["text"] = text
            cnt += 1
            if "test" in data_type and not test_label:
                new_datas.append(item)
                continue

            labels = data["label"]
            item["entity_list"] = []
            for ent_type in labels:
                ent_type_set.add(ent_type)
                for ent_text in labels[ent_type]:
                    item["entity_list"].append({"text": ent_text, "type": ent_type})
            new_datas.append(item)

    out_path = os.path.join(out_dir, data_type + ".json")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    if "test" in data_type and not test_label:
        json.dump(new_datas, open(out_path, "w", encoding="utf-8"), ensure_ascii=False)
        return

    if "train" in data_type:
        # 添加char_span,token_span成最终数据集
        dataset = handle_normal_dataset(new_datas, ignore_subword_match=False)
        if not split_train_valid:
            json.dump(dataset, open(out_path, "w", encoding="utf-8"), ensure_ascii=False)
        else:
            train_dataset, valid_dataset = train_test_split(dataset, train_size=split_radio, shuffle=shuffle)
            json.dump(train_dataset, open(os.path.join(out_dir, "train_data.json"), "w", encoding="utf-8"),
                      ensure_ascii=False)
            json.dump(valid_dataset, open(os.path.join(out_dir, "valid_data.json"), "w", encoding="utf-8"),
                      ensure_ascii=False)

        # 保存实体类型信息
        ent_type_list = sorted(ent_type_set)
        ent2id = {ent: ind for ind, ent in enumerate(ent_type_list)}
        json.dump(
            ent2id, open(os.path.join(out_dir, "ent2id.json"), "w", encoding="utf-8"), ensure_ascii=False
        )
    else:
        # 添加char_span,token_span成最终数据集
        dataset = handle_normal_dataset(new_datas, ignore_subword_match=False)
        json.dump(dataset, open(out_path, "w", encoding="utf-8"), ensure_ascii=False)


if __name__ == "__main__":
    data_dir = "/datas/"
    out_dir = data_dir + "/tplinker/"
    train_path = data_dir + "/ner/train.json"
    valid_path = data_dir + "/ner/dev.json"
    test_path = data_dir + "/ner/test.json"
    main(train_path, "train_data", out_dir)
    main(valid_path, "valid_data", out_dir)
    main(test_path, "test_data", out_dir, test_label=True)
