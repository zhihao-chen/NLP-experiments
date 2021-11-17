# -*- coding: utf8 -*-
"""
======================================
    Project Name: NLP
    File Name: utils_ee
    Author: czh
    Create Date: 2021/9/6
--------------------------------------
    Change Activity: 
======================================
"""
# 事件抽取util
import codecs
import re
import copy
import json
from pathlib import Path
from typing import List, Dict, Union

from seqeval.metrics.sequence_labeling import get_entities
from torch.utils.data import Dataset

from nlp.processors.utils_ner import get_entities as extract_entities
from nlp.models.model_util import Indexer


def filter_annotations(sample, start_ind, end_ind):
    """
    filter annotations in [start_ind, end_ind]
    :param sample:
    :param start_ind:
    :param end_ind:
    :return:
    """
    filter_res = {}
    limited_span = [start_ind, end_ind]

    if "relation_list" in sample:
        filter_res["relation_list"] = filter_spans(sample["relation_list"], start_ind, end_ind)

    if "entity_list" in sample:
        filter_res["entity_list"] = filter_spans(sample["entity_list"], start_ind, end_ind)

    if "event_list" in sample:
        sub_event_list = []
        for event in sample["event_list"]:
            event_cp = copy.deepcopy(event)
            if "trigger" in event_cp:
                trigger_tok_span = event["trigger_tok_span"]
                trigger_ch_span = event["trigger_char_span"]
                if type(trigger_tok_span[0]) is list:
                    new_tok_span = []
                    new_ch_span = []
                    for sp_idx, tok_sp in enumerate(trigger_tok_span):
                        if span_contains(limited_span, tok_sp):
                            new_tok_span.append(tok_sp)
                            new_ch_span.append(trigger_ch_span[sp_idx])
                    event_cp["trigger_tok_span"] = new_tok_span
                    event_cp["trigger_char_span"] = new_ch_span

                    if len(event_cp["trigger_tok_span"]) == 0:
                        del event_cp["trigger"]
                        del event_cp["trigger_tok_span"]
                        del event_cp["trigger_char_span"]
                else:
                    if not span_contains(limited_span, trigger_tok_span):
                        del event_cp["trigger"]
                        del event_cp["trigger_tok_span"]
                        del event_cp["trigger_char_span"]

            new_arg_list = []
            for arg in event_cp["argument_list"]:
                tok_span = arg["tok_span"]
                ch_span = arg["char_span"]
                if type(tok_span[0]) is list:
                    new_tok_span = []
                    new_ch_span = []
                    for sp_idx, tok_sp in enumerate(tok_span):
                        if span_contains(limited_span, tok_sp):
                            new_tok_span.append(tok_sp)
                            new_ch_span.append(ch_span[sp_idx])
                    arg["tok_span"] = new_tok_span
                    arg["char_span"] = new_ch_span

                    if len(arg["tok_span"]) > 0:
                        new_arg_list.append(arg)
                else:
                    if span_contains(limited_span, tok_span):
                        new_arg_list.append(arg)

            new_trigger_list = []
            for trigger in event_cp.get("trigger_list", []):
                tok_span = trigger["tok_span"]
                ch_span = trigger["char_span"]
                if type(tok_span[0]) is list:
                    new_tok_span = []
                    new_ch_span = []
                    for sp_idx, tok_sp in enumerate(tok_span):
                        if span_contains(limited_span, tok_sp):
                            new_tok_span.append(tok_sp)
                            new_ch_span.append(ch_span[sp_idx])
                    trigger["tok_span"] = new_tok_span
                    trigger["char_span"] = new_ch_span

                    if len(trigger["tok_span"]) > 0:
                        new_trigger_list.append(trigger)
                else:
                    if span_contains(limited_span, tok_span):
                        new_trigger_list.append(trigger)

            if len(new_arg_list) > 0 or "trigger" in event_cp:
                event_cp["argument_list"] = new_arg_list
                if len(new_trigger_list) > 0:
                    event_cp["trigger_list"] = new_trigger_list
                sub_event_list.append(event_cp)
        filter_res["event_list"] = sub_event_list

    if "open_spo_list" in sample:
        sub_open_spo_list = []
        for spo in sample["open_spo_list"]:
            new_spo = []
            bad_spo = False
            for arg in spo:
                if span_contains(limited_span, arg["tok_span"]):
                    new_spo.append(arg)
                elif not span_contains(limited_span, arg["tok_span"]) \
                        and arg["type"] in {"predicate", "object", "subject"}:
                    bad_spo = True
                    break
            if not bad_spo:
                sub_open_spo_list.append(new_spo)

        filter_res["open_spo_list"] = sub_open_spo_list
    return filter_res


def filter_spans(inp_list, start_ind, end_ind):
    limited_span = [start_ind, end_ind]
    filter_res = []
    for item in inp_list:
        if any("tok_span" in k and not span_contains(limited_span, v) for k, v in item.items()):
            pass
        else:
            filter_res.append(item)

    return filter_res


def span_offset(sample_spans, tok_level_offset, char_level_offset):
    """
    add offset
    :param sample_spans:
    :param tok_level_offset:
    :param char_level_offset:
    :return:
    """

    def list_add(ori_list, add_num):
        if len(ori_list) > 0 and type(ori_list[0]) is list:
            return [[e + add_num for e in sub_list] for sub_list in ori_list]
        else:
            return [e + add_num for e in ori_list]

    annotations = {}
    if "relation_list" in sample_spans:
        annotations["relation_list"] = copy.deepcopy(sample_spans["relation_list"])
        for rel in annotations["relation_list"]:
            rel["subj_tok_span"] = list_add(rel["subj_tok_span"], tok_level_offset)
            rel["obj_tok_span"] = list_add(rel["obj_tok_span"], tok_level_offset)
            rel["subj_char_span"] = list_add(rel["subj_char_span"], char_level_offset)
            rel["obj_char_span"] = list_add(rel["obj_char_span"], char_level_offset)

    if "entity_list" in sample_spans:
        annotations["entity_list"] = copy.deepcopy(sample_spans["entity_list"])
        for ent in annotations["entity_list"]:
            ent["tok_span"] = list_add(ent["tok_span"], tok_level_offset)
            ent["char_span"] = list_add(ent["char_span"], char_level_offset)

    if "event_list" in sample_spans:
        annotations["event_list"] = copy.deepcopy(sample_spans["event_list"])
        for event in annotations["event_list"]:
            if "trigger" in event:
                event["trigger_tok_span"] = list_add(event["trigger_tok_span"], tok_level_offset)
                event["trigger_char_span"] = list_add(event["trigger_char_span"], char_level_offset)
            for arg in event["argument_list"]:
                arg["tok_span"] = list_add(arg["tok_span"], tok_level_offset)
                arg["char_span"] = list_add(arg["char_span"], char_level_offset)
    if "open_spo_list" in sample_spans:
        annotations["open_spo_list"] = copy.deepcopy(sample_spans["open_spo_list"])
        for spo in annotations["open_spo_list"]:
            for arg in spo:
                arg["tok_span"] = list_add(arg["tok_span"], tok_level_offset)
                arg["char_span"] = list_add(arg["char_span"], char_level_offset)
    return annotations


class MyDatasets(Dataset):
    def __init__(self, datasets, max_seq_length=512, data_type='train',
                 bert_vocab_dict=None, additional_preprocess=None):
        self.token_level = "subword"
        self.max_seq_length = max_seq_length
        self.slid_len = max_seq_length
        self.data_type = data_type
        self.additional_preprocess = additional_preprocess

        self.key2dict = {"subword_list": bert_vocab_dict}

        self.data = self.__convert_to_features(datasets)

    def __len__(self):
        return len(self.data)

    def get_data_anns(self):
        data_anns = {}
        for sample in self.data:
            for k, v in sample.items():
                if k in {"entity_list", "relation_list", "event_list", "open_spo_list"}:
                    data_anns.setdefault(k, []).extend(v)
        return data_anns

    def __convert_to_features(self, datas):
        index_datas = []
        for i, sample in enumerate(datas):
            sample = self.choose_features_by_token_level4sample(sample, token_level=self.token_level)
            sample = self.choose_spans_by_token_level4sample(sample, self.token_level)
            if self.additional_preprocess is not None:
                sample = self.additional_preprocess(sample)
            samples = self.split_into_short_samples(sample, max_seq_len=self.max_seq_length, sliding_len=self.slid_len,
                                                    data_type=self.data_type, token_level=self.token_level)
            index_samples = self.index_features(data=samples, language='ch', key2dict=self.key2dict,
                                                max_seq_len=self.max_seq_length)
            for index_sample in index_samples:
                index_datas.append(index_sample)
        return index_datas

    def __getitem__(self, idx):
        sample = self.data[idx]

        return sample

    @staticmethod
    def index_features(data, language, key2dict, max_seq_len, max_char_num_in_tok=None):
        """
        :param language:
        :param data:
        :param key2dict: feature key to dict for indexing
        :param max_seq_len:
        :param max_char_num_in_tok: max character number in a token, truncate or pad to this length
        :return:
        """
        # map for replacing key names
        key_map = {
            "subword_list": "subword_input_ids",
        }

        for i, sample in enumerate(data):
            features = sample["features"]
            features["token_type_ids"] = [0] * len(features["tok2char_span"])
            features["attention_mask"] = [1] * len(features["tok2char_span"])
            features["char_list"] = list(sample["text"])

            sep = " " if language == "en" else ""
            features["word_list"] += ["[PAD]"] * (max_seq_len - len(features["word_list"]))
            fin_features = {
                "padded_text": sep.join(features["word_list"]),
                "word_list": features["word_list"],
            }
            for f_key, tags in features.items():
                # features need indexing and padding
                if f_key in key2dict.keys():
                    tag2id = key2dict[f_key]

                    if f_key == "ner_tag_list":
                        spe_tag_dict = {"[UNK]": tag2id["O"], "[PAD]": tag2id["O"]}
                    else:
                        spe_tag_dict = {"[UNK]": tag2id["[UNK]"], "[PAD]": tag2id["[PAD]"]}

                    indexer = Indexer(tag2id, max_seq_len, spe_tag_dict)
                    if f_key == "dependency_list":
                        fin_features[key_map[f_key]] = indexer.index_tag_list_w_matrix_pos(tags)
                    elif f_key == "char_list" and max_char_num_in_tok is not None:
                        char_input_ids = indexer.index_tag_list(tags)
                        # padding character ids
                        char_input_ids_padded = []
                        for span in features["tok2char_span"]:
                            char_ids = char_input_ids[span[0]:span[1]]

                            if len(char_ids) < max_char_num_in_tok:
                                char_ids.extend([0] * (max_char_num_in_tok - len(char_ids)))
                            else:
                                char_ids = char_ids[:max_char_num_in_tok]
                            char_input_ids_padded.extend(char_ids)
                        fin_features[key_map[f_key]] = char_input_ids_padded  # torch.LongTensor(char_input_ids_padded)
                    else:
                        fin_features[key_map[f_key]] = indexer.index_tag_list(
                            tags)  # torch.LongTensor(indexer.index_tag_list(tags))

                # features only need padding
                elif f_key in {"token_type_ids", "attention_mask"}:
                    # torch.LongTensor(Indexer.pad2length(tags, 0, max_seq_len))
                    fin_features[f_key] = Indexer.pad2length(tags, 0,
                                                             max_seq_len)
                elif f_key == "tok2char_span":
                    fin_features[f_key] = Indexer.pad2length(tags, [0, 0], max_seq_len)
                elif f_key == "pos_tag_list_csp":
                    tag2id = key2dict["pos_tag_list"]
                    fin_features["pos_tag_points"] = list(
                        {(pos["tok_span"][0], pos["tok_span"][1] - 1, tag2id[pos["type"]])
                         for pos in tags})
                elif f_key == "dependency_list_csp":
                    tag2id = key2dict["dependency_list"]
                    t2t_points = list(
                        {(deprel["subj_tok_span"][1] - 1, deprel["obj_tok_span"][1] - 1,
                          tag2id[deprel["predicate"]] + len(tag2id)) for deprel in tags}
                    )
                    h2h_points = list(
                        {(deprel["subj_tok_span"][0], deprel["obj_tok_span"][0],
                          tag2id[deprel["predicate"]]) for deprel in tags}
                    )
                    fin_features["deprel_points_hnt"] = h2h_points + t2t_points

            sample["features"] = fin_features
            yield sample

    @staticmethod
    def choose_features_by_token_level4sample(sample, token_level='subword', do_lower_case=False):
        features = sample["features"]
        if token_level == "subword":
            subword2word_id = features["subword2word_id"]
            new_features = {
                "subword_list": features["subword_list"],
                "tok2char_span": features["subword2char_span"],
                "word_list": [features["word_list"][wid] for wid in subword2word_id],
            }
            if "ner_tag_list" in features:
                new_features["ner_tag_list"] = [features["ner_tag_list"][wid] for wid in subword2word_id]
            if "pos_tag_list" in features:
                new_features["pos_tag_list"] = [features["pos_tag_list"][wid] for wid in subword2word_id]
            if "subword_dependency_list" in features:
                new_features["dependency_list"] = features["subword_dependency_list"]
            if "pos_tag_list_csp" in features:
                new_features["pos_tag_list_csp"] = features["pos_tag_list_csp"]
            if "dependency_list_csp" in features:
                new_features["dependency_list_csp"] = features["dependency_list_csp"]
            sample["features"] = new_features
        else:
            subwd_list = [w.lower() for w in features["word_list"]] if do_lower_case else features["word_list"]
            new_features = {
                "word_list": features["word_list"],
                "subword_list": subwd_list,
                "tok2char_span": features["word2char_span"],
            }
            if "ner_tag_list" in features:
                new_features["ner_tag_list"] = features["ner_tag_list"]
            if "pos_tag_list" in features:
                new_features["pos_tag_list"] = features["pos_tag_list"]
            if "word_dependency_list" in features:
                new_features["dependency_list"] = features["word_dependency_list"]
            if "pos_tag_list_csp" in features:
                new_features["pos_tag_list_csp"] = features["pos_tag_list_csp"]
            if "dependency_list_csp" in features:
                new_features["dependency_list_csp"] = features["dependency_list_csp"]
            sample["features"] = new_features
        return sample

    @staticmethod
    def choose_spans_by_token_level4sample(sample, token_level):
        tok_key = "subwd_span" if token_level == "subword" else "wd_span"

        def choose_sps(a_list):
            for item in a_list:
                if type(item) is list:
                    choose_sps(item)
                elif type(item) is dict:
                    add_dict = {}
                    for key, value in item.items():
                        if tok_key in key:
                            add_tok_sp_k = re.sub(tok_key, "tok_span", key)
                            add_dict[add_tok_sp_k] = value
                        elif type(value) is list:
                            choose_sps(value)
                    item.update(add_dict)

        for k, v in sample.items():
            if type(v) is list:
                choose_sps(v)
            elif type(v) is dict:
                for val in v.values():
                    if type(val) is list:
                        choose_sps(val)
        return sample

    @staticmethod
    def split_into_short_samples(sample, max_seq_len, sliding_len, data_type, token_level='subword',
                                 wordpieces_prefix="##", early_stop=True,
                                 drop_neg_samples=False):
        """
        split long samples into short samples
        :param sample: original data
        :param max_seq_len: the max sequence length of a subtext
        :param sliding_len: the size of the sliding window
        :param data_type: train, valid, test
        :param token_level:
        :param wordpieces_prefix:
        :param early_stop:
        :param drop_neg_samples:
        :return:
        """

        idx = sample["id"]
        text = sample["text"]
        features = sample["features"]
        tokens = features["subword_list"] if token_level == "subword" else features["word_list"]
        tok2char_span = features["tok2char_span"]
        # split by sliding window
        new_samples = []
        for start_ind in range(0, len(tokens), sliding_len):
            if token_level == "subword":
                while wordpieces_prefix in tokens[start_ind]:
                    start_ind -= 1
            end_ind = start_ind + max_seq_len

            # split text
            char_span_list = tok2char_span[start_ind:end_ind]
            char_span = (char_span_list[0][0], char_span_list[-1][1])
            sub_text = text[char_span[0]:char_span[1]]

            # offsets
            tok_level_offset, char_level_offset = start_ind, char_span[0]

            # split features
            short_word_list = features["word_list"][start_ind:end_ind]
            short_subword_list = features["subword_list"][start_ind:end_ind]
            split_features = {"word_list": short_word_list,
                              "subword_list": short_subword_list,
                              "tok2char_span": [[char_sp[0] - char_level_offset, char_sp[1] - char_level_offset]
                                                for char_sp in features["tok2char_span"][start_ind:end_ind]],
                              }
            if "pos_tag_list" in features:
                split_features["pos_tag_list"] = features["pos_tag_list"][start_ind:end_ind]
            if "ner_tag_list" in features:
                split_features["ner_tag_list"] = features["ner_tag_list"][start_ind:end_ind]
            new_sample = {
                "id": idx,
                "text": sub_text,
                "features": split_features,
                "tok_level_offset": tok_level_offset,
                "char_level_offset": char_level_offset,
            }
            if "entity_list" in sample:
                new_sample["entity_list"] = []
            if "relation_list" in sample:
                new_sample["relation_list"] = []
            if "event_list" in sample:
                new_sample["event_list"] = []
            if "open_spo_list" in sample:
                new_sample["open_spo_list"] = []

            # if train or debug, filter annotations
            if data_type not in {"train", "debug"}:
                if len(sub_text) > 0:
                    new_samples.append(new_sample)
                if end_ind > len(tokens):
                    break
            else:
                # if train data, need to filter annotations in the subtext
                filtered_res = filter_annotations(sample, start_ind, end_ind)
                if "entity_list" in filtered_res:
                    new_sample["entity_list"] = filtered_res["entity_list"]
                if "relation_list" in filtered_res:
                    new_sample["relation_list"] = filtered_res["relation_list"]
                if "event_list" in filtered_res:
                    new_sample["event_list"] = filtered_res["event_list"]
                if "open_spo_list" in filtered_res:
                    new_sample["open_spo_list"] = filtered_res["open_spo_list"]

                # do not introduce excessive negative samples
                if drop_neg_samples and data_type == "train":
                    if ("entity_list" not in new_sample or len(new_sample["entity_list"]) == 0) \
                            and ("relation_list" not in new_sample or len(new_sample["relation_list"]) == 0) \
                            and ("event_list" not in new_sample or len(new_sample["event_list"]) == 0) \
                            and ("open_spo_list" not in new_sample or len(new_sample["open_spo_list"]) == 0):
                        continue

                # offset
                anns = span_offset(new_sample, - tok_level_offset, - char_level_offset)
                new_sample = {**new_sample, **anns}
                new_samples.append(new_sample)
                if early_stop and end_ind > len(tokens):
                    break
        return new_samples


class InputExample(object):
    """A single training/test example for token classification."""

    def __init__(self, guid, text_a, arguments):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            text_a: list. The words of the sequence.
            arguments: dict.
        """
        self.guid = guid
        self.text_a = text_a
        self.arguments = arguments

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, input_len, segment_ids, label_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids
        self.input_len = input_len

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class DataProcessor(object):
    def get_train_examples(self):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @staticmethod
    def unique_list(inp_list):
        out_list = []
        memory = set()
        for item in inp_list:
            mem = str(item)
            if type(item) is dict:
                mem = str(dict(sorted(item.items())))
            if mem not in memory:
                out_list.append(item)
                memory.add(mem)
        return out_list

    @staticmethod
    def clean_text(text):
        text = re.sub(r"�", "", text)
        # text = re.sub("([,;.?!]+)", r" \1 ", text)
        text = re.sub(r"\ufeff", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    # DuEE1.0 苏剑林格式
    @classmethod
    def read_event_schema(cls, file_name_or_path: Union[str, Path] = None, alist: List = None):
        id2label, label2id = {}, {}
        event_type_dict = {}
        n = 0
        datas = []
        if file_name_or_path:
            with codecs.open(file_name_or_path, encoding="utf8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    line = json.loads(line)
                    datas.append(line)
        elif alist:
            datas = alist
        else:
            raise ValueError
        for line in datas:
            event_type = line['event_type']
            if event_type not in event_type_dict:
                event_type_dict[event_type] = {}
            event_type_dict[event_type]["name"] = line['class']
            for role in line['role_list']:
                if "name" in role:
                    name = role["name"]
                else:
                    name = role["role"]
                event_type_dict[event_type][role["role"]] = name
                key = (event_type, role['role'])
                id2label[n] = key
                label2id[key] = n
                n += 1
        num_labels = len(id2label)
        return id2label, label2id, num_labels, event_type_dict

    # DuEE1.0 苏剑林格式
    @staticmethod
    def read_json(file_name_or_path):
        lines = []
        with codecs.open(file_name_or_path, encoding='utf8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                line = json.loads(line)
                arguments = {}
                event_list = line.get("event_list", None)
                if event_list:
                    for event in line['event_list']:
                        for argument in event['arguments']:
                            key = argument['argument']
                            value = (event['event_type'], argument['role'])
                            arguments[key] = value
                # (text, {argument: (event_type, role)})
                lines.append((line['text'], arguments))
        return lines


def get_argument_for_seq(pred_labels, id2label: Dict, suffix=None):
    """
    针对序列标注方式，提取argument, 参考苏剑林的代码https://github.com/bojone/lic2020_baselines/blob/master/ee.py
    :param pred_labels:
    :param id2label:
    :param suffix: 'BIOS', 'BIEOS', 'BIO'
    :return: [[label, start_id, end_id]]
    """
    label_entities = []
    if suffix in ["BIOS", 'BIO']:
        label_entities = extract_entities(pred_labels, id2label, suffix)
    elif suffix == "BIEOS":
        label_entities = get_entities(pred_labels)
    else:
        arguments = []
        starting = False
        for i, label in enumerate(pred_labels):
            if label > 0:
                if label % 2 == 1:
                    starting = True
                    arguments.append([[i], id2label[(label - 1) // 2]])
                elif starting:
                    arguments[-1][0].append(i)
                else:
                    starting = False
            else:
                starting = False
        for w, label in arguments:
            start = w[0]
            end = w[-1]
            label_entities.append([label, start, end])
    return label_entities


def join_segs(segs, sep=None):
    if len(segs) == 0:
        return ""
    if sep is not None:
        return " ".join(segs)

    text = segs[0]
    for seg in segs[1:]:
        if text != "" and seg != "" and \
                re.match("^[a-zA-Z]$", text[-1]) is not None and \
                re.match("^[a-zA-Z]$", seg[0]) is not None:
            text += " "
        else:
            pass
        text += seg
    return text


def unique_list(inp_list):
    out_list = []
    memory = set()
    for item in inp_list:
        mem = str(item)
        if type(item) is dict:
            mem = str(dict(sorted(item.items())))
        if mem not in memory:
            out_list.append(item)
            memory.add(mem)
    return out_list


def extract_ent_fr_txt_by_char_sp(char_span, text, language='ch'):
    segs = [text[char_span[idx]:char_span[idx + 1]] for idx in range(0, len(char_span), 2)]

    if language == "en":
        return join_segs(segs, " ")
    else:
        return join_segs(segs)


def get_char2tok_span(tok2char_span):
    """
    get a map from character level index to token level span
    e.g. "She is singing" -> [
                             [0, 1], [0, 1], [0, 1], # She
                             [-1, -1] # whitespace
                             [1, 2], [1, 2], # is
                             [-1, -1] # whitespace
                             [2, 3], [2, 3], [2, 3], [2, 3], [2, 3], [2, 3], [2, 3] # singing
                             ]

     tok2char_span： a map from token index to character level span
    """

    # get the number of characters
    char_num = None
    for tok_ind in range(len(tok2char_span) - 1, -1, -1):
        if tok2char_span[tok_ind][1] != 0:
            char_num = tok2char_span[tok_ind][1]
            break

    # build a map: char index to token level span
    char2tok_span = [[-1, -1] for _ in range(char_num)]  # 除了空格，其他字符均有对应token
    for tok_ind, char_sp in enumerate(tok2char_span):
        for char_ind in range(char_sp[0], char_sp[1]):
            tok_sp = char2tok_span[char_ind]
            # 因为在bert中，char to tok 也可能出现1对多的情况，比如韩文。
            # 所以char_span的pos1以第一个tok_ind为准，pos2以最后一个tok_ind为准
            if tok_sp[0] == -1:  # 第一次赋值以后不再修改
                tok_sp[0] = tok_ind
            tok_sp[1] = tok_ind + 1  # 每一次都更新
    return char2tok_span


def span_contains(sp1, sp2):
    if len(sp2) == 0:
        return True
    span1 = sorted(sp1) if len(sp1) > 2 else sp1
    span2 = sorted(sp2) if len(sp2) > 2 else sp2
    return span1[0] <= span2[0] < span2[-1] <= span1[-1]


def tok_span2char_span(tok_span, tok2char_span):
    char_span = []
    if len(tok_span) == 0:
        return []

    for idx in range(0, len(tok_span), 2):
        tk_sp = [tok_span[idx], tok_span[idx + 1]]
        if tk_sp[-1] == 0:
            return tok_span
        try:
            char_span_list = tok2char_span[tk_sp[0]:tk_sp[1]]
            char_span.extend([char_span_list[0][0], char_span_list[-1][1]])
        except Exception as e:
            print("error in tok_span2char_span function!")
            raise e
    return char_span


def merge_spans(spans, text=None):
    """
    merge continuous spans
    :param spans: [1, 2, 2, 3]
    :param text:
    :return: [1, 3]
    """
    new_spans = []
    for pid, pos in enumerate(spans):
        p = pos
        if pid == 0 or pid % 2 != 0 or pid % 2 == 0 and p != new_spans[-1]:
            new_spans.append(pos)
        elif pid % 2 == 0 and p == new_spans[-1]:
            new_spans.pop()

    new_spans_ = []
    if text is not None:  # merge spans if only blanks between them
        for pid, pos in enumerate(new_spans):
            if pid != 0 and pid % 2 == 0 and re.match(r"^\s+$", text[new_spans[pid - 1]:pos]) is not None:
                new_spans_.pop()
            else:
                new_spans_.append(pos)
        new_spans = new_spans_

    return new_spans


def ids2span(ids):
    """
    parse ids to spans, e.g. [1, 2, 3, 4, 7, 8, 9] -> [1, 5, 7, 10]
    :param ids: id list
    :return:
    """
    spans = []
    pre = -10
    for pos in ids:
        if pos - 1 != pre:
            spans.append(pre + 1)
            spans.append(pos)
        pre = pos
    spans.append(pre + 1)
    spans = spans[1:]
    return spans


def spans2ids(spans):
    """
    parse spans to ids, e.g. [1, 5, 7, 10] -> [1, 2, 3, 4, 7, 8, 9]
    :param spans:
    :return:
    """
    ids = []
    for i in range(0, len(spans), 2):
        ids.extend(list(range(spans[i], spans[i + 1])))
    return ids


def decompose2splits(data):
    """
    decompose combined samples to splits by the list "splits"
    :param data:
    :return:
    """
    new_data = []
    for sample in data:
        if "components" in sample:
            text = sample["text"]
            tok2char_span = sample["features"]["tok2char_span"]
            # decompose
            for spl in sample["components"]:
                split_sample = {
                    "id": spl["id"],
                    "tok_level_offset": spl["offset_in_ori_txt"]["tok_level_offset"],
                    "char_level_offset": spl["offset_in_ori_txt"]["char_level_offset"],
                }
                text_tok_span = spl["offset_in_this_comb"]
                char_sp_list = tok2char_span[text_tok_span[0]:text_tok_span[1]]

                text_char_span = [char_sp_list[0][0], char_sp_list[-1][1]]
                assert text_char_span[0] < text_char_span[1]

                # text
                split_sample["text"] = text[text_char_span[0]:text_char_span[1]]
                # filter annotations
                filtered_sample = filter_annotations(sample, text_tok_span[0], text_tok_span[1])
                if "entity_list" in filtered_sample:
                    split_sample["entity_list"] = filtered_sample["entity_list"]
                if "relation_list" in filtered_sample:
                    split_sample["relation_list"] = filtered_sample["relation_list"]
                if "event_list" in filtered_sample:
                    split_sample["event_list"] = filtered_sample["event_list"]
                if "open_spo_list" in filtered_sample:
                    split_sample["open_spo_list"] = filtered_sample["open_spo_list"]
                # recover spans
                anns = span_offset(split_sample, -text_tok_span[0], -text_char_span[0])
                split_sample = {**split_sample, **anns}
                new_data.append(split_sample)
        else:
            new_data.append(sample)
    return new_data
