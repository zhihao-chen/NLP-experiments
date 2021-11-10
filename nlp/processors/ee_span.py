# -*- coding: utf8 -*-
"""
======================================
    Project Name: NLP
    File Name: ee_span
    Author: czh
    Create Date: 2021/9/23
--------------------------------------
    Change Activity: 
======================================
"""
import logging
import os
import re
import codecs
import json
import copy
import unicodedata
from abc import ABC

from nlp.processors.utils_ee import (DataProcessor, extract_ent_fr_txt_by_char_sp,
                                     unique_list, get_char2tok_span, tok_span2char_span)
from nlp.utils.tokenizers import ChineseWordTokenizer

logger = logging.getLogger(__name__)


class TpLinkerEEProcessor(DataProcessor, ABC):
    def __init__(self, data_dir,
                 language='ch',
                 do_lower_case=False,
                 tokenizer=None,
                 ori_data_format='duee1'):
        super(TpLinkerEEProcessor, self).__init__()

        self.__data_dir = data_dir
        self.__do_lower_case = do_lower_case
        self.__word_tokenizer = ChineseWordTokenizer()
        self.__stanza_tokenizer = tokenizer
        self.__language = language
        self.__ori_data_format = ori_data_format

    def get_train_examples(self):
        return self.__read_json(os.path.join(self.__data_dir, 'train.json'), 'train')

    def get_dev_examples(self):
        return self.__read_json(os.path.join(self.__data_dir, 'dev.json'), 'dev')

    def get_test_examples(self):
        return self.__read_json(os.path.join(self.__data_dir, 'test.json'), 'test')

    def __read_json(self, file_name_or_path, data_type):
        lines = []
        with codecs.open(file_name_or_path, encoding='utf8') as f:
            i = 1
            for line in f:
                line = line.strip()
                if not line:
                    continue
                sample = json.loads(line)
                sample['id'] = "{}_{}".format(data_type, i)
                sample = self.__process_sample(sample)
                lines.append(sample)
        return lines

    def __process_sample(self, sample):
        if self.__ori_data_format == 'duee1':
            sample = self.__trans_duee(sample)
        sample = self.__pre_check_data_annotation(sample)
        sample = self.__create_features(sample)
        sample = self.__add_tok_span(sample)
        sample = self.__check_tok_span(sample)
        return sample

    @staticmethod
    def __labels():
        label_dicts = [
            {"event_type": "PERUP", "role_list": [{"role": "PER", "name": "晋升人员"}, {"role": "CORG", "name": "当前所在公司"},
                                                  {"role": "CDEP", "name": "当前所在部门"},
                                                  {"role": "CTITLE", "name": "当前职位/职级"},
                                                  {"role": "LORG", "name": "晋升前公司"}, {"role": "LDEP", "name": "晋升前部门"},
                                                  {"role": "LTITLE", "name": "晋升前职位/职级"},
                                                  {"role": "CHA", "name": "主要业务"}], "class": "人员晋升"},
            {"event_type": "BUSEXP", "role_list": [{"role": "ORG", "name": "公司名称"}, {"role": "CHA", "name": "业务方向"},
                                                   {"role": "BTIME", "name": "计划时间"}, {"role": "PER", "name": "主要人员"},
                                                   {"role": "BPTR", "name": "合作伙伴"}], "class": "业务拓展"},
            {"event_type": "ORGFIN",
             "role_list": [{"role": "CIRCLE", "name": "融资轮数"}, {"role": "MONEY", "name": "融资金额"},
                           {"role": "ORG", "name": "融资公司"}, {"role": "INV", "name": "投资方"},
                           {"role": "CHA", "name": "融资业务方向"}], "class": "企业融资"},
            {"event_type": "ADMPEN",
             "role_list": [{"role": "ORG", "name": "处罚公司"}, {"role": "PTYP", "name": "处罚内容/违反条例"},
                           {"role": "MONEY", "name": "处罚金额"}], "class": "行政处罚"},
            {"event_type": "JOBREC",
             "role_list": [{"role": "HCHA", "name": "业务关键词"}, {"role": "HTITLE", "name": "职位关键词"},
                           {"role": "HLEVEL", "name": "职级关键词"}], "class": "职位招聘"},
            {"event_type": "STRCOO", "role_list": [{"role": "ORG", "name": "公司名称"}, {"role": "CHA", "name": "业务方向"}],
             "class": "战略合作"},
            {"event_type": "OTHER", "role_list": [{"role": "ORG", "name": "公司名称"}, {"role": "CHA", "name": "主要业务"},
                                                  {"role": "PER", "name": "晋升人员"}, {"role": "BPRT", "name": "合作伙伴"}],
             "class": "其它"}
        ]
        return label_dicts

    def get_labels(self):
        file_path = os.path.join(self.__data_dir, 'event_schema.json')
        if os.path.exists(file_path):
            id2label, label2id, num_labels, event_type_dict = self.read_event_schema(file_name_or_path=file_path)
        else:
            label_dicts = self.__labels()
            id2label, label2id, num_labels, event_type_dict = self.read_event_schema(alist=label_dicts)
        return id2label, label2id, num_labels, event_type_dict

    @staticmethod
    def __trans_duee(sample):
        text = sample["text"]

        # event list
        if "event_list" in sample:  # train or valid data
            normal_event_list = []
            for event in sample["event_list"]:
                normal_event = copy.deepcopy(event)

                # rm whitespaces
                trigger = normal_event.get("trigger", None)
                if trigger:
                    clean_tri = normal_event["trigger"].lstrip()
                    normal_event["trigger_start_index"] += len(normal_event["trigger"]) - len(clean_tri)
                    normal_event["trigger"] = clean_tri.rstrip()

                    normal_event["trigger_char_span"] = [normal_event["trigger_start_index"],
                                                         normal_event["trigger_start_index"] + len(
                                                             normal_event["trigger"])]
                    char_span = normal_event["trigger_char_span"]
                    assert text[char_span[0]:char_span[1]] == normal_event["trigger"]
                    del normal_event["trigger_start_index"]

                normal_arg_list = []
                for arg in normal_event["arguments"]:
                    # clean whitespaces
                    clean_arg = arg["argument"].lstrip()
                    arg["argument_start_index"] += len(arg["argument"]) - len(clean_arg)
                    arg["argument"] = clean_arg.rstrip()

                    char_span = [arg["argument_start_index"],
                                 arg["argument_start_index"] + len(arg["argument"])]
                    assert text[char_span[0]:char_span[1]] == arg["argument"]
                    normal_arg_list.append({
                        "text": arg["argument"],
                        "type": arg["role"],
                        "char_span": char_span,
                    })
                normal_event["argument_list"] = normal_arg_list
                del normal_event["arguments"]
                normal_event_list.append(normal_event)
            sample["event_list"] = normal_event_list
            return sample

    def __pre_check_data_annotation(self, sample):
        def check_ent_span(entity_list, text_):
            for ent in entity_list:
                ent_ext_fr_span = extract_ent_fr_txt_by_char_sp(ent["char_span"], text_, self.__language)
                if ent["text"] != ent_ext_fr_span:
                    raise Exception("char span error: ent_text: {} != ent_ext_fr_span: {}".format(ent["text"],
                                                                                                  ent_ext_fr_span))

        text = sample["text"]

        if "entity_list" in sample:
            check_ent_span(sample["entity_list"], text)

        if "relation_list" in sample:
            entities_fr_rel = []
            for rel in sample["relation_list"]:
                entities_fr_rel.append({
                    "text": rel["subject"],
                    "char_span": [*rel["subj_char_span"]]
                })

                entities_fr_rel.append({
                    "text": rel["object"],
                    "char_span": [*rel["obj_char_span"]]
                })
            entities_fr_rel = unique_list(entities_fr_rel)
            check_ent_span(entities_fr_rel, text)

            entities_mem = {str({"text": ent["text"], "char_span": ent["char_span"]})
                            for ent in sample["entity_list"]}
            for ent in entities_fr_rel:
                if str(ent) not in entities_mem:
                    raise Exception("entity list misses some entities in relation list")

        if "event_list" in sample:
            entities_fr_event = []
            for event in sample["event_list"]:
                if "trigger" in event:
                    if type(event["trigger_char_span"][0]) is list:
                        for ch_sp in event["trigger_char_span"]:
                            entities_fr_event.append({
                                "text": event["trigger"],
                                "char_span": [*ch_sp],
                            })
                    else:
                        entities_fr_event.append({
                            "text": event["trigger"],
                            "char_span": [*event["trigger_char_span"]]
                        })
                for arg in event["argument_list"]:
                    if type(arg["char_span"][0]) is list:
                        for ch_sp in arg["char_span"]:
                            entities_fr_event.append({
                                "text": arg["text"],
                                "char_span": [*ch_sp],
                            })
                    else:
                        entities_fr_event.append({
                            "text": arg["text"],
                            "char_span": [*arg["char_span"]]
                        })

            entities_fr_event = unique_list(entities_fr_event)
            check_ent_span(entities_fr_event, text)

        if "open_spo_list" in sample:
            for spo in sample["open_spo_list"]:
                check_ent_span(spo, text)
        return sample

    @staticmethod
    def __get_all_possible_char_spans(sample):
        sp_list = []
        if "entity_list" in sample:
            sp_list.extend([ent["char_span"] for ent in sample["entity_list"]])

        if "relation_list" in sample:
            sp_list.extend([spo["subj_char_span"] for spo in sample["relation_list"]])
            sp_list.extend([spo["obj_char_span"] for spo in sample["relation_list"]])

        if "event_list" in sample:
            for event in sample["event_list"]:
                if "trigger" in event:
                    if type(event["trigger_char_span"][0]) is list:
                        sp_list.extend(event["trigger_char_span"])
                    else:
                        sp_list.append(event["trigger_char_span"])
                for arg in event["argument_list"]:
                    if type(arg["char_span"][0]) is list:
                        sp_list.extend(arg["char_span"])
                    else:
                        sp_list.append(arg["char_span"])

        if "open_spo_list" in sample:
            for spo in sample["open_spo_list"]:
                for arg in spo:
                    # ent_list.append(arg["text"])
                    if "char_span" not in arg or len(arg["char_span"]) == 0:
                        continue
                    sp_list.append(arg["char_span"])

        return sp_list

    @staticmethod
    def __rm_accents(ss):
        return "".join(c for c in unicodedata.normalize('NFD', ss) if unicodedata.category(c) != 'Mn')

    def __create_features(self, sample):
        """
        :param sample:
        :return:
        """
        # create features
        text = sample["text"]

        # word level
        word_features = {}
        if "word_list" not in sample or "word2char_span" not in sample:
            # generate word_list, word2char_span
            all_sps = self.__get_all_possible_char_spans(sample)
            wd_tok_res = self.__word_tokenizer.tokenize_plus(text, span_list=all_sps)
            sample["word_list"] = wd_tok_res["word_list"]
            sample["word2char_span"] = wd_tok_res["word2char_span"]

        codes = self.__stanza_tokenizer.encode_plus_fr_words(sample["word_list"],
                                                             sample["word2char_span"],
                                                             return_offsets_mapping=True,
                                                             add_special_tokens=False,
                                                             )
        subword_features = {
            "subword_list": codes["subword_list"],
            "subword2char_span": codes["offset_mapping"],
        }

        for key in {"ner_tag_list", "word_list", "pos_tag_list", "dependency_list", "word2char_span",
                    "dependency_list_csp", "pos_tag_list_csp"}:
            if key in sample:
                word_features[key] = sample[key]
                del sample[key]

        sample["features"] = word_features

        # generate subword2word_id
        char2word_span = get_char2tok_span(sample["features"]["word2char_span"])

        subword2word_id = []
        for subw_id, char_sp in enumerate(subword_features["subword2char_span"]):
            wd_sps = char2word_span[char_sp[0]:char_sp[1]]
            assert wd_sps[0][0] == wd_sps[-1][1] - 1  # the same word idx
            subword2word_id.append(wd_sps[0][0])

        # generate word2subword_span
        word2subword_span = [[-1, -1] for _ in range(len(sample["features"]["word_list"]))]
        for subw_id, wid in enumerate(subword2word_id):
            if word2subword_span[wid][0] == -1:
                word2subword_span[wid][0] = subw_id
            word2subword_span[wid][1] = subw_id + 1

        # add subword level features into the feature list
        sample["features"] = {
            **sample["features"],
            **subword_features,
            "subword2word_id": subword2word_id,
            "word2subword_span": word2subword_span,
        }

        # check
        feats = sample["features"]
        num_words = len(word2subword_span)
        for k in {"ner_tag_list", "pos_tag_list", "word2char_span", "word_list"}:
            if k in feats:
                assert len(feats[k]) == num_words
        assert len(feats["subword_list"]) == len(feats["subword2char_span"]) == len(subword2word_id)
        for subw_id, wid in enumerate(subword2word_id):
            subw = sample["features"]["subword_list"][subw_id]
            word = sample["features"]["word_list"][wid]

            subw_ = re.sub(r"##", "", subw)

            if re.match(r"^[\uAC00-\uD7FFh]+$", word) is not None:  # skip korean
                continue
            word = self.__rm_accents(word)
            subw_ = self.__rm_accents(subw_)

            try:
                if self.__do_lower_case:
                    assert subw_.lower() in word.lower() or subw_ == "[UNK]"
                else:
                    assert subw_ in word or subw_ == "[UNK]"
            except Exception as e:
                print("subw({}) not in word({})".format(subw_, word))
                raise e

        for subw_id, char_sp in enumerate(feats["subword2char_span"]):
            subw = sample["features"]["subword_list"][subw_id]
            subw = re.sub(r"##", "", subw)
            subw_extr = sample["text"][char_sp[0]:char_sp[1]]

            if re.match(r"^[\uAC00-\uD7FFh]+$", subw_extr) is not None:
                continue
            subw_extr = self.__rm_accents(subw_extr)
            subw = self.__rm_accents(subw)
            try:
                if self.__do_lower_case:
                    assert subw_extr.lower() == subw.lower() or subw == "[UNK]"
                else:
                    assert subw_extr == subw or subw == "[UNK]"
            except Exception as e:
                print("subw_extr({}) != subw({})".format(subw_extr, subw))
                raise e

        return sample

    @staticmethod
    def __add_tok_span(sample):
        """
        add token level span according to the character spans, character level spans are required
        """

        def char_span2tok_span(char_span, char2tok_span):
            tok_span = []
            for idx in range(0, len(char_span), 2):  # len(char_span) > 2 if discontinuous entity
                if char_span[-1] == 0:
                    return char_span
                ch_sp = [char_span[idx], char_span[idx + 1]]
                tok_span_list = char2tok_span[ch_sp[0]:ch_sp[1]]
                try:
                    tok_span.extend([tok_span_list[0][0], tok_span_list[-1][1]])
                except Exception as e:
                    print("error in char_span2tok_span!")
                    raise e
            return tok_span

        char2word_span = get_char2tok_span(sample["features"]["word2char_span"])
        char2subwd_span = get_char2tok_span(sample["features"]["subword2char_span"])

        def add_sps(a_list):
            for item in a_list:
                if type(item) is list:
                    add_sps(item)
                elif type(item) is dict:
                    add_dict = {}
                    for key, value in item.items():
                        if "char_span" in key:
                            add_wd_sp_k = re.sub("char_span", "wd_span", key)
                            add_subwd_sp_k = re.sub("char_span", "subwd_span", key)

                            if type(value[0]) is list:
                                add_dict[add_wd_sp_k] = [char_span2tok_span(ch_sp, char2word_span) for ch_sp in value]
                                add_dict[add_subwd_sp_k] = [char_span2tok_span(ch_sp, char2subwd_span) for ch_sp in
                                                            value]
                            else:
                                assert type(value[0]) is int
                                add_dict[add_wd_sp_k] = char_span2tok_span(value, char2word_span)
                                add_dict[add_subwd_sp_k] = char_span2tok_span(value, char2subwd_span)
                        elif type(value) is list:
                            add_sps(value)
                    item.update(add_dict)

        for k, v in sample.items():
            if type(v) is list:
                add_sps(v)
            elif type(v) is dict:
                for val in v.values():
                    if type(val) is list:
                        add_sps(val)
        return sample

    def __extract_ent_fr_txt_by_tok_sp(self, tok_span, tok2char_span, text):
        char_span = tok_span2char_span(tok_span, tok2char_span)
        return extract_ent_fr_txt_by_char_sp(char_span, text, self.__language)

    def __check_tok_span(self, sample):
        """
        check if text is equal to the one extracted by the annotated token level spans
        :param sample:
        :return:
        """

        text = sample["text"]
        word2char_span = sample["features"]["word2char_span"]
        subword2char_span = sample["features"]["subword2char_span"]

        if "entity_list" in sample:
            bad_entities = []
            for ent in sample["entity_list"]:
                word_span = ent["wd_span"]
                subword_span = ent["subwd_span"]
                ent_wd = self.__extract_ent_fr_txt_by_tok_sp(word_span, word2char_span, text)
                ent_subwd = self.__extract_ent_fr_txt_by_tok_sp(subword_span, subword2char_span, text)

                if not (ent_wd == ent_subwd == ent["text"]):
                    bad_ent = copy.deepcopy(ent)
                    bad_ent["extr_ent_wd"] = ent_wd
                    bad_ent["extr_ent_subwd"] = ent_subwd
                    bad_entities.append(bad_ent)
            if len(bad_entities) > 0:
                print(text)
                print(bad_entities)
                print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")

        if "relation_list" in sample:
            bad_rels = []
            for rel in sample["relation_list"]:
                subj_wd_span = rel["subj_wd_span"]
                obj_wd_span = rel["obj_wd_span"]
                subj_subwd_span = rel["subj_subwd_span"]
                obj_subwd_span = rel["obj_subwd_span"]

                subj_wd = self.__extract_ent_fr_txt_by_tok_sp(subj_wd_span, word2char_span, text)
                obj_wd = self.__extract_ent_fr_txt_by_tok_sp(obj_wd_span, word2char_span, text)
                subj_subwd = self.__extract_ent_fr_txt_by_tok_sp(subj_subwd_span, subword2char_span, text)
                obj_subwd = self.__extract_ent_fr_txt_by_tok_sp(obj_subwd_span, subword2char_span, text)

                if not (subj_wd == rel["subject"] == subj_subwd and obj_wd == rel["object"] == obj_subwd):
                    bad_rel = copy.deepcopy(rel)
                    bad_rel["extr_subj_wd"] = subj_wd
                    bad_rel["extr_subj_subwd"] = subj_subwd
                    bad_rel["extr_obj_wd"] = obj_wd
                    bad_rel["extr_obj_subwd"] = obj_subwd
                    bad_rels.append(bad_rel)
            if len(bad_rels) > 0:
                print(text)
                print(bad_rels)
                print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")

        if "event_list" in sample:
            bad_events = []
            for event in sample["event_list"]:
                bad_event = copy.deepcopy(event)
                bad = False

                if "trigger" in event:
                    trigger_wd_span = event["trigger_wd_span"]
                    trigger_subwd_span = event["trigger_subwd_span"]

                    if type(trigger_wd_span[0]) is list or type(trigger_subwd_span[0]) is list:
                        pass
                    else:
                        trigger_wd_span = [trigger_wd_span, ]
                        trigger_subwd_span = [trigger_subwd_span, ]

                    for sp_idx, wd_sp in enumerate(trigger_wd_span):
                        subwd_sp = trigger_subwd_span[sp_idx]
                        extr_trigger_wd = self.__extract_ent_fr_txt_by_tok_sp(wd_sp, word2char_span, text)
                        extr_trigger_subwd = self.__extract_ent_fr_txt_by_tok_sp(subwd_sp, subword2char_span, text)

                        if not (extr_trigger_wd == extr_trigger_subwd == event["trigger"]):
                            bad = True
                            bad_event.setdefault("extr_trigger_wd", []).append(extr_trigger_wd)
                            bad_event.setdefault("extr_trigger_subwd", []).append(extr_trigger_subwd)

                for arg in bad_event["argument_list"]:
                    arg_wd_span = arg["wd_span"]
                    arg_subwd_span = arg["subwd_span"]
                    if type(arg_wd_span[0]) is list or type(arg_subwd_span[0]) is list:
                        pass
                    else:
                        arg_wd_span = [arg_wd_span, ]
                        arg_subwd_span = [arg_subwd_span, ]

                    for sp_idx, wd_sp in enumerate(arg_wd_span):
                        subwd_sp = arg_subwd_span[sp_idx]
                        extr_arg_wd = self.__extract_ent_fr_txt_by_tok_sp(wd_sp, word2char_span, text)
                        extr_arg_subwd = self.__extract_ent_fr_txt_by_tok_sp(subwd_sp, subword2char_span, text)

                        if not (extr_arg_wd == extr_arg_subwd == arg["text"]):
                            bad = True
                            bad_event.setdefault("extr_arg_wd", []).append(extr_arg_wd)
                            bad_event.setdefault("extr_arg_subwd", []).append(extr_arg_subwd)

                if bad:
                    bad_events.append(bad_event)
            if len(bad_events) > 0:
                print(text)
                print(bad_events)
                print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")

        return sample
