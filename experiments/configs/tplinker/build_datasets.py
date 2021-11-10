# -*- coding: utf8 -*-
"""
======================================
    Project Name: NLP
    File Name: build_datasets
    Author: czh
    Create Date: 2021/8/16
--------------------------------------
    Change Activity: 
======================================
"""
import json
import os
import logging
from tqdm import tqdm
import yaml
import codecs
from pprint import pprint

from transformers import BertTokenizerFast
from nlp.utils.tplinker_utils import Preprocessor


logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

config = yaml.load(open("build_data_config.yaml", "r"), Loader=yaml.FullLoader)
tokenizer = BertTokenizerFast.from_pretrained(config["bert_path"], add_special_tokens=False, do_lower_case=False)
tokenize = tokenizer.tokenize
get_tok2char_span_map = lambda text: tokenizer.encode_plus(text,
                                                           return_offsets_mapping=True,
                                                           add_special_tokens=False)["offset_mapping"]
preprocessor = Preprocessor(tokenize_func=tokenize, get_tok2char_span_map_func=get_tok2char_span_map)


def check_tok_span(dataset):

    def extr_ent(text_, tok_span_, tok2char_span_):
        char_span_list = tok2char_span_[tok_span_[0]:tok_span_[1]]
        char_span = (char_span_list[0][0], char_span_list[-1][1])
        decoded_ent = text_[char_span[0]:char_span[1]]
        return decoded_ent

    span_error_memory = set()
    for sample in tqdm(dataset, desc="check tok spans"):
        text = sample["text"]
        tok2char_span = get_tok2char_span_map(text)
        for ent in sample["entity_list"]:
            tok_span = ent["tok_span"]
            if extr_ent(text, tok_span, tok2char_span) != ent["text"]:
                span_error_memory.add(
                    "extr ent: {}---gold ent: {}".format(extr_ent(text, tok_span, tok2char_span), ent["text"]))

        for rel in sample["relation_list"]:
            subj_tok_span, obj_tok_span = rel["subj_tok_span"], rel["obj_tok_span"]
            if extr_ent(text, subj_tok_span, tok2char_span) != rel["subject"]:
                span_error_memory.add(
                    "extr: {}---gold: {}".format(extr_ent(text, subj_tok_span, tok2char_span), rel["subject"]))
            if extr_ent(text, obj_tok_span, tok2char_span) != rel["object"]:
                span_error_memory.add(
                    "extr: {}---gold: {}".format(extr_ent(text, obj_tok_span, tok2char_span), rel["object"]))

    return span_error_memory


def main():
    exp_name = config["exp_name"]
    data_in_dir = os.path.join(config["data_in_dir"], exp_name)
    data_out_dir = os.path.join(config["data_out_dir"], exp_name)
    if not os.path.exists(data_out_dir):
        os.makedirs(data_out_dir)

    file_name2data = {}
    for name in ['train', 'dev', 'test']:
        filename = os.path.join(data_in_dir, name + '_raw.json')
        data = []
        with codecs.open(filename, encoding='utf8') as fr:
            lines = json.load(fr)
            for line in lines:
                # line = line.strip()
                if not line:
                    continue
                # line_json = json.loads(line)
                data.append({"text": line["text"], "triple_list": line["spo_list"]})
        file_name2data[name] = data
    # file_name2data["test"] = file_name2data["dev"]

    ori_format = config["ori_data_format"]
    if ori_format != "tplinker":  # if tplinker, skip transforming
        for file_name, data in file_name2data.items():
            if "train" in file_name:
                data_type = "train"
            elif "dev" in file_name:
                data_type = "valid"
            elif "test" in file_name:
                data_type = "test"
            else:
                raise ValueError(r"'data_type' must be 'train', 'valid' or 'test'")
            data = preprocessor.transform_data(data, ori_format=ori_format, dataset_type=data_type, add_id=True)
            file_name2data[file_name] = data

    # clean, add char span, tok span
    # collect relations
    # check tok spans
    rel_set = set()
    ent_set = set()
    error_statistics = {}
    for file_name, data in file_name2data.items():
        assert len(data) > 0
        if "relation_list" in data[0]:  # train or valid data
            # rm redundant whitespaces
            # separate by whitespaces
            print(data[0])
            data = preprocessor.clean_data_wo_span(data, separate=config["separate_char_by_white"])
            print(data[0])
            error_statistics[file_name] = {}
            # add char span
            if config["add_char_span"]:
                data, miss_sample_list = preprocessor.add_char_span(data, config["ignore_subword"])
                error_statistics[file_name]["miss_samples"] = len(miss_sample_list)
                print(data[0])

            # clean
            data, bad_samples_w_char_span_error = preprocessor.clean_data_w_span(data)
            error_statistics[file_name]["char_span_error"] = len(bad_samples_w_char_span_error)

            # collect relation types and entity types
            for sample in tqdm(data, desc="building relation type set and entity type set"):
                if "entity_list" not in sample:
                    # if "entity_list" not in sample, generate entity list with default type
                    ent_list = []
                    for rel in sample["relation_list"]:
                        ent_list.append({
                            "text": rel["subject"],
                            "type": "DEFAULT" if "subject_type" not in rel else rel["subject_type"],
                            "char_span": rel["subj_char_span"],
                        })
                        ent_list.append({
                            "text": rel["object"],
                            "type": "DEFAULT" if "object_type" not in rel else rel["object_type"],
                            "char_span": rel["obj_char_span"],
                        })
                    sample["entity_list"] = ent_list

                for ent in sample["entity_list"]:
                    ent_set.add(ent["type"])

                for rel in sample["relation_list"]:
                    rel_set.add(rel["predicate"])

            # add tok span
            data = preprocessor.add_tok_span(data)
            print(data[0])
            # check tok span
            if config["check_tok_span"]:
                span_error_memory = check_tok_span(data)
                if len(span_error_memory) > 0:
                    print(span_error_memory)
                error_statistics[file_name]["tok_span_error"] = len(span_error_memory)

            file_name2data[file_name] = data
    pprint(error_statistics)

    rel_set = sorted(rel_set)
    rel2id = {rel: ind for ind, rel in enumerate(rel_set)}

    ent_set = sorted(ent_set)
    ent2id = {ent: ind for ind, ent in enumerate(ent_set)}

    data_statistics = {
        "relation_type_num": len(rel2id),
        "entity_type_num": len(ent2id),
    }

    for file_name, data in file_name2data.items():
        data_path = os.path.join(data_out_dir, "{}.json".format(file_name))
        json.dump(data, open(data_path, "w", encoding="utf-8"), ensure_ascii=False)
        logging.info("{} is output to {}".format(file_name, data_path))
        data_statistics[file_name] = len(data)

    rel2id_path = os.path.join(data_out_dir, "rel2id.json")
    json.dump(rel2id, open(rel2id_path, "w", encoding="utf-8"), ensure_ascii=False)
    logging.info("rel2id is output to {}".format(rel2id_path))

    ent2id_path = os.path.join(data_out_dir, "ent2id.json")
    json.dump(ent2id, open(ent2id_path, "w", encoding="utf-8"), ensure_ascii=False)
    logging.info("ent2id is output to {}".format(ent2id_path))

    data_statistics_path = os.path.join(data_out_dir, "data_statistics.txt")
    json.dump(data_statistics, open(data_statistics_path, "w", encoding="utf-8"), ensure_ascii=False, indent=4)
    logging.info("data_statistics is output to {}".format(data_statistics_path))

    pprint(data_statistics)


if __name__ == "__main__":
    main()
