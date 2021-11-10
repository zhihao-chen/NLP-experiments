# -*- coding: utf8 -*-
"""
======================================
    Project Name: NLP
    File Name: tplinker_utils
    Author: czh
    Create Date: 2021/8/16
--------------------------------------
    Change Activity: 
======================================
"""
import os
import json
import re
import copy
from tqdm import tqdm

import torch


class HandshakingTaggingScheme(object):
    """docstring for HandshakingTaggingScheme"""

    def __init__(self, rel2id, max_seq_len):
        super(HandshakingTaggingScheme, self).__init__()
        self.rel2id = rel2id
        self.id2rel = {ind: rel for rel, ind in rel2id.items()}

        self.tag2id_ent = {
            "O": 0,
            "ENT-H2T": 1,  # entity head to entity tail
        }
        self.id2tag_ent = {id_: tag for tag, id_ in self.tag2id_ent.items()}

        self.tag2id_head_rel = {
            "O": 0,
            "REL-SH2OH": 1,  # subject head to object head
            "REL-OH2SH": 2,  # object head to subject head
        }
        self.id2tag_head_rel = {id_: tag for tag, id_ in self.tag2id_head_rel.items()}

        self.tag2id_tail_rel = {
            "O": 0,
            "REL-ST2OT": 1,  # subject tail to object tail
            "REL-OT2ST": 2,  # object tail to subject tail
        }
        self.id2tag_tail_rel = {id_: tag for tag, id_ in self.tag2id_tail_rel.items()}

        # mapping shaking sequence and matrix
        self.matrix_size = max_seq_len
        # e.g. [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]
        self.shaking_ind2matrix_ind = [(ind, end_ind) for ind in range(self.matrix_size) for end_ind in
                                       list(range(self.matrix_size))[ind:]]

        self.matrix_ind2shaking_ind = []
        for j in range(self.matrix_size):
            self.matrix_ind2shaking_ind.append([0 for _ in range(self.matrix_size)])
        for shaking_ind, matrix_ind in enumerate(self.shaking_ind2matrix_ind):
            self.matrix_ind2shaking_ind[matrix_ind[0]][matrix_ind[1]] = shaking_ind

    def get_spots(self, sample):
        """
        entity spot and tail_rel spot: (span_pos1, span_pos2, tag_id)
        head_rel spot: (rel_id, span_pos1, span_pos2, tag_id)
        """
        ent_matrix_spots, head_rel_matrix_spots, tail_rel_matrix_spots = [], [], []

        for rel in sample["relation_list"]:
            subj_tok_span = rel["subj_tok_span"]
            obj_tok_span = rel["obj_tok_span"]
            ent_matrix_spots.append((subj_tok_span[0], subj_tok_span[1] - 1, self.tag2id_ent["ENT-H2T"]))
            ent_matrix_spots.append((obj_tok_span[0], obj_tok_span[1] - 1, self.tag2id_ent["ENT-H2T"]))

            if subj_tok_span[0] <= obj_tok_span[0]:
                head_rel_matrix_spots.append((self.rel2id[rel["predicate"]], subj_tok_span[0], obj_tok_span[0],
                                              self.tag2id_head_rel["REL-SH2OH"]))
            else:
                head_rel_matrix_spots.append((self.rel2id[rel["predicate"]], obj_tok_span[0], subj_tok_span[0],
                                              self.tag2id_head_rel["REL-OH2SH"]))

            if subj_tok_span[1] <= obj_tok_span[1]:
                tail_rel_matrix_spots.append((self.rel2id[rel["predicate"]], subj_tok_span[1] - 1, obj_tok_span[1] - 1,
                                              self.tag2id_tail_rel["REL-ST2OT"]))
            else:
                tail_rel_matrix_spots.append((self.rel2id[rel["predicate"]], obj_tok_span[1] - 1, subj_tok_span[1] - 1,
                                              self.tag2id_tail_rel["REL-OT2ST"]))

        return ent_matrix_spots, head_rel_matrix_spots, tail_rel_matrix_spots

    def sharing_spots2shaking_tag(self, spots):
        """
        convert spots to shaking seq tag
        spots: [(start_ind, end_ind, tag_id), ], for entiy
        return:
            shake_seq_tag: (shaking_seq_len, )
        """
        shaking_seq_len = self.matrix_size * (self.matrix_size + 1) // 2
        shaking_seq_tag = torch.zeros(shaking_seq_len).long()
        for sp in spots:
            shaking_ind = self.matrix_ind2shaking_ind[sp[0]][sp[1]]
            shaking_seq_tag[shaking_ind] = sp[2]
        return shaking_seq_tag

    def spots2shaking_tag(self, spots):
        """
        convert spots to shaking seq tag
        spots: [(rel_id, start_ind, end_ind, tag_id), ], for head relation and tail relation
        return:
            shake_seq_tag: (rel_size, shaking_seq_len, )
        """
        shaking_seq_len = self.matrix_size * (self.matrix_size + 1) // 2
        shaking_seq_tag = torch.zeros(len(self.rel2id), shaking_seq_len).long()
        for sp in spots:
            shaking_ind = self.matrix_ind2shaking_ind[sp[1]][sp[2]]
            shaking_seq_tag[sp[0]][shaking_ind] = sp[3]
        return shaking_seq_tag

    def sharing_spots2shaking_tag4batch(self, batch_spots):
        """
        convert spots to batch shaking seq tag
        因长序列的stack是费时操作，所以写这个函数用作生成批量shaking tag
        如果每个样本生成一条shaking tag再stack，一个32的batch耗时1s，太昂贵
        spots: [(start_ind, end_ind, tag_id), ], for entiy
        return:
            batch_shake_seq_tag: (batch_size, shaking_seq_len)
        """
        shaking_seq_len = self.matrix_size * (self.matrix_size + 1) // 2
        batch_shaking_seq_tag = torch.zeros(len(batch_spots), shaking_seq_len).long()
        for batch_id, spots in enumerate(batch_spots):
            for sp in spots:
                shaking_ind = self.matrix_ind2shaking_ind[sp[0]][sp[1]]
                tag_id = sp[2]
                batch_shaking_seq_tag[batch_id][shaking_ind] = tag_id
        return batch_shaking_seq_tag

    def spots2shaking_tag4batch(self, batch_spots):
        """
        convert spots to batch shaking seq tag
        spots: [(rel_id, start_ind, end_ind, tag_id), ], for head relation and tail_relation
        return:
            batch_shake_seq_tag: (batch_size, rel_size, shaking_seq_len)
        """
        shaking_seq_len = self.matrix_size * (self.matrix_size + 1) // 2
        batch_shaking_seq_tag = torch.zeros(len(batch_spots), len(self.rel2id), shaking_seq_len).long()
        for batch_id, spots in enumerate(batch_spots):
            for sp in spots:
                shaking_ind = self.matrix_ind2shaking_ind[sp[1]][sp[2]]
                tag_id = sp[3]
                rel_id = sp[0]
                batch_shaking_seq_tag[batch_id][rel_id][shaking_ind] = tag_id
        return batch_shaking_seq_tag

    def get_spots_fr_shaking_tag(self, shaking_tag):
        """
        shaking_tag -> spots
        shaking_tag: (rel_size, shaking_seq_len)
        spots: [(rel_id, start_ind, end_ind, tag_id), ]
        """
        spots = []

        for shaking_inds in shaking_tag.nonzero():
            rel_id = shaking_inds[0].item()
            tag_id = shaking_tag[rel_id][shaking_inds[1]].item()
            matrix_inds = self.shaking_ind2matrix_ind[shaking_inds[1]]
            spot = (rel_id, matrix_inds[0], matrix_inds[1], tag_id)
            spots.append(spot)
        return spots

    def get_sharing_spots_fr_shaking_tag(self, shaking_tag):
        """
        shaking_tag -> spots
        shaking_tag: (shaking_seq_len, )
        spots: [(start_ind, end_ind, tag_id), ]
        """
        spots = []

        for shaking_ind in shaking_tag.nonzero():
            shaking_ind_ = shaking_ind[0].item()
            tag_id = shaking_tag[shaking_ind_]
            matrix_inds = self.shaking_ind2matrix_ind[shaking_ind_]
            spot = (matrix_inds[0], matrix_inds[1], tag_id)
            spots.append(spot)
        return spots

    def decode_rel_fr_shaking_tag(self, text, ent_shaking_tag, head_rel_shaking_tag, tail_rel_shaking_tag,
                                  tok2char_span, tok_offset=0, char_offset=0):
        """
        ent shaking tag: (shaking_seq_len, )
        head rel and tail rel shaking_tag: size = (rel_size, shaking_seq_len, )
        """
        rel_list = []

        ent_matrix_spots = self.get_sharing_spots_fr_shaking_tag(ent_shaking_tag)
        head_rel_matrix_spots = self.get_spots_fr_shaking_tag(head_rel_shaking_tag)
        tail_rel_matrix_spots = self.get_spots_fr_shaking_tag(tail_rel_shaking_tag)

        # entity
        head_ind2entities = {}
        for sp in ent_matrix_spots:
            tag_id = sp[2]
            if tag_id != self.tag2id_ent["ENT-H2T"]:
                continue

            char_span_list = tok2char_span[sp[0]:sp[1] + 1]
            char_sp = [char_span_list[0][0], char_span_list[-1][1]]
            ent_text = text[char_sp[0]:char_sp[1]]

            head_key = sp[0]  # take head as the key to entity list start with the head token
            if head_key not in head_ind2entities:
                head_ind2entities[head_key] = []
            head_ind2entities[head_key].append({
                "text": ent_text,
                "tok_span": [sp[0], sp[1] + 1],
                "char_span": char_sp,
            })

        # tail relation
        tail_rel_memory_set = set()
        for sp in tail_rel_matrix_spots:
            rel_id = sp[0]
            tag_id = sp[3]
            if tag_id == self.tag2id_tail_rel["REL-ST2OT"]:
                tail_rel_memory = "{}-{}-{}".format(rel_id, sp[1], sp[2])
                tail_rel_memory_set.add(tail_rel_memory)
            elif tag_id == self.tag2id_tail_rel["REL-OT2ST"]:
                tail_rel_memory = "{}-{}-{}".format(rel_id, sp[2], sp[1])
                tail_rel_memory_set.add(tail_rel_memory)

        # head relation
        for sp in head_rel_matrix_spots:
            rel_id = sp[0]
            tag_id = sp[3]

            if tag_id == self.tag2id_head_rel["REL-SH2OH"]:
                subj_head_key, obj_head_key = sp[1], sp[2]
            elif tag_id == self.tag2id_head_rel["REL-OH2SH"]:
                subj_head_key, obj_head_key = sp[2], sp[1]
            else:
                subj_head_key, obj_head_key = -1, -1

            if subj_head_key not in head_ind2entities or obj_head_key not in head_ind2entities:
                # no entity start with subj_head_key and obj_head_key
                continue
            subj_list = head_ind2entities[subj_head_key]  # all entities start with this subject head
            obj_list = head_ind2entities[obj_head_key]  # all entities start with this object head

            # go over all subj-obj pair to check whether the relation exists
            for subj in subj_list:
                for obj in obj_list:
                    tail_rel_memory = "{}-{}-{}".format(rel_id, subj["tok_span"][1] - 1, obj["tok_span"][1] - 1)
                    if tail_rel_memory not in tail_rel_memory_set:
                        # no such relation
                        continue

                    rel_list.append({
                        "subject": subj["text"],
                        "object": obj["text"],
                        "subj_tok_span": [subj["tok_span"][0] + tok_offset, subj["tok_span"][1] + tok_offset],
                        "obj_tok_span": [obj["tok_span"][0] + tok_offset, obj["tok_span"][1] + tok_offset],
                        "subj_char_span": [subj["char_span"][0] + char_offset, subj["char_span"][1] + char_offset],
                        "obj_char_span": [obj["char_span"][0] + char_offset, obj["char_span"][1] + char_offset],
                        "predicate": self.id2rel[rel_id],
                    })
        return rel_list


class DataMaker4Bert(object):
    def __init__(self, tokenizer, handshaking_tagger):
        self.tokenizer = tokenizer
        self.handshaking_tagger = handshaking_tagger

    def get_indexed_data(self, data, max_seq_len, data_type="train"):
        indexed_samples = []
        for ind, sample in tqdm(enumerate(data), desc="Generate indexed train or valid data"):
            text = sample["text"]
            # codes for bert input
            codes = self.tokenizer.encode_plus(text,
                                               return_offsets_mapping=True,
                                               add_special_tokens=False,
                                               max_length=max_seq_len,
                                               truncation=True,
                                               pad_to_max_length=True)
            # tagging
            spots_tuple = None
            if data_type != "test":
                spots_tuple = self.handshaking_tagger.get_spots(sample)

            # get codes
            input_ids = torch.tensor(codes["input_ids"]).long()
            attention_mask = torch.tensor(codes["attention_mask"]).long()
            token_type_ids = torch.tensor(codes["token_type_ids"]).long()
            tok2char_span = codes["offset_mapping"]

            sample_tp = (sample, input_ids, attention_mask, token_type_ids, tok2char_span, spots_tuple)
            indexed_samples.append(sample_tp)
        return indexed_samples

    def generate_batch(self, batch_data, data_type="train"):
        sample_list = []
        input_ids_list = []
        attention_mask_list = []
        token_type_ids_list = []
        tok2char_span_list = []

        ent_spots_list = []
        head_rel_spots_list = []
        tail_rel_spots_list = []

        for tp in batch_data:
            sample_list.append(tp[0])
            input_ids_list.append(tp[1])
            attention_mask_list.append(tp[2])
            token_type_ids_list.append(tp[3])
            tok2char_span_list.append(tp[4])

            if data_type != "test":
                ent_matrix_spots, head_rel_matrix_spots, tail_rel_matrix_spots = tp[5]
                ent_spots_list.append(ent_matrix_spots)
                head_rel_spots_list.append(head_rel_matrix_spots)
                tail_rel_spots_list.append(tail_rel_matrix_spots)

        # @specific: indexed by bert tokenizer
        batch_input_ids = torch.stack(input_ids_list, dim=0)
        batch_attention_mask = torch.stack(attention_mask_list, dim=0)
        batch_token_type_ids = torch.stack(token_type_ids_list, dim=0)

        batch_ent_shaking_tag, batch_head_rel_shaking_tag, batch_tail_rel_shaking_tag = None, None, None
        if data_type != "test":
            batch_ent_shaking_tag = self.handshaking_tagger.sharing_spots2shaking_tag4batch(ent_spots_list)
            batch_head_rel_shaking_tag = self.handshaking_tagger.spots2shaking_tag4batch(head_rel_spots_list)
            batch_tail_rel_shaking_tag = self.handshaking_tagger.spots2shaking_tag4batch(tail_rel_spots_list)

        return (sample_list, batch_input_ids, batch_attention_mask, batch_token_type_ids, tok2char_span_list,
                batch_ent_shaking_tag, batch_head_rel_shaking_tag, batch_tail_rel_shaking_tag)


class DataMaker4BiLSTM(object):
    def __init__(self, text2indices, get_tok2char_span_map, handshaking_tagger):
        self.text2indices = text2indices
        self.handshaking_tagger = handshaking_tagger
        self.get_tok2char_span_map = get_tok2char_span_map

    def get_indexed_data(self, data, max_seq_len, data_type="train"):
        indexed_samples = []
        for ind, sample in tqdm(enumerate(data), desc="Generate indexed train or valid data"):
            text = sample["text"]

            # tagging
            spots_tuple = None
            if data_type != "test":
                spots_tuple = self.handshaking_tagger.get_spots(sample)
            tok2char_span = self.get_tok2char_span_map(text)
            tok2char_span.extend([(-1, -1)] * (max_seq_len - len(tok2char_span)))
            input_ids = self.text2indices(text, max_seq_len)

            sample_tp = (sample,
                         input_ids,
                         tok2char_span,
                         spots_tuple,
                         )
            indexed_samples.append(sample_tp)
        return indexed_samples

    def generate_batch(self, batch_data, data_type="train"):
        sample_list = []
        input_ids_list = []
        tok2char_span_list = []

        ent_spots_list = []
        head_rel_spots_list = []
        tail_rel_spots_list = []

        for tp in batch_data:
            sample_list.append(tp[0])
            input_ids_list.append(tp[1])
            tok2char_span_list.append(tp[2])

            if data_type != "test":
                ent_matrix_spots, head_rel_matrix_spots, tail_rel_matrix_spots = tp[3]
                ent_spots_list.append(ent_matrix_spots)
                head_rel_spots_list.append(head_rel_matrix_spots)
                tail_rel_spots_list.append(tail_rel_matrix_spots)

        batch_input_ids = torch.stack(input_ids_list, dim=0)

        batch_ent_shaking_tag, batch_head_rel_shaking_tag, batch_tail_rel_shaking_tag = None, None, None
        if data_type != "test":
            batch_ent_shaking_tag = self.handshaking_tagger.sharing_spots2shaking_tag4batch(ent_spots_list)
            batch_head_rel_shaking_tag = self.handshaking_tagger.spots2shaking_tag4batch(head_rel_spots_list)
            batch_tail_rel_shaking_tag = self.handshaking_tagger.spots2shaking_tag4batch(tail_rel_spots_list)

        return (sample_list, batch_input_ids, tok2char_span_list, batch_ent_shaking_tag, batch_head_rel_shaking_tag,
                batch_tail_rel_shaking_tag)


class DefaultLogger:
    def __init__(self, log_path, project, run_name, run_id, hyperparameter):
        self.log_path = log_path
        log_dir = "/".join(self.log_path.split("/")[:-1])
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        self.run_id = run_id
        self.log("============================================================================")
        self.log("project: {}, run_name: {}, run_id: {}\n".format(project, run_name, run_id))
        hyperparameters_format = "--------------hypter_parameters------------------- \n{}\n----------------------------"
        self.log(hyperparameters_format.format(json.dumps(hyperparameter, indent=4)))

    def log(self, text):
        text = "run_id: {}, {}".format(self.run_id, text)
        print(text)
        open(self.log_path, "a", encoding="utf-8").write("{}\n".format(text))


class Preprocessor:
    """
    1. transform the dataset to normal format, which can fit in our codes
    2. add token level span to all entities in the relations, which will be used in tagging phase
    """

    def __init__(self, tokenize_func, get_tok2char_span_map_func):
        self._tokenize = tokenize_func
        self._get_tok2char_span_map = get_tok2char_span_map_func

    def transform_data(self, data, ori_format, dataset_type, add_id=True):
        """
        This function can only deal with three original format used in the previous works.
        If you want to feed new dataset to the model, just define your own function to transform data.
        data: original data
        ori_format: "casrel", "joint_re", "raw_nyt"
        dataset_type: "train", "valid", "test"; only for generate id for the data
        """
        normal_sample_list = []
        for ind, sample in tqdm(enumerate(data), desc="Transforming data format"):
            if ori_format == "casrel":
                text = sample["text"]
                rel_list = sample["triple_list"]
                subj_key, pred_key, obj_key = "subject", "predicate", "object"
            elif ori_format == "etl_span":
                text = " ".join(sample["tokens"])
                rel_list = sample["spo_list"]
                subj_key, pred_key, obj_key = 0, 1, 2
            elif ori_format == "raw_nyt":
                text = sample["sentText"]
                rel_list = sample["relationMentions"]
                subj_key, pred_key, obj_key = "em1Text", "label", "em2Text"
            else:
                raise ValueError("ori_format must in ['casrel', 'joint_re', 'raw_nyt']")

            normal_sample = {
                "text": text,
            }
            if add_id:
                normal_sample["id"] = "{}_{}".format(dataset_type, ind)
            normal_rel_list = []
            for rel in rel_list:
                normal_rel = {
                    "subject": rel[subj_key],
                    "predicate": rel[pred_key],
                    "object": rel[obj_key]
                }
                # normal_rel = rel
                normal_rel_list.append(normal_rel)
            normal_sample["relation_list"] = normal_rel_list
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
                end_ind = start_ind + max_seq_len

                char_span_list = tok2char_span[start_ind:end_ind]
                char_level_span = [char_span_list[0][0], char_span_list[-1][1]]
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
                else:  # train or valid dataset, only save spo and entities in the subtext
                    # spo
                    sub_rel_list = []
                    for rel in sample["relation_list"]:
                        subj_tok_span = rel["subj_tok_span"]
                        obj_tok_span = rel["obj_tok_span"]
                        # if subject and object are both in this subtext, add this spo to new sample
                        if subj_tok_span[0] >= start_ind and subj_tok_span[1] <= end_ind \
                                and obj_tok_span[0] >= start_ind and obj_tok_span[1] <= end_ind:
                            new_rel = copy.deepcopy(rel)
                            new_rel["subj_tok_span"] = [subj_tok_span[0] - start_ind,
                                                        subj_tok_span[1] - start_ind]  # start_ind: tok level offset
                            new_rel["obj_tok_span"] = [obj_tok_span[0] - start_ind, obj_tok_span[1] - start_ind]
                            new_rel["subj_char_span"][0] -= char_level_span[0]  # char level offset
                            new_rel["subj_char_span"][1] -= char_level_span[0]
                            new_rel["obj_char_span"][0] -= char_level_span[0]
                            new_rel["obj_char_span"][1] -= char_level_span[0]
                            sub_rel_list.append(new_rel)

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

                    # event
                    if "event_list" in sample:
                        sub_event_list = []
                        for event in sample["event_list"]:
                            trigger_tok_span = event["trigger_tok_span"]
                            if trigger_tok_span[1] > end_ind or trigger_tok_span[0] < start_ind:
                                continue
                            new_event = copy.deepcopy(event)
                            new_arg_list = []
                            for arg in new_event["argument_list"]:
                                if arg["tok_span"][0] >= start_ind and arg["tok_span"][1] <= end_ind:
                                    new_arg_list.append(arg)
                            new_event["argument_list"] = new_arg_list
                            sub_event_list.append(new_event)
                        new_sample["event_list"] = sub_event_list  # maybe empty

                    new_sample["entity_list"] = sub_ent_list  # maybe empty
                    new_sample["relation_list"] = sub_rel_list  # maybe empty
                    split_sample_list.append(new_sample)

                # all segments covered, no need to continue
                if end_ind > len(tokens):
                    break

            new_sample_list.extend(split_sample_list)
        return new_sample_list

    @staticmethod
    def _clean_sp_char(dataset):

        def clean_text(text):
            text = re.sub("�", "", text)
            # text = re.sub("([A-Za-z]+)", r" \1 ", text)
            # text = re.sub("(\d+)", r" \1 ", text)
            # text = re.sub("\s+", " ", text).strip()
            return text

        for sample in tqdm(dataset, desc="Clean"):
            sample["text"] = clean_text(sample["text"])
            for rel in sample["relation_list"]:
                rel["subject"] = clean_text(rel["subject"])
                rel["object"] = clean_text(rel["object"])
        return dataset

    @staticmethod
    def clean_data_wo_span(ori_data, separate=False, data_type="train"):
        """
        rm duplicate whitespaces
        and add whitespaces around tokens to keep special characters from them
        """

        def clean_text(text):
            text = re.sub(r"\s+", " ", text).strip()
            if separate:
                text = re.sub("([^A-Za-z0-9])", r" \1 ", text)
                text = re.sub(r"\s+", " ", text).strip()
            return text

        for sample in tqdm(ori_data, desc="clean data"):
            sample["text"] = clean_text(sample["text"])
            if data_type == "test":
                continue
            for rel in sample["relation_list"]:
                rel["subject"] = clean_text(rel["subject"])
                rel["object"] = clean_text(rel["object"])
        return ori_data

    @staticmethod
    def clean_data_w_span(ori_data):
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
            for rel in sample["relation_list"]:
                # rm whitespaces
                rel["subject"], rel["subj_char_span"] = strip_white(rel["subject"], rel["subj_char_span"])
                rel["object"], rel["obj_char_span"] = strip_white(rel["object"], rel["obj_char_span"])

                subj_char_span = rel["subj_char_span"]
                obj_char_span = rel["obj_char_span"]
                if rel["subject"] not in text or rel["subject"] != text[subj_char_span[0]:subj_char_span[1]] or \
                        rel["object"] not in text or rel["object"] != text[obj_char_span[0]:obj_char_span[1]]:
                    rel["stake"] = 0
                    bad = True

            if bad:
                bad_samples.append(copy.deepcopy(sample))

            new_rel_list = [rel for rel in sample["relation_list"] if "stake" not in rel]
            if len(new_rel_list) > 0:
                sample["relation_list"] = new_rel_list
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

    @staticmethod
    def _get_ent2char_spans(text, entities, ignore_subword_match=True):
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
                if not ignore_subword_match and re.match(r"\d+",
                                                         target_ent):  # avoid matching a inner number of a number
                    if (m.span()[0] - 1 >= 0 and re.match(r"\d", text_cp[m.span()[0] - 1])) or \
                            (m.span()[1] < len(text_cp) and re.match(r"\d", text_cp[m.span()[1]])):
                        continue
                span = [m.span()[0], m.span()[1] - 2] if ignore_subword_match else m.span()
                spans.append(span)
            ent2char_spans[ent] = spans
        return ent2char_spans

    def add_char_span(self, dataset, ignore_subword_match=True):
        miss_sample_list = []
        for sample in tqdm(dataset, desc="adding char level spans"):
            entities = [rel["subject"] for rel in sample["relation_list"]]
            entities.extend([rel["object"] for rel in sample["relation_list"]])
            if "entity_list" in sample:
                entities.extend([ent["text"] for ent in sample["entity_list"]])
            ent2char_spans = self._get_ent2char_spans(sample["text"], entities,
                                                      ignore_subword_match=ignore_subword_match)

            new_relation_list = []
            for rel in sample["relation_list"]:
                subj_char_spans = ent2char_spans[rel["subject"]]
                obj_char_spans = ent2char_spans[rel["object"]]
                for subj_sp in subj_char_spans:
                    for obj_sp in obj_char_spans:
                        new_relation_list.append({
                            "subject": rel["subject"],
                            "object": rel["object"],
                            "subj_char_span": subj_sp,
                            "obj_char_span": obj_sp,
                            "predicate": rel["predicate"],
                            "subject_type": rel["subject_type"],
                            "object_type": rel["object_type"]
                        })

            if len(sample["relation_list"]) > len(new_relation_list):
                miss_sample_list.append(sample)
            sample["relation_list"] = new_relation_list

            if "entity_list" in sample:
                new_ent_list = []
                for ent in sample["entity_list"]:
                    for char_sp in ent2char_spans[ent["text"]]:
                        new_ent_list.append({
                            "text": ent["text"],
                            "type": ent["type"],
                            "char_span": char_sp,
                        })
                sample["entity_list"] = new_ent_list
        return dataset, miss_sample_list

    def add_tok_span(self, dataset):
        """
        dataset must has char level span
        """

        def char_span2tok_span(char_span_, char2tok_span_):
            tok_span_list = char2tok_span_[char_span_[0]:char_span_[1]]
            tok_span = [tok_span_list[0][0], tok_span_list[-1][1]]
            return tok_span

        for sample in tqdm(dataset, desc="adding token level spans"):
            char2tok_span = self._get_char2tok_span(sample["text"])
            for rel in sample["relation_list"]:
                subj_char_span = rel["subj_char_span"]
                obj_char_span = rel["obj_char_span"]
                rel["subj_tok_span"] = char_span2tok_span(subj_char_span, char2tok_span)
                rel["obj_tok_span"] = char_span2tok_span(obj_char_span, char2tok_span)
            for ent in sample["entity_list"]:
                char_span = ent["char_span"]
                ent["tok_span"] = char_span2tok_span(char_span, char2tok_span)
            if "event_list" in sample:
                for event in sample["event_list"]:
                    event["trigger_tok_span"] = char_span2tok_span(event["trigger_char_span"], char2tok_span)
                    for arg in event["argument_list"]:
                        arg["tok_span"] = char_span2tok_span(arg["char_span"], char2tok_span)
        return dataset
