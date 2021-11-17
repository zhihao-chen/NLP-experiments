import copy
import re
import logging
import random
from abc import ABCMeta, abstractmethod

import torch
import networkx as nx

from nlp.models.model_util import Indexer
from nlp.processors.utils_ee import (unique_list,
                                     tok_span2char_span,
                                     extract_ent_fr_txt_by_char_sp,
                                     span_contains,
                                     merge_spans,
                                     join_segs)


class Tagger(metaclass=ABCMeta):
    @classmethod
    def additional_preprocess(cls, sample, **kwargs):
        return sample

    @abstractmethod
    def get_tag_size(self):
        pass

    @abstractmethod
    def get_tag_points(self, sample):
        """
        This function is for generating tag points

        sample: an example
        return points for tagging
        point: (start_pos, end_pos, tag_id)
        """
        pass

    @abstractmethod
    def tag(self, data):
        """
        This function is for generating tag points in batch

        data: examples
        return: data with points
        """
        pass

    @abstractmethod
    def decode(self, sample, pred_tags, pred_outs=None):
        """
        decoding function: to extract results by the predicted tag
        :param sample: an example (to offer text, tok2char_span for decoding)
        :param pred_tags: predicted tag id tensors converted from the outputs of the forward function,
                          it is a tuple or a single tag tensor
        :param pred_outs: the outputs of the forward function
        :return: predicted example
        """
        pass

    def decode_batch(self, sample_list, batch_pred_tags, batch_pred_outputs=None):
        pred_sample_list = []
        for ind in range(len(sample_list)):
            sample = sample_list[ind]
            pred_tags = [batch_pred_tag[ind] for batch_pred_tag in batch_pred_tags]
            pred_outs = [batch_pred_out[ind] for batch_pred_out in batch_pred_outputs] \
                if batch_pred_outputs is not None else None
            pred_sample = self.decode(sample, pred_tags, pred_outs)  # decoding one sample
            pred_sample_list.append(pred_sample)
        return pred_sample_list


class Tagger4SpanNER(Tagger):
    def __init__(self, data_anns):
        """
        :param data_anns: annotations, used to generate entity type and relation type dicts
        """
        super().__init__()
        ent_type_set = set()

        # entity type
        ent_type_set |= {ent["type"] for ent in data_anns["entity_list"]}
        ent_type_set = sorted(ent_type_set)
        self.ent_tag2id = {ent: idx for idx, ent in enumerate(ent_type_set)}
        self.id2ent_tag = {idx: t for t, idx in self.ent_tag2id.items()}

    def get_tag_size(self):
        return len(self.ent_tag2id)

    def get_tag_points(self, sample):
        points = set()

        if "entity_list" in sample:
            for ent in sample["entity_list"]:
                points.add((ent["tok_span"][0],
                            ent["tok_span"][1] - 1,
                            self.ent_tag2id[ent["type"]])
                           )
        return points

    def tag(self, data):
        for sample in data:
            sample["ent_points"] = self.get_tag_points(sample)

    def decode(self, sample, pred_tags, pred_outs=None):
        rel_list, ent_list = [], []
        predicted_shaking_tag = pred_tags[0]
        shk_points = Indexer.shaking_seq2points(predicted_shaking_tag)

        sample_idx, text = sample["id"], sample["text"]
        tok2char_span = sample["features"]["tok2char_span"]

        for sp in shk_points:
            ent_type = self.id2ent_tag[sp[2]]
            # for an entity, the start position can not be larger than the end pos.
            assert sp[0] <= sp[1]

            char_span_list = tok2char_span[sp[0]:sp[1] + 1]
            char_sp = [char_span_list[0][0], char_span_list[-1][1]]
            ent_text = text[char_sp[0]:char_sp[1]]
            entity = {
                "type": ent_type,
                "text": ent_text,
                "tok_span": [sp[0], sp[1] + 1],
                "char_span": char_sp,
            }
            ent_list.append(entity)

        pred_sample = copy.deepcopy(sample)
        pred_sample["entity_list"] = ent_list
        return pred_sample


class Tagger4TPLinkerPlus(Tagger):
    @classmethod
    def additional_preprocess(cls, sample, **kwargs):
        assert "entity_list" in sample
        fin_ent_list = copy.deepcopy(sample["entity_list"])
        fin_rel_list = copy.deepcopy(sample["relation_list"]) if "relation_list" in sample else []

        # add default entity type
        add_default_entity_type = kwargs["add_default_entity_type"]
        if add_default_entity_type is True:
            for ent in sample["entity_list"]:
                fin_ent_list.append({
                    "text": ent["text"],
                    "type": "EXT:DEFAULT",
                    "char_span": ent["char_span"],
                    "tok_span": ent["tok_span"],
                })

        classify_entities_by_relation = kwargs["classify_entities_by_relation"]
        if classify_entities_by_relation:
            for rel in fin_rel_list:
                # add rel types to entities
                fin_ent_list.append({
                    "text": rel["subject"],
                    "type": "REL:{}".format(rel["predicate"]),
                    "char_span": rel["subj_char_span"],
                    "tok_span": rel["subj_tok_span"],
                })
                fin_ent_list.append({
                    "text": rel["object"],
                    "type": "REL:{}".format(rel["predicate"]),
                    "char_span": rel["obj_char_span"],
                    "tok_span": rel["obj_tok_span"],
                })

        sample["entity_list"] = unique_list(fin_ent_list)
        sample["relation_list"] = unique_list(fin_rel_list)
        return sample

    def __init__(self, data_anns, **kwargs):
        """
        :param data: all data, used to generate entity type and relation type dicts
        """
        super().__init__()
        # generate entity type and relation type dicts
        rel_type_set = set()
        ent_type_set = set()

        # entity type
        ent_type_set |= {ent["type"] for ent in data_anns["entity_list"]}

        # relation type
        rel_type_set |= {rel["predicate"] for rel in data_anns["relation_list"]}
        rel_type_set = sorted(rel_type_set)
        ent_type_set = sorted(ent_type_set)
        self.ent2id = {ent: ind for ind, ent in enumerate(ent_type_set)}
        self.rel2id = {rel: ind for ind, rel in enumerate(rel_type_set)}
        self.id2ent = {ind: ent for ent, ind in self.ent2id.items()}
        self.id2rel = {ind: rel for rel, ind in self.rel2id.items()}

        self.separator = "\u2E80"
        self.rel_link_types = {"SH2OH",  # subject head to object head
                               "OH2SH",  # object head to subject head
                               "ST2OT",  # subject tail to object tail
                               "OT2ST",  # object tail to subject tail
                               }

        self.add_h2t_n_t2h_links = False
        if "add_h2t_n_t2h_links" in kwargs and kwargs["add_h2t_n_t2h_links"] is True:
            self.rel_link_types = self.rel_link_types.union({
                "SH2OT",  # subject head to object tail
                "OT2SH",  # object tail to subject head
                "ST2OH",  # subject tail to object head
                "OH2ST",  # object head to subject tail
            })
            self.add_h2t_n_t2h_links = True

        self.classify_entities_by_relation = kwargs["classify_entities_by_relation"]

        self.tags = {self.separator.join([rel, lt]) for rel in self.rel2id.keys() for lt in self.rel_link_types}
        self.tags |= {self.separator.join([ent, "EH2ET"]) for ent in
                      self.ent2id.keys()}  # EH2ET: entity head to entity tail

        self.tags = sorted(self.tags)
        self.tag2id = {t: idx for idx, t in enumerate(self.tags)}
        self.id2tag = {idx: t for t, idx in self.tag2id.items()}
        print(">>>>>>>>>>>>>>>>>>>>> tag_size: {} >>>>>>>>>>>>>>>>>>>>>>>".format(len(self.tag2id)))

    def get_tag_size(self):
        return len(self.tag2id)

    def tag(self, data):
        for sample in data:
            sample["tag_points"] = self.get_tag_points(sample)

    def get_tag_points(self, sample):
        """
        matrix_points: [(tok_pos1, tok_pos2, tag_id), ]
        """
        matrix_points = set()

        if "entity_list" in sample:
            for ent in sample["entity_list"]:
                ent_type = ent["type"]
                if ent_type not in self.ent2id:
                    logging.warning("ent_type: {} is not in training set".format(ent_type))
                    continue
                matrix_points.add(
                    (ent["tok_span"][0], ent["tok_span"][1] - 1,
                     self.tag2id[self.separator.join([ent_type, "EH2ET"])]))

        if "relation_list" in sample:
            for rel in sample["relation_list"]:
                subj_tok_span = rel["subj_tok_span"]
                obj_tok_span = rel["obj_tok_span"]
                rel = rel["predicate"]

                if rel not in self.rel2id:
                    logging.warning("rel: {} is not in the training set".format(rel))
                    continue

                # add related boundaries
                if subj_tok_span[0] <= obj_tok_span[0]:
                    matrix_points.add(
                        (subj_tok_span[0], obj_tok_span[0], self.tag2id[self.separator.join([rel, "SH2OH"])]))
                else:
                    matrix_points.add(
                        (obj_tok_span[0], subj_tok_span[0], self.tag2id[self.separator.join([rel, "OH2SH"])]))

                if subj_tok_span[1] <= obj_tok_span[1]:
                    matrix_points.add(
                        (subj_tok_span[1] - 1, obj_tok_span[1] - 1, self.tag2id[self.separator.join([rel, "ST2OT"])]))
                else:
                    matrix_points.add(
                        (obj_tok_span[1] - 1, subj_tok_span[1] - 1, self.tag2id[self.separator.join([rel, "OT2ST"])]))

                if self.add_h2t_n_t2h_links:
                    if subj_tok_span[0] <= obj_tok_span[1] - 1:
                        matrix_points.add(
                            (subj_tok_span[0], obj_tok_span[1] - 1, self.tag2id[self.separator.join([rel, "SH2OT"])]))
                    else:
                        matrix_points.add(
                            (obj_tok_span[1] - 1, subj_tok_span[0], self.tag2id[self.separator.join([rel, "OT2SH"])]))
                    if subj_tok_span[1] - 1 <= obj_tok_span[0]:
                        matrix_points.add(
                            (subj_tok_span[1] - 1, obj_tok_span[0], self.tag2id[self.separator.join([rel, "ST2OH"])]))
                    else:
                        matrix_points.add(
                            (obj_tok_span[0], subj_tok_span[1] - 1, self.tag2id[self.separator.join([rel, "OH2ST"])]))
        return list(matrix_points)

    def decode(self, sample, pred_tags, pred_outs=None):
        rel_list, ent_list = [], []
        pred_hsk_tag = pred_tags[0]
        hsk_points = Indexer.shaking_seq2points(pred_hsk_tag)
        pred_conf = None
        matrix_idx2shaking_idx = None
        if pred_outs is not None:
            pred_conf = torch.sigmoid(pred_outs[0])
            shaking_seq_len = pred_conf.size()[0]
            matrix_size = int((2 * shaking_seq_len + 0.25) ** 0.5 - 0.5)
            matrix_idx2shaking_idx = Indexer.get_matrix_idx2shaking_idx(matrix_size)

        sample_idx, text = sample["id"], sample["text"]
        tok2char_span = sample["features"]["tok2char_span"]

        # entity
        for pt in hsk_points:
            tag = self.id2tag[pt[2]]
            ent_type, link_type = tag.split(self.separator)
            # for an entity, the start position can not be larger than the end pos.
            if link_type != "EH2ET" or pt[0] > pt[1]:
                continue
            tok_sp = [pt[0], pt[1] + 1]
            char_span_list = tok2char_span[tok_sp[0]:tok_sp[1]]
            char_sp = [char_span_list[0][0], char_span_list[-1][1]]
            if char_sp[1] == 0:  # if [PAD] tokens are included, char_sp would be [*, 0]
                continue
            ent_text = text[char_sp[0]:char_sp[1]]
            conf = 1.
            if pred_conf is not None:
                shaking_idx = matrix_idx2shaking_idx[pt[0]][pt[1]]
                conf = pred_conf[shaking_idx][pt[2]].item()
            entity = {
                "type": ent_type,
                "text": ent_text,
                "tok_span": tok_sp,
                "char_span": char_sp,
                "conf": round(conf, 5),
            }
            ent_list.append(entity)

        # cand_ent_list4rel = []
        rel2candidate_ents = {
            "DEFAULT": []
        }
        for ent in ent_list:
            if "MASK:" in ent["type"]:
                continue
            if self.classify_entities_by_relation:
                if "REL:" in ent["type"]:
                    new_ent = copy.deepcopy(ent)
                    rel = re.sub("REL:", "", new_ent["type"])
                    if rel not in rel2candidate_ents:
                        rel2candidate_ents[rel] = []
                    rel2candidate_ents[rel].append(new_ent)
            else:
                rel2candidate_ents["DEFAULT"].append(ent)

        rel2link_type_map = {}
        for pt in hsk_points:
            tag = self.id2tag[pt[2]]
            rel, link_type = tag.split(self.separator)
            if link_type == "EH2ET":
                continue

            rel2link_type_map.setdefault(rel, {})
            if link_type[0] == "S":
                index_pair = "{},{}".format(pt[0], pt[1])

            else:
                # reverse
                index_pair = "{},{}".format(pt[1], pt[0])
                lt_split = link_type.split("2")
                link_type = "2".join([lt_split[1], lt_split[0]])

            rel2link_type_map[rel].setdefault(index_pair, {})

            if pred_conf is not None:
                shaking_idx = matrix_idx2shaking_idx[pt[0]][pt[1]]
                conf = pred_conf[shaking_idx][pt[2]].item()
                rel2link_type_map[rel][index_pair][link_type] = conf
            else:
                rel2link_type_map[rel][index_pair][link_type] = 1.

        for rel, link_type_map in rel2link_type_map.items():
            cand_ent_list4rel = rel2candidate_ents.get(rel, []) \
                if self.classify_entities_by_relation else rel2candidate_ents["DEFAULT"]

            for subj in cand_ent_list4rel:
                for obj in cand_ent_list4rel:
                    h2h_ids = "{},{}".format(subj["tok_span"][0], obj["tok_span"][0])
                    t2t_ids = "{},{}".format(subj["tok_span"][1] - 1, obj["tok_span"][1] - 1)
                    h2t_ids = "{},{}".format(subj["tok_span"][0], obj["tok_span"][1] - 1)
                    t2h_ids = "{},{}".format(subj["tok_span"][1] - 1, obj["tok_span"][0])

                    rel_exist = False
                    edge_conf = 1.
                    if self.add_h2t_n_t2h_links:
                        if h2h_ids in link_type_map and "SH2OH" in link_type_map[h2h_ids] and \
                                t2t_ids in link_type_map and "ST2OT" in link_type_map[t2t_ids] and \
                                h2t_ids in link_type_map and "SH2OT" in link_type_map[h2t_ids] and \
                                t2h_ids in link_type_map and "ST2OH" in link_type_map[t2h_ids]:
                            rel_exist = True
                            edge_conf = (link_type_map[h2h_ids]["SH2OH"] * link_type_map[t2t_ids]["ST2OT"]
                                         * link_type_map[h2t_ids]["SH2OT"] * link_type_map[t2h_ids]["ST2OH"]) ** 0.25
                    else:
                        if h2h_ids in link_type_map and "SH2OH" in link_type_map[h2h_ids] and \
                                t2t_ids in link_type_map and "ST2OT" in link_type_map[t2t_ids]:
                            rel_exist = True
                            edge_conf = (link_type_map[h2h_ids]["SH2OH"] * link_type_map[t2t_ids]["ST2OT"]) ** 0.5

                    if rel_exist:
                        rel_list.append({
                            "subject": subj["text"],
                            "object": obj["text"],
                            "subj_tok_span": [subj["tok_span"][0], subj["tok_span"][1]],
                            "obj_tok_span": [obj["tok_span"][0], obj["tok_span"][1]],
                            "subj_char_span": [subj["char_span"][0], subj["char_span"][1]],
                            "obj_char_span": [obj["char_span"][0], obj["char_span"][1]],
                            "predicate": rel,
                            "conf": round((edge_conf * subj["conf"] * obj["conf"]) ** (1 / 3), 5)
                        })

        pred_sample = sample

        # filter extra relations
        pred_sample["relation_list"] = unique_list(
            [rel for rel in rel_list if "EXT:" not in rel["predicate"]])
        # filter extra entities
        ent_types2filter = {"REL:", "EXT:"}
        ent_filter_pattern = "({})".format("|".join(ent_types2filter))
        pred_sample["entity_list"] = unique_list(
            [ent for ent in ent_list if re.search(ent_filter_pattern, ent["type"]) is None])

        return pred_sample


class Tagger4RAIN(Tagger4TPLinkerPlus):
    def __init__(self, data_anns, **kwargs):
        """
        :param data_anns: all data annatations, used to generate entity type and relation type dicts
        """
        super(Tagger4RAIN, self).__init__(data_anns, **kwargs)

        self.rel_tags = {self.separator.join([rel, lt]) for rel in self.rel2id.keys() for lt in self.rel_link_types}
        self.rel_tag2id = {t: idx for idx, t in enumerate(sorted(self.rel_tags))}
        self.id2rel_tag = {idx: t for t, idx in self.rel_tag2id.items()}

        self.ent_tag2id = self.ent2id
        self.id2ent_tag = {idx: t for t, idx in self.ent_tag2id.items()}

        print(">>>>>>>>>>>>>>>> ent_tag_size: {}; rel_tag_size: {} >>>>>>>>>>>>>>>>>>>>".format(len(self.ent_tag2id),
                                                                                                len(self.rel_tag2id)))

    def get_tag_size(self):
        return len(self.ent_tag2id), len(self.rel_tag2id)

    def tag(self, data):
        for sample in data:
            ent_points, rel_points = self.get_tag_points(sample)
            sample["ent_points"] = ent_points
            sample["rel_points"] = rel_points

    def get_tag_points(self, sample):
        """
        matrix_points: [(tok_pos1, tok_pos2, tag_id), ]
        """
        ent_matrix_points, rel_matrix_points = [], []

        if "entity_list" in sample:
            for ent in sample["entity_list"]:
                tag = ent["type"]
                if tag not in self.ent_tag2id:
                    logging.warning("ent_type: {} is not in training set".format(tag))
                    continue
                point = (ent["tok_span"][0], ent["tok_span"][-1] - 1,
                         self.ent_tag2id[tag],
                         )
                ent_matrix_points.append(point)

        if "relation_list" in sample:
            for rel in sample["relation_list"]:
                subj_tok_span = rel["subj_tok_span"]
                obj_tok_span = rel["obj_tok_span"]
                rel = rel["predicate"]

                if rel not in self.rel2id:
                    logging.warning("rel: {} is not in the training set".format(rel))
                    continue

                # add related boundaries
                rel_matrix_points.append(
                    (subj_tok_span[0], obj_tok_span[0], self.rel_tag2id[self.separator.join([rel, "SH2OH"])]))
                rel_matrix_points.append(
                    (subj_tok_span[1] - 1, obj_tok_span[1] - 1, self.rel_tag2id[self.separator.join([rel, "ST2OT"])]))

                if self.add_h2t_n_t2h_links:
                    rel_matrix_points.append(
                        (subj_tok_span[0], obj_tok_span[1] - 1, self.rel_tag2id[self.separator.join([rel, "SH2OT"])]))
                    rel_matrix_points.append(
                        (subj_tok_span[1] - 1, obj_tok_span[0], self.rel_tag2id[self.separator.join([rel, "ST2OH"])]))

        return unique_list(ent_matrix_points), unique_list(rel_matrix_points)

    def decode(self, sample, pred_tags, pred_outs=None):
        """
        sample: to provide tok2char_span map and text
        pred_tags: predicted tags
        """
        rel_list, ent_list = [], []
        pred_ent_tag, pred_rel_tag = pred_tags[0], pred_tags[1]
        pred_ent_conf, pred_rel_conf = None, None
        matrix_idx2shaking_idx = None
        if pred_outs is not None:
            pred_ent_conf, pred_rel_conf = torch.sigmoid(pred_outs[0]), torch.sigmoid(pred_outs[1])
            shaking_seq_len = pred_ent_conf.size()[0]
            matrix_size = int((2 * shaking_seq_len + 0.25) ** 0.5 - 0.5)
            matrix_idx2shaking_idx = Indexer.get_matrix_idx2shaking_idx(matrix_size)
        ent_points = Indexer.shaking_seq2points(pred_ent_tag)
        rel_points = Indexer.matrix2points(pred_rel_tag)

        sample_idx, text = sample["id"], sample["text"]
        tok2char_span = sample["features"]["tok2char_span"]

        # entity
        for pt in ent_points:
            ent_tag = self.id2ent_tag[pt[2]]
            ent_type = ent_tag
            tok_sp = [pt[0], pt[1] + 1]
            char_span_list = tok2char_span[tok_sp[0]:tok_sp[1]]
            char_sp = [char_span_list[0][0], char_span_list[-1][1]]
            if char_sp[1] == 0:  # if [PAD] tokens are included, char_sp would be [*, 0]
                continue
            ent_text = text[char_sp[0]:char_sp[1]]
            conf = 1.
            if pred_ent_conf is not None:
                shaking_idx = matrix_idx2shaking_idx[pt[0]][pt[1]]
                conf = pred_ent_conf[shaking_idx][pt[2]].item()
            entity = {
                "type": ent_type,
                "text": ent_text,
                "tok_span": tok_sp,
                "char_span": char_sp,
                "conf": round(conf, 5),
            }
            ent_list.append(entity)

        # cand_ent_list4rel = []
        rel2candidate_ents = {
            "DEFAULT": []
        }
        for ent in ent_list:
            if "MASK:" in ent["type"]:
                continue
            if self.classify_entities_by_relation:
                if "REL:" in ent["type"]:
                    new_ent = copy.deepcopy(ent)
                    rel = re.sub("REL:", "", new_ent["type"])
                    if rel not in rel2candidate_ents:
                        rel2candidate_ents[rel] = []
                    rel2candidate_ents[rel].append(new_ent)
            else:
                rel2candidate_ents["DEFAULT"].append(ent)

        rel2link_type_map = {}
        for pt in rel_points:
            tag = self.id2rel_tag[pt[2]]
            rel, link_type = tag.split(self.separator)
            if rel not in rel2link_type_map:
                rel2link_type_map[rel] = {}
            index_pair = "{},{}".format(pt[0], pt[1])
            if index_pair not in rel2link_type_map[rel]:
                rel2link_type_map[rel][index_pair] = {}
            rel2link_type_map[rel][index_pair][link_type] = pred_rel_conf[pt[0]][pt[1]][pt[2]].item() \
                if pred_rel_conf is not None else 1.

        for rel, link_type_map in rel2link_type_map.items():
            cand_ent_list4rel = rel2candidate_ents.get(rel, []) \
                if self.classify_entities_by_relation else rel2candidate_ents["DEFAULT"]
            for subj in cand_ent_list4rel:
                for obj in cand_ent_list4rel:
                    h2h_ids = "{},{}".format(subj["tok_span"][0], obj["tok_span"][0])
                    t2t_ids = "{},{}".format(subj["tok_span"][1] - 1, obj["tok_span"][1] - 1)
                    h2t_ids = "{},{}".format(subj["tok_span"][0], obj["tok_span"][1] - 1)
                    t2h_ids = "{},{}".format(subj["tok_span"][1] - 1, obj["tok_span"][0])

                    rel_exist = False
                    edge_conf = 1.
                    if self.add_h2t_n_t2h_links:
                        if h2h_ids in link_type_map and "SH2OH" in link_type_map[h2h_ids] and \
                                t2t_ids in link_type_map and "ST2OT" in link_type_map[t2t_ids] and \
                                h2t_ids in link_type_map and "SH2OT" in link_type_map[h2t_ids] and \
                                t2h_ids in link_type_map and "ST2OH" in link_type_map[t2h_ids]:
                            rel_exist = True
                            edge_conf = (link_type_map[h2h_ids]["SH2OH"] * link_type_map[t2t_ids]["ST2OT"]
                                         * link_type_map[h2t_ids]["SH2OT"] * link_type_map[t2h_ids]["ST2OH"]) ** 0.25
                    else:
                        if h2h_ids in link_type_map and "SH2OH" in link_type_map[h2h_ids] and \
                                t2t_ids in link_type_map and "ST2OT" in link_type_map[t2t_ids]:
                            rel_exist = True
                            edge_conf = (link_type_map[h2h_ids]["SH2OH"] * link_type_map[t2t_ids]["ST2OT"]) ** 0.5

                    if rel_exist:
                        rel_list.append({
                            "subject": subj["text"],
                            "object": obj["text"],
                            "subj_tok_span": [subj["tok_span"][0], subj["tok_span"][1]],
                            "obj_tok_span": [obj["tok_span"][0], obj["tok_span"][1]],
                            "subj_char_span": [subj["char_span"][0], subj["char_span"][1]],
                            "obj_char_span": [obj["char_span"][0], obj["char_span"][1]],
                            "predicate": rel,
                            "conf": round((edge_conf * subj["conf"] * obj["conf"]) ** (1 / 3), 5)
                        })

        pred_sample = sample

        # filter extra relations
        pred_sample["relation_list"] = unique_list(
            [rel for rel in rel_list if "EXT:" not in rel["predicate"]])
        # filter extra entities
        ent_types2filter = {"REL:", "EXT:"}
        ent_filter_pattern = "({})".format("|".join(ent_types2filter))
        pred_sample["entity_list"] = unique_list(
            [ent for ent in ent_list if re.search(ent_filter_pattern, ent["type"]) is None])

        return pred_sample


def create_rebased_ee_tagger(base_class):
    class REBasedEETagger(base_class):
        def __init__(self, data_anns, *args, **kwargs):
            super(REBasedEETagger, self).__init__(data_anns, *args, **kwargs)
            self.event_type2arg_rols = {}

            for event in data_anns["event_list"]:
                event_type = event["event_type"]
                for arg in event["argument_list"]:
                    if event_type not in self.event_type2arg_rols:
                        self.event_type2arg_rols[event_type] = set()
                    self.event_type2arg_rols[event_type].add(arg["type"])

        @classmethod
        def additional_preprocess(cls, sample, **kwargs):
            separator = "\u2E82"
            # transform event list to relation list and entity list
            fin_ent_list = []
            fin_rel_list = []
            for event in sample["event_list"]:
                fin_ent_list.append({
                    "text": event["trigger"],
                    "type": "EE:{}{}{}".format("Trigger", separator, event["event_type"]),
                    "char_span": event["trigger_char_span"],
                    "tok_span": event["trigger_tok_span"],
                })
                for arg in event["argument_list"]:
                    fin_ent_list.append({
                        "text": arg["text"],
                        "type": "EE:{}{}{}".format("Argument", separator, arg["type"]),
                        "char_span": arg["char_span"],
                        "tok_span": arg["tok_span"],
                    })
                    fin_rel_list.append({
                        "subject": arg["text"],
                        "subj_char_span": arg["char_span"],
                        "subj_tok_span": arg["tok_span"],
                        "object": event["trigger"],
                        "obj_char_span": event["trigger_char_span"],
                        "obj_tok_span": event["trigger_tok_span"],
                        "predicate": "ARG2TRI",
                    })
                    fin_rel_list.append({
                        "subject": arg["text"],
                        "subj_char_span": arg["char_span"],
                        "subj_tok_span": arg["tok_span"],
                        "object": event["trigger"],
                        "obj_char_span": event["trigger_char_span"],
                        "obj_tok_span": event["trigger_tok_span"],
                        "predicate": "EE:{}{}{}".format(arg["type"], separator, event["event_type"]),
                    })
            sample["relation_list"] = unique_list(fin_rel_list)

            # extend original entity list
            if "entity_list" in sample:
                fin_ent_list.extend(sample["entity_list"])
            sample["entity_list"] = unique_list(fin_ent_list)
            return super().additional_preprocess(sample, **kwargs)

        def decode(self, sample, pred_tags, pred_outs=None):
            pred_sample = super(REBasedEETagger, self).decode(sample, pred_tags, pred_outs)
            pred_sample = self._trans2ee(pred_sample)
            # filter extra entities and relations
            pred_sample["entity_list"] = [ent for ent in pred_sample["entity_list"] if "EE:" not in ent["type"]]
            pred_sample["relation_list"] = [rel for rel in pred_sample["relation_list"] if
                                            "EE:" not in rel["predicate"]]
            return pred_sample

        def _trans2ee(self, sample):
            # filter tags with EE:
            new_rel_list, new_ent_list = [], []
            for rel in sample["relation_list"]:
                if rel["predicate"].split(":")[0] == "EE":
                    new_rel = copy.deepcopy(rel)
                    new_rel["predicate"] = re.sub(r"EE:", "", new_rel["predicate"])
                    new_rel_list.append(new_rel)
            for ent in sample["entity_list"]:
                if ent["type"].split(":")[0] == "EE":
                    new_ent = copy.deepcopy(ent)
                    new_ent["type"] = re.sub(r"EE:", "", new_ent["type"])
                    new_ent_list.append(new_ent)
            rel_list, ent_list = new_rel_list, new_ent_list

            # decoding
            separator = "\u2E82"
            type_wise_edges = []
            arg_offset2roles = {}
            arg_mark2arg = {}
            tri_offset2event_types = {}
            tri_mark2trigger = {}
            for ent in ent_list:
                arg_tri, role = ent["type"].split(separator)
                tok_offset = "{},{}".format(*ent["tok_span"])
                if arg_tri == "Argument":
                    arg_offset2roles.setdefault(tok_offset, set()).add(role)

                    arg = copy.deepcopy(ent)
                    arg["type"] = role
                    arg_mark2arg["{},{}".format(tok_offset, role)] = arg

                elif arg_tri == "Trigger":
                    event_type = role
                    tri_offset2event_types.setdefault(tok_offset, set()).add(event_type)
                    tri_mark2trigger["{},{}".format(tok_offset, event_type)] = {
                        "trigger": ent["text"],
                        "trigger_tok_span": ent["tok_span"],
                        "trigger_char_span": ent["char_span"],
                        "event_type": event_type,
                    }

            for rel in rel_list:
                if rel["predicate"] == "ARG2TRI":
                    arg_offset = "{},{}".format(*rel["subj_tok_span"])
                    tri_mark = "{},{}".format(*rel["obj_tok_span"])
                    arg_roles = arg_offset2roles.get(arg_offset, set())
                    event_types = tri_offset2event_types.get(tri_mark, set())
                    if len(arg_roles) == 1 and \
                            len(event_types) == 1:
                        arg_role = list(arg_roles)[0]
                        event_type = list(event_types)[0]
                        if arg_role in self.event_type2arg_rols[event_type]:
                            rel_cp = copy.deepcopy(rel)
                            rel_cp["predicate"] = separator.join([arg_role, event_type])
                            type_wise_edges.append(rel_cp)
                else:
                    type_wise_edges.append(rel)

            tri_mark2args = {}
            arg_used_mem = set()
            for edge in type_wise_edges:
                arg_role, event_type = edge["predicate"].split(separator)
                tri_mark = "{},{},{}".format(*edge["obj_tok_span"], event_type)
                tri_mark2trigger[tri_mark] = {
                    "trigger": edge["object"],
                    "trigger_tok_span": edge["obj_tok_span"],
                    "trigger_char_span": edge["obj_char_span"],
                    "event_type": event_type,
                }
                tri_mark2args.setdefault(tri_mark, []).append({
                    "text": edge["subject"],
                    "char_span": edge["subj_char_span"],
                    "tok_span": edge["subj_tok_span"],
                    "type": arg_role,
                })
                arg_mark = "{},{},{}".format(*edge["subj_tok_span"], arg_role)
                arg_used_mem.add(arg_mark)

            event_list = []
            for trigger_mark, trigger in tri_mark2trigger.items():
                arg_list = unique_list(tri_mark2args.get(trigger_mark, []))
                if len(arg_list) == 0:  # if it is a single-node trigger, add all possible arguments
                    arg_list = []
                    for arg_mark, arg in arg_mark2arg.items():
                        if arg_mark not in arg_used_mem and \
                                arg["type"] in self.event_type2arg_rols[trigger["event_type"]]:
                            arg_list.append(arg)
                            arg_mark = "{},{},{}".format(*arg["tok_span"], arg["type"])
                            arg_used_mem.add(arg_mark)

                event_list.append({
                    **trigger,
                    "argument_list": arg_list,
                })

            sample["event_list"] = event_list
            return sample

    return REBasedEETagger


def create_rebased_tfboys4doc_ee_tagger(base_class):
    class REBasedTFBoys4DocEETagger(base_class):
        def __init__(self, data_anns, *args, **kwargs):
            super(REBasedTFBoys4DocEETagger, self).__init__(data_anns, *args, **kwargs)
            self.event_type2arg_rols = {}
            for event in data_anns["event_list"]:
                event_type = event["event_type"]
                for arg in event["argument_list"]:
                    self.event_type2arg_rols.setdefault(event_type, set()).add(arg["type"])

            self.dtm_arg_type_by_edges = kwargs["dtm_arg_type_by_edges"]

        @classmethod
        def additional_preprocess(cls, sample, **kwargs):
            separator = "\u2E82"
            fin_ent_list = []
            fin_rel_list = []
            for event in sample["event_list"]:
                event_type = event["event_type"]
                arg_list = copy.deepcopy(event["argument_list"])

                if "trigger" in event:
                    pseudo_arg = {
                        "type": "Trigger",
                        "char_span": event["trigger_char_span"],
                        "tok_span": event["trigger_tok_span"],
                        "text": event["trigger"],
                    }
                    arg_list += [pseudo_arg]

                for i, arg_i in enumerate(arg_list):
                    ch_sp_list_i = arg_i["char_span"]
                    tk_sp_list_i = arg_i["tok_span"]
                    if type(arg_i["char_span"][0]) is not list:
                        ch_sp_list_i = [arg_i["char_span"], ]
                        tk_sp_list_i = [arg_i["tok_span"], ]

                    for sp_idx, ch_sp in enumerate(ch_sp_list_i):
                        tk_sp = tk_sp_list_i[sp_idx]
                        fin_ent_list.append({
                            "text": arg_i["text"],
                            "type": "EE:{}{}{}".format(event_type, separator, arg_i["type"]),
                            "char_span": ch_sp,
                            "tok_span": tk_sp,
                        })

                    for j, arg_j in enumerate(arg_list):
                        assert type(arg_j["char_span"][0]) is list
                        ch_sp_list_j = arg_j["char_span"]
                        tk_sp_list_j = arg_j["tok_span"]

                        for sp_idx_i, ch_sp_i in enumerate(ch_sp_list_i):
                            for sp_idx_j, ch_sp_j in enumerate(ch_sp_list_j):
                                tk_sp_i = tk_sp_list_i[sp_idx_i]
                                tk_sp_j = tk_sp_list_j[sp_idx_j]

                                fin_rel_list.append({
                                    "subject": arg_i["text"],
                                    "subj_char_span": ch_sp_i,
                                    "subj_tok_span": tk_sp_i,
                                    "object": arg_j["text"],
                                    "obj_char_span": ch_sp_j,
                                    "obj_tok_span": tk_sp_j,
                                    "predicate": "EE:{}".format(separator.join(["IN_SAME_EVENT", event_type])),
                                })
                                if kwargs["dtm_arg_type_by_edges"]:
                                    fin_rel_list.append({
                                        "subject": arg_i["text"],
                                        "subj_char_span": ch_sp_i,
                                        "subj_tok_span": tk_sp_i,
                                        "object": arg_j["text"],
                                        "obj_char_span": ch_sp_j,
                                        "obj_tok_span": tk_sp_j,
                                        "predicate": "EE:{}".format(separator.join([arg_i["type"], arg_j["type"]])),
                                    })

            sample["entity_list"] = fin_ent_list
            sample["relation_list"] = fin_rel_list
            return super().additional_preprocess(sample, **kwargs)

        def decode(self, sample, pred_tags, pred_outs=None):
            pred_sample = super(REBasedTFBoys4DocEETagger, self).decode(sample, pred_tags, pred_outs)
            pred_sample = self._trans(pred_sample)

            del pred_sample["entity_list"]
            del pred_sample["relation_list"]
            return pred_sample

        def _trans(self, sample):
            rel_list = sample["relation_list"]
            ent_list = sample["entity_list"]

            # choose tags with EE:
            new_rel_list, new_ent_list = [], []
            for rel in rel_list:
                if rel["predicate"].split(":")[0] == "EE":
                    new_rel = copy.deepcopy(rel)
                    new_rel["predicate"] = re.sub(r"EE:", "", new_rel["predicate"])
                    new_rel_list.append(new_rel)
            for ent in ent_list:
                if ent["type"].split(":")[0] == "EE":
                    new_ent = copy.deepcopy(ent)
                    new_ent["type"] = re.sub(r"EE:", "", new_ent["type"])
                    new_ent_list.append(new_ent)

            # decoding
            tok2char_span = sample["features"]["tok2char_span"]
            text = sample["text"]
            separator = "\u2E82"
            event2graph = {}
            offsets2arg_pair_rel = {}
            for rel in new_rel_list:
                subj_offset_str = "{},{}".format(*rel["subj_tok_span"])
                obj_offset_str = "{},{}".format(*rel["obj_tok_span"])

                if "IN_SAME_EVENT" in rel["predicate"]:
                    _, event_type = rel["predicate"].split(separator)
                    if event_type not in event2graph:
                        event2graph[event_type] = nx.Graph()
                    event2graph[event_type].add_edge(subj_offset_str, obj_offset_str)
                else:
                    offset_str4arg_pair = separator.join([subj_offset_str, obj_offset_str])
                    if offset_str4arg_pair not in offsets2arg_pair_rel:
                        offsets2arg_pair_rel[offset_str4arg_pair] = set()
                    offsets2arg_pair_rel[offset_str4arg_pair].add(rel["predicate"])

            event2role_map = {}
            for ent in new_ent_list:
                event_type, role = ent["type"].split(separator)
                offset_str = "{},{}".format(*ent["tok_span"])
                if event_type not in event2role_map:
                    event2role_map[event_type] = {}
                if offset_str not in event2role_map[event_type]:
                    event2role_map[event_type][offset_str] = set()
                event2role_map[event_type][offset_str].add(role)

                if event_type not in event2graph:
                    event2graph[event_type] = nx.Graph()
                event2graph[event_type].add_node(offset_str)

            # find events (cliques) under every event type
            event_list = []
            for event_type, graph in event2graph.items():
                role_map = event2role_map.get(event_type, dict())
                cliques = list(nx.find_cliques(graph))  # all maximal cliques

                for cli in cliques:
                    event = {
                        "event_type": event_type,
                    }
                    arguments = []
                    for offset_str in cli:
                        start, end = offset_str.split(",")
                        tok_span = [int(start), int(end)]
                        char_span = tok_span2char_span(tok_span, tok2char_span)
                        arg_text = extract_ent_fr_txt_by_char_sp(char_span, text)
                        role_set = role_map.get(offset_str, set())

                        if self.dtm_arg_type_by_edges:
                            role_set_fin = set()
                            if len(role_set) == 1:
                                role_set_fin.add(list(role_set)[0])
                            else:  # determine the role by the edge
                                min_edge_num = 1 << 31
                                can_role_set = set()
                                for offset_str_j in cli:
                                    arg_p_set = offsets2arg_pair_rel.get(separator.join([offset_str, offset_str_j]),
                                                                         set())
                                    if len(arg_p_set) != 0 and len(arg_p_set) < min_edge_num:
                                        min_edge_num = len(arg_p_set)
                                        can_role_set = {arg_p.split(separator)[0] for arg_p in arg_p_set}
                                role_set_fin = can_role_set

                            if len(role_set_fin) == 1:
                                role_set = role_set_fin

                        for role in role_set:
                            if role in self.event_type2arg_rols[event_type] or role == "Trigger":
                                arguments.append({
                                    "text": arg_text,
                                    "type": role,
                                    "char_span": char_span,
                                    "tok_span": tok_span,
                                })

                    # combine arg spans:
                    arguments_combined = []
                    arg_text2args = {}
                    for arg in arguments:
                        arg_text2args.setdefault(separator.join([arg["type"], arg["text"]]), []).append(arg)
                    for role_argtext, args in arg_text2args.items():
                        new_tk_sps = [a["tok_span"] for a in args]
                        new_ch_sps = [a["char_span"] for a in args]
                        role, arg_text = role_argtext.split(separator)
                        arguments_combined.append({
                            "text": arg_text,
                            "type": role,
                            "char_span": new_ch_sps,
                            "tok_span": new_tk_sps,
                        })
                    arguments = arguments_combined

                    # find trigger
                    new_argument_list = []
                    triggers = []
                    for arg in arguments:
                        if arg["type"] == "Trigger":
                            triggers.append(arg)
                        else:
                            new_argument_list.append(arg)

                    if len(triggers) > 0:
                        trigger = random.choice(triggers)
                        event["trigger"] = trigger["text"]
                        event["trigger_tok_span"] = trigger["tok_span"]
                        event["trigger_char_span"] = trigger["char_span"]
                        event["trigger_list"] = triggers

                    # if the role sets corresponding to the nodes are all empty,
                    # this clique is invalid and the corresponding event without argument list and triggers
                    # will not be appended into the event list.
                    if len(new_argument_list) > 0 or "trigger" in event:
                        event["argument_list"] = new_argument_list
                        event_list.append(event)

            sample["event_list"] = event_list
            return sample

    return REBasedTFBoys4DocEETagger


def create_rebased_tfboys_tagger(base_class):
    class REBasedTFBoysTagger(base_class):
        def __init__(self, data_anns, *args, **kwargs):
            super(REBasedTFBoysTagger, self).__init__(data_anns, *args, **kwargs)
            self.event_type2arg_rols = {}

            for event in data_anns["event_list"]:
                event_type = event["event_type"]
                for arg in event["argument_list"]:
                    self.event_type2arg_rols.setdefault(event_type, set()).add(arg["type"])

            self.dtm_arg_type_by_edges = kwargs["dtm_arg_type_by_edges"]

        @classmethod
        def additional_preprocess(cls, sample, **kwargs):
            separator = "\u2E82"
            fin_ent_list = []
            fin_rel_list = []
            for event in sample["event_list"]:
                event_type = event["event_type"]
                arg_list = copy.deepcopy(event["argument_list"])

                if "trigger" in event:
                    pseudo_arg = {
                        "type": "Trigger",
                        "char_span": event["trigger_char_span"],
                        "tok_span": event["trigger_tok_span"],
                        "text": event["trigger"],
                    }

                    arg_list += [pseudo_arg]

                for i, arg_i in enumerate(arg_list):
                    ch_sp_list_i = arg_i["char_span"]
                    tk_sp_list_i = arg_i["tok_span"]
                    if type(arg_i["char_span"][0]) is not list:
                        ch_sp_list_i = [arg_i["char_span"], ]
                        tk_sp_list_i = [arg_i["tok_span"], ]

                    for sp_idx, ch_sp in enumerate(ch_sp_list_i):
                        tk_sp = tk_sp_list_i[sp_idx]
                        fin_ent_list.append({
                            "text": arg_i["text"],
                            "type": "EE:{}{}{}".format(event_type, separator, arg_i["type"]),
                            "char_span": ch_sp,
                            "tok_span": tk_sp,
                        })

                    for j, arg_j in enumerate(arg_list):
                        ch_sp_list_j = arg_j["char_span"]
                        tk_sp_list_j = arg_j["tok_span"]
                        if type(arg_j["char_span"][0]) is not list:
                            ch_sp_list_j = [arg_j["char_span"], ]
                            tk_sp_list_j = [arg_j["tok_span"], ]

                        for sp_idx_i, ch_sp_i in enumerate(ch_sp_list_i):
                            for sp_idx_j, ch_sp_j in enumerate(ch_sp_list_j):
                                tk_sp_i = tk_sp_list_i[sp_idx_i]
                                tk_sp_j = tk_sp_list_j[sp_idx_j]

                                fin_rel_list.append({
                                    "subject": arg_i["text"],
                                    "subj_char_span": ch_sp_i,
                                    "subj_tok_span": tk_sp_i,
                                    "object": arg_j["text"],
                                    "obj_char_span": ch_sp_j,
                                    "obj_tok_span": tk_sp_j,
                                    "predicate": "EE:{}".format(separator.join(["IN_SAME_EVENT", event_type])),
                                })
                                if kwargs["dtm_arg_type_by_edges"]:
                                    fin_rel_list.append({
                                        "subject": arg_i["text"],
                                        "subj_char_span": ch_sp_i,
                                        "subj_tok_span": tk_sp_i,
                                        "object": arg_j["text"],
                                        "obj_char_span": ch_sp_j,
                                        "obj_tok_span": tk_sp_j,
                                        "predicate": "EE:{}".format(separator.join([arg_i["type"], arg_j["type"]])),
                                    })

            sample["entity_list"] = fin_ent_list
            sample["relation_list"] = fin_rel_list
            return super().additional_preprocess(sample, **kwargs)

        def decode(self, sample, pred_tags, pred_outs):
            pred_sample = super(REBasedTFBoysTagger, self).decode(sample, pred_tags, pred_outs)
            pred_sample = self._trans(pred_sample)

            del pred_sample["entity_list"]
            del pred_sample["relation_list"]
            return pred_sample

        def _trans(self, sample):
            rel_list = sample["relation_list"]
            ent_list = sample["entity_list"]

            # choose tags with EE:
            new_rel_list, new_ent_list = [], []
            for rel in rel_list:
                if rel["predicate"].split(":")[0] == "EE":
                    new_rel = copy.deepcopy(rel)
                    new_rel["predicate"] = re.sub(r"EE:", "", new_rel["predicate"])
                    new_rel_list.append(new_rel)
            for ent in ent_list:
                if ent["type"].split(":")[0] == "EE":
                    new_ent = copy.deepcopy(ent)
                    new_ent["type"] = re.sub(r"EE:", "", new_ent["type"])
                    new_ent_list.append(new_ent)

            # decoding
            tok2char_span = sample["features"]["tok2char_span"]
            text = sample["text"]
            separator = "\u2E82"
            event2graph = {}
            offsets2arg_pair_rel = {}
            for rel in new_rel_list:
                subj_offset_str = "{},{}".format(*rel["subj_tok_span"])
                obj_offset_str = "{},{}".format(*rel["obj_tok_span"])

                if "IN_SAME_EVENT" in rel["predicate"]:
                    _, event_type = rel["predicate"].split(separator)
                    if event_type not in event2graph:
                        event2graph[event_type] = nx.Graph()
                    event2graph[event_type].add_edge(subj_offset_str, obj_offset_str)
                else:
                    offset_str4arg_pair = separator.join([subj_offset_str, obj_offset_str])
                    if offset_str4arg_pair not in offsets2arg_pair_rel:
                        offsets2arg_pair_rel[offset_str4arg_pair] = set()
                    offsets2arg_pair_rel[offset_str4arg_pair].add(rel["predicate"])

            event2role_map = {}
            for ent in new_ent_list:
                event_type, role = ent["type"].split(separator)
                offset_str = "{},{}".format(*ent["tok_span"])
                if event_type not in event2role_map:
                    event2role_map[event_type] = {}
                if offset_str not in event2role_map[event_type]:
                    event2role_map[event_type][offset_str] = set()
                event2role_map[event_type][offset_str].add(role)

                if event_type not in event2graph:
                    event2graph[event_type] = nx.Graph()
                event2graph[event_type].add_node(offset_str)

            # find events (cliques) under every event type
            event_list = []
            for event_type, graph in event2graph.items():
                role_map = event2role_map.get(event_type, dict())
                cliques = list(nx.find_cliques(graph))  # all maximal cliques

                for cli in cliques:
                    event = {
                        "event_type": event_type,
                    }
                    arguments = []
                    for offset_str in cli:
                        start, end = offset_str.split(",")
                        tok_span = [int(start), int(end)]
                        char_span = tok_span2char_span(tok_span, tok2char_span)
                        arg_text = extract_ent_fr_txt_by_char_sp(char_span, text)
                        role_set = role_map.get(offset_str, set())

                        if self.dtm_arg_type_by_edges:
                            role_set_fin = set()
                            if len(role_set) == 1:
                                role_set_fin.add(list(role_set)[0])
                            else:  # determine the role by the edge
                                min_edge_num = 1 << 31
                                can_role_set = set()
                                for offset_str_j in cli:
                                    arg_p_set = offsets2arg_pair_rel.get(separator.join([offset_str, offset_str_j]),
                                                                         set())
                                    if len(arg_p_set) != 0 and len(arg_p_set) < min_edge_num:
                                        min_edge_num = len(arg_p_set)
                                        can_role_set = {arg_p.split(separator)[0] for arg_p in arg_p_set}
                                role_set_fin = can_role_set

                            if len(role_set_fin) == 1:
                                role_set = role_set_fin

                        for role in role_set:
                            if role in self.event_type2arg_rols[event_type] or role == "Trigger":
                                arguments.append({
                                    "text": arg_text,
                                    "type": role,
                                    "char_span": char_span,
                                    "tok_span": tok_span,
                                })

                    # find trigger
                    new_argument_list = []
                    triggers = []
                    for arg in arguments:
                        if arg["type"] == "Trigger":
                            triggers.append(arg)
                        else:
                            new_argument_list.append(arg)

                    if len(triggers) > 0:
                        trigger = random.choice(triggers)
                        event["trigger"] = trigger["text"]
                        event["trigger_tok_span"] = trigger["tok_span"]
                        event["trigger_char_span"] = trigger["char_span"]
                        event["trigger_list"] = triggers

                    # if the role sets corresponding to the nodes are all empty,
                    # this clique is invalid and the corresponding event without argument list and triggers
                    # will not be appended into the event list.
                    if len(new_argument_list) > 0 or "trigger" in event:
                        event["argument_list"] = new_argument_list
                        event_list.append(event)

            sample["event_list"] = event_list
            return sample

    return REBasedTFBoysTagger


def create_rebased_discontinuous_ner_tagger(base_class):
    # 0129
    class REBasedDiscontinuousNERTagger(base_class):
        def __init__(self, *arg, **kwargs):
            super(REBasedDiscontinuousNERTagger, self).__init__(*arg, **kwargs)
            self.language = kwargs["language"]
            self.use_bound = kwargs["use_bound"]
            self.seg_tag_scheme = kwargs["seg_tag_scheme"]

        @classmethod
        def additional_preprocess(cls, sample, **kwargs):

            use_bound = kwargs["use_bound"]
            seg_tag_scheme = kwargs["seg_tag_scheme"]

            new_tag_sep = "\u2E82"
            text = sample["text"]
            new_ent_list = []
            new_rel_list = []
            for ent in sample["entity_list"]:
                assert len(ent["char_span"]) == len(ent["tok_span"])
                ent_type = ent["type"]

                ch_sp = [ent["char_span"][0], ent["char_span"][-1]]
                tok_sp = [ent["tok_span"][0], ent["tok_span"][-1]]

                # boundary
                if use_bound:
                    new_ent_list.append({
                        "text": text[ch_sp[0]:ch_sp[1]],
                        "type": new_tag_sep.join([ent_type, "BOUNDARY"]),
                        "char_span": ch_sp,
                        "tok_span": tok_sp,
                    })

                for idx_i in range(0, len(ent["char_span"]), 2):
                    seg_i_ch_span = [ent["char_span"][idx_i], ent["char_span"][idx_i + 1]]
                    seg_i_tok_span = [ent["tok_span"][idx_i], ent["tok_span"][idx_i + 1]]

                    position_tag = None
                    if seg_tag_scheme == "BIS":
                        if idx_i == 0:
                            position_tag = "B"
                        else:
                            position_tag = "I"
                        if len(ent["char_span"]) == 2:
                            position_tag = "S"
                    elif seg_tag_scheme == "I":
                        position_tag = "I"
                    assert position_tag is not None

                    new_ent_type = "{}{}{}".format(ent_type, new_tag_sep, position_tag)

                    new_ent_list.append({
                        "text": text[seg_i_ch_span[0]:seg_i_ch_span[1]],
                        "type": new_ent_type,
                        "char_span": seg_i_ch_span,
                        "tok_span": seg_i_tok_span,
                    })
                    for idx_j in range(idx_i + 2, len(ent["char_span"]), 2):
                        seg_j_ch_span = [ent["char_span"][idx_j], ent["char_span"][idx_j + 1]]
                        seg_j_tok_span = [ent["tok_span"][idx_j], ent["tok_span"][idx_j + 1]]
                        new_rel_list.append({
                            "subject": text[seg_i_ch_span[0]:seg_i_ch_span[1]],
                            "subj_char_span": seg_i_ch_span,
                            "subj_tok_span": seg_i_tok_span,
                            "object": text[seg_j_ch_span[0]:seg_j_ch_span[1]],
                            "obj_char_span": seg_j_ch_span,
                            "obj_tok_span": seg_j_tok_span,
                            "predicate": "{}{}{}".format(ent_type, new_tag_sep, "SAME_ENT"),
                        })
                        new_rel_list.append({
                            "subject": text[seg_j_ch_span[0]:seg_j_ch_span[1]],
                            "subj_char_span": seg_j_ch_span,
                            "subj_tok_span": seg_j_tok_span,
                            "object": text[seg_i_ch_span[0]:seg_i_ch_span[1]],
                            "obj_char_span": seg_i_ch_span,
                            "obj_tok_span": seg_i_tok_span,
                            "predicate": "{}{}{}".format(ent_type, new_tag_sep, "SAME_ENT"),
                        })
            sample["entity_list"] = new_ent_list
            sample["relation_list"] = new_rel_list
            return sample

        def decode(self, sample, pred_tags, pred_outs):
            pred_sample = super(REBasedDiscontinuousNERTagger, self).decode(sample, pred_tags, pred_outs)
            return self._trans(pred_sample)

        def _trans(self, ori_sample):
            # decoding
            ent_list = ori_sample["entity_list"]
            rel_list = ori_sample["relation_list"]
            text = ori_sample["text"]
            tok2char_span = ori_sample["features"]["tok2char_span"]

            new_ent_list = []
            new_tag_sep = "\u2E82"
            ent_type2anns = {}

            # map boudaries by entity type
            # map entities by type
            for ent in ent_list:
                ent_type, pos_tag = ent["type"].split(new_tag_sep)
                ent["type"] = pos_tag
                if ent_type not in ent_type2anns:
                    ent_type2anns[ent_type] = {
                        "seg_list": [],
                        "rel_list": [],
                        "boundaries": [],
                        "continuous_entity_list": [],
                    }

                if ent["type"] == "BOUNDARY":
                    ent_type2anns[ent_type]["boundaries"].append(ent)
                elif ent["type"] in {"B", "I"}:
                    ent_type2anns[ent_type]["seg_list"].append(ent)
                else:
                    assert ent["type"] == "S"
                    ent_type2anns[ent_type]["continuous_entity_list"].append(ent)

            # map relations by entity type
            for rel in rel_list:
                ent_type, rel_tag = rel["predicate"].split(new_tag_sep)
                rel["predicate"] = rel_tag
                assert rel_tag == "SAME_ENT"
                subj_tok_span = rel["subj_tok_span"]
                obj_tok_span = rel["obj_tok_span"]
                if span_contains(subj_tok_span, obj_tok_span) or span_contains(obj_tok_span, subj_tok_span):
                    continue
                if ent_type in ent_type2anns:
                    ent_type2anns[ent_type]["rel_list"].append(rel)

            for ent_type, anns in ent_type2anns.items():

                if self.seg_tag_scheme == "BIS":
                    for c_ent in anns["continuous_entity_list"]:
                        c_ent["type"] = ent_type
                        new_ent_list.append(c_ent)

                def extr_disc(bd_span):
                    sub_seg_list = anns["seg_list"]
                    sub_rel_list = anns["rel_list"]
                    if bd_span is not None:
                        # select nodes and edges in this region
                        sub_seg_list = [seg for seg in anns["seg_list"] if
                                        span_contains(bd_span, seg["tok_span"])]
                        sub_rel_list = [rel_ for rel_ in anns["rel_list"]
                                        if span_contains(bd_span, rel_["subj_tok_span"])
                                        and span_contains(bd_span, rel_["obj_tok_span"])]

                    offset2seg_types = {}
                    graph = nx.Graph()
                    for seg in sub_seg_list:
                        offset_key = "{},{}".format(*seg["tok_span"])
                        if offset_key not in offset2seg_types:
                            offset2seg_types[offset_key] = set()
                        offset2seg_types[offset_key].add(seg["type"])
                        graph.add_node(offset_key)  # add a segment (a node)

                    for rel_ in sub_rel_list:
                        subj_offset_key = "{},{}".format(*rel_["subj_tok_span"])
                        obj_offset_key = "{},{}".format(*rel_["obj_tok_span"])
                        if rel_["predicate"] == "SAME_ENT":
                            graph.add_edge(subj_offset_key, obj_offset_key)  # add an edge between 2 segments

                    cliques = []
                    if bd_span is not None:
                        for cli in nx.find_cliques(graph):  # find all maximal cliques,
                            # filter invalid ones that do not include boundary tokens
                            if any(int(n.split(",")[0]) == bd_span[0] for n in cli) and \
                                    any(int(n.split(",")[1]) == bd_span[1] for n in cli):
                                cliques.append(cli)
                    else:
                        cliques = nx.find_cliques(graph)

                    for cli in cliques:
                        # 0129
                        if self.seg_tag_scheme == "BIS" and \
                                not any(n in offset2seg_types and "B" in offset2seg_types[n] for n in cli):
                            continue
                        spans = []
                        for n in cli:
                            start, end = n.split(",")
                            spans.append([int(start), int(end)])
                        tok_span = []
                        last_end = -10
                        for sp in sorted(spans, key=lambda sp_: sp_[0]):
                            if sp[0] < last_end:
                                continue
                            tok_span.extend(sp)
                            last_end = sp[1]

                        tok_span = merge_spans(tok_span)
                        char_span = tok_span2char_span(tok_span, tok2char_span)
                        new_ent_list.append({
                            "text": extract_ent_fr_txt_by_char_sp(char_span, text, self.language),
                            "type": ent_type,
                            "char_span": char_span,
                            "tok_span": tok_span,
                        })

                if self.use_bound:
                    for boundary in anns["boundaries"]:
                        bound_span = boundary["tok_span"]
                        extr_disc(bound_span)
                else:
                    extr_disc(None)

            pred_sample = copy.deepcopy(ori_sample)
            pred_sample["entity_list"] = new_ent_list
            del pred_sample["relation_list"]
            return pred_sample

    return REBasedDiscontinuousNERTagger


def create_rebased_oie_tagger(base_class):
    class REBasedOIETagger(base_class):

        def __init__(self, data_anns, *arg, **kwargs):
            super(REBasedOIETagger, self).__init__(data_anns, *arg, **kwargs)
            self.language = kwargs["language"]
            self.add_obj_placeholders = any("[OBJ]" in arg["text"] for spo in data_anns["open_spo_list"]
                                            for arg in spo if arg["type"] == "predicate")

        @classmethod
        def additional_preprocess(cls, data):

            new_tag_sep = "\u2E82"
            new_data = []

            for sample in data:
                new_sample = copy.deepcopy(sample)
                text = sample["text"]
                new_ent_list = []
                new_rel_list = []
                tok2char_span = sample["features"]["tok2char_span"]
                spo_span_set = set()
                for spo in sample["open_spo_list"]:
                    seg_list = []
                    obj_list = []
                    pred = None
                    predicate_prefix = None
                    predicate_suffix = None

                    # segment list and next edge
                    for arg in spo:
                        arg_type = arg["type"]
                        if arg_type == "predicate_prefix":
                            predicate_prefix = arg["text"]
                            continue
                        elif arg_type == "predicate_suffix":
                            predicate_suffix = arg["text"]
                            continue
                        elif arg_type == "predicate":
                            pred = arg
                            if len(pred["char_span"]) == 0:  # if no char span and type == "predicate",
                                # it is a predefined predicate that does not exist in the text,
                                continue
                        elif arg_type == "object":
                            obj_list.append(arg)

                        # append segments and generate next edges
                        for idx_i in range(0, len(arg["tok_span"]), 2):
                            tok_sp_i = [arg["tok_span"][idx_i], arg["tok_span"][idx_i + 1]]
                            ch_sp_i = [arg["char_span"][idx_i], arg["char_span"][idx_i + 1]]
                            seg_list.append({
                                "type": arg_type,
                                "text": text[ch_sp_i[0]:ch_sp_i[1]],
                                "char_span": ch_sp_i,
                                "tok_span": tok_sp_i,
                            })

                            # next edges
                            idx_j = idx_i + 2
                            if idx_j + 1 <= len(arg["tok_span"]):
                                tok_sp_j = [arg["tok_span"][idx_j], arg["tok_span"][idx_j + 1]]
                                ch_sp_j = [arg["char_span"][idx_j], arg["char_span"][idx_j + 1]]
                                new_rel_list.append({
                                    "subject": text[ch_sp_i[0]:ch_sp_i[1]],
                                    "subj_char_span": ch_sp_i,
                                    "subj_tok_span": tok_sp_i,
                                    "object": text[ch_sp_j[0]:ch_sp_j[1]],
                                    "obj_char_span": ch_sp_j,
                                    "obj_tok_span": tok_sp_j,
                                    "predicate": "NEXT",
                                })

                    new_ent_list.extend(seg_list)

                    # spo visible area
                    # spo_ch_area = [99999, -1]
                    spo_tok_area = [99999, -1]
                    for seg in seg_list:
                        spo_tok_area[0] = min(seg["tok_span"][0], spo_tok_area[0])
                        spo_tok_area[1] = max(seg["tok_span"][1], spo_tok_area[1])
                    spo_span_set.add((spo_tok_area[0], spo_tok_area[1]))

                    # generate edges between segments
                    for seg_i in seg_list:
                        for seg_j in seg_list:
                            # spo clique
                            new_rel_list.append({
                                "subject": seg_i["text"],
                                "subj_char_span": seg_i["char_span"],
                                "subj_tok_span": seg_i["tok_span"],
                                "object": seg_j["text"],
                                "obj_char_span": seg_j["char_span"],
                                "obj_tok_span": seg_j["tok_span"],
                                "predicate": "IN_SPO",
                            })
                            # role pair
                            new_rel_list.append({
                                "subject": seg_i["text"],
                                "subj_char_span": seg_i["char_span"],
                                "subj_tok_span": seg_i["tok_span"],
                                "object": seg_j["text"],
                                "obj_char_span": seg_j["char_span"],
                                "obj_tok_span": seg_j["tok_span"],
                                "predicate": new_tag_sep.join(["ROLE_PAIR", seg_i["type"], seg_j["type"]]),  # "IN_SPO",
                            })
                            # if predefined predicate
                            if pred is not None and len(pred["char_span"]) == 0:
                                new_rel_list.append({
                                    "subject": seg_i["text"],
                                    "subj_char_span": seg_i["char_span"],
                                    "subj_tok_span": seg_i["tok_span"],
                                    "object": seg_j["text"],
                                    "obj_char_span": seg_j["char_span"],
                                    "obj_tok_span": seg_j["tok_span"],
                                    "predicate": new_tag_sep.join(["PREDEFINED_CLI", pred["text"]]),
                                })

                    if pred is not None:
                        # predicate prefix/suffix
                        if predicate_prefix is not None:
                            pred_tok_sp_b = pred["tok_span"][:2]
                            pred_ch_sp_b = pred["char_span"][:2]
                            new_ent_list.append({
                                "type": new_tag_sep.join(["PRED_PREFIX", predicate_prefix]),
                                "text": text[pred_ch_sp_b[0]:pred_ch_sp_b[1]],
                                "char_span": pred_ch_sp_b,
                                "tok_span": pred_tok_sp_b,
                            })
                        if predicate_suffix is not None:
                            pred_tok_sp_e = pred["tok_span"][-2:]
                            pred_ch_sp_e = pred["char_span"][-2:]
                            new_ent_list.append({
                                "type": new_tag_sep.join(["PRED_SUFFIX", predicate_suffix]),
                                "text": text[pred_ch_sp_e[0]:pred_ch_sp_e[1]],
                                "char_span": pred_ch_sp_e,
                                "tok_span": pred_tok_sp_e,
                            })
                        # predicate to object edges
                        pred_cp = pred["text"][:]
                        obj_pre_list = []  # for predicate segment before an object
                        for idx in range(0, len(pred["char_span"]), 2):
                            ch_sp_start, ch_sp_end = pred["char_span"][idx], pred["char_span"][idx + 1]
                            tok_sp_start, tok_sp_end = pred["tok_span"][idx], pred["tok_span"][idx + 1]

                            p_txt = text[ch_sp_start:ch_sp_end]
                            pred_cp = pred_cp.lstrip(p_txt)
                            pred_cp = pred_cp.lstrip(" ")
                            if pred_cp[:5] == "[OBJ]":
                                obj_pre_list.append({
                                    "text": p_txt,
                                    "char_span": [ch_sp_start, ch_sp_end],
                                    "tok_span": [tok_sp_start, tok_sp_end],
                                })
                                pred_cp = pred_cp.lstrip("[OBJ]")

                        for idx, pre in enumerate(obj_pre_list):
                            new_rel_list.append({
                                "subject": pre["text"],
                                "subj_char_span": pre["char_span"],
                                "subj_tok_span": pre["tok_span"],
                                "object": obj_list[idx]["text"],
                                "obj_char_span": obj_list[idx]["char_span"],
                                "obj_tok_span": obj_list[idx]["tok_span"],
                                "predicate": "PRED_TO_OBJ",
                            })

                # spo span area
                for tok_sp in spo_span_set:
                    spo_ch_spans = tok2char_span[tok_sp[0]:tok_sp[1]]
                    # try:
                    spo_ch_span = [spo_ch_spans[0][0], spo_ch_spans[-1][1]]
                    # except Exception:
                    #     print("dec")
                    new_ent_list.append({
                        "type": "MASK:SPO_AREA",
                        "text": text[spo_ch_span[0]:spo_ch_span[1]],
                        "char_span": spo_ch_span,
                        "tok_span": tok_sp,
                    })

                new_sample["entity_list"] = new_ent_list
                new_sample["relation_list"] = new_rel_list
                new_data.append(new_sample)
            return new_data

        def decode(self, sample, pred_tags, pred_outs):
            pred_sample = super(REBasedOIETagger, self).decode(sample, pred_tags, pred_outs)
            return self._trans(pred_sample)

        def _trans(self, ori_sample):
            new_tag_sep = "\u2E82"
            sample_id = ori_sample["id"]

            text = ori_sample["text"]
            tok2char_span = ori_sample["features"]["tok2char_span"]
            open_spo_list = []

            spo_span_set = set()
            ent_list = []
            for seg in ori_sample["entity_list"]:
                if seg["type"] == "MASK:SPO_AREA":
                    spo_span_set.add((seg["tok_span"][0], seg["tok_span"][1]))
                else:
                    ent_list.append(seg)

            if sample_id in {40440, 19811, 1928}:
                print("trans debug")
                print("????")

            for spo_span in spo_span_set:
                sub_ent_list = [ent for ent in ent_list if span_contains(spo_span, ent["tok_span"])]
                sub_rel_list = [rel for rel in ori_sample["relation_list"]
                                if span_contains(spo_span, rel["subj_tok_span"])
                                and span_contains(spo_span, rel["obj_tok_span"])]
                predefined_spo_graph_map = {}
                spo_graph = nx.Graph()
                seg2roles = {}
                next_edge_set = set()
                pred2obj_set = set()
                edge2role_pair = {}
                prefix_map, suffix_map = {}, {}

                for rel in sub_rel_list:
                    offset_str_seg_i = "{},{}".format(*rel["subj_tok_span"])
                    offset_str_seg_j = "{},{}".format(*rel["obj_tok_span"])

                    if rel["predicate"] == "IN_SPO":
                        spo_graph.add_edge(offset_str_seg_i, offset_str_seg_j)
                    if "ROLE_PAIR" in rel["predicate"]:
                        edge_str = "-".join([offset_str_seg_i, offset_str_seg_j])
                        _, role_i, role_j = rel["predicate"].split(new_tag_sep)
                        edge2role_pair.setdefault(edge_str, set())
                        edge2role_pair[edge_str].add(new_tag_sep.join([role_i, role_j]))

                    elif rel["predicate"] == "NEXT":
                        next_edge_set.add("-".join([offset_str_seg_i, offset_str_seg_j]))
                    elif rel["predicate"] == "PRED_TO_OBJ":
                        pred2obj_set.add("-".join([offset_str_seg_i, offset_str_seg_j]))
                    elif "PREDEFINED_CLI" in rel["predicate"]:
                        _, predefined_cli = rel["predicate"].split(new_tag_sep)
                        if predefined_cli not in predefined_spo_graph_map:
                            predefined_spo_graph_map[predefined_cli] = nx.Graph()
                        predefined_spo_graph_map[predefined_cli].add_edge(offset_str_seg_i, offset_str_seg_j)

                # seg2roles
                for seg in sub_ent_list:
                    offset_str_seg = "{},{}".format(*seg["tok_span"])
                    if "PRED_PREFIX" in seg["type"]:
                        _, prefix = seg["type"].split(new_tag_sep)
                        prefix_map[offset_str_seg] = prefix
                    if "PRED_SUFFIX" in seg["type"]:
                        _, suffix = seg["type"].split(new_tag_sep)
                        suffix_map[offset_str_seg] = suffix
                    spo_graph.add_node(offset_str_seg)
                    seg2roles.setdefault(offset_str_seg, set()).add(seg["type"])

                # predefined predicate
                cli2pred = {}
                for pred, graph in predefined_spo_graph_map.items():
                    for cli in nx.find_cliques(graph):
                        cli2pred[str(sorted(cli))] = pred

                def get_role(seg_offset_str, clique):
                    """
                    :param seg_offset_str: "3,6"
                    :param clique: for voting
                    :return: the role of the corresponding segment
                    """
                    # 用seg2roles确定角色类型，有多个角色用角色边
                    cand_role_list = seg2roles.get(seg_offset_str, set())

                    if len(cand_role_list) == 1:
                        return cand_role_list
                    else:
                        # vote
                        role2votes = {}
                        for seg_offset_str_j in clique:
                            if seg_offset_str_j != seg_offset_str:
                                rps = edge2role_pair.get("-".join([seg_offset_str, seg_offset_str_j]), set())
                                for rp in rps:
                                    vote_role = rp.split(new_tag_sep)[0]
                                    role2votes[vote_role] = role2votes.get(vote_role, 0) + 1

                                rps = edge2role_pair.get("-".join([seg_offset_str_j, seg_offset_str]), set())
                                for rp in rps:
                                    # try:
                                    vote_role = rp.split(new_tag_sep)[1]
                                    # except Exception:
                                    #     print("!")
                                    role2votes[vote_role] = role2votes.get(vote_role, 0) + 1

                        try:
                            max_vote_num = sorted(role2votes.items(), key=lambda x: x[1])[-1][1]
                            cand_role_list = {role_ for role_, vt in role2votes.items() if vt == max_vote_num}
                        except Exception:  # noqa no roles can be decoded from edges
                            return cand_role_list
                    return cand_role_list

                ori_cliques = list(nx.find_cliques(spo_graph))
                # cliques belong to this spo span
                cliques = [cli for cli in ori_cliques
                           if any(int(n.split(",")[0]) == spo_span[0] for n in cli)
                           and any(int(n.split(",")[1]) == spo_span[1] for n in cli)]

                for cli in cliques:  # all spo clique
                    arg_list = []
                    predicate = None
                    start_sp2obj = {}
                    next_map = {}
                    for e in next_edge_set:
                        offset_str_seg_i, offset_str_seg_j = e.split("-")
                        if offset_str_seg_i in cli and offset_str_seg_j in cli \
                                and get_role(offset_str_seg_i, cli) == get_role(offset_str_seg_j, cli):
                            next_map[offset_str_seg_i] = offset_str_seg_j

                    for offset_str in cli:  # all arguments
                        if offset_str not in next_map.values():  # if a beginner
                            # 确定role type
                            cand_roles = get_role(offset_str, cli)
                            tok_span = [int(idx) for idx in offset_str.split(",")]
                            char_span = tok_span2char_span(tok_span, tok2char_span)
                            # arg_txt = text[char_span[0]:char_span[1]]

                            # 循环next链接直到末尾
                            mem = [offset_str, ]
                            point = offset_str
                            while point in next_map:
                                next_seg_offset_str = next_map[point]
                                if next_seg_offset_str in mem:  # no circle
                                    break

                                # 指向的下一个元素必须与第一个元素类型一致，否则丢弃
                                next_seg_role = get_role(next_seg_offset_str, cli)
                                if next_seg_role == cand_roles:
                                    mem.append(next_seg_offset_str)

                                    new_tok_sp = [int(idx) for idx in next_seg_offset_str.split(",")]
                                    new_ch_sp = tok_span2char_span(new_tok_sp, tok2char_span)
                                    tok_span.extend(new_tok_sp)
                                    char_span.extend(new_ch_sp)
                                    point = next_seg_offset_str
                                else:
                                    break

                            # generate arguments
                            # 注意predicate和object要特殊处理：[OBJ]
                            for role in cand_roles:
                                arg = {
                                    "tok_span": tok_span,
                                    "char_span": char_span,
                                    "text": extract_ent_fr_txt_by_char_sp(char_span, text),
                                    "type": role,
                                }
                                if role == "predicate":
                                    predicate = arg
                                elif role == "object":
                                    start_sp2obj["{},{}".format(*tok_span)] = arg
                                else:
                                    arg_list.append(arg)
                    # add [OBJ] to predicate
                    if predicate is not None:  # if predicate exists
                        if self.add_obj_placeholders:  # add placeholders [OBJ]
                            new_pred_segs = []
                            for idx in range(0, len(predicate["char_span"]), 2):
                                ch_sp_start, ch_sp_end = predicate["char_span"][idx], predicate["char_span"][idx + 1]
                                tok_sp_start, tok_sp_end = predicate["tok_span"][idx], predicate["tok_span"][idx + 1]

                                new_pred_segs.append(text[ch_sp_start:ch_sp_end])
                                p_sub_offset_str = "{},{}".format(tok_sp_start, tok_sp_end)
                                for obj_star_offset_str, obj in start_sp2obj.items():
                                    if "-".join([p_sub_offset_str, obj_star_offset_str]) in pred2obj_set:
                                        new_pred_segs.append("[OBJ]")
                                        arg_list.append(obj)  # append objects according to [OBJ]s in the predicate

                            predicate["text"] = join_segs(new_pred_segs)
                            arg_list.append(predicate)
                        else:
                            arg_list.append(predicate)
                            arg_list.extend(start_sp2obj.values())

                        # add prefix and suffix
                        pred_tok_b = "{},{}".format(*predicate["tok_span"][:2])
                        pred_tok_e = "{},{}".format(*predicate["tok_span"][-2:])
                        if pred_tok_b in prefix_map:
                            arg_list.append({
                                "text": prefix_map[pred_tok_b],
                                "type": "predicate_prefix",
                                "tok_span": [],
                                "char_span": [],
                            })
                        if pred_tok_e in suffix_map:
                            arg_list.append({
                                "text": suffix_map[pred_tok_e],
                                "type": "predicate_suffix",
                                "tok_span": [],
                                "char_span": [],
                            })
                    else:
                        # if no predicate, append objects directly
                        arg_list.extend(start_sp2obj.values())
                        # if a predefined predicate exists
                        if str(sorted(cli)) in cli2pred:
                            arg_list.append({
                                "tok_span": [],
                                "char_span": [],
                                "text": cli2pred[str(sorted(cli))],
                                "type": "predicate",
                            })
                    open_spo_list.append(arg_list)

            pred_sample = ori_sample
            filtered_open_spo_list = []
            for spo in open_spo_list:
                type_map = {}
                for arg in spo:
                    type_map[arg["type"]] = type_map.get(arg["type"], 0) + 1
                if type_map.get("subject", 0) > 1 or type_map.get("predicate", 0) > 1:
                    continue
                filtered_open_spo_list.append(spo)
            pred_sample["open_spo_list"] = filtered_open_spo_list
            return pred_sample

    return REBasedOIETagger
