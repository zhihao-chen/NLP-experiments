# -*- coding: utf8 -*-
"""
======================================
    Project Name: NLP
    File Name: tplinker_plus_utils
    Author: czh
    Create Date: 2021/8/16
--------------------------------------
    Change Activity: 
======================================
"""
import re
import copy
import json
import os
import datetime
from tqdm import tqdm
from pathlib import Path
from typing import Union

import torch
from pydantic import dataclasses, Field

now_day = datetime.datetime.now().strftime("%Y-%m-%d")


@dataclasses.dataclass
class DataAndTrainArguments:
    bert_name_or_path: Union[str, Path] = Field(
        default="hfl/chinese-roberta-wwm-ext",
        description="pretrained bert directory (absolute path) or transformer bert model name. "
                    "bert-base-cased， chinese-bert-wwm-ext-hit"
    )
    tokenizer_name: Union[str, Path] = Field(
        default="",
        description="Pretrained tokenizer name or path if not the same as model_name"
    )
    bert_config_name: Union[str, Path] = Field(
        default="",
        description="Pretrained config name or path if not the same as model_name"
    )
    data_dir: Union[str, Path] = Field(
        default="",
        description="The input data dir. Should contain the training files for the CoNLL-2003 NER task."
    )
    task_name: str = Field(
        default='ner',
        description="Task Name. This will be used for directory name to distinguish different task."
    )
    model_type: str = Field(
        default='bert',
        description="Model type selected in the list: [bert, nezha]"
    )
    cache_dir: Union[str, Path] = Field(
        default="",
        description="Where do you want to store the pre-trained models downloaded from s3"
    )
    pretrained_word_embedding_path: Union[str, Path] = Field(
        default="../../pretrained_emb/glove_300_nyt.emb",
        description="Valid only if 'model_type' is 'BiLSTM'"
    )
    output_dir: Union[str, Path] = Field(
        default=f"../experiments/output_{task_name.default}_{model_type.default}/",
        description="The output directory where the model predictions and checkpoints will be written."
    )
    log_dir: Union[str, Path] = "../experiments/output_file_dir/default.log"
    tensorboard_log_dir: Union[str, Path] = "../experiments/tensorboard/tplinker_plus_ner/"
    path_to_save_model: Union[str, Path] = output_dir
    save_res_dir: Union[str, Path] = "../experiments/output_file_dir/tplinker_plus_ner/eval/"
    model_state_dict_path: Union[str, Path] = Field(
        default=path_to_save_model,
        description="if not fr scratch, set a model_state_dict_path only if 'fr_scratch' is False")
    train_data_name: str = Field(default="train_data.json")
    valid_data_name: str = Field(default="valid_data.json")
    test_data_name: str = Field(default="test_data.json")
    rel2id: str = "rel2id.json"
    ent2id: str = Field(
        default="end2id.json",
        description="entity mapping to id"
    )
    token2idx: str = Field(default="token2idx.json", description="Valid only if 'model_type' is 'BiLSTM'")
    local_rank: int = Field(default=-1, description="For distributed training: local_rank")
    do_lower_case: bool = True

    enc_hidden_size: int = Field(default=300, description="Valid only if 'model_type' is 'BiLSTM'")
    dec_hidden_size: int = Field(default=600, description="Valid only if 'model_type' is 'BiLSTM'")
    emb_dropout: float = Field(default=0.1, description="Valid only if 'model_type' is 'BiLSTM'")
    rnn_dropout: float = Field(default=0.1, description="Valid only if 'model_type' is 'BiLSTM'")
    word_embedding_dim: int = Field(default=300, description="Valid only if 'model_type' is 'BiLSTM'")

    train_batch_size: int = 8
    eval_batch_size: int = 8
    epochs: int = 30
    seed: int = 2333
    log_interval: int = 10
    max_seq_len: int = 128
    sliding_len: int = 20
    num_workers: int = 0
    last_k_model: int = 1
    patience: int = Field(5, description="Early stopping steps")
    lr: float = Field(default=5e-5, description="Learning rate")

    shaking_type: str = Field(
        default="cln_plus",
        description="cat, cat_plus, cln, cln_plus; Experiments show that cat/cat_plus work better with BiLSTM, "
                    "while cln/cln_plus work better with BERT. The results in the paper are produced by 'cat'. "
                    "So, if you want to reproduce the results, 'cat' is enough, no matter for BERT or BiLSTM."
    )
    inner_enc_type: str = Field(
        default="lstm",
        description="valid only if cat_plus or cln_plus is set. It is the way how to encode inner tokens between "
                    "each token pairs. If you only want to reproduce the results, just leave it alone."
    )
    match_pattern: str = Field(
        default="whole_text",
        description="only_head_text (nyt_star, webnlg_star), whole_text (nyt, webnlg), only_head_index, whole_span"
    )

    n_gpu: str = Field(default="0,1,2,3", description="Specify the CUDA_VISIBLE_DEVICES")

    logger: str = Field(default="default", description="wandb or default")
    f1_2_save: int = Field(default=0, description="when to save the model state dict")
    fr_scratch: bool = Field(default=True, description="whether train from scratch")
    fr_last_checkpoint: bool = Field(default=False, description="Whether continue to train from last checkpoint")
    note: str = Field(default="start from scratch", description="write down notes here if you want, it will be logged")

    scheduler: str = Field(
        default="CAWR",
        description="'CAWR' is 'CosineAnnealingWarmRestarts', 'Step' is 'StepLR'")
    save_res: bool = False
    score: bool = Field(default=True, description="score: set true only if test set is tagged")
    force_split: bool = False
    eval_all_checkpoints: bool = False
    ghm: bool = Field(
        default=False,
        description="set True if you want to use GHM to adjust the weights of gradients, "
                    "this will speed up the training process and might improve the results. "
                    "(Note that ghm in current version is unstable now, may hurt the results)"
    )
    tok_pair_sample_rate: int = Field(
        1,
        description="(0, 1] How many percent of token paris you want to sample for training, "
                    "this would slow down the training if set to less than 1. "
                    "It is only helpful when your GPU memory is not enought for the training."
    )
    T_mult: int = Field(default=1, description="CosineAnnealingWarmRestarts.Only if 'scheduler' is 'CAWR'")
    rewarm_epoch_num: int = Field(default=2, description="CosineAnnealingWarmRestarts.Only if 'scheduler' is 'CAWR'")
    decay_rate: float = Field(default=0.999, description="StepLR.Only if 'scheduler' is 'Step'")
    decay_steps: int = Field(default=100, description="StepLR.Only if 'scheduler' is 'Step'")
    fp16: bool = False
    gradient_accumulation_steps: int = Field(
        default=1,
        description="Number of updates steps to accumulate before performing a backward/update pass."
    )
    warmup_proportion: float = Field(
        default=0.1,
        description="Proportion of training to perform linear learning rate warmup for,E.g., 0.1 = 10% of training."
    )
    evaluate_during_training: bool = Field(
        default=False,
        description="Whether to run evaluation during training at each logging step."
    )
    save_steps: int = Field(
        default=500,
        description="Save checkpoint every X updates steps."
    )
    logging_steps: int = Field(
        default=500,
        description="Log every X updates steps."
    )
    dist_emb_size: int = Field(
        default=-1,
        description="-1: do not use distance embedding; other number: need to be larger than the max_seq_len "
                    "of the inputs.set -1 if you only want to reproduce the results in the paper."
    )
    ent_add_dist: bool = Field(
        default=False,
        description="set true if you want add distance embeddings for each token pairs. (for entity decoder)"
    )
    rel_add_dist: bool = Field(default=False, description="the same as above (for relation decoder)")


class HandshakingTaggingScheme(object):
    def __init__(self, rel2id, max_seq_len, entity_type2id):
        super().__init__()
        self.rel2id = rel2id
        self.id2rel = {ind: rel for rel, ind in rel2id.items()}

        self.separator = "\u2E80"
        self.link_types = {"SH2OH",  # subject head to object head
                           "OH2SH",  # object head to subject head
                           "ST2OT",  # subject tail to object tail
                           "OT2ST",  # object tail to subject tail
                           }
        self.tags = {self.separator.join([rel, lt]) for rel in self.rel2id.keys() for lt in self.link_types}

        self.ent2id = entity_type2id
        self.id2ent = {ind: ent for ent, ind in self.ent2id.items()}
        self.tags |= {self.separator.join([ent, "EH2ET"]) for ent in
                      self.ent2id.keys()}  # EH2ET: entity head to entity tail

        self.tags = sorted(self.tags)

        self.tag2id = {t: idx for idx, t in enumerate(self.tags)}
        self.id2tag = {idx: t for t, idx in self.tag2id.items()}
        self.matrix_size = max_seq_len

        # map
        # e.g. [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]  上三角矩阵
        self.shaking_idx2matrix_idx = [(ind, end_ind) for ind in range(self.matrix_size) for end_ind in
                                       list(range(self.matrix_size))[ind:]]

        self.matrix_idx2shaking_idx = []
        for j in range(self.matrix_size):
            self.matrix_idx2shaking_idx.append([0 for _ in range(self.matrix_size)])
        for shaking_ind, matrix_ind in enumerate(self.shaking_idx2matrix_idx):
            self.matrix_idx2shaking_idx[matrix_ind[0]][matrix_ind[1]] = shaking_ind

    def get_tag_size(self):
        return len(self.tag2id)

    def get_spots(self, sample):
        """
        matrix_spots: [(tok_pos1, tok_pos2, tag_id), ]
        """
        matrix_spots = []
        spot_memory_set = set()

        def add_spot(spot):
            memory = "{},{},{}".format(*spot)
            if memory not in spot_memory_set:
                matrix_spots.append(spot)
                spot_memory_set.add(memory)

        # # if entity_list exist, need to distinguish entity types
        # if self.ent2id is not None and "entity_list" in sample:
        for ent in sample["entity_list"]:
            add_spot(
                (ent["tok_span"][0], ent["tok_span"][1] - 1, self.tag2id[self.separator.join([ent["type"], "EH2ET"])]))

        for rel in sample["relation_list"]:
            subj_tok_span = rel["subj_tok_span"]
            obj_tok_span = rel["obj_tok_span"]
            rel = rel["predicate"]
            # if self.ent2id is None: # set all entities to default type
            # add_spot((subj_tok_span[0], subj_tok_span[1] - 1, self.tag2id[self.separator.join(["DEFAULT", "EH2ET"])]))
            # add_spot((obj_tok_span[0], obj_tok_span[1] - 1, self.tag2id[self.separator.join(["DEFAULT", "EH2ET"])]))
            if subj_tok_span[0] <= obj_tok_span[0]:
                add_spot((subj_tok_span[0], obj_tok_span[0], self.tag2id[self.separator.join([rel, "SH2OH"])]))
            else:
                add_spot((obj_tok_span[0], subj_tok_span[0], self.tag2id[self.separator.join([rel, "OH2SH"])]))
            if subj_tok_span[1] <= obj_tok_span[1]:
                add_spot((subj_tok_span[1] - 1, obj_tok_span[1] - 1, self.tag2id[self.separator.join([rel, "ST2OT"])]))
            else:
                add_spot((obj_tok_span[1] - 1, subj_tok_span[1] - 1, self.tag2id[self.separator.join([rel, "OT2ST"])]))
        return matrix_spots

    def spots2shaking_tag(self, spots):
        """
        convert spots to matrix tag
        spots: [(start_ind, end_ind, tag_id), ]
        return:
            shaking_tag: (shaking_seq_len, tag_size)
        """
        shaking_seq_len = self.matrix_size * (self.matrix_size + 1) // 2
        shaking_tag = torch.zeros(shaking_seq_len, len(self.tag2id)).long()
        for sp in spots:
            shaking_idx = self.matrix_idx2shaking_idx[sp[0]][sp[1]]
            shaking_tag[shaking_idx][sp[2]] = 1
        return shaking_tag

    def spots2shaking_tag4batch(self, batch_spots):
        """
        batch_spots: a batch of spots, [spots1, spots2, ...]
            spots: [(start_ind, end_ind, tag_id), ]
        return:
            batch_shaking_tag: (batch_size, shaking_seq_len, tag_size)
        """
        shaking_seq_len = self.matrix_size * (self.matrix_size + 1) // 2
        batch_shaking_tag = torch.zeros(len(batch_spots), shaking_seq_len, len(self.tag2id)).long()
        for batch_id, spots in enumerate(batch_spots):
            for sp in spots:
                shaking_idx = self.matrix_idx2shaking_idx[sp[0]][sp[1]]
                batch_shaking_tag[batch_id][shaking_idx][sp[2]] = 1
        return batch_shaking_tag

    def get_spots_fr_shaking_tag(self, shaking_tag):
        """
        shaking_tag -> spots
        shaking_tag: (shaking_seq_len, tag_id)
        spots: [(start_ind, end_ind, tag_id), ]
        """
        spots = []
        nonzero_points = torch.nonzero(shaking_tag, as_tuple=False)
        for point in nonzero_points:
            shaking_idx, tag_idx = point[0].item(), point[1].item()
            pos1, pos2 = self.shaking_idx2matrix_idx[shaking_idx]
            spot = (pos1, pos2, tag_idx)
            spots.append(spot)
        return spots

    def decode_rel(self,
                   text,
                   shaking_tag,
                   tok2char_span,
                   tok_offset=0, char_offset=0):
        """
        shaking_tag: (shaking_seq_len, tag_id_num)
        """
        rel_list = []
        matrix_spots = self.get_spots_fr_shaking_tag(shaking_tag)

        # entity
        head_ind2entities = {}
        ent_list = []
        for sp in matrix_spots:
            tag = self.id2tag[sp[2]]
            ent_type, link_type = tag.split(self.separator)
            # for an entity, the start position can not be larger than the end pos.
            if link_type != "EH2ET" or sp[0] > sp[1]:
                continue

            char_span_list = tok2char_span[sp[0]:sp[1] + 1]
            char_sp = [char_span_list[0][0], char_span_list[-1][1]]
            ent_text = text[char_sp[0]:char_sp[1]]
            entity = {
                "type": ent_type,
                "text": ent_text,
                "tok_span": [sp[0], sp[1] + 1],
                "char_span": char_sp,
            }
            head_key = str(sp[0])  # take ent_head_pos as the key to entity list
            if head_key not in head_ind2entities:
                head_ind2entities[head_key] = []
            head_ind2entities[head_key].append(entity)
            ent_list.append(entity)

        # tail link
        tail_link_memory_set = set()
        for sp in matrix_spots:
            tag = self.id2tag[sp[2]]
            rel, link_type = tag.split(self.separator)
            if link_type == "ST2OT":
                tail_link_memory = self.separator.join([rel, str(sp[0]), str(sp[1])])
                tail_link_memory_set.add(tail_link_memory)
            elif link_type == "OT2ST":
                tail_link_memory = self.separator.join([rel, str(sp[1]), str(sp[0])])
                tail_link_memory_set.add(tail_link_memory)

        # head link
        for sp in matrix_spots:
            tag = self.id2tag[sp[2]]
            rel, link_type = tag.split(self.separator)

            if link_type == "SH2OH":
                subj_head_key, obj_head_key = str(sp[0]), str(sp[1])
            elif link_type == "OH2SH":
                subj_head_key, obj_head_key = str(sp[1]), str(sp[0])
            else:
                continue

            if subj_head_key not in head_ind2entities or obj_head_key not in head_ind2entities:
                # no entity start with subj_head_key and obj_head_key
                continue

            subj_list = head_ind2entities[subj_head_key]  # all entities start with this subject head
            obj_list = head_ind2entities[obj_head_key]  # all entities start with this object head

            # go over all subj-obj pair to check whether the tail link exists
            for subj in subj_list:
                for obj in obj_list:
                    tail_link_memory = self.separator.join(
                        [rel, str(subj["tok_span"][1] - 1), str(obj["tok_span"][1] - 1)])
                    if tail_link_memory not in tail_link_memory_set:
                        # no such relation
                        continue
                    rel_list.append({
                        "subject": subj["text"],
                        "object": obj["text"],
                        "subj_tok_span": [subj["tok_span"][0] + tok_offset, subj["tok_span"][1] + tok_offset],
                        "obj_tok_span": [obj["tok_span"][0] + tok_offset, obj["tok_span"][1] + tok_offset],
                        "subj_char_span": [subj["char_span"][0] + char_offset, subj["char_span"][1] + char_offset],
                        "obj_char_span": [obj["char_span"][0] + char_offset, obj["char_span"][1] + char_offset],
                        "predicate": rel,
                    })
            # recover the positons in the original text
            for ent in ent_list:
                ent["char_span"] = [ent["char_span"][0] + char_offset, ent["char_span"][1] + char_offset]
                ent["tok_span"] = [ent["tok_span"][0] + tok_offset, ent["tok_span"][1] + tok_offset]

        return rel_list, ent_list

    @staticmethod
    def trans2ee(rel_list, ent_list):
        sepatator = "_"  # \u2E80
        # trigger_set, arg_iden_set, arg_class_set = set(), set(), set()
        trigger_offset2vote = {}
        trigger_offset2trigger_text = {}
        trigger_offset2trigger_char_span = {}
        # get candidate trigger types from relation
        for rel in rel_list:
            trigger_offset = rel["obj_tok_span"]
            trigger_offset_str = "{},{}".format(trigger_offset[0], trigger_offset[1])
            trigger_offset2trigger_text[trigger_offset_str] = rel["object"]
            trigger_offset2trigger_char_span[trigger_offset_str] = rel["obj_char_span"]
            _, event_type = rel["predicate"].split(sepatator)

            if trigger_offset_str not in trigger_offset2vote:
                trigger_offset2vote[trigger_offset_str] = {}
            trigger_offset2vote[trigger_offset_str][event_type] = trigger_offset2vote[trigger_offset_str].get(
                event_type, 0) + 1

        # get candidate trigger types from entity types
        for ent in ent_list:
            t1, t2 = ent["type"].split(sepatator)
            assert t1 == "Trigger" or t1 == "Argument"
            if t1 == "Trigger":  # trigger
                event_type = t2
                trigger_span = ent["tok_span"]
                trigger_offset_str = "{},{}".format(trigger_span[0], trigger_span[1])
                trigger_offset2trigger_text[trigger_offset_str] = ent["text"]
                trigger_offset2trigger_char_span[trigger_offset_str] = ent["char_span"]
                if trigger_offset_str not in trigger_offset2vote:
                    trigger_offset2vote[trigger_offset_str] = {}
                trigger_offset2vote[trigger_offset_str][event_type] = trigger_offset2vote[trigger_offset_str].get(
                    event_type, 0) + 1.1  # if even, entity type makes the call

        # voting
        tirigger_offset2event = {}
        for trigger_offet_str, event_type2score in trigger_offset2vote.items():
            event_type = sorted(event_type2score.items(), key=lambda x: x[1], reverse=True)[0][0]
            tirigger_offset2event[trigger_offet_str] = event_type  # final event type

        # generate event list
        trigger_offset2arguments = {}
        for rel in rel_list:
            trigger_offset = rel["obj_tok_span"]
            argument_role, event_type = rel["predicate"].split(sepatator)
            trigger_offset_str = "{},{}".format(trigger_offset[0], trigger_offset[1])
            if tirigger_offset2event[trigger_offset_str] != event_type:  # filter false relations
                # set_trace()
                continue

            # append arguments
            if trigger_offset_str not in trigger_offset2arguments:
                trigger_offset2arguments[trigger_offset_str] = []
            trigger_offset2arguments[trigger_offset_str].append({
                "text": rel["subject"],
                "type": argument_role,
                "char_span": rel["subj_char_span"],
                "tok_span": rel["subj_tok_span"],
            })
        event_list = []
        for trigger_offset_str, event_type in tirigger_offset2event.items():
            arguments = trigger_offset2arguments[
                trigger_offset_str] if trigger_offset_str in trigger_offset2arguments else []
            event = {
                "trigger": trigger_offset2trigger_text[trigger_offset_str],
                "trigger_char_span": trigger_offset2trigger_char_span[trigger_offset_str],
                "trigger_tok_span": trigger_offset_str.split(","),
                "trigger_type": event_type,
                "argument_list": arguments,
            }
            event_list.append(event)
        return event_list


class DataMaker4Bert:
    def __init__(self, tokenizer, shaking_tagger):
        self.tokenizer = tokenizer
        self.shaking_tagger = shaking_tagger

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
            matrix_spots = None
            if data_type != "test":
                matrix_spots = self.shaking_tagger.get_spots(sample)

            # get codes
            input_ids = torch.tensor(codes["input_ids"]).long()
            attention_mask = torch.tensor(codes["attention_mask"]).long()
            token_type_ids = torch.tensor(codes["token_type_ids"]).long()
            tok2char_span = codes["offset_mapping"]

            sample_tp = (sample,
                         input_ids,
                         attention_mask,
                         token_type_ids,
                         tok2char_span,
                         matrix_spots,
                         )
            indexed_samples.append(sample_tp)
        return indexed_samples

    def generate_batch(self, batch_data, data_type="train"):
        sample_list = []
        input_ids_list = []
        attention_mask_list = []
        token_type_ids_list = []
        tok2char_span_list = []
        matrix_spots_list = []

        for tp in batch_data:
            sample_list.append(tp[0])
            input_ids_list.append(tp[1])
            attention_mask_list.append(tp[2])
            token_type_ids_list.append(tp[3])
            tok2char_span_list.append(tp[4])
            if data_type != "test":
                matrix_spots_list.append(tp[5])

        # @specific: indexed by bert tokenizer
        batch_input_ids = torch.stack(input_ids_list, dim=0)
        batch_attention_mask = torch.stack(attention_mask_list, dim=0)
        batch_token_type_ids = torch.stack(token_type_ids_list, dim=0)

        batch_shaking_tag = None
        if data_type != "test":
            batch_shaking_tag = self.shaking_tagger.spots2shaking_tag4batch(matrix_spots_list)

        return (sample_list, batch_input_ids, batch_attention_mask, batch_token_type_ids,
                tok2char_span_list, batch_shaking_tag)


class DataMaker4BiLSTM:
    def __init__(self, text2indices, get_tok2char_span_map, shaking_tagger):
        self.text2indices = text2indices
        self.shaking_tagger = shaking_tagger
        self.get_tok2char_span_map = get_tok2char_span_map

    def get_indexed_data(self, data, max_seq_len, data_type="train"):
        indexed_samples = []
        for ind, sample in tqdm(enumerate(data), desc="Generate indexed train or valid data"):
            text = sample["text"]
            # tagging
            matrix_spots = None
            if data_type != "test":
                matrix_spots = self.shaking_tagger.get_spots(sample)
            tok2char_span = self.get_tok2char_span_map(text)
            tok2char_span.extend([(-1, -1)] * (max_seq_len - len(tok2char_span)))
            input_ids = self.text2indices(text, max_seq_len)

            sample_tp = (sample,
                         input_ids,
                         tok2char_span,
                         matrix_spots,
                         )
            indexed_samples.append(sample_tp)
        return indexed_samples

    def generate_batch(self, batch_data, data_type="train"):
        sample_list = []
        input_ids_list = []
        tok2char_span_list = []
        matrix_spots_list = []

        for tp in batch_data:
            sample_list.append(tp[0])
            input_ids_list.append(tp[1])
            tok2char_span_list.append(tp[2])
            if data_type != "test":
                matrix_spots_list.append(tp[3])

        batch_input_ids = torch.stack(input_ids_list, dim=0)

        batch_shaking_tag = None
        if data_type != "test":
            batch_shaking_tag = self.shaking_tagger.spots2shaking_tag4batch(matrix_spots_list)

        return sample_list, batch_input_ids, tok2char_span_list, batch_shaking_tag


class DefaultLogger:
    def __init__(self, log_path, project, run_name, run_id, hyper_parameter=None):
        self.log_path = log_path
        log_dir = "/".join(self.log_path.split("/")[:-1])
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        self.run_id = run_id
        self.log("============================================================================")
        self.log("project: {}, run_name: {}, run_id: {}\n".format(project, run_name, run_id))
        if hyper_parameter:
            hyper_parameters_format = "--------------hyper_parameters----------------- \n{}\n--------------------------"
            self.log(hyper_parameters_format.format(json.dumps(hyper_parameter, indent=4)))

    def log(self, text):
        text = "run_id: {}, {}".format(self.run_id, text)
        print(text)
        open(self.log_path, "a", encoding="utf-8").write("{}\n".format(text))


class Preprocessor:
    """
    1. transform the dataset to normal format, which can fit in our codes
    2. add token level span to all entities in the relations, which will be used in tagging phase
    """

    def __init__(self, tokenizer, get_tok2char_span_map_func):
        self.tokenizer = tokenizer
        self._tokenize = tokenizer.tokenize
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
                subj_key, pred_key, obj_key = 0, 1, 2
            elif ori_format == "etl_span":
                text = " ".join(sample["tokens"])
                rel_list = sample["spo_list"]
                subj_key, pred_key, obj_key = 0, 1, 2
            elif ori_format == "raw_nyt":
                text = sample["sentText"]
                rel_list = sample["relationMentions"]
                subj_key, pred_key, obj_key = "em1Text", "label", "em2Text"
            else:
                raise ValueError("ori_fomat must in ['casrel', 'joint_re', 'raw_nyt']")

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
                    "object": rel[obj_key],
                }
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
                    if sample.get("relation_list", None):
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
                    if sample.get("entity_list", None):
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
                text = re.sub(r"([^A-Za-z0-9])", r" \1 ", text)
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
                    if (m.span()[0] - 1 >= 0 and re.match(r"\d", text_cp[m.span()[0] - 1])) or (
                            m.span()[1] < len(text_cp) and re.match(r"\d", text_cp[m.span()[1]])):
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
