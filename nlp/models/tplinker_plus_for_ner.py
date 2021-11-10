# -*- coding: utf8 -*-
"""
======================================
    Project Name: NLP
    File Name: tplinker_plus_for_ner
    Author: czh
    Create Date: 2021/8/20
--------------------------------------
    Change Activity: 
======================================
"""
import logging
import os
import re
import json
import glob
import time
import copy
from tqdm import tqdm
from pprint import pprint
from typing import Union, List, Dict

import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from transformers import BertModel, BertTokenizerFast, get_linear_schedule_with_warmup, set_seed
from accelerate import Accelerator

from nlp.models.bert_for_relation_extraction import TPLinkerPlusBert
from nlp.models.nezha import NeZhaConfig, NeZhaModel
from nlp.utils.tplinker_plus_utils import DataAndTrainArguments, Preprocessor
from nlp.utils.tplinker_plus_ner_util import DataMaker4Bert, HandshakingTaggingScheme, MetricsCalculator
from nlp.tools.common import init_logger, save_model, prepare_device


# 整理数据，用Dataloader加载
class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


def get_last_k_paths(path_list, k):
    path_list = sorted(
        path_list, key=lambda x: int(re.search(r"(\d+)", x.split(os.sep)[-1]).group(1))
    )
    return path_list[-k:]


def filter_duplicates(ent_list):
    """
    过滤重复实体
    """
    ent_memory_set = set()
    filtered_ent_list = []
    for ent in ent_list:
        ent_memory = "{}\u2E80{}\u2E80{}".format(
            ent["tok_span"][0], ent["tok_span"][1], ent["type"]
        )
        if ent_memory not in ent_memory_set:
            filtered_ent_list.append(ent)
            ent_memory_set.add(ent_memory)

    return filtered_ent_list


def get_test_prf(pred_sample_list, gold_test_data, metrics, pattern="whole_text"):
    """
    测试集Precision,Recall,F1
    要求测试集必须已标注（按验证集格式标注）
    """
    text_id2gold_n_pred = {}  # text id to gold and pred results

    for sample in gold_test_data:
        text_id = sample["id"]
        text_id2gold_n_pred[text_id] = {
            "gold_entity_list": sample["entity_list"],
        }

    for sample in pred_sample_list:
        text_id = sample["id"]
        text_id2gold_n_pred[text_id]["pred_entity_list"] = sample["entity_list"]

    ere_cpg_dict = {
        "ent_cpg": [0, 0, 0],
    }
    for gold_n_pred in text_id2gold_n_pred.values():
        gold_ent_list = gold_n_pred["gold_entity_list"]
        pred_ent_list = (
            gold_n_pred["pred_entity_list"] if "pred_entity_list" in gold_n_pred else []
        )
        metrics.cal_ent_cpg(pred_ent_list, gold_ent_list, ere_cpg_dict, pattern)

    ent_prf = metrics.get_prf_scores(
        ere_cpg_dict["ent_cpg"][0],
        ere_cpg_dict["ent_cpg"][1],
        ere_cpg_dict["ent_cpg"][2],
    )
    prf_dict = {
        "ent_prf": ent_prf,
    }
    return prf_dict


class TPLinkerPlusForNER:
    def __init__(self, args: DataAndTrainArguments):
        self.accelerator = Accelerator(fp16=args.fp16)
        self.summer_writer = SummaryWriter(log_dir=args.tensorboard_log_dir)
        self.logger = init_logger(args.log_dir)

        self.config = args
        self.model_type = args.model_type.lower()
        self.device, list_ids = prepare_device(args.n_gpu)
        self.num_gpu = len(list_ids)

        data_home = self.config.data_dir
        self.experiment_name = self.config.task_name
        self.train_data_path = os.path.join(data_home, self.experiment_name, self.config.train_data_name)
        self.valid_data_path = os.path.join(data_home, self.experiment_name, self.config.valid_data_name)
        self.test_data_path = os.path.join(data_home, self.experiment_name, self.config.test_data_name)
        self.rel2id_path = os.path.join(data_home, self.experiment_name, self.config.rel2id)
        self.ent2id_path = os.path.join(data_home, self.experiment_name, self.config.ent2id)
        self.__global_step = 0
        self.__steps_trained_in_current_epoch = 0

        model_state_dict_dir = self.config.path_to_save_model
        if not os.path.exists(model_state_dict_dir):
            os.makedirs(model_state_dict_dir)

        self.tokenizer = BertTokenizerFast.from_pretrained(args.tokenizer_name if args.tokenizer_name
                                                           else args.bert_name_or_path,
                                                           do_lower_case=args.do_lower_case,
                                                           add_special_tokens=False,
                                                           cache_dir=args.cache_dir if args.cache_dir else None)
        self.preprocessor = Preprocessor(tokenizer=self.tokenizer,
                                         get_tok2char_span_map_func=self.__get_tok2char_span_map)
        self.max_seq_len = args.max_seq_len
        self.ent2id = json.load(open(self.ent2id_path, "r", encoding="utf-8"))
        self.handshaking_tagger = HandshakingTaggingScheme(self.ent2id, self.max_seq_len)
        self.tag_size = self.handshaking_tagger.get_tag_size()
        self.data_maker = DataMaker4Bert(self.tokenizer, self.handshaking_tagger)

        self.metrics = MetricsCalculator(self.handshaking_tagger)

    def init_env(self):
        os.environ["TOKENIZERS_PARALLELISM"] = "true"
        torch.backends.cudnn.deterministic = True
        set_seed(self.config.seed)

    def init_others(self, max_seq_len):
        self.handshaking_tagger = HandshakingTaggingScheme(self.ent2id, max_seq_len)
        self.tag_size = self.handshaking_tagger.get_tag_size()
        self.data_maker = DataMaker4Bert(self.tokenizer, self.handshaking_tagger)

        self.metrics = MetricsCalculator(self.handshaking_tagger)

    def __load_train_and_valid_data(self):
        # 读取数据
        train_data = json.load(open(self.train_data_path, "r", encoding="utf-8"))
        valid_data = json.load(open(self.valid_data_path, "r", encoding="utf-8"))
        all_data = train_data + valid_data
        max_tok_num = 0

        for sample in all_data:
            tokens = self.tokenizer.tokenize(sample['text'])
            max_tok_num = max(max_tok_num, len(tokens))

        if max_tok_num > self.config.max_seq_len:
            train_data = self.preprocessor.split_into_short_samples(train_data,
                                                                    self.config.max_seq_len,
                                                                    sliding_len=self.config.sliding_len,
                                                                    encoder=self.config.model_type
                                                                    )
            valid_data = self.preprocessor.split_into_short_samples(valid_data,
                                                                    self.config.max_seq_len,
                                                                    sliding_len=self.config.sliding_len,
                                                                    encoder=self.config.model_type
                                                                    )

        self.logger.info("train: {}".format(len(train_data)), "valid: {}".format(len(valid_data)))
        return train_data, valid_data, max_tok_num

    def __get_data_loader(self, train_data, valid_data, max_seq_len):
        indexed_train_data = self.data_maker.get_indexed_data(train_data, max_seq_len)
        indexed_valid_data = self.data_maker.get_indexed_data(valid_data, max_seq_len)

        train_dataloader = DataLoader(MyDataset(indexed_train_data),
                                      batch_size=self.config.train_batch_size,
                                      shuffle=True,
                                      num_workers=self.config.num_workers,
                                      drop_last=False,
                                      collate_fn=self.data_maker.generate_batch,
                                      )
        valid_dataloader = DataLoader(MyDataset(indexed_valid_data),
                                      batch_size=self.config.eval_batch_size,
                                      shuffle=True,
                                      num_workers=self.config.num_workers,
                                      drop_last=False,
                                      collate_fn=self.data_maker.generate_batch,
                                      )
        return train_dataloader, valid_dataloader

    def __load_test_datas(self):
        # Load Data
        test_data_path_dict = {}
        for file_path in glob.glob(self.test_data_path):
            file_name = re.search(r"(.*?)\.json", file_path.split(os.sep)[-1]).group(1)
            test_data_path_dict[file_name] = file_path

        test_data_dict = {}
        for file_name, path in test_data_path_dict.items():
            test_data_dict[file_name] = json.load(open(path, "r", encoding="utf-8"))

        all_data = []
        for data in list(test_data_dict.values()):
            all_data.extend(data)

        max_tok_num = 0
        for sample in tqdm(all_data, desc="Calculate the max token number"):
            tokens = self.tokenizer.tokenize(sample["text"])
            max_tok_num = max(len(tokens), max_tok_num)
        return all_data, max_tok_num, test_data_dict

    def process_predict_data(self, data: Union[List[Dict], List[str], str], max_seq_len=None):
        if isinstance(data, str):
            data = [{"id": 0, "text": data}]
        if isinstance(data, list) and isinstance(data[0], str):
            temp = []
            for i, t in enumerate(data):
                if t:
                    temp.append({"id": i, "text": t})
            data = temp
        ori_test_data = copy.deepcopy(data)
        if not max_seq_len:
            max_tok_num = 0
            for sample in tqdm(data, desc="Calculate the max token number"):
                tokens = self.tokenizer.tokenize(sample["text"])
                max_tok_num = max(len(tokens), max_tok_num)
            max_seq_len = max_tok_num
        if max_seq_len > self.max_seq_len:
            max_seq_len = self.max_seq_len
            data = self.preprocessor.split_into_short_samples(data,
                                                              max_seq_len,
                                                              sliding_len=self.config.sliding_len,
                                                              data_type='test')
        return data, ori_test_data, max_seq_len

    def __get_tok2char_span_map(self, text):
        return self.tokenizer.encode_plus(text, return_offsets_mapping=True, add_special_tokens=False)["offset_mapping"]

    def init_model(self, tag_size):
        if self.model_type == "bert":
            encoder = BertModel.from_pretrained(self.config.bert_name_or_path)
        elif self.model_type == "nezha":
            config = NeZhaConfig.from_pretrained(self.config.bert_config_name if self.config.bert_config_name else
                                                 self.config.bert_name_or_path)
            encoder = NeZhaModel(config)
        else:
            raise ValueError("'model_type' must be 'bert' or 'nezha'")

        ent_extractor = TPLinkerPlusBert(encoder,
                                         tag_size,
                                         self.config.shaking_type,
                                         self.config.inner_enc_type,
                                         self.config.tok_pair_sample_rate
                                         )
        return ent_extractor

    def __loss_func(self, y_pred, y_true, metrics):
        return metrics.loss_func(y_pred, y_true, ghm=self.config.ghm)

    def load_trained_model(self):
        # Load trained model path
        model_state_dir = self.config.model_state_dict_path
        run_id2model_state_paths = {}
        for dirs in os.listdir(model_state_dir):
            if "eval_results" in dirs:
                continue
            if not self.config.eval_all_checkpoints:
                if re.match(r"checkpoint.*", dirs):
                    continue
            root = os.path.join(model_state_dir, dirs)
            for file_name in os.listdir(root):
                run_id = root.split(os.sep)[-2].split("-")[-1] if model_state_dir == "./wandb" else \
                    root.split(os.sep)[-1].split("-")[-1]
                if re.match(r"pytorch_model.*\.bin", file_name):
                    if run_id not in run_id2model_state_paths:
                        run_id2model_state_paths[run_id] = []
                    model_state_path = os.path.join(root, file_name)
                    run_id2model_state_paths[run_id].append(model_state_path)

        assert len(run_id2model_state_paths) != 0, "未加载到已训练模型，请检查路径及run_id"

        # only last k models
        k = self.config.last_k_model
        for run_id, path_list in run_id2model_state_paths.items():
            if len(path_list) > 1:
                run_id2model_state_paths[run_id] = get_last_k_paths(path_list, k)

        return run_id2model_state_paths

    def __init_optimizer_scheduler(self, model, train_dataloader):
        # optimizer
        init_learning_rate = float(self.config.lr)
        optimizer = torch.optim.AdamW(model.parameters(), lr=init_learning_rate)

        if self.config.scheduler == "CAWR":
            t_mult = self.config.T_mult
            rewarm_epoch_num = self.config.rewarm_epoch_num
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
                                                                             len(train_dataloader) * rewarm_epoch_num,
                                                                             t_mult)

        elif self.config.scheduler == "Step":
            decay_rate = self.config.decay_rate
            decay_steps = self.config.decay_steps
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=decay_steps, gamma=decay_rate)

        elif self.config.scheduler == "ReduceLROnPlateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", verbose=True, patience=6)
        else:
            t_total = len(train_dataloader) // self.config.gradient_accumulation_steps * self.config.epochs
            scheduler = get_linear_schedule_with_warmup(optimizer,
                                                        num_warmup_steps=int(t_total * self.config.warmup_proportion),
                                                        num_training_steps=t_total)

        return optimizer, scheduler

    def restore_from_last_checkpoint(self, train_dataloader):
        # Check if continuing training from a checkpoint
        if os.path.exists(self.config.model_state_dict_path) and "checkpoint" in self.config.model_state_dict_path:
            # set global_step to gobal_step of last saved checkpoint from model path
            self.__global_step = int(self.config.model_state_dict_path.split("-")[-1].split("/")[0])
            epochs_trained = self.__global_step // (len(train_dataloader) // self.config.gradient_accumulation_steps)
            self.__steps_trained_in_current_epoch = self.__global_step % (len(train_dataloader) //
                                                                          self.config.gradient_accumulation_steps)
            self.logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            self.logger.info("  Continuing training from epoch %d", epochs_trained)
            self.logger.info("  Continuing training from global step %d", self.__global_step)
            self.logger.info("  Will skip the first %d steps in the first epoch", self.__steps_trained_in_current_epoch)

    def restore(self, model):
        model_state_dict_path = self.config.model_state_dict_path
        if os.path.isdir(model_state_dict_path):
            file_name = os.path.join(model_state_dict_path, "pytorch_model.bin")
            if not os.path.exists(file_name):
                raise ValueError(f"The dir of '{model_state_dict_path}' has not 'pytorch_model.bin'")
            else:
                model_state_dict_path = file_name
        elif os.path.isfile(model_state_dict_path):
            if not re.match(r"pytorch_model.*\.bin", model_state_dict_path):
                raise ValueError(f"The file of '{model_state_dict_path}' not is 'pytorch_model.bin'")
        model.load_state_dict(torch.load(model_state_dict_path, map_location=self.device))
        self.logger.info(
            "---------model state {} loaded -------------".format(model_state_dict_path.split("/")[-1]))

    # valid step
    def __valid_step(self, batch_valid_data, ent_extractor, metrics):
        sample_list, input_ids, attention_mask, token_type_ids, tok2char_span_list, shaking_tag = batch_valid_data

        input_ids, attention_mask, token_type_ids, shaking_tag = (input_ids.to(self.device),
                                                                  attention_mask.to(self.device),
                                                                  token_type_ids.to(self.device),
                                                                  shaking_tag.to(self.device))

        with torch.no_grad():
            pred_shaking_outputs, _ = ent_extractor(input_ids,
                                                    attention_mask,
                                                    token_type_ids,
                                                    )

        pred_shaking_tag = (pred_shaking_outputs > 0.).long()
        sample_acc = metrics.get_sample_accuracy(pred_shaking_tag,
                                                 shaking_tag)

        cpg_dict = metrics.get_cpg(sample_list,
                                   tok2char_span_list,
                                   pred_shaking_tag,
                                   self.config.match_pattern)
        return sample_acc.item(), cpg_dict

    def __train(self, model, optimizer, scheduler, train_dataloader, valid_dataloader, tag_size, ep):
        model.train()

        t_ep = time.time()
        total_loss, total_sample_acc = 0., 0.
        for batch_ind, batch_train_data in enumerate(train_dataloader):
            t_batch = time.time()
            if self.__steps_trained_in_current_epoch > 0:
                self.__steps_trained_in_current_epoch -= 1
                continue
            sample_list, input_ids, attention_mask, token_type_ids, tok2char_span_list, shaking_tag = batch_train_data

            input_ids, attention_mask, token_type_ids, shaking_tag = (input_ids.to(self.device),
                                                                      attention_mask.to(self.device),
                                                                      token_type_ids.to(self.device),
                                                                      shaking_tag.to(self.device))

            # zero the parameter gradients
            optimizer.zero_grad()
            pred_small_shaking_outputs, sampled_tok_pair_indices = model(input_ids,
                                                                         attention_mask,
                                                                         token_type_ids
                                                                         )

            # sampled_tok_pair_indices: (batch_size, ~segment_len)
            # batch_small_shaking_tag: (batch_size, ~segment_len, tag_size)
            batch_small_shaking_tag = shaking_tag.gather(1, sampled_tok_pair_indices[:, :, None].repeat(1, 1, tag_size))

            loss = self.__loss_func(pred_small_shaking_outputs, batch_small_shaking_tag, self.metrics)
            if self.num_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if self.config.gradient_accumulation_steps > 1:
                loss = loss / self.config.gradient_accumulation_steps
            self.accelerator.backward(loss)
            self.summer_writer.add_scalar("loss/Train", loss, self.__global_step)

            pred_small_shaking_tag = (pred_small_shaking_outputs > 0.).long()
            sample_acc = self.metrics.get_sample_accuracy(pred_small_shaking_tag,
                                                          batch_small_shaking_tag)

            total_loss += loss
            total_sample_acc += sample_acc
            avg_loss = total_loss / (batch_ind + 1)
            if (batch_ind + 1) % self.config.gradient_accumulation_steps == 0:
                optimizer.step()
                # scheduler
                if self.config.scheduler == "ReduceLROnPlateau":
                    scheduler.step(avg_loss)
                else:
                    scheduler.step()

                optimizer.zero_grad()
                self.__global_step += 1

            avg_sample_acc = total_sample_acc / (batch_ind + 1)

            batch_print_format = "\rproject: {}, Epoch: {}/{}, batch: {}/{}, train_loss: {}, " + \
                                 "t_sample_acc: {}," + "lr: {}, batch_time: {}, total_time: {} -------------"

            self.logger.info(batch_print_format.format(self.experiment_name,
                                                       ep + 1, self.config.epochs,
                                                       batch_ind + 1, len(train_dataloader),
                                                       avg_loss,
                                                       avg_sample_acc,
                                                       optimizer.param_groups[0]['lr'],
                                                       time.time() - t_batch,
                                                       time.time() - t_ep,
                                                       ))

            if self.config.logger == "wandb" and batch_ind % self.config.log_interval == 0:
                self.logger.log({
                    "epoch": ep,
                    "train_loss": avg_loss,
                    "train_small_shaking_seq_acc": avg_sample_acc,
                    "learning_rate": optimizer.param_groups[0]['lr'],
                    "time": time.time() - t_ep,
                })
            if self.config.evaluate_during_training and self.config.local_rank in [-1, 0] and \
                    self.config.logging_steps > 0 and self.__global_step % self.config.logging_steps == 0:
                f1_score = self.__valid(model, valid_dataloader)
                self.logger.info(f"epoch: {ep}\tglobal steps: {self.__global_step}\tf1_score: {f1_score}")
            if self.config.local_rank in [-1, 0] and self.config.save_steps > 0 and \
                    self.__global_step % self.config.save_steps == 0:
                output_dir = os.path.join(self.config.output_dir, "checkpoint-{}".format(self.__global_step))
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                save_model(model, os.path.join(output_dir, 'pytorch_model.bin'))
                self.logger.info("Saving model checkpoint to %s", output_dir)
                self.tokenizer.save_vocabulary(output_dir)
                torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                self.logger.info("Saving optimizer and scheduler states to %s", output_dir)

    def __valid(self, model, dataloader):
        # valid
        model.eval()

        t_ep = time.time()
        total_sample_acc = 0.
        total_cpg_dict = {}
        for batch_ind, batch_valid_data in enumerate(tqdm(dataloader, desc="Validating")):
            sample_acc, cpg_dict = self.__valid_step(batch_valid_data, model, self.metrics)
            total_sample_acc += sample_acc

            # init total_cpg_dict
            for k in cpg_dict.keys():
                if k not in total_cpg_dict:
                    total_cpg_dict[k] = [0, 0, 0]

            for k, cpg in cpg_dict.items():
                for idx, n in enumerate(cpg):
                    total_cpg_dict[k][idx] += cpg[idx]

        avg_sample_acc = total_sample_acc / len(dataloader)

        if "ent_cpg" in total_cpg_dict:
            ent_prf = self.metrics.get_prf_scores(total_cpg_dict["ent_cpg"][0], total_cpg_dict["ent_cpg"][1],
                                                  total_cpg_dict["ent_cpg"][2])
            final_score = ent_prf[2]
            log_dict = {
                "val_shaking_tag_acc": avg_sample_acc,
                "val_ent_prec": ent_prf[0],
                "val_ent_recall": ent_prf[1],
                "val_ent_f1": ent_prf[2],
                "time": time.time() - t_ep,
            }
        elif "trigger_iden_cpg" in total_cpg_dict:
            trigger_iden_prf = self.metrics.get_prf_scores(total_cpg_dict["trigger_iden_cpg"][0],
                                                           total_cpg_dict["trigger_iden_cpg"][1],
                                                           total_cpg_dict["trigger_iden_cpg"][2])
            trigger_class_prf = self.metrics.get_prf_scores(total_cpg_dict["trigger_class_cpg"][0],
                                                            total_cpg_dict["trigger_class_cpg"][1],
                                                            total_cpg_dict["trigger_class_cpg"][2])
            arg_iden_prf = self.metrics.get_prf_scores(total_cpg_dict["arg_iden_cpg"][0],
                                                       total_cpg_dict["arg_iden_cpg"][1],
                                                       total_cpg_dict["arg_iden_cpg"][2])
            arg_class_prf = self.metrics.get_prf_scores(total_cpg_dict["arg_class_cpg"][0],
                                                        total_cpg_dict["arg_class_cpg"][1],
                                                        total_cpg_dict["arg_class_cpg"][2])
            final_score = arg_class_prf[2]
            log_dict = {
                "val_shaking_tag_acc": avg_sample_acc,
                "val_trigger_iden_prec": trigger_iden_prf[0],
                "val_trigger_iden_recall": trigger_iden_prf[1],
                "val_trigger_iden_f1": trigger_iden_prf[2],
                "val_trigger_class_prec": trigger_class_prf[0],
                "val_trigger_class_recall": trigger_class_prf[1],
                "val_trigger_class_f1": trigger_class_prf[2],
                "val_arg_iden_prec": arg_iden_prf[0],
                "val_arg_iden_recall": arg_iden_prf[1],
                "val_arg_iden_f1": arg_iden_prf[2],
                "val_arg_class_prec": arg_class_prf[0],
                "val_arg_class_recall": arg_class_prf[1],
                "val_arg_class_f1": arg_class_prf[2],
                "time": time.time() - t_ep,
            }
        else:
            log_dict = {}
            final_score = 0.0

        self.logger.log(level=logging.INFO, msg=json.dumps(log_dict, ensure_ascii=False, indent=2))
        pprint(log_dict)

        return final_score

    def train_and_valid(self):
        train_data, valid_data, max_tok_num = self.__load_train_and_valid_data()
        self.max_seq_len = min(max_tok_num, self.max_seq_len)
        self.init_others(self.max_seq_len)

        train_dataloader, valid_dataloader = self.__get_data_loader(train_data, valid_data, self.max_seq_len)
        model = self.init_model(self.tag_size)
        if not self.config.fr_scratch:
            if self.config.fr_last_checkpoint:
                self.restore_from_last_checkpoint(train_dataloader)
            else:
                self.restore(model)

        model.to(self.device)
        optimizer, scheduler = self.__init_optimizer_scheduler(model, train_dataloader)

        max_f1 = 0.0
        steps = 0.0
        for ep in range(self.config.epochs):
            self.__train(model, optimizer, scheduler, train_dataloader, valid_dataloader, self.tag_size, ep)
            valid_f1 = self.__valid(model, valid_dataloader)

            if valid_f1 >= max_f1:
                max_f1 = valid_f1
                if valid_f1 > self.config.f1_2_save:  # save the best model
                    output_dir = self.config.path_to_save_model
                    save_model(model, os.path.join(output_dir, f'pytorch_model_{ep}.bin'))
                    self.tokenizer.save_vocabulary(output_dir)
                    self.logger.info(f"save model to {output_dir}")
                steps = 0
            else:
                steps += 1
            self.logger.info("Current avf_f1: {}, Best f1: {}".format(valid_f1, max_f1))
            if steps > self.config.patience:
                break

    def evaluate(self):
        test_data, max_tok_num, test_data_dict = self.__load_test_datas()
        split_test_data = False
        if max_tok_num > self.max_seq_len:
            split_test_data = True
            self.logger.info(
                "max_tok_num: {}, lagger than max_test_seq_len: {}, test data will be split!".format(
                    max_tok_num, self.max_seq_len
                )
            )
        else:
            self.logger.info(
                "max_tok_num: {}, less than or equal to max_test_seq_len: {}, no need to split!".format(
                    max_tok_num, self.max_seq_len
                )
            )
        max_seq_len = min(max_tok_num, self.max_seq_len)
        self.init_others(max_seq_len)

        ori_test_data_dict = copy.deepcopy(test_data_dict)
        if split_test_data:
            test_data_dict = {}
            for file_name, data in ori_test_data_dict.items():
                test_data_dict[file_name] = self.preprocessor.split_into_short_samples(
                    data,
                    max_seq_len,
                    sliding_len=self.config.sliding_len,
                    encoder=self.config.model_type,
                    data_type="test",
                )

        ent_extractor = self.init_model(self.tag_size)
        run_id2model_state_paths = self.load_trained_model()
        # predict
        res_dict = {}
        predict_statistics = {}
        for file_name, short_data in test_data_dict.items():
            ori_test_data = ori_test_data_dict[file_name]
            for run_id, model_path_list in run_id2model_state_paths.items():
                save_dir4run = os.path.join(self.config.save_res_dir, run_id)
                if self.config.save_res and not os.path.exists(save_dir4run):
                    os.makedirs(save_dir4run)

                for i, model_state_path in enumerate(model_path_list):
                    res = re.search(r"(\d+)", model_state_path.split(os.sep)[-1])
                    if res:
                        res_num = res.group(1)
                    else:
                        res_num = i
                    save_path = os.path.join(
                        save_dir4run, "{}_res_{}.json".format(file_name, res_num)
                    )

                    if os.path.exists(save_path):
                        pred_sample_list = [
                            json.loads(line) for line in open(save_path, "r", encoding="utf-8")
                        ]
                        self.logger.info("{} already exists, load it directly!".format(save_path))
                    else:
                        # load model state
                        ent_extractor.load_state_dict(torch.load(model_state_path, map_location=self.device))
                        ent_extractor.eval()
                        self.logger.info("run_id: {}, model state {} loaded".format(
                            run_id, model_state_path.split(os.sep)[-1]))

                        # predict
                        pred_sample_list = self.predict(short_data, ent_extractor, ori_test_data, max_seq_len,
                                                        batch_size=self.config.eval_batch_size)

                    res_dict[save_path] = pred_sample_list
                    predict_statistics[save_path] = len(
                        [s for s in pred_sample_list if len(s["entity_list"]) > 0]
                    )
        pprint(predict_statistics)

        # check
        for path, res in res_dict.items():
            for sample in tqdm(res, desc="check char span"):
                text = sample["text"]
                for ent in sample["entity_list"]:
                    assert ent["text"] == text[ent["char_span"][0]: ent["char_span"][1]], f"{ent}\t{text}"

        # save
        if self.config.save_res:
            for path, res in res_dict.items():
                with open(path, "w", encoding="utf-8") as file_out:
                    for sample in tqdm(res, desc="Output"):
                        if len(sample["entity_list"]) == 0:
                            continue
                        json_line = json.dumps(sample, ensure_ascii=False)
                        file_out.write("{}\n".format(json_line))
        if self.config.score:
            filepath2scores = {}
            for file_path, pred_samples in res_dict.items():
                file_name = re.search(r"(.*?)_res_\d+\.json", file_path.split(os.sep)[-1]).group(1)
                gold_test_data = ori_test_data_dict[file_name]
                prf_dict = get_test_prf(
                    pred_samples,
                    gold_test_data,
                    self.metrics,
                    pattern=self.config.match_pattern,
                )
                filepath2scores[file_path] = prf_dict
            print("---------------- Results -----------------------")
            pprint(filepath2scores)

    def predict(self, test_data: Union[List, str], model=None, ori_test_data=None, max_seq_len=512, batch_size=1):
        """
        test_data: if split, it would be samples with subtext
        ori_test_data: the original data has not been split, used to get original text here
        """
        if model is None:
            model = self.init_model(self.tag_size)
            self.restore(model)
        model.to(self.device)
        indexed_test_data = self.data_maker.get_indexed_data(
            test_data, max_seq_len, data_type="test"
        )  # fill up to max_seq_len
        test_dataloader = DataLoader(
            MyDataset(indexed_test_data),
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            drop_last=False,
            collate_fn=lambda data_batch: self.data_maker.generate_batch(
                data_batch, data_type="test"
            ),
        )

        pred_sample_list = []
        for batch_test_data in tqdm(test_dataloader, desc="Predicting"):
            batch_input_ids, batch_attention_mask, batch_token_type_ids = None, None, None
            sample_list, tok2char_span_list = [], []
            if self.model_type in ["bert", "nezha"]:
                (
                    sample_list,
                    batch_input_ids,
                    batch_attention_mask,
                    batch_token_type_ids,
                    tok2char_span_list,
                    _,
                ) = batch_test_data

                batch_input_ids, batch_attention_mask, batch_token_type_ids = (
                    batch_input_ids.to(self.device),
                    batch_attention_mask.to(self.device),
                    batch_token_type_ids.to(self.device),
                )

            elif self.model_type in {
                "bilstm",
            }:
                sample_list, batch_input_ids, tok2char_span_list, _ = batch_test_data

                batch_input_ids = batch_input_ids.to(self.device)
            model.eval()
            with torch.no_grad():
                if self.model_type in ["bert", "nezha"]:
                    batch_pred_shaking_tag, _ = model(
                        batch_input_ids, batch_attention_mask, batch_token_type_ids,
                    )
                elif self.model_type in {
                    "bilstm",
                }:
                    batch_pred_shaking_tag, _ = model(batch_input_ids)

            batch_pred_shaking_tag = (batch_pred_shaking_tag > 0.0).long()
            if max_seq_len > self.max_seq_len or self.config.force_split:
                split_test_data = True
                print("force to split the test dataset!")
            else:
                split_test_data = False

            for ind in range(len(sample_list)):
                gold_sample = sample_list[ind]
                text = gold_sample["text"]
                text_id = gold_sample["id"]
                tok2char_span = tok2char_span_list[ind]
                pred_shaking_tag = batch_pred_shaking_tag[ind]
                tok_offset, char_offset = 0, 0
                if split_test_data:
                    tok_offset, char_offset = (
                        gold_sample["tok_offset"],
                        gold_sample["char_offset"],
                    )
                ent_list = self.handshaking_tagger.decode_ent(
                    text,
                    pred_shaking_tag,
                    tok2char_span,
                    tok_offset=tok_offset,
                    char_offset=char_offset,
                )
                pred_sample_list.append(
                    {"text": text, "id": text_id, "entity_list": ent_list, }
                )

        # merge
        text_id2pred_res = {}
        for sample in pred_sample_list:
            text_id = sample["id"]
            if text_id not in text_id2pred_res:
                text_id2pred_res[text_id] = {
                    "ent_list": sample["entity_list"],
                }
            else:
                text_id2pred_res[text_id]["ent_list"].extend(sample["entity_list"])

        text_id2text = {sample["id"]: sample["text"] for sample in ori_test_data}
        merged_pred_sample_list = []
        for text_id, pred_res in text_id2pred_res.items():
            filtered_ent_list = filter_duplicates(pred_res["ent_list"])
            merged_pred_sample_list.append(
                {
                    "id": text_id,
                    "text": text_id2text[text_id],
                    "entity_list": filtered_ent_list,
                }
            )

        return merged_pred_sample_list
