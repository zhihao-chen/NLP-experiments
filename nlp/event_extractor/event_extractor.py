# -*- coding: utf8 -*-
"""
======================================
    Project Name: EventExtraction
    File Name: event_extractor
    Author: czh
    Create Date: 2021/9/15
--------------------------------------
    Change Activity: 
======================================
"""
import codecs
import glob
import json
import os
import re
import time
from pathlib import Path
from typing import List, Union

import torch
import torch.nn as nn
from accelerate import Accelerator
from torch.optim import AdamW
from torch.utils.data import TensorDataset, RandomSampler, SequentialSampler, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import get_scheduler

from nlp.models.bert_for_ee import MODEL_TYPE_CLASSES
from nlp.models.bert_for_ee_tplinker import TpLinkerForEE
from nlp.utils.ee_arguments import DataAndTrainArguments
from nlp.tools.common import init_logger, prepare_device, seed_everything
from nlp.processors.ee_seq import EEProcessor, convert_examples_to_features, collate_fn, tokenize
from nlp.processors.utils_ee import get_argument_for_seq


class EventExtractor(object):
    def __init__(self, args: DataAndTrainArguments, state: str = "train", model_path: Union[str, Path] = None):
        self.__args = args
        self.__logger = init_logger()
        self.__tensorboard_log_dir = args.output_dir
        self.__processor = EEProcessor(self.__args.data_dir)
        self.__id2label, self.__label2id, self.__num_labels, self.__event_type_dict = self.__processor.get_labels()

        self.__logger.info("label2id: ", self.__label2id)
        self.__device = torch.device("cpu")
        cuda_number = str(self.__args.cuda_number)
        self.__n_gpu = len(cuda_number.split(','))

        self.__global_step = 0
        self.__steps_trained_in_current_epoch = 0

        if state == "train":
            config, self.__tokenizer, self.__model = self.__init_ee_model()
            if not self.__args.from_scratch and self.__args.model_sate_dict_path:
                self.__restore(self.__model)
        elif state == "pred":
            model_path = self.__check_model_path(model_path)
            config, self.__tokenizer, self.__model = self.__init_ee_model(model_name_or_path=model_path)
        else:
            raise ValueError("state must be 'train' or 'pred'")

    @property
    def id2label(self):
        return self.__id2label

    @property
    def label2id(self):
        return self.__label2id

    @property
    def configs(self):
        return self.__args.__dict__

    @property
    def event_type_dict(self):
        return self.__event_type_dict

    @staticmethod
    def __check_model_path(model_path):
        if os.path.exists(model_path):
            if os.path.isdir(model_path):
                file_name = ""
                file_list = list(glob.glob(model_path+'/*.bin'))
                for name in file_list:
                    if re.search(r"args\.bin", name):
                        continue
                    if re.search(r'pytorch_model|=\d+.*', name):
                        file_name = name
                if not os.path.exists(file_name):
                    raise ValueError(f"The dir of '{model_path}' has not model file")
                else:
                    model_path = file_name
            elif os.path.isfile(model_path):
                if not re.search(r'(?:pytorch_model|=\d+.*)\.bin', model_path):
                    raise ValueError(f"The file of '{model_path}' not is model file")
        else:
            raise ValueError(f"The dir of '{model_path}' has not model file")
        return model_path

    def __check_best_model_path(self):
        if self.__args.model_type in self.__args.output_dir:
            if "best_model" in self.__args.output_dir:
                model_path = self.__args.output_dir
            else:
                model_path = os.path.join(self.__args.output_dir, "best_model")
        else:
            model_path = os.path.join(self.__args.output_dir, f"{self.__args.model_type}/best_model")
        return model_path

    def __set_device(self):
        if not self.__args.no_cuda:
            cuda_number = str(self.__args.cuda_number)
        else:
            cuda_number = ""
        self.__device, list_ids = prepare_device(cuda_number)

    def __init_train_and_evl_env(self):
        time_ = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())
        accelerator_ = Accelerator(fp16=self.__args.fp16)

        seed_everything(self.__args.seed)

        if not os.path.exists(self.__args.output_dir):
            os.mkdir(self.__args.output_dir)
        self.__args.output_dir = self.__args.output_dir + '{}'.format(self.__args.model_type)
        if not os.path.exists(self.__args.output_dir):
            os.mkdir(self.__args.output_dir)
        if os.path.exists(self.__args.output_dir) and os.listdir(
                self.__args.output_dir) and not self.__args.overwrite_output_dir:
            raise ValueError(
                "Output directory ({}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome.".format(self.__args.output_dir))
        log_dir = os.path.join(self.__args.output_dir, 'logs')
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)
        self.__logger = init_logger(log_file=log_dir + f'/{self.__args.model_type}-'
                                                       f'{self.__args.task_name}-{time_}.log')
        cuda_number = str(self.__args.cuda_number)
        if self.__n_gpu > 1 and not self.__args.no_cuda:
            self.__device = accelerator_.device
        else:
            self.__device, list_ids = prepare_device(cuda_number)

        self.__logger.warning(
            "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
            self.__args.local_rank, self.__device, self.__n_gpu, bool(self.__args.local_rank != -1), self.__args.fp16, )
        self.__tensorboard_log_dir = os.path.join(self.__args.output_dir, "tensorboard_log_dir")
        summary_writer = SummaryWriter(log_dir=self.__tensorboard_log_dir)

        return accelerator_, summary_writer

    def __init_ee_model(self, model_name_or_path=None):
        self.__args.model_type = self.__args.model_type.lower()
        config_class, tokenizer_class, model_class = MODEL_TYPE_CLASSES[self.__args.model_type]
        config = config_class.from_pretrained(self.__args.config_name if self.__args.config_name
                                              else self.__args.model_name_or_path,
                                              num_labels=self.__num_labels)
        tokenizer = tokenizer_class.from_pretrained(self.__args.tokenizer_name if self.__args.tokenizer_name
                                                    else self.__args.model_name_or_path,
                                                    do_lower_case=self.__args.do_lower_case)
        if model_name_or_path:
            model = model_class(config=config, train_config=self.__args)
            model.load_state_dict(torch.load(model_name_or_path, map_location=self.__device))
        else:
            model = model_class.from_pretrained(self.__args.model_name_or_path,
                                                from_tf=bool(".ckpt" in self.__args.model_name_or_path),
                                                config=config, train_config=self.__args,
                                                cache_dir=self.__args.cache_dir if self.__args.cache_dir else None)
        return config, tokenizer, model

    def __init_optimizer(self, total, parameters, warmup_steps):
        optimizer = AdamW(parameters, lr=self.__args.learning_rate, eps=self.__args.adam_epsilon)
        scheduler = get_scheduler(self.__args.scheduler_type, optimizer=optimizer, num_warmup_steps=warmup_steps,
                                  num_training_steps=total)
        # Check if saved optimizer or scheduler states exist
        if os.path.isfile(os.path.join(self.__args.model_name_or_path, "optimizer.pt")) and os.path.isfile(
                os.path.join(self.__args.model_name_or_path, "scheduler.pt")):
            # Load in optimizer and scheduler states
            optimizer.load_state_dict(torch.load(os.path.join(self.__args.model_name_or_path, "optimizer.pt")))
            scheduler.load_state_dict(torch.load(os.path.join(self.__args.model_name_or_path, "scheduler.pt")))

        return optimizer, scheduler

    def __get_parameters(self, model):
        no_decay = ["bias", "LayerNorm.weight", "LayerNorm.bias"]
        bert_param_optimizer = list(model.bert.named_parameters())
        crf_param_optimizer = list(model.crf.named_parameters())
        linear_param_optimizer = list(model.classifier.named_parameters())
        optimizer_grouped_parameters = [
            {'params': [p for n, p in bert_param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay': self.__args.weight_decay, 'lr': self.__args.learning_rate},
            {'params': [p for n, p in bert_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
             'lr': self.__args.learning_rate},

            {'params': [p for n, p in crf_param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay': self.__args.weight_decay, 'lr': self.__args.crf_learning_rate},
            {'params': [p for n, p in crf_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
             'lr': self.__args.crf_learning_rate},

            {'params': [p for n, p in linear_param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay': self.__args.weight_decay, 'lr': self.__args.crf_learning_rate},
            {'params': [p for n, p in linear_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
             'lr': self.__args.crf_learning_rate}
        ]
        return optimizer_grouped_parameters

    def __get_all_checkpoint_file(self):
        checkpoints = list(
            os.path.dirname(c) for c in glob.glob(self.__args.output_dir + "/checkpoint*/pytorch_model.bin",
                                                  recursive=True))
        return checkpoints

    def __get_last_checkpoint_file(self):
        return self.__get_all_checkpoint_file()[-1]

    def __restore(self, model):
        model_state_dict_path = self.__args.model_sate_dict_path
        if os.path.isdir(model_state_dict_path):
            file_name = os.path.join(model_state_dict_path, "pytorch_model.bin")
            files = list(glob.glob(model_state_dict_path+'/*.bin'))
            for name in files:
                if re.search(r"args\.bin", name):
                    continue
                if re.search(r'pytorch_model|=\d+.*', name):
                    file_name = name
            if not os.path.exists(file_name):
                raise ValueError(f"The dir of '{model_state_dict_path}' has not model file")
            else:
                model_state_dict_path = file_name
        elif os.path.isfile(model_state_dict_path):
            if not re.search(r'(?:pytorch_model|=\d+.*)\.bin', model_state_dict_path):
                raise ValueError(f"The file of '{model_state_dict_path}' not is model file")
        model.load_state_dict(torch.load(model_state_dict_path, map_location=self.__device))
        self.__logger.info(
            "---------model state {} loaded -------------".format(model_state_dict_path.split("/")[-1]))

    def __restore_from_last_checkpoint(self, train_dataloader):
        # Check if continuing training from a checkpoint
        last_checkpoint_file = self.__get_last_checkpoint_file()
        if os.path.exists(last_checkpoint_file) and "checkpoint" in last_checkpoint_file:
            # set global_step to gobal_step of last saved checkpoint from model path
            self.__global_step = int(last_checkpoint_file.split("-")[-1].split("/")[0])
            epochs_trained = self.__global_step // (len(train_dataloader) // self.__args.gradient_accumulation_steps)
            self.__steps_trained_in_current_epoch = self.__global_step % (len(train_dataloader) //
                                                                          self.__args.gradient_accumulation_steps)
            self.__logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            self.__logger.info("  Continuing training from epoch %d", epochs_trained)
            self.__logger.info("  Continuing training from global step %d", self.__global_step)
            self.__logger.info("  Will skip the first %d steps in the first epoch",
                               self.__steps_trained_in_current_epoch)

    def __load_examples(self, data_type="train"):
        self.__logger.info("Creating features from dataset file at %s", self.__args.data_dir)
        if data_type == 'train':
            examples = self.__processor.get_train_examples()
        elif data_type == 'dev':
            examples = self.__processor.get_dev_examples()
        else:
            examples = self.__processor.get_test_examples()
        return examples

    def __load_and_cache_examples(self, tokenizer, examples, data_type='train', max_seq_length=512):
        cached_features_file = os.path.join(self.__args.data_dir, 'cached_crf-{}_{}_{}_{}'.format(
            data_type,
            list(filter(None, self.__args.model_name_or_path.split('/'))).pop(),
            str(self.__args.train_max_seq_length if data_type == 'train' else self.__args.eval_max_seq_length),
            str(self.__args.task_name.lower())))
        if os.path.exists(cached_features_file) and not self.__args.overwrite_cache:
            self.__logger.info("Loading features from cached file %s", cached_features_file)
            features = torch.load(cached_features_file)
        else:
            features = convert_examples_to_features(examples=examples,
                                                    tokenizer=tokenizer,
                                                    label2id=self.__label2id,
                                                    max_seq_length=max_seq_length,
                                                    cls_token=tokenizer.cls_token,
                                                    sep_token=tokenizer.sep_token,
                                                    pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                                                    data_type=data_type,
                                                    do_lower_case=self.__args.do_lower_case,
                                                    )
            if self.__args.local_rank in [-1, 0]:
                self.__logger.info("Saving features into cached file %s", cached_features_file)
                torch.save(features, cached_features_file)
        # Convert to Tensors and build dataset
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.long)
        all_lens = torch.tensor([f.input_len for f in features], dtype=torch.long)
        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_lens, all_label_ids)
        return dataset

    @staticmethod
    def __get_prf_scores(correct_num, pred_num, gold_num, eval_type):
        """
        get precision, recall, and F1 score
        :param correct_num:
        :param pred_num:
        :param gold_num:
        :param eval_type:
        :return:
        """
        if correct_num == pred_num == gold_num == 0:
            return 1.2333, 1.2333, 1.2333  # highlight this info by illegal outputs instead of outputting 0.

        minimum = 1e-20
        precision = correct_num / (pred_num + minimum)
        recall = correct_num / (gold_num + minimum)
        f1 = 2 * precision * recall / (precision + recall + minimum)

        results = {f"{eval_type}_precision": round(precision, 5),
                   f"{eval_type}_recall": round(recall, 5),
                   f"{eval_type}_f1": round(f1, 5)}
        return results

    def train_and_valid(self):
        """
        在训练集上训练模型，并且在验证集上验证模型指标
        :return:
        """
        """ Train the model """
        accelerator, writer = self.__init_train_and_evl_env()
        n_gpu = self.__n_gpu
        train_examples = self.__load_examples(data_type="train")
        train_dataset = self.__load_and_cache_examples(self.__tokenizer,
                                                       train_examples,
                                                       data_type='train',
                                                       max_seq_length=self.__args.train_max_seq_length)
        self.__args.train_batch_size = self.__args.per_gpu_train_batch_size * max(1, n_gpu)
        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=self.__args.train_batch_size,
                                      collate_fn=collate_fn)
        if self.__args.max_steps > 0:
            t_total = self.__args.max_steps
            self.__args.num_train_epochs = self.__args.max_steps // (len(train_dataloader) //
                                                                     self.__args.gradient_accumulation_steps) + 1
        else:
            t_total = len(train_dataloader) // self.__args.gradient_accumulation_steps * self.__args.num_train_epochs

        # Prepare optimizer and schedule (linear warmup and decay)
        optimizer_grouped_parameters = self.__get_parameters(self.__model)
        warmup_steps = int(t_total * self.__args.warmup_proportion)
        optimizer, scheduler = self.__init_optimizer(t_total, optimizer_grouped_parameters, warmup_steps)
        if n_gpu > 1:
            model, optimizer, train_dataloader = accelerator.prepare(self.__model, optimizer, train_dataloader)
        else:
            self.__model.to(self.__device)
        # Train!
        self.__logger.info("***** Running training *****")
        self.__logger.info("  Num examples = %d", len(train_dataset))
        self.__logger.info("  Num Epochs = %d", self.__args.num_train_epochs)
        self.__logger.info("  Instantaneous batch size per GPU = %d", self.__args.per_gpu_train_batch_size)
        self.__logger.info("  Gradient Accumulation steps = %d", self.__args.gradient_accumulation_steps)
        self.__logger.info("  Total optimization steps = %d", t_total)

        if not self.__args.from_scratch:
            if self.__args.from_last_checkpoint:
                self.__restore_from_last_checkpoint(train_dataloader)

        self.__model.zero_grad()
        best_f1 = 0.0
        best_epoch = 0
        tolerance_num = 0
        total_loss = 0
        for epoch in range(int(self.__args.num_train_epochs)):
            tr_loss, logging_loss = 0.0, 0.0
            self.__model.train()
            for step, batch in tqdm(enumerate(train_dataloader), desc=f"Training epoch {epoch}-loss {total_loss} "):
                # Skip past any already trained steps if resuming training
                if self.__steps_trained_in_current_epoch > 0:
                    self.__steps_trained_in_current_epoch -= 1
                    continue
                batch = tuple(t.to(self.__device) for t in batch)
                inputs = {"input_ids": batch[0],
                          "attention_mask": batch[1],
                          "labels": batch[3],
                          "token_type_ids": batch[2]}

                outputs = self.__model(**inputs)
                loss, logits = outputs  # model outputs are always tuple in pytorch-transformers (see doc)

                if n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu parallel training
                if self.__args.gradient_accumulation_steps > 1:
                    loss = loss / self.__args.gradient_accumulation_steps
                accelerator.backward(loss)
                if self.__args.logging_steps > 0 and self.__global_step % self.__args.logging_steps == 0:
                    writer.add_scalar("Loss/train", loss.item(), global_step=self.__global_step)
                tr_loss += loss.item()
                if (step + 1) % self.__args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    optimizer.zero_grad()
                    self.__global_step += 1
                    if self.__args.local_rank in [-1, 0] and self.__args.save_steps > 0 and \
                            self.__global_step % self.__args.save_steps == 0:
                        # Save model checkpoint
                        output_dir = os.path.join(self.__args.output_dir, "checkpoint-{}".format(self.__global_step))
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        model_to_save = (
                            self.__model.module if hasattr(self.__model, "module") else self.__model
                        )  # Take care of distributed/parallel training
                        model_to_save.save_pretrained(output_dir)
                        torch.save(self.__args, os.path.join(output_dir, "training_args.bin"))
                        self.__logger.info("Saving model checkpoint to %s", output_dir)
                        self.__tokenizer.save_vocabulary(output_dir)
                        torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                        torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                        self.__logger.info("Saving optimizer and scheduler states to %s", output_dir)

            if self.__args.local_rank in [-1, 0] and \
                    self.__args.evaluate_during_training and self.__args.do_eval_per_epoch:
                # Log metrics
                # Only evaluate when single GPU otherwise metrics may not average well
                self.__logger.info(f"Evaluating model at {epoch + 1} time")
                result = self.__evaluate()
                f1 = result["argument_hard"]["argument_hard_f1"]
                writer.add_scalar("Loss/eval", result['loss'], global_step=self.__global_step)
                if f1 > best_f1:
                    best_f1 = f1
                    best_epoch = epoch
                    tolerance_num = 0
                    model_to_save = (
                        self.__model.module if hasattr(self.__model, "module") else self.__model
                    )  # Take care of distributed/parallel training
                    output_dir = os.path.join(self.__args.output_dir, "best_model")
                    if not os.path.exists(output_dir):
                        os.mkdir(output_dir)
                    model_to_save.save_pretrained(output_dir)
                    self.__tokenizer.save_vocabulary(output_dir)
                else:
                    tolerance_num += 1

            total_loss = tr_loss / self.__global_step
            if 'cuda' in str(self.__device):
                torch.cuda.empty_cache()
            if self.__args.early_stop and tolerance_num > self.__args.tolerance:
                break
        return best_f1, best_epoch

    def __evaluate(self, data_type="dev", prefix=""):
        """
        验证模型的指标
        :param data_type: 当为"dev"时，则表示在dev data上验证模型指标；若为"test"，则表示在test data上验证模型指标。
        注意，当为"test"时，test data必须有标注标签
        :return:
        """
        eval_output_dir = self.__args.output_dir
        if not os.path.exists(eval_output_dir) and self.__args.local_rank in [-1, 0]:
            os.makedirs(eval_output_dir)
        eval_examples = self.__load_examples(data_type)
        eval_dataset = self.__load_and_cache_examples(self.__tokenizer,
                                                      eval_examples,
                                                      data_type=data_type,
                                                      max_seq_length=self.__args.eval_max_seq_length)
        self.__args.eval_batch_size = self.__args.per_gpu_eval_batch_size * max(1, self.__n_gpu)
        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=self.__args.eval_batch_size,
                                     collate_fn=collate_fn)
        # Eval!
        self.__logger.info("***** Running evaluation %s *****", prefix)
        self.__logger.info("  Num examples = %d", len(eval_dataset))
        self.__logger.info("  Batch size = %d", self.__args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0

        event_type_results = [0, 0, 0]  # [correct_num, pred_num, gold_num]
        argument_soft_result = [0, 0, 0]  # 部分匹配role, argument
        argument_hard_results = [0, 0, 0]  # 严格匹配event_type, role, argument

        if isinstance(self.__model, nn.DataParallel):
            self.__model = self.__model.module
        for step, batch in tqdm(enumerate(eval_dataloader), desc="Evaluation"):
            batch_eval_example = eval_examples[step * self.__args.eval_batch_size:
                                               (step + 1) * self.__args.eval_batch_size]
            assert len(batch_eval_example) <= self.__args.eval_batch_size

            self.__model.eval()
            batch = tuple(t.to(self.__device) for t in batch)
            with torch.no_grad():
                inputs = {"input_ids": batch[0],
                          "attention_mask": batch[1],
                          "labels": batch[3],
                          "token_type_ids": batch[2]}
                outputs = self.__model(**inputs)
                tmp_eval_loss, logits = outputs[:2]
                mask = inputs['attention_mask']
                tags = self.__model.crf.decode(logits, mask)
            if self.__n_gpu > 1:
                tmp_eval_loss = tmp_eval_loss.mean()  # mean() to average on multi-gpu parallel evaluating
            eval_loss += tmp_eval_loss.item()
            nb_eval_steps += 1

            tags = tags.squeeze(0).cpu().numpy().tolist()
            for i, example in enumerate(batch_eval_example):
                gold_arguments_hard = set()
                pred_arguments_hard = set()
                gold_event_type = set()
                gold_arguments_soft = set()
                pred_arguments_soft = set()
                pred_event_type = set()

                for k, v in example.arguments.items():
                    if self.__args.task_name.lower() == 'ee':
                        gold_event_type.add(v[0])
                        gold_arguments_soft.add('-'.join([v[1], k]))
                    if isinstance(v, str):
                        gold_arguments_hard.add('-'.join([v, k]))
                    elif isinstance(v, tuple):
                        gold_arguments_hard.add('-'.join(v + (k,)))

                text = example.text_a
                pred_entities = get_argument_for_seq(tags[i][1:-1], self.__id2label)

                for tag, s, e in pred_entities:
                    t = text[s: e + 1]
                    if self.__args.task_name.lower() == 'ee':
                        pred_event_type.add(tag[0])
                        pred_arguments_soft.add('-'.join([tag[1], t]))
                    if isinstance(tag, str):
                        pred_arguments_hard.add('-'.join([tag, t]))
                    elif isinstance(tag, tuple):
                        pred_arguments_hard.add('-'.join(tag + (t,)))

                if self.__args.task_name.lower() == 'ee':
                    correct_arg_num_soft = len(pred_arguments_soft.intersection(gold_arguments_soft))
                    correct_event_type_num = len(pred_event_type.intersection(gold_event_type))

                    event_type_results[0] += correct_event_type_num
                    event_type_results[1] += len(pred_event_type)
                    event_type_results[2] += len(gold_event_type)

                    argument_soft_result[0] += correct_arg_num_soft
                    argument_soft_result[1] += len(pred_arguments_soft)
                    argument_soft_result[2] += len(gold_arguments_soft)

                correct_arg_num_hard = len(pred_arguments_hard.intersection(gold_arguments_hard))
                argument_hard_results[0] += correct_arg_num_hard
                argument_hard_results[1] += len(pred_arguments_hard)
                argument_hard_results[2] += len(gold_arguments_hard)

        eval_loss = eval_loss / nb_eval_steps
        results = {"loss": eval_loss,
                   "event_type": self.__get_prf_scores(*event_type_results, eval_type="event_type"),
                   "argument_hard": self.__get_prf_scores(*argument_hard_results, eval_type="argument_hard"),
                   "argument_soft": self.__get_prf_scores(*argument_soft_result, eval_type="argument_soft")}

        self.__logger.info("***** Eval results for %s datasets *****", data_type)
        info = json.dumps(results, ensure_ascii=False, indent=2)
        self.__logger.info(info)
        return results

    def evaluate(self, data_type: str = "dev", eval_all_checkpoints: bool = False):
        """
        在验证集或测试集上验证模型的指标
        :param data_type: 当为"dev"时，则表示在dev data上验证模型指标；若为"test"，则表示在test data上验证模型指标。
        注意，当为"test"时，test data必须有标注标签
        :param eval_all_checkpoints:
        :return:
        """
        self.__set_device()
        if eval_all_checkpoints:
            checkpoints = self.__get_all_checkpoint_file()
        else:
            model_path = self.__check_best_model_path()
            checkpoints = [model_path]

        results = {}
        for checkpoint in checkpoints:
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            prefix = checkpoint.split('/')[-1] if checkpoint.find('checkpoint') != -1 else ""
            model_path = self.__check_model_path(checkpoint)
            config, self.__tokenizer, self.__model = self.__init_ee_model(model_path)
            self.__model.to(self.__device)
            result = self.__evaluate(data_type, prefix)
            if global_step:
                result = {"{}_{}".format(global_step, k): v for k, v in result.items()}
            results.update(result)

        output_eval_file = os.path.join(self.__args.output_dir, f"{data_type}_results.txt")
        with codecs.open(output_eval_file, "w") as fw:
            for key in sorted(results.keys()):
                fw.write("{} = {}\n".format(key, str(results[key])))

    def predict(self, data_type: str = None,
                input_texts: Union[List[str], str] = None,
                pred_output_dir: Union[str, Path] = None):
        """
        预测函数
        :param data_type: 只能是'test'，或者None。若为test则表示在测试数据集上预测
        :param input_texts: 若不为空，则表示是预测新的数据
        :param pred_output_dir: 若不为空，则表示将预测结果写入指定位置保存，可以是目录，也可以是文件
        :return:
            {
                "text":"",
                "event_list": [
                    {
                        "event_type":"",
                        'event_type_name':
                        "arguments": [
                            {
                                "role":"",
                                "role_name":"",
                                "argument": ""
                            }
                        ]
                    }
                ]
            }
        """
        if pred_output_dir and not os.path.exists(pred_output_dir):
            os.makedirs(pred_output_dir)
        if data_type == "test":
            test_examples = self.__load_examples(data_type='test')
            texts = [i.text_a for i in test_examples]
        else:
            if input_texts:
                if isinstance(input_texts, List):
                    texts = input_texts
                elif isinstance(input_texts, str):
                    texts = [input_texts]
                else:
                    raise ValueError
            else:
                raise ValueError("'data_type' must be 'test' or 'input_texts' not empty")

        if isinstance(self.__model, nn.DataParallel):
            model = self.__model.module
        else:
            model = self.__model

        results = []
        for step, text in tqdm(enumerate(texts), desc="Predicting"):
            tokens = tokenize(text, self.__tokenizer.vocab, do_lower_case=self.__args.do_lower_case)
            if len(tokens) > self.__args.eval_max_seq_length - 2:
                tokens = tokens[: (self.__args.eval_max_seq_length - 2)]
            tokens = ["[CLS]"] + tokens + ["[SEP]"]
            input_ids = self.__tokenizer.convert_tokens_to_ids(tokens)
            input_mask = [1] * len(input_ids)
            segment_ids = [0] * len(input_ids)
            input_ids = torch.LongTensor([input_ids]).to(self.__device)
            input_mask = torch.LongTensor([input_mask]).to(self.__device)
            segment_ids = torch.LongTensor([segment_ids]).to(self.__device)
            model.eval()
            with torch.no_grad():
                inputs = {"input_ids": input_ids,
                          "attention_mask": input_mask,
                          "labels": None,
                          "token_type_ids": segment_ids}
                outputs = model(**inputs)
                logits = outputs[0]
                tags = model.crf.decode(logits, inputs['attention_mask'])
                tags = tags.squeeze(0).cpu().numpy().tolist()
            preds = tags[0][1:-1]  # [CLS]XXXX[SEP]
            label_entities = get_argument_for_seq(preds, self.__id2label)
            pred_arguments = {text[s: e + 1]: tag for tag, s, e in label_entities}
            result = {"text": text, "event_list": []}
            temp = {}
            for k, v in pred_arguments.items():
                event_type = v[0]
                role_name = self.__event_type_dict[v[0]][v[1]]
                event_type_name = self.__event_type_dict[v[0]]['name']
                if event_type not in temp:
                    temp[event_type] = {"type_name": event_type_name, "arguments": []}
                arguments = {
                        'role': v[1],
                        'role_name': role_name,
                        'argument': k
                    }
                if arguments not in temp[event_type]["arguments"]:
                    temp[event_type]["arguments"].append(arguments)
            for k, v in temp.items():
                result["event_list"].append({"event_type": k,
                                             "event_type_name": v['type_name'],
                                             'arguments': v['arguments']})
            results.append(result)
            yield result
        if pred_output_dir:
            if os.path.isdir(pred_output_dir):
                output_predict_file = os.path.join(pred_output_dir, "test_crf_prediction.json")
            elif os.path.isfile(pred_output_dir):
                output_predict_file = pred_output_dir
            else:
                raise ValueError
            with codecs.open(output_predict_file, 'w', encoding='utf8') as fw:
                for res in results:
                    fw.write(json.dumps(res, ensure_ascii=False) + '\n')


class EventExtractorTpLinker(object):
    def __init__(self, args: DataAndTrainArguments, state: str = "train", model_path: Union[str, Path] = None):
        self.__args = args
        self.__logger = init_logger()
        self.__tensorboard_log_dir = args.output_dir
        self.__processor = EEProcessor(self.__args.data_dir)
        self.__id2label, self.__label2id, self.__num_labels, self.__event_type_dict = self.__processor.get_labels()

        self.__logger.info("label2id: ", self.__label2id)
        self.__device = torch.device("cpu")
        cuda_number = str(self.__args.cuda_number)
        self.__n_gpu = len(cuda_number.split(','))

        self.__global_step = 0
        self.__steps_trained_in_current_epoch = 0

    def __set_device(self):
        if not self.__args.no_cuda:
            cuda_number = str(self.__args.cuda_number)
        else:
            cuda_number = ""
        self.__device, list_ids = prepare_device(cuda_number)

    def __init_train_and_evl_env(self):
        time_ = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())
        accelerator_ = Accelerator(fp16=self.__args.fp16)

        seed_everything(self.__args.seed)

        if not os.path.exists(self.__args.output_dir):
            os.mkdir(self.__args.output_dir)
        self.__args.output_dir = self.__args.output_dir + '{}'.format(self.__args.model_type)
        if not os.path.exists(self.__args.output_dir):
            os.mkdir(self.__args.output_dir)
        if os.path.exists(self.__args.output_dir) and os.listdir(
                self.__args.output_dir) and not self.__args.overwrite_output_dir:
            raise ValueError(
                "Output directory ({}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome.".format(self.__args.output_dir))
        log_dir = os.path.join(self.__args.output_dir, 'logs')
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)
        self.__logger = init_logger(log_file=log_dir + f'/{self.__args.model_type}-'
                                                       f'{self.__args.task_name}-{time_}.log')
        cuda_number = str(self.__args.cuda_number)
        if self.__n_gpu > 1 and not self.__args.no_cuda:
            self.__device = accelerator_.device
        else:
            self.__device, list_ids = prepare_device(cuda_number)

        self.__logger.warning(
            "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
            self.__args.local_rank, self.__device, self.__n_gpu, bool(self.__args.local_rank != -1), self.__args.fp16, )
        self.__tensorboard_log_dir = os.path.join(self.__args.output_dir, "tensorboard_log_dir")
        summary_writer = SummaryWriter(log_dir=self.__tensorboard_log_dir)

        return accelerator_, summary_writer
