# -*- coding: utf8 -*-
"""
======================================
    Project Name: NLP
    File Name: train_and_valid
    Author: czh
    Create Date: 2021/9/28
--------------------------------------
    Change Activity: 
======================================
"""
import os
import re
from glob import glob
import json
import datetime
import time
from tqdm import tqdm
from pprint import pprint

import sys
sys.path.append('/data/chenzhihao/NLP')

import torch
from torch.utils.data import DataLoader
import numpy as np

from nlp.models.bert_for_ee_tplinker import TpLinkerForEE
from nlp.processors.utils_ee import MyDatasets, decompose2splits, span_offset, unique_list
from nlp.processors.ee_span import TpLinkerEEProcessor
from nlp.utils.tokenizers import BertTokenizerAlignedWithStanza
from nlp.metrics.tplinker_metric import MetricsCalculator
from nlp.utils.taggers import create_rebased_tfboys_tagger, Tagger4RAIN
from nlp.tools.format_conv import merge_events4doc_ee
from nlp.tools.common import init_logger

logger = init_logger()
root_path = "/data/chenzhihao/NLP"

os.environ["TOKENIZERS_PARALLELISM"] = "true"
torch.backends.cudnn.deterministic = True  # noqa


configs = {
    "pretrained_model_path": "hfl/chinese-roberta-wwm-ext",
    "do_lower_case": False,
    "language": "ch",
    "data_dir": root_path + "/datas/duee1",
    "ori_data_format": "duee1",
    "use_ghm": False,
    "lr_rate": 3e-5,
    "scheduler": "CAWR",
    "device": "cuda:0",
    "cache_dir": root_path + "/datas/duee1/caches",
    "max_seq_length": 256,
    "train_batch_size": 1,
    "dev_batch_size": 1,
    "num_workers": 4,
    "epochs": 10,
    "gradient_accumulation_steps": 8,
    "model_state_dict_path": None,
    "model_bag_size": 1,
    "dir_to_save_model": root_path + "/datas/output_dir"
}

addtional_preprocessing_config = {
    "add_default_entity_type": False,
    "classify_entities_by_relation": False,
    "dtm_arg_type_by_edges": True,
}

# tagger config
tagger_config = {
    "dtm_arg_type_by_edges": addtional_preprocessing_config["dtm_arg_type_by_edges"],
    "classify_entities_by_relation": addtional_preprocessing_config["classify_entities_by_relation"],
    "add_h2t_n_t2h_links": False,
    "language": configs["language"],
}

# optimizers and schedulers
optimizer_config = {
    "class_name": "Adam",
    "parameters": {
    }
}

scheduler_dict = {
    "CAWR": {
        # CosineAnnealingWarmRestarts
        "name": "CAWR",
        "T_mult": 1,
        "rewarm_epochs": 4,
    },
    "StepLR": {
        "name": "StepLR",
        "decay_rate": 0.999,
        "decay_steps": 100,
    },
}

model_settings = {
    "pos_tag_emb_config": None,
    "ner_tag_emb_config": None,
    "subwd_encoder_config": {
        "pretrained_model_path": configs["pretrained_model_path"],
        "finetune": True,
        "use_last_k_layers": 1,
        "wordpieces_prefix": "##"},
    "dep_config": None,
    "handshaking_kernel_config": {
        "ent_shaking_type": "cln+bilstm",
        "rel_shaking_type": "cln",
        "ent_dist_emb_dim": 64,
        "rel_dist_emb_dim": 128},
    "use_attns4rel": True,
    "ent_dim": 1024,
    "rel_dim": 1024,
    "span_len_emb_dim": -1,
    "emb_ent_info2rel": False,
    "golden_ent_cla_guide": False,
    "loss_func": "mce_loss",
    "loss_weight": 0.5,
    "loss_weight_recover_steps": 0,
    "pred_threshold": 0.,
}

trainer_config = {
    "run_name": "duee1_test",
    "exp_name": "duee1",
    "scheduler_config": scheduler_dict[configs["scheduler"]],
    "log_interval": 10,
}

device = torch.device(configs["device"])
bert_tokenizer = BertTokenizerAlignedWithStanza.from_pretrained(configs["pretrained_model_path"],
                                                                add_special_tokens=False,
                                                                do_lower_case=configs["do_lower_case"],
                                                                stanza_language=configs["language"])
vocab_dict = {w: i for i, w in enumerate(bert_tokenizer.vocab)}

processor = TpLinkerEEProcessor(data_dir=configs["data_dir"],
                                language=configs["language"],
                                do_lower_case=configs["do_lower_case"],
                                tokenizer=bert_tokenizer,
                                ori_data_format=configs["ori_data_format"])


class DefaultLogger:
    def __init__(self, log_path, project, run_name, run_id, config2log):
        self.log_path = log_path
        log_dir = "/".join(self.log_path.split("/")[:-1])
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        self.run_id = run_id
        self.line = "============================================================================"
        self.log("project: {}, run_name: {}, run_id: {}\n".format(project, run_name, run_id))
        self.log({
            "config": config2log,
        })

    def log(self, content):
        log_dict = {
            "run_id": self.run_id,
            "log_text": content,
        }
        open(self.log_path, "a", encoding="utf-8").write("{}\n{}".format(self.line, json.dumps(log_dict, indent=4)))


class Evaluator:
    def __init__(self, model, cache_dir):
        self.model = model
        self.decoder = model.tagger
        self.device = device
        self.cache_dir = cache_dir

    def _predict_step_debug(self, batch_predict_data):
        sample_list = batch_predict_data["sample_list"]
        pred_tags = batch_predict_data["golden_tags"]
        pred_sample_list = self.decoder.decode_batch(sample_list, pred_tags)
        return pred_sample_list

    def _predict_debug(self, dataloader, golden_data):
        # predict
        total_pred_sample_list = []
        for batch_ind, batch_predict_data in enumerate(tqdm(dataloader, desc="debug: predicting")):
            pred_sample_list = self._predict_step_debug(batch_predict_data)
            total_pred_sample_list.extend(pred_sample_list)
        pred_data = self._alignment(total_pred_sample_list, golden_data)
        return pred_data

    def check_tagging_n_decoding(self, dataloader, golden_data):
        pred_data = self._predict_debug(dataloader, golden_data)
        return self.score(pred_data, golden_data, "debug")

    # predict step
    def _predict_step(self, batch_predict_data):
        sample_list = batch_predict_data["sample_list"]
        del batch_predict_data["sample_list"]
        if "golden_tags" in batch_predict_data:
            del batch_predict_data["golden_tags"]

        for k, v in batch_predict_data.items():
            if k == "padded_text_list":
                for sent in v:
                    sent.to(self.device)
            else:
                batch_predict_data[k] = v.to(self.device)

        with torch.no_grad():
            pred_outputs = self.model(**batch_predict_data)

        if type(pred_outputs) == tuple:
            pred_tags = [self.model.pred_output2pred_tag(pred_out) for pred_out in pred_outputs]
            pred_outputs = pred_outputs
        else:
            pred_tags = [self.model.pred_output2pred_tag(pred_outputs), ]
            pred_outputs = [pred_outputs, ]

        pred_sample_list = self.decoder.decode_batch(sample_list, pred_tags, pred_outputs)
        return pred_sample_list

    @staticmethod
    def _alignment(pred_sample_list, golden_data):
        # decompose to splits
        pred_sample_list = decompose2splits(pred_sample_list)

        # merge and alignment
        id2text = {sample["id"]: sample["text"] for sample in golden_data}
        merged_pred_samples = {}
        for sample in pred_sample_list:
            id_ = sample["id"]
            # recover spans by offsets
            anns = span_offset(sample, sample["tok_level_offset"], sample["char_level_offset"])
            sample = {**sample, **anns}

            # merge
            if id_ not in merged_pred_samples:
                merged_pred_samples[id_] = {
                    "id": id_,
                    "text": id2text[id_],
                }
            if "entity_list" in sample:
                if "entity_list" not in merged_pred_samples[id_]:
                    merged_pred_samples[id_]["entity_list"] = []
                merged_pred_samples[id_]["entity_list"].extend(sample["entity_list"])
            if "relation_list" in sample:
                if "relation_list" not in merged_pred_samples[id_]:
                    merged_pred_samples[id_]["relation_list"] = []
                merged_pred_samples[id_]["relation_list"].extend(sample["relation_list"])
            if "event_list" in sample:
                if "event_list" not in merged_pred_samples[id_]:
                    merged_pred_samples[id_]["event_list"] = []
                merged_pred_samples[id_]["event_list"].extend(sample["event_list"])
            if "open_spo_list" in sample:
                if "open_spo_list" not in merged_pred_samples[id_]:
                    merged_pred_samples[id_]["open_spo_list"] = []
                merged_pred_samples[id_]["open_spo_list"].extend(sample["open_spo_list"])

        # alignment by id (in order)
        pred_data = []
        assert len(merged_pred_samples) == len(golden_data)
        for sample in golden_data:
            id_ = sample["id"]
            pred_data.append(merged_pred_samples[id_])

        def arg_to_str(arg):
            if len(arg["tok_span"]) > 0 and type(arg["tok_span"][0]) is list:
                arg_str = "-".join([arg["type"], arg["text"]])
            else:
                s = ",".join([str(idx) for idx in arg["tok_span"]]) if len(arg["tok_span"]) > 0 else arg["text"]
                arg_str = "-".join([arg["type"], s])
            return arg_str

        for sample in pred_data:
            if "entity_list" in sample:
                sample["entity_list"] = unique_list(sample["entity_list"])
            if "relation_list" in sample:
                sample["relation_list"] = unique_list(sample["relation_list"])
            if "event_list" in sample:
                sample["event_list"] = unique_list(sample["event_list"])

                new_event_list = []

                def to_e_set(event):
                    arg_list = event["argument_list"] + [
                        {"type": "Trigger", "tok_span": event["trigger_tok_span"], "text": event["trigger"]}, ] \
                        if "trigger" in event else event["argument_list"]
                    event_set = {arg_to_str(arg) for arg in arg_list}
                    return event_set

                for e_i in sample["event_list"]:
                    event_set_i = to_e_set(e_i)
                    if any(event_set_i.issubset(to_e_set(e_j))
                           and len(event_set_i) < len(to_e_set(e_j))
                           for e_j in sample["event_list"]):  # if ei is a proper subset of another event, skip
                        continue
                    new_event_list.append(e_i)
                sample["event_list"] = new_event_list

            if "open_spo_list" in sample:
                sample["open_spo_list"] = unique_list(sample["open_spo_list"])

                # filter redundant spo
                new_spo_list = []
                for spo_i in sample["open_spo_list"]:
                    spo_set_i = {arg_to_str(arg) for arg in spo_i}
                    if any(spo_set_i.issubset({arg_to_str(arg) for arg in spo_j})
                           and len(spo_set_i) < len({arg_to_str(arg) for arg in spo_j})
                           for spo_j in sample["open_spo_list"]):  # if spo_i is a proper subset of another spo, skip
                        continue
                    new_spo_list.append(spo_i)
                sample["open_spo_list"] = new_spo_list

        # >>>>>>>>>>>>>>>>>>>>> merge events for document level EE task >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        if any(type(arg["tok_span"][0]) is list
               for sample in golden_data
               for event in sample.get("event_list", [])
               for arg in event["argument_list"]):  # if document level event extraction
            for sample in pred_data:
                sample["event_list"] = merge_events4doc_ee(sample["event_list"], 0.6)
        return pred_data

    def predict(self, dataloader, golden_data):
        # predict
        self.model.eval()
        time_str = datetime.datetime.now().strftime("%Y-%m-%d.%H-%M-%S")
        cache_file_name = "pred_data_cache_{}.json".format(time_str)
        cache_file_path = os.path.join(self.cache_dir, cache_file_name)
        with open(cache_file_path, "w", encoding="utf-8") as file_out:
            for batch_ind, batch_predict_data in enumerate(tqdm(dataloader, desc="predicting")):
                pred_sample_list = self._predict_step(batch_predict_data)
                # write to disk in case the program carsh
                for sample in pred_sample_list:
                    file_out.write("{}\n".format(json.dumps(sample, ensure_ascii=False)))
        total_pred_sample_list = load_data(cache_file_path)
        pred_data = self._alignment(total_pred_sample_list, golden_data)
        return pred_data

    def score(self, pred_data, golden_data, data_filename=""):
        """
        :param pred_data:
        :param golden_data:
        :param data_filename: just for logging
        :return:
        """
        return self.model.metrics_cal.score(pred_data, golden_data, data_filename)


class Trainer:
    def __init__(self, model, dataloader):
        self.model = model
        self.tagger = model.tagger
        self.device = device
        self.optimizer = optimizer
        self.logger = logger
        self.dataloader = dataloader

        self.run_name = trainer_config["run_name"]
        self.exp_name = trainer_config["exp_name"]
        self.logger_interval = trainer_config["log_interval"]

        scheduler_config = trainer_config["scheduler_config"]
        self.scheduler_name = scheduler_config["name"]
        self.scheduler = None

        if self.scheduler_name == "CAWR":
            t_mult = scheduler_config["T_mult"]
            rewarm_steps = scheduler_config["rewarm_epochs"] * len(dataloader)
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, rewarm_steps, t_mult)

        elif self.scheduler_name == "ReduceLROnPlateau":
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", verbose=True, patience=6)

        elif self.scheduler_name == "Step":
            decay_rate = scheduler_config["decay_rate"]
            decay_steps = scheduler_config["decay_steps"]
            self.scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=decay_steps, gamma=decay_rate)

    # train step
    def train_step(self, batch_train_data, step, gradient_accumulation_steps=1):
        golden_tags = batch_train_data["golden_tags"]
        golden_tags = [tag.to(self.device) for tag in golden_tags]

        del batch_train_data["sample_list"]
        del batch_train_data["golden_tags"]
        for k, v in batch_train_data.items():
            if k == "padded_text_list":
                for sent in v:
                    sent.to(self.device)
            else:
                batch_train_data[k] = v.to(self.device)

        pred_outputs = self.model(**batch_train_data)
        if type(pred_outputs) is not tuple:  # make it a tuple
            pred_outputs = (pred_outputs, )

        metrics_dict = self.model.get_metrics(pred_outputs, golden_tags)
        loss = metrics_dict["loss"]

        if gradient_accumulation_steps > 1:
            loss = loss / gradient_accumulation_steps
        loss.backward()
        if step % gradient_accumulation_steps == 0:
            self.optimizer.step()
            self.optimizer.zero_grad()

        return metrics_dict

    def train(self, ep_, num_epoch, gradient_accumulation_steps=1):
        # train
        self.model.train()
        self.model.zero_grad()
        t_ep = time.time()
        fin_metrics_dict = {}
        dataloader = self.dataloader
        for batch_ind, batch_train_data in enumerate(dataloader):
            t_batch = time.time()
            metrics_dict = self.train_step(batch_train_data, batch_ind+1, gradient_accumulation_steps)

            log_dict = {
                "learning_rate": self.optimizer.param_groups[0]['lr'],
                "time": time.time() - t_ep,
            }
            metrics_log = ""
            for key, met in metrics_dict.items():
                fin_metrics_dict[key] = fin_metrics_dict.get(key, 0) + met.item()
                avg_met = fin_metrics_dict[key] / (batch_ind + 1)
                metrics_log += "train_{}: {:.5}, ".format(key, avg_met)
                log_dict["train_{}".format(key)] = avg_met

            # scheduler
            if self.scheduler_name == "ReduceLROnPlateau":
                self.scheduler.step(fin_metrics_dict["loss"] / (batch_ind + 1))
            else:
                self.scheduler.step()

            batch_print_format = "\rexp: {}, run_name: {}, Epoch: {}/{}, batch: {}/{}, {}" + \
                                 "lr: {:.5}, batch_time: {:.5}, total_time: {:.5} -------------"

            print(batch_print_format.format(self.exp_name, self.run_name,
                                            ep_ + 1, num_epoch,
                                            batch_ind + 1, len(dataloader),
                                            metrics_log,
                                            self.optimizer.param_groups[0]['lr'],
                                            time.time() - t_batch,
                                            time.time() - t_ep,
                                            ), end="")

            if type(self.logger) is DefaultLogger and (batch_ind + 1) == len(dataloader):
                # if logger is not wandb, only log once at the end
                self.logger.log(log_dict)


def load_data(path, lines=None, mute=False):
    filename = path.split("/")[-1]
    try:
        data = []
        with open(path, "r", encoding="utf-8") as file_in, \
                tqdm(desc="loading data {}".format(filename), total=lines) as bar:
            if lines is not None:
                print("max number is set: {}".format(lines))

            for line in file_in:
                data.append(json.loads(line))
                if not mute:
                    bar.update()
                if lines is not None and len(data) == lines:
                    break
        if len(data) == 1:
            data = data[0]

    except json.decoder.JSONDecodeError:
        print("loading data: {}".format(filename))
        data = json.load(open(path, "r", encoding="utf-8"))
        if lines is not None:
            print("total number is set: {}".format(lines))
            data = data[:lines]
        sample_num = len(data) if type(data) == list else 1
        print("done! {} samples are loaded!".format(sample_num))
    return data


def worker_init_fn4map(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def get_score_fr_path(model_path):
    return float(re.search(r"_([\d\.]+)\.pt", model_path.split("/")[-1]).group(1))  # noqa


id2label, label2id, num_labels, event_type_dict = processor.get_labels()

metrics_cal = MetricsCalculator(configs["use_ghm"])


tagger_class_name = create_rebased_tfboys_tagger(Tagger4RAIN)


def additional_preprocess(data):
    return tagger_class_name.additional_preprocess(data, **addtional_preprocessing_config)


train_data = processor.get_train_examples()
valid_data = processor.get_dev_examples()

train_dataset = MyDatasets(train_data, max_seq_length=configs["max_seq_length"],
                           data_type='train', bert_vocab_dict=vocab_dict, additional_preprocess=additional_preprocess)


tagger = tagger_class_name(train_dataset.get_data_anns(), **tagger_config)

my_model = TpLinkerForEE(tagger=tagger, metrics_cal=metrics_cal, **model_settings)
my_model.to(device)

collate_fn = my_model.generate_batch

# optimizer
optimizer_class_name = getattr(torch.optim, optimizer_config["class_name"])
optimizer = optimizer_class_name(my_model.parameters(), lr=float(configs["lr_rate"]), **optimizer_config["parameters"])

train_dataloader = DataLoader(dataset=train_dataset, batch_size=configs["train_batch_size"],
                              shuffle=True, num_workers=configs["num_workers"], drop_last=False,
                              collate_fn=collate_fn, worker_init_fn=worker_init_fn4map)

valid_dataset = MyDatasets(valid_data, max_seq_length=configs["max_seq_length"],
                           data_type='valid', bert_vocab_dict=vocab_dict)
valid_dataloader = DataLoader(dataset=valid_dataset, batch_size=configs["dev_batch_size"],
                              shuffle=False, num_workers=configs["num_workers"],
                              drop_last=False, collate_fn=collate_fn, worker_init_fn=worker_init_fn4map)

# load pretrained model
if configs["model_state_dict_path"] is not None:
    my_model.load_state_dict(torch.load(configs["model_state_dict_path"]))
    print("model state loaded: {}".format("/".join(configs["model_state_dict_path"].split("/")[-2:])))

trainer = Trainer(my_model, train_dataloader)

evaluator = Evaluator(my_model, configs["cache_dir"])

# train and valid
score_dict4comparing = {}
best_f1 = 0.0
tolerance = 0
epochs = configs["epochs"]
for ep in range(epochs):
    # train
    trainer.train(ep, epochs, configs["gradient_accumulation_steps"])
    # valid
    pred_samples = evaluator.predict(valid_dataloader, valid_data)
    score_dict = evaluator.score(pred_samples, valid_data, "val")
    logger.log(score_dict)
    dataset2score_dict = {
        "valid_data.json": score_dict,
    }

    for metric_key, current_val_score in score_dict.items():
        if "f1" not in metric_key:
            continue

        if metric_key not in score_dict4comparing:
            score_dict4comparing[metric_key] = {
                "current": 0.0,
                "best": 0.0,
            }
        score_dict4comparing[metric_key]["current"] = current_val_score
        score_dict4comparing[metric_key]["best"] = max(current_val_score,
                                                       score_dict4comparing[metric_key]["best"])

        if "f1" in metric_key:
            if current_val_score > best_f1:
                best_f1 = current_val_score
                tolerance = 0
            else:
                tolerance += 1
        # save models
        if current_val_score > 0. and configs["model_bag_size"] > 0:
            dir_to_save_model_this_key = os.path.join(configs["dir_to_save_model"], metric_key)
            if not os.path.exists(dir_to_save_model_this_key):
                os.makedirs(dir_to_save_model_this_key)

            # save model state
            torch.save(my_model.state_dict(),
                       os.path.join(dir_to_save_model_this_key,
                                    "model_state_dict_{}_{:.5}.pt".format(ep, current_val_score * 100)))

            # all state paths
            model_state_path_list = glob("{}/model_state_*".format(dir_to_save_model_this_key))
            # sorted by scores
            sorted_model_state_path_list = sorted(model_state_path_list,
                                                  key=get_score_fr_path)
            # only save <model_bag_size> model states
            if len(sorted_model_state_path_list) > configs["model_bag_size"]:
                os.remove(sorted_model_state_path_list[0])  # remove the state dict with the minimum score
    pprint(dataset2score_dict)
    pprint(score_dict4comparing)

test_data = processor.get_test_examples()
test_dataset = MyDatasets(test_data, data_type='test', bert_vocab_dict=vocab_dict)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=1, collate_fn=collate_fn, drop_last=False)
pred_samples = evaluator.predict(test_dataloader, test_data)
score_dict = evaluator.score(pred_samples, test_data, "test")
print("test_result: ", score_dict)
