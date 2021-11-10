# -*- coding: utf8 -*-
"""
======================================
    Project Name: NLP
    File Name: train_tplinker
    Author: czh
    Create Date: 2021/8/16
--------------------------------------
    Change Activity: 
======================================
"""
import json
import os
from tqdm import tqdm
from pprint import pprint
from transformers import BertTokenizerFast, AutoModel
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import glob
import time
from nlp.utils.tplinker_utils import (Preprocessor, DefaultLogger, HandshakingTaggingScheme,
                                      DataMaker4Bert, DataMaker4BiLSTM)
from nlp.models.bert_for_relation_extraction import TPLinkerBert, TPLinkerBiLSTM
from nlp.metrics.metric import TPLinkerMetricsCalculator

from experiments.configs.tplinker import tplinker_config as config
import numpy as np


config = config.train_config
hyper_parameters = config["hyper_parameters"]

os.environ["TOKENIZERS_PARALLELISM"] = "true"
# os.environ["CUDA_VISIBLE_DEVICES"] = str(config["device_num"])
device = torch.device("cuda:1")

# for reproductivity
torch.manual_seed(hyper_parameters["seed"])  # pytorch random seed
torch.backends.cudnn.deterministic = True

data_home = config["data_home"]
experiment_name = config["exp_name"]
train_data_path = os.path.join(data_home, experiment_name, config["train_data"])
valid_data_path = os.path.join(data_home, experiment_name, config["valid_data"])
rel2id_path = os.path.join(data_home, experiment_name, config["rel2id"])

logger = DefaultLogger(config["log_path"], experiment_name, config["run_name"], config["run_id"], hyper_parameters)
model_state_dict_dir = config["path_to_save_model"]
if not os.path.exists(model_state_dict_dir):
    os.makedirs(model_state_dict_dir)


def init_bilstm():
    tokenize = lambda text: text.split(" ")

    def get_tok2char_span_map(text):
        tokens = text.split(" ")
        tok2char_span = []
        char_num = 0
        for tok in tokens:
            tok2char_span.append((char_num, char_num + len(tok)))
            char_num += len(tok) + 1  # +1: whitespace
        return tok2char_span

    return tokenize, get_tok2char_span_map


def init_bert():
    # @specific
    tokenizer = BertTokenizerFast.from_pretrained(config["bert_path"], add_special_tokens=False, do_lower_case=False)
    tokenize = tokenizer.tokenize
    get_tok2char_span_map = lambda text: tokenizer.encode_plus(text, return_offsets_mapping=True,
                                                               add_special_tokens=False)["offset_mapping"]
    return tokenizer, tokenize, get_tok2char_span_map


def load_and_trans_data(preprocessor, tokenize):
    train_data = json.load(open(train_data_path, "r", encoding="utf-8"))
    valid_data = json.load(open(valid_data_path, "r", encoding="utf-8"))
    # train and valid max token num
    max_tok_num = 0
    all_data = train_data + valid_data

    for sample in all_data:
        tokens = tokenize(sample["text"])
        max_tok_num = max(max_tok_num, len(tokens))

    if max_tok_num > hyper_parameters["max_seq_len"]:
        train_data = preprocessor.split_into_short_samples(train_data, hyper_parameters["max_seq_len"],
                                                           sliding_len=hyper_parameters["sliding_len"],
                                                           encoder=config["encoder"]
                                                           )
        valid_data = preprocessor.split_into_short_samples(valid_data, hyper_parameters["max_seq_len"],
                                                           sliding_len=hyper_parameters["sliding_len"],
                                                           encoder=config["encoder"]
                                                           )

    print("train: {}".format(len(train_data)), "valid: {}".format(len(valid_data)))
    return train_data, valid_data, max_tok_num


def get_token_idx():
    token2idx_path = os.path.join(data_home, experiment_name, config["token2idx"])
    token2idx = json.load(open(token2idx_path, "r", encoding="utf-8"))
    idx2token = {idx: tok for tok, idx in token2idx.items()}
    return token2idx, idx2token


class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


def load_dataloader(data_maker, data, max_seq_len):
    indexed_data = data_maker.get_indexed_data(data, max_seq_len)
    dataloader = DataLoader(MyDataset(indexed_data),
                            batch_size=hyper_parameters["batch_size"],
                            shuffle=True,
                            num_workers=6,
                            drop_last=False,
                            collate_fn=data_maker.generate_batch,
                            )
    return dataloader


def init_bert_model(max_seq_len, rel2id):
    # # Model
    encoder = AutoModel.from_pretrained(config["bert_path"])
    hidden_size = encoder.config.hidden_size
    fake_inputs = torch.zeros([hyper_parameters["batch_size"], max_seq_len, hidden_size]).to(device)
    rel_extractor = TPLinkerBert(encoder,
                                 len(rel2id),
                                 hyper_parameters["shaking_type"],
                                 hyper_parameters["inner_enc_type"],
                                 hyper_parameters["dist_emb_size"],
                                 hyper_parameters["ent_add_dist"],
                                 hyper_parameters["rel_add_dist"],
                                 )
    return rel_extractor, fake_inputs


def init_bilstm_model(max_seq_len, rel2id, get_tok2char_span_map, handshaking_tagger):
    token2idx, idx2token = get_token_idx()

    def text2indices(text, max_seq_len):
        input_ids = []
        tokens = text.split(" ")
        for tok in tokens:
            if tok not in token2idx:
                input_ids.append(token2idx['<UNK>'])
            else:
                input_ids.append(token2idx[tok])
        if len(input_ids) < max_seq_len:
            input_ids.extend([token2idx['<PAD>']] * (max_seq_len - len(input_ids)))
        input_ids = torch.tensor(input_ids[:max_seq_len])
        return input_ids

    data_maker = DataMaker4BiLSTM(text2indices, get_tok2char_span_map, handshaking_tagger)

    glove = Glove()
    glove = glove.load(config["pretrained_word_embedding_path"])

    # prepare embedding matrix
    word_embedding_init_matrix = np.random.normal(-1, 1,
                                                  size=(len(token2idx), hyper_parameters["word_embedding_dim"]))
    count_in = 0

    # 在预训练词向量中的用该预训练向量
    # 不在预训练集里的用随机向量
    for ind, tok in tqdm(idx2token.items(), desc="Embedding matrix initializing..."):
        if tok in glove.dictionary:
            count_in += 1
            word_embedding_init_matrix[ind] = glove.word_vectors[glove.dictionary[tok]]

    print(
        "{:.4f} tokens are in the pretrain word embedding matrix".format(count_in / len(idx2token)))  # 命中预训练词向量的比例
    word_embedding_init_matrix = torch.FloatTensor(word_embedding_init_matrix)

    fake_inputs = torch.zeros(
        [hyper_parameters["batch_size"], max_seq_len, hyper_parameters["dec_hidden_size"]]).to(
        device)
    rel_extractor = TPLinkerBiLSTM(word_embedding_init_matrix,
                                   hyper_parameters["emb_dropout"],
                                   hyper_parameters["enc_hidden_size"],
                                   hyper_parameters["dec_hidden_size"],
                                   hyper_parameters["rnn_dropout"],
                                   len(rel2id),
                                   hyper_parameters["shaking_type"],
                                   hyper_parameters["inner_enc_type"],
                                   hyper_parameters["dist_emb_size"],
                                   hyper_parameters["ent_add_dist"],
                                   hyper_parameters["rel_add_dist"],
                                   )
    return rel_extractor, fake_inputs, data_maker


# Metrics
def bias_loss(weights=None):
    if weights is not None:
        weights = torch.FloatTensor(weights).to(device)
    cross_en = nn.CrossEntropyLoss(weight=weights)
    return lambda pred, target: cross_en(pred.view(-1, pred.size()[-1]), target.view(-1))


def set_bert(rel2id):
    tokenizer, tokenize, get_tok2char_span_map = init_bert()

    preprocessor = Preprocessor(tokenize_func=tokenize, get_tok2char_span_map_func=get_tok2char_span_map)

    train_data, valid_data, max_tok_num = load_and_trans_data(preprocessor, tokenize)
    # # Tagger (Decoder)
    max_seq_len = min(max_tok_num, hyper_parameters["max_seq_len"])

    handshaking_tagger = HandshakingTaggingScheme(rel2id=rel2id, max_seq_len=max_seq_len)
    data_maker = DataMaker4Bert(tokenizer, handshaking_tagger)

    train_dataloader = load_dataloader(data_maker, train_data, max_seq_len)
    valid_dataloader = load_dataloader(data_maker, valid_data, max_seq_len)
    rel_extractor, fake_inputs = init_bert_model(max_seq_len, rel2id)
    rel_extractor = rel_extractor.to(device)
    metrics = TPLinkerMetricsCalculator(handshaking_tagger)

    return train_dataloader, valid_dataloader, rel_extractor, metrics


def set_bilstm(rel2id):
    tokenize, get_tok2char_span_map = init_bilstm()
    preprocessor = Preprocessor(tokenize_func=tokenize, get_tok2char_span_map_func=get_tok2char_span_map)

    train_data, valid_data, max_tok_num = load_and_trans_data(preprocessor, tokenize)
    # # Tagger (Decoder)
    max_seq_len = min(max_tok_num, hyper_parameters["max_seq_len"])

    handshaking_tagger = HandshakingTaggingScheme(rel2id=rel2id, max_seq_len=max_seq_len)
    rel_extractor, fake_inputs, data_maker = init_bilstm_model(max_seq_len, rel2id, get_tok2char_span_map,
                                                               handshaking_tagger)
    rel_extractor = rel_extractor.to(device)
    train_dataloader = load_dataloader(data_maker, train_data, max_seq_len)
    valid_dataloader = load_dataloader(data_maker, valid_data, max_seq_len)
    metrics = TPLinkerMetricsCalculator(handshaking_tagger)
    return train_dataloader, valid_dataloader, rel_extractor, metrics


# train step
def train_step(batch_train_data, optimizer, loss_weights, rel_extractor, metrics, loss_func):
    if config["encoder"] == "BERT":
        sample_list, batch_input_ids, batch_attention_mask, batch_token_type_ids, tok2char_span_list, \
        batch_ent_shaking_tag, batch_head_rel_shaking_tag, batch_tail_rel_shaking_tag = batch_train_data
        batch_input_ids, batch_attention_mask, batch_token_type_ids, batch_ent_shaking_tag, batch_head_rel_shaking_tag, \
        batch_tail_rel_shaking_tag = (batch_input_ids.to(device),
                                      batch_attention_mask.to(device),
                                      batch_token_type_ids.to(device),
                                      batch_ent_shaking_tag.to(device),
                                      batch_head_rel_shaking_tag.to(device),
                                      batch_tail_rel_shaking_tag.to(device)
                                      )
        ent_shaking_outputs, head_rel_shaking_outputs, tail_rel_shaking_outputs = rel_extractor(batch_input_ids,
                                                                                                batch_attention_mask,
                                                                                                batch_token_type_ids,
                                                                                                )

    elif config["encoder"] in {"BiLSTM", }:
        sample_list, batch_input_ids, tok2char_span_list, batch_ent_shaking_tag, batch_head_rel_shaking_tag, \
        batch_tail_rel_shaking_tag = batch_train_data

        batch_input_ids, batch_ent_shaking_tag, batch_head_rel_shaking_tag, \
        batch_tail_rel_shaking_tag = (batch_input_ids.to(device),
                                      batch_ent_shaking_tag.to(device),
                                      batch_head_rel_shaking_tag.to(device),
                                      batch_tail_rel_shaking_tag.to(device)
                                      )
        ent_shaking_outputs, head_rel_shaking_outputs, tail_rel_shaking_outputs = rel_extractor(batch_input_ids)
    # zero the parameter gradients
    optimizer.zero_grad()

    w_ent, w_rel = loss_weights["ent"], loss_weights["rel"]
    loss = w_ent * loss_func(ent_shaking_outputs, batch_ent_shaking_tag) + \
           w_rel * loss_func(head_rel_shaking_outputs, batch_head_rel_shaking_tag) + \
           w_rel * loss_func(tail_rel_shaking_outputs, batch_tail_rel_shaking_tag)

    loss.backward()
    optimizer.step()

    ent_sample_acc = metrics.get_sample_accuracy(ent_shaking_outputs, batch_ent_shaking_tag)
    head_rel_sample_acc = metrics.get_sample_accuracy(head_rel_shaking_outputs, batch_head_rel_shaking_tag)
    tail_rel_sample_acc = metrics.get_sample_accuracy(tail_rel_shaking_outputs, batch_tail_rel_shaking_tag)

    return loss.item(), ent_sample_acc.item(), head_rel_sample_acc.item(), tail_rel_sample_acc.item()


# valid step
def valid_step(batch_valid_data, rel_extractor, metrics):
    if config["encoder"] == "BERT":
        sample_list, batch_input_ids, batch_attention_mask, batch_token_type_ids, tok2char_span_list, \
        batch_ent_shaking_tag, batch_head_rel_shaking_tag, batch_tail_rel_shaking_tag = batch_valid_data

        batch_input_ids, batch_attention_mask, batch_token_type_ids, batch_ent_shaking_tag, batch_head_rel_shaking_tag, \
        batch_tail_rel_shaking_tag = (batch_input_ids.to(device),
                                      batch_attention_mask.to(device),
                                      batch_token_type_ids.to(device),
                                      batch_ent_shaking_tag.to(device),
                                      batch_head_rel_shaking_tag.to(device),
                                      batch_tail_rel_shaking_tag.to(device)
                                      )

    elif config["encoder"] in {"BiLSTM", }:
        sample_list, batch_input_ids, tok2char_span_list, batch_ent_shaking_tag, batch_head_rel_shaking_tag, \
        batch_tail_rel_shaking_tag = batch_valid_data

        batch_input_ids, batch_ent_shaking_tag, batch_head_rel_shaking_tag, batch_tail_rel_shaking_tag = (
        batch_input_ids.to(device),
        batch_ent_shaking_tag.to(device),
        batch_head_rel_shaking_tag.to(device),
        batch_tail_rel_shaking_tag.to(device)
        )

    with torch.no_grad():
        if config["encoder"] == "BERT":
            ent_shaking_outputs, head_rel_shaking_outputs, tail_rel_shaking_outputs = rel_extractor(batch_input_ids,
                                                                                                    batch_attention_mask,
                                                                                                    batch_token_type_ids,
                                                                                                    )
        elif config["encoder"] in {"BiLSTM", }:
            ent_shaking_outputs, head_rel_shaking_outputs, tail_rel_shaking_outputs = rel_extractor(batch_input_ids)

    ent_sample_acc = metrics.get_sample_accuracy(ent_shaking_outputs, batch_ent_shaking_tag)
    head_rel_sample_acc = metrics.get_sample_accuracy(head_rel_shaking_outputs, batch_head_rel_shaking_tag)
    tail_rel_sample_acc = metrics.get_sample_accuracy(tail_rel_shaking_outputs, batch_tail_rel_shaking_tag)

    rel_cpg = metrics.get_rel_cpg(sample_list, tok2char_span_list, ent_shaking_outputs, head_rel_shaking_outputs,
                                  tail_rel_shaking_outputs, hyper_parameters["match_pattern"])

    return ent_sample_acc.item(), head_rel_sample_acc.item(), tail_rel_sample_acc.item(), rel_cpg


def train_n_valid(rel_extractor, rel2id, metrics, train_dataloader, valid_dataloader,
                  loss_func, num_epoch):
    # optimizer
    init_learning_rate = float(hyper_parameters["lr"])
    optimizer = torch.optim.Adam(rel_extractor.parameters(), lr=init_learning_rate)

    if hyper_parameters["scheduler"] == "CAWR":
        T_mult = hyper_parameters["T_mult"]
        rewarm_epoch_num = hyper_parameters["rewarm_epoch_num"]
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
                                                                         len(train_dataloader) * rewarm_epoch_num,
                                                                         T_mult)

    elif hyper_parameters["scheduler"] == "Step":
        decay_rate = hyper_parameters["decay_rate"]
        decay_steps = hyper_parameters["decay_steps"]
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=decay_steps, gamma=decay_rate)

    if not config["fr_scratch"]:
        model_state_path = config["model_state_dict_path"]
        rel_extractor.load_state_dict(torch.load(model_state_path))
        print("------------model state {} loaded ----------------".format(model_state_path.split("/")[-1]))

    def train(dataloader, ep):
        # train
        rel_extractor.train()

        t_ep = time.time()
        start_lr = optimizer.param_groups[0]['lr']
        total_loss, total_ent_sample_acc, total_head_rel_sample_acc, total_tail_rel_sample_acc = 0., 0., 0., 0.
        for batch_ind, batch_train_data in enumerate(dataloader):
            t_batch = time.time()
            z = (2 * len(rel2id) + 1)
            steps_per_ep = len(dataloader)
            total_steps = hyper_parameters["loss_weight_recover_steps"] + 1  # + 1 avoid division by zero error
            current_step = steps_per_ep * ep + batch_ind

            # 动态权重，随着step加大，w_ent的权重递减，w_rel权重递增。也就是开始关注实体，先保证实体抽准确，后面再越来越关注关系抽取
            w_ent = max(1 / z + 1 - current_step / total_steps, 1 / z)
            w_rel = min((len(rel2id) / z) * current_step / total_steps, (len(rel2id) / z))
            loss_weights = {"ent": w_ent, "rel": w_rel}

            loss, ent_sample_acc, head_rel_sample_acc, tail_rel_sample_acc = train_step(batch_train_data,
                                                                                        optimizer,
                                                                                        loss_weights,
                                                                                        rel_extractor,
                                                                                        metrics,
                                                                                        loss_func)
            scheduler.step()

            total_loss += loss
            total_ent_sample_acc += ent_sample_acc
            total_head_rel_sample_acc += head_rel_sample_acc
            total_tail_rel_sample_acc += tail_rel_sample_acc

            avg_loss = total_loss / (batch_ind + 1)
            avg_ent_sample_acc = total_ent_sample_acc / (batch_ind + 1)
            avg_head_rel_sample_acc = total_head_rel_sample_acc / (batch_ind + 1)
            avg_tail_rel_sample_acc = total_tail_rel_sample_acc / (batch_ind + 1)

            batch_print_format = "\rproject: {}, run_name: {}, Epoch: {}/{}, batch: {}/{}, train_loss: {}, " + \
                                 "t_ent_sample_acc: {}, t_head_rel_sample_acc: {}, t_tail_rel_sample_acc: {}," + \
                                 "lr: {}, batch_time: {}, total_time: {} -------------"

            print(batch_print_format.format(experiment_name, config["run_name"],
                                            ep + 1, num_epoch,
                                            batch_ind + 1, len(dataloader),
                                            avg_loss,
                                            avg_ent_sample_acc,
                                            avg_head_rel_sample_acc,
                                            avg_tail_rel_sample_acc,
                                            optimizer.param_groups[0]['lr'],
                                            time.time() - t_batch,
                                            time.time() - t_ep,
                                            ), end="")

            if config["logger"] == "wandb" and batch_ind % hyper_parameters["log_interval"] == 0:
                logger.log({
                    "train_loss": avg_loss,
                    "train_ent_seq_acc": avg_ent_sample_acc,
                    "train_head_rel_acc": avg_head_rel_sample_acc,
                    "train_tail_rel_acc": avg_tail_rel_sample_acc,
                    "learning_rate": optimizer.param_groups[0]['lr'],
                    "time": time.time() - t_ep,
                })

        if config["logger"] != "wandb":  # only log once for training if logger is not wandb
            logger.log({
                "train_loss": avg_loss,
                "train_ent_seq_acc": avg_ent_sample_acc,
                "train_head_rel_acc": avg_head_rel_sample_acc,
                "train_tail_rel_acc": avg_tail_rel_sample_acc,
                "learning_rate": optimizer.param_groups[0]['lr'],
                "time": time.time() - t_ep,
            })

    def valid(dataloader, ep):
        # valid
        rel_extractor.eval()

        t_ep = time.time()
        total_ent_sample_acc, total_head_rel_sample_acc, total_tail_rel_sample_acc = 0., 0., 0.
        total_rel_correct_num, total_rel_pred_num, total_rel_gold_num = 0, 0, 0
        for batch_ind, batch_valid_data in enumerate(tqdm(dataloader, desc="Validating")):
            ent_sample_acc, head_rel_sample_acc, tail_rel_sample_acc, rel_cpg = valid_step(batch_valid_data,
                                                                                           rel_extractor,
                                                                                           metrics)

            total_ent_sample_acc += ent_sample_acc
            total_head_rel_sample_acc += head_rel_sample_acc
            total_tail_rel_sample_acc += tail_rel_sample_acc

            total_rel_correct_num += rel_cpg[0]
            total_rel_pred_num += rel_cpg[1]
            total_rel_gold_num += rel_cpg[2]

        avg_ent_sample_acc = total_ent_sample_acc / len(dataloader)
        avg_head_rel_sample_acc = total_head_rel_sample_acc / len(dataloader)
        avg_tail_rel_sample_acc = total_tail_rel_sample_acc / len(dataloader)

        rel_prf = metrics.get_prf_scores(total_rel_correct_num, total_rel_pred_num, total_rel_gold_num)

        log_dict = {
            "val_ent_seq_acc": avg_ent_sample_acc,
            "val_head_rel_acc": avg_head_rel_sample_acc,
            "val_tail_rel_acc": avg_tail_rel_sample_acc,
            "val_prec": rel_prf[0],
            "val_recall": rel_prf[1],
            "val_f1": rel_prf[2],
            "time": time.time() - t_ep,
        }
        logger.log(log_dict)
        pprint(log_dict)

        return rel_prf[2]

    max_f1 = 0.
    for ep in range(num_epoch):
        train(train_dataloader, ep)
        valid_f1 = valid(valid_dataloader, ep)

        if valid_f1 >= max_f1:
            max_f1 = valid_f1
            if valid_f1 > config["f1_2_save"]:  # save the best model
                modle_state_num = len(glob.glob(model_state_dict_dir + "/model_state_dict_*.pt"))
                torch.save(rel_extractor.state_dict(), os.path.join(model_state_dict_dir,
                                                                    "model_state_dict_{}.pt".format(modle_state_num)))
                # scheduler_state_num = len(glob.glob(schedule_state_dict_dir + "/scheduler_state_dict_*.pt"))
                # torch.save(scheduler.state_dict(), os.path.join(schedule_state_dict_dir,
                #                                                 "scheduler_state_dict_{}.pt".format(scheduler_state_num)))
        print("Current avf_f1: {}, Best f1: {}".format(valid_f1, max_f1))


def main():
    rel2id = json.load(open(rel2id_path, "r", encoding="utf-8"))
    loss_func = bias_loss()

    if config["encoder"] == "BERT":
        train_dataloader, valid_dataloader, rel_extractor, metrics = set_bert(rel2id)
        train_n_valid(rel_extractor, rel2id, metrics, train_dataloader, valid_dataloader, loss_func,
                      hyper_parameters["epochs"])
    elif config["encoder"] in {"BiLSTM", }:
        train_dataloader, valid_dataloader, rel_extractor, metrics = set_bilstm(rel2id)
        train_n_valid(rel_extractor, rel2id, metrics, train_dataloader, valid_dataloader, loss_func,
                      hyper_parameters["epochs"])


if __name__ == "__main__":
    main()
