#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: czh
@email: zhihao.chen@kuwo.cn
@date: 2022/6/29 15:15
"""
# bert-whitening 实验，无监督语义匹配
# 参考https://github.com/zhoujx4/NLP-Series-sentence-embeddings
import os
import sys
import pickle
from tqdm import tqdm

import torch
import numpy as np
import scipy.stats
import torch.nn.functional as nnf
from transformers import BertModel, BertTokenizer
sys.path.append('/data2/work2/chenzhihao/NLP')

from nlp.processors.semantic_match_preprocessor import load_data


def get_embedding(output, pooling_strategy='cls'):
    hidden_states = output.hidden_states
    if pooling_strategy == 'cls':
        output_hidden_state = output.last_hidden_state[:, 0, :]
    elif pooling_strategy == 'last_avg':
        output_hidden_state = output.last_hidden_state.mean(dim=1)
    elif pooling_strategy == 'first_last_avg':
        output_hidden_state = hidden_states[-1] + hidden_states[1]
        output_hidden_state = output_hidden_state.mean(dim=1)
    elif pooling_strategy == 'last2avg':
        output_hidden_state = hidden_states[-1] + hidden_states[-2]
        output_hidden_state = output_hidden_state.mean(dim=1)
    else:
        raise ValueError("'pooling_strategy' must one of [fist-last-avg, last-avg, last2avg, cls]")
    vec = output_hidden_state.cpu().numpy()[0]
    return vec


def sent_to_vec(sent, tokenizer, model, pooling_strategy, max_seq_length, device):
    inputs = tokenizer(sent, return_tensors="pt", padding=True, truncation=True, max_length=max_seq_length)
    inputs['input_ids'] = inputs['input_ids'].to(device)
    inputs['token_type_ids'] = inputs['token_type_ids'].to(device)
    inputs['attention_mask'] = inputs['attention_mask'].to(device)

    with torch.no_grad():
        output = model(**inputs, return_dict=True, output_hidden_states=True)
    vec = get_embedding(output, pooling_strategy)
    return vec


def sents_to_vecs(sent_lst, tokenizer, model, pooling_strategy, max_seq_length, device):
    vec_list = []
    for sent in tqdm(sent_lst, desc="sentence to vector"):
        vec = sent_to_vec(sent, tokenizer, model, pooling_strategy, max_seq_length, device)
        vec_list.append(vec)
    assert len(vec_list) == len(sent_lst)
    vectors = np.asarray(vec_list)
    return vectors


def compute_kernel_bias(vecs):
    """计算kernel和bias
    最后的变换：y = (x + bias).dot(kernel)
    """
    vecs = np.concatenate(vecs, axis=0)
    mu = vecs.mean(axis=0, keepdims=True)
    cov = np.cov(vecs.T)
    u, s, vh = np.linalg.svd(cov)
    w = np.dot(u, np.diag(s ** 0.5))
    w = np.linalg.inv(w.T)
    return w, -mu


def normalize(vecs):
    """标准化
    """
    return vecs / (vecs ** 2).sum(axis=1, keepdims=True) ** 0.5


def transform_and_normalize(vecs, kernel, bias, dim):
    """应用变换，然后标准化
    """
    if not (kernel is None or bias is None):
        vecs = (vecs + bias).dot(kernel[:, :dim])
    return normalize(vecs)


def save_whiten(weight_save_path, kernel, bias):
    whiten = {
        'kernel': kernel,
        'bias': bias
    }
    with open(weight_save_path, 'wb') as f:
        pickle.dump(whiten, f)


def load_whiten(weight_save_path):
    with open(weight_save_path, 'rb') as f:
        whiten = pickle.load(f)
    kernel = whiten['kernel']
    bias = whiten['bias']
    return kernel, bias


def train(train_datas, model, tokenizer, config):
    sent_list = [x[0] for x in train_datas] + [x[1] for x in train_datas]
    print("Transfer sentences to BERT embedding vectors.")
    vectors_train = sents_to_vecs(sent_list, tokenizer, model,
                                  config['pooling_strategy'], config['max_seq_length'], config['device'])
    print("vector_trains shape: ", vectors_train.shape)
    print("Compute kernel and bias.")
    kernel, bias = compute_kernel_bias([vectors_train])
    save_whiten(config['model_save_path'], kernel, bias)


def test(test_datas, model, tokenizer, config):
    label_list = [int(x[2]) for x in test_datas]
    label_list = np.array(label_list)

    sent1_embeddings, sent2_embeddings = [], []
    for sent in tqdm(test_datas, total=len(test_datas), desc="get sentence embeddings!"):
        vec = sent_to_vec(sent[0], tokenizer, model, config['pooling_strategy'],
                          config['max_seq_length'], config['device'])
        sent1_embeddings.append(vec)
        vec = sent_to_vec(sent[1], tokenizer, model, config['pooling_strategy'],
                          config['max_seq_length'], config['device'])
        sent2_embeddings.append(vec)
    print(f"load kernel and bias from '{config['model_save_path']}'")
    kernel, bias = load_whiten(config['model_save_path'])

    target_embeddings = np.vstack(sent1_embeddings)
    target_embeddings = transform_and_normalize(target_embeddings, kernel, bias, config['dim'])  # whitening
    source_embeddings = np.vstack(sent2_embeddings)
    source_embeddings = transform_and_normalize(source_embeddings, kernel, bias, config['dim'])  # whitening

    similarity_list = nnf.cosine_similarity(torch.Tensor(target_embeddings),
                                            torch.tensor(source_embeddings))
    similarity_list = similarity_list.cpu().numpy()
    corrcoef = scipy.stats.spearmanr(label_list, similarity_list).correlation
    return corrcoef


def main():
    root_path = "/data2/work2/chenzhihao/NLP"
    config = {
        'model_type': "roberta",
        'model_name': "/data2/work2/chenzhihao/NLP/pretrained_models/chinese-roberta-wwm-ext",
        'output_dir': root_path + f"/experiments/output_file_dir/semantic_match",
        'max_seq_length': 64,
        'dim': 768,
        'data_type': "STS-B",
        'train_dataset': "STS-B.train.data",
        'valid_dataset': "STS-B.valid.data",
        'test_dataset': "STS-B.test.data",
        'pooling_strategy': 'first_last_avg',
        'cuda_number': 1
    }
    data_dir = "/data2/work2/chenzhihao/datasets/chinese-semantics-match-dataset/" + config['data_type']
    if not os.path.exists(data_dir):
        raise ValueError(f"The path of '{data_dir}' not exist")
    output_dir = config['output_dir'] + f"/{config['data_type']}-bert-whitening-{config['model_type']}"
    model_save_path = output_dir + f"/{config['pooling_strategy']}-whiten.pkl"
    config['model_save_path'] = model_save_path
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    device = torch.device(f"cuda:{config['cuda_number']}") if torch.cuda.is_available() else torch.device("cpu")
    config['device'] = device

    print("******* Init Bert model and tokenizer ******")
    model = BertModel.from_pretrained(config['model_name']).to(device)
    tokenizer = BertTokenizer.from_pretrained(config['model_name'])

    print("****** Loading datasets ******")
    train_data = load_data(os.path.join(data_dir, config['train_dataset']))
    valid_data = load_data(os.path.join(data_dir, config['valid_dataset']))
    test_data = load_data(os.path.join(data_dir, config['test_dataset']))

    print("****** Training ******")
    train(train_data, model, tokenizer, config)

    print("****** Testing on valid datasets ******")
    corr_coef = test(valid_data, model, tokenizer, config)
    print("valid corrcoef: {}".format(corr_coef))

    print("****** Testing on test datasets ******")
    corr_coef = test(test_data, model, tokenizer, config)
    print("test corrcoef: {}".format(corr_coef))


if __name__ == "__main__":
    main()
