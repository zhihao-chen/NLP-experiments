#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: czh
@email: zhihao.chen@kuwo.cn
@date: 2022/6/30 10:56
"""
import pickle
import numpy as np


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
