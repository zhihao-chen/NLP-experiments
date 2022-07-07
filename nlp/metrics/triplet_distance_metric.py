#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: czh
@email: zhihao.chen@kuwo.cn
@date: 2022/6/30 11:01
"""
# The metric for the triplet loss
from torch.nn import functional as nnf


def cosin(x, y):
    return 1 - nnf.cosine_similarity(x, y)


def euclidean(x, y):
    return nnf.pairwise_distance(x, y, p=2)


def manhattan(x, y):
    return nnf.pairwise_distance(x, y, p=1)
