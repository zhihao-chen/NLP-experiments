# -*- coding: utf8 -*-
"""
======================================
    Project Name: NLP
    File Name: idcnn_for_crf
    Author: czh
    Create Date: 2022/2/22
--------------------------------------
    Change Activity: 
======================================
"""
import numpy as np
import torch
import torch.nn as nn
from nlp.layers.cnn import IDCNN
from nlp.layers.crf import CRF
from torch.nn import functional as func


class IDCNNForCRF(nn.Module):
    def __init__(self,
                 vocab_size,
                 word_embedding_dim,
                 word2id,
                 num_tag,
                 embedding_file=None,
                 dropout_rate=0.5,
                 nil=True):
        super(IDCNNForCRF, self).__init__()
        self.embedding = nn.Embedding(vocab_size, word_embedding_dim)
        self.embedding_file = embedding_file

        self.embedding.weight.data.copy_(
            torch.from_numpy(
                self.get_embedding(vocab_size,
                                   word_embedding_dim,
                                   word2id,
                                   nil)))
        self.idcnn = IDCNN(input_size=word_embedding_dim, filters=64)
        self.linear = nn.Linear(64, 256)
        self.out = nn.Linear(256, num_tag)

        self.crf = CRF(num_tags=num_tag)
        self.dropout_layer = nn.Dropout(dropout_rate)

    def forward(self, inputs, length, labels=None):
        embeddings = self.embedding(inputs)
        embeddings = self.dropout_layer(embeddings)
        out = self.idcnn(embeddings, length)
        out = self.linear(out)
        out = self.out(out)
        logits = func.dropout(out, p=0.1, training=self.training)
        output = {'logits': logits}
        if labels is not None:
            loss = -1 * self.crf(emissions=logits, tags=labels)
            output["loss"] = loss
        return output

    def parse_word_vector(self, word_index, embedding_dim):
        pre_trained_wordvector = {}
        f = open(self.embedding_file, encoding='utf-8')
        fr = f.readlines()
        for line in fr[1:]:
            lines = line.strip().split(' ')
            word = lines[0]
            if len(word) == 1:
                if word_index.get(word) is not None:
                    vector = [float(f) for f in lines[1:embedding_dim + 1]]
                    pre_trained_wordvector[word] = vector
                else:
                    continue
            else:
                continue
        return pre_trained_wordvector

    def get_embedding(self, vocab_size, embedding_dim, word2id, nil=True):
        print('Get embedding...')
        embedding_matrix = np.zeros((vocab_size, embedding_dim), dtype=np.float32)
        if not nil:
            pre_trained_wordector = self.parse_word_vector(word2id, embedding_dim)
            for word, idx in word2id.items():
                try:
                    word_vector = pre_trained_wordector[word]
                    embedding_matrix[id] = word_vector
                except:
                    continue
        print('Get embedding done!')
        return embedding_matrix
