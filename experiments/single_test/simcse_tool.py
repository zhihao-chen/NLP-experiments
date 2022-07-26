#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: czh
@email: zhihao.chen@kuwo.cn
@date: 2022/7/25 17:25
"""
# 参考https://github.com/princeton-nlp/SimCSE/blob/main/simcse/tool.py
import os
import logging
from tqdm import tqdm
from typing import List, Tuple, Union

import numpy as np
import torch
from transformers import BertTokenizer, BertModel, BertConfig
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import models, SentenceTransformer, util

from nlp.models.sentence_embedding_models import SimCSEModel


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class SimCSE(object):
    """
        A class for embedding sentences, calculating similarities, and retriving sentences by SimCSE.
    """
    def __init__(self,
                 bert_model,
                 tokenizer=None,
                 device: str = None,
                 num_cells: int = 100,
                 num_cells_in_search: int = 10):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        self.model = bert_model

        self.tokenizer = tokenizer

        self.index = None
        self.is_faiss_index = False
        self.num_cells = num_cells
        self.num_cells_in_search = num_cells_in_search

    def encode(self, sentence: Union[str, List[str]],
               device: str = None,
               return_numpy: bool = False,
               normalize_to_unit: bool = True,
               keepdim: bool = False,
               batch_size: int = 64,
               max_length: int = 128) -> Union[np.ndarray, torch.Tensor]:
        target_device = self.device if device is None else device
        self.model = self.model.to(target_device)

        single_sentence = False
        if isinstance(sentence, str):
            sentence = [sentence]
            single_sentence = True

        embedding_list = []
        with torch.no_grad():
            total_batch = len(sentence) // batch_size + (1 if len(sentence) % batch_size > 0 else 0)
            for batch_id in tqdm(range(total_batch)):
                inputs = self.tokenizer(
                    sentence[batch_id * batch_size:(batch_id + 1) * batch_size],
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt"
                )
                inputs = {k: v.to(target_device) for k, v in inputs.items()}
                embeddings = self.model.get_sent_embed(**inputs)
                if normalize_to_unit:
                    embeddings = embeddings / embeddings.norm(dim=1, keepdim=True)
                embedding_list.append(embeddings.cpu())
        embeddings = torch.cat(embedding_list, 0)

        if single_sentence and not keepdim:
            embeddings = embeddings[0]

        if return_numpy and not isinstance(embeddings, np.ndarray):
            return embeddings.numpy()
        return embeddings

    def similarity(self, queries: Union[str, List[str]],
                   keys: Union[str, List[str], np.ndarray],
                   device: str = None) -> Union[float, np.ndarray]:

        query_vecs = self.encode(queries, device=device, return_numpy=True)  # suppose N queries

        if not isinstance(keys, np.ndarray):
            key_vecs = self.encode(keys, device=device, return_numpy=True)  # suppose M keys
        else:
            key_vecs = keys

        # check whether N == 1 or M == 1
        single_query, single_key = len(query_vecs.shape) == 1, len(key_vecs.shape) == 1
        if single_query:
            query_vecs = query_vecs.reshape(1, -1)
        if single_key:
            key_vecs = key_vecs.reshape(1, -1)

        # returns an N*M similarity array
        similarities = cosine_similarity(query_vecs, key_vecs)

        if single_query:
            similarities = similarities[0]
            if single_key:
                similarities = float(similarities[0])

        return similarities

    def build_index(self, sentences_or_file_path: Union[str, List[str]],
                    use_faiss: bool = None,
                    faiss_fast: bool = False,
                    device: str = None,
                    batch_size: int = 64):

        if use_faiss is None or use_faiss:
            try:
                import faiss
                assert hasattr(faiss, "IndexFlatIP")
                use_faiss = True
            except:
                logger.warning(
                    "Fail to import faiss. If you want to use faiss, install faiss through PyPI. "
                    "Now the program continues with brute force search.")
                use_faiss = False

        # if the input sentence is a string, we assume it's the path of file that stores various sentences
        if isinstance(sentences_or_file_path, str):
            sentences = []
            with open(sentences_or_file_path, "r") as f:
                logging.info("Loading sentences from %s ..." % sentences_or_file_path)
                for line in tqdm(f):
                    sentences.append(line.rstrip())
            sentences_or_file_path = sentences

        logger.info("Encoding embeddings for sentences...")
        embeddings = self.encode(sentences_or_file_path, device=device, batch_size=batch_size, normalize_to_unit=True,
                                 return_numpy=True)

        logger.info("Building index...")
        self.index = {"sentences": sentences_or_file_path}

        if use_faiss:
            quantizer = faiss.IndexFlatIP(embeddings.shape[1])
            if faiss_fast:
                index = faiss.IndexIVFFlat(quantizer, embeddings.shape[1],
                                           min(self.num_cells, len(sentences_or_file_path)))
            else:
                index = quantizer

            if (self.device == "cuda" and device != "cpu") or device == "cuda":
                if hasattr(faiss, "StandardGpuResources"):
                    logger.info("Use GPU-version faiss")
                    res = faiss.StandardGpuResources()
                    res.setTempMemory(20 * 1024 * 1024 * 1024)
                    index = faiss.index_cpu_to_gpu(res, 0, index)
                else:
                    logger.info("Use CPU-version faiss")
            else:
                logger.info("Use CPU-version faiss")

            if faiss_fast:
                index.train(embeddings.astype(np.float32))
            index.add(embeddings.astype(np.float32))
            index.nprobe = min(self.num_cells_in_search, len(sentences_or_file_path))
            self.is_faiss_index = True
        else:
            index = embeddings
            self.is_faiss_index = False
        self.index["index"] = index
        logger.info("Finished")

    def add_to_index(self, sentences_or_file_path: Union[str, List[str]],
                     device: str = None,
                     batch_size: int = 64):

        # if the input sentence is a string, we assume it's the path of file that stores various sentences
        if isinstance(sentences_or_file_path, str):
            sentences = []
            with open(sentences_or_file_path, "r") as f:
                logging.info("Loading sentences from %s ..." % sentences_or_file_path)
                for line in tqdm(f):
                    sentences.append(line.rstrip())
            sentences_or_file_path = sentences

        logger.info("Encoding embeddings for sentences...")
        embeddings = self.encode(sentences_or_file_path, device=device, batch_size=batch_size, normalize_to_unit=True,
                                 return_numpy=True)

        if self.is_faiss_index:
            self.index["index"].add(embeddings.astype(np.float32))
        else:
            self.index["index"] = np.concatenate((self.index["index"], embeddings))
        self.index["sentences"] += sentences_or_file_path
        logger.info("Finished")

    def search(self, queries: Union[str, List[str]],
               device: str = None,
               threshold: float = 0.6,
               top_k: int = 5) -> Union[List[Tuple[str, float]], List[List[Tuple[str, float]]]]:

        if not self.is_faiss_index:
            if isinstance(queries, list):
                combined_results = []
                for query in queries:
                    results = self.search(query, device)
                    combined_results.append(results)
                return combined_results

            similarities = self.similarity(queries, self.index["index"]).tolist()
            id_and_score = []
            for i, s in enumerate(similarities):
                if s >= threshold:
                    id_and_score.append((i, s))
            id_and_score = sorted(id_and_score, key=lambda x: x[1], reverse=True)[:top_k]
            results = [(self.index["sentences"][idx], score) for idx, score in id_and_score]
            return results
        else:
            query_vecs = self.encode(queries, device=device, normalize_to_unit=True, keepdim=True, return_numpy=True)

            distance, idx = self.index["index"].search(query_vecs.astype(np.float32), top_k)

            def pack_single_result(dist, idx):
                results = [(self.index["sentences"][i], s) for i, s in zip(idx, dist) if s >= threshold]
                return results

            if isinstance(queries, list):
                combined_results = []
                for i in range(len(queries)):
                    results = pack_single_result(distance[i], idx[i])
                    combined_results.append(results)
                return combined_results
            else:
                return pack_single_result(distance[0], idx[0])


def test_1(query_list, example_list):
    model_name_or_path = "/data2/work2/chenzhihao/NLP/experiments/output_file_dir/semantic_match/STS-B-unsupsimcse2-roberta-2022-07-06_04"
    bert_name_or_path = "/data2/work2/chenzhihao/pretrained_models/chinese-roberta-wwm-ext"
    device = "cuda:7"

    config = {
        'do_mlm': False,  # "Whether to use MLM auxiliary objective."
        'mlp_only_train': False,  # "Use MLP only during training"
        'pooling_strategy': "first-last-avg",  # first-last-avg, last-avg, cls, pooler, last2avg
        'temp': 0.05,  # "Temperature for softmax."
        'device': device
    }

    bert_config = BertConfig.from_pretrained(bert_name_or_path)
    tokenizer = BertTokenizer.from_pretrained(bert_name_or_path)
    bert_model = SimCSEModel(model_args=config, bert_config=bert_config, pooling_strategy=config['pooling_strategy'])
    bert_model.load_state_dict(torch.load(model_name_or_path + "/pytorch_model.bin", map_location=device))

    simcse = SimCSE(bert_model=bert_model, tokenizer=tokenizer, device=device)

    print("\n=========Calculate cosine similarities between queries and sentences============\n")
    similarities = simcse.similarity(query_list, example_list)
    print(similarities)

    print("\n=========Naive brute force search============\n")
    simcse.build_index(example_list, use_faiss=False)
    results = simcse.search(query_list, top_k=10)
    for i, result in enumerate(results):
        print("Retrieval results for query: {}".format(query_list[i]))
        for sentence, score in result:
            print("    {}  (cosine similarity: {:.4f})".format(sentence, score))
        print("")

    # print("\n=========Search with Faiss backend============\n")
    # simcse.build_index(example_list, use_faiss=True)
    # results = simcse.search(queries)
    # for i, result in enumerate(results):
    #     print("Retrieval results for query: {}".format(example_list[i]))
    #     for sentence, score in result:
    #         print("    {}  (cosine similarity: {:.4f})".format(sentence, score))
    #     print("")


def test2(query_list, example_list):
    root_dir = "/data2/work2/chenzhihao/NLP/experiments/"
    model_name_or_path = root_dir + "output_file_dir/semantic_match/STS-B-unsup_simcse-roberta-2022-07-01_09"
    device = "cuda:7"
    max_seq_length = 128
    word_embedding_model = models.Transformer(model_name_or_path)

    # Apply mean pooling to get one fixed sized sentence vector
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                                   pooling_mode='cls',
                                   pooling_mode_mean_tokens=False,
                                   pooling_mode_cls_token=True,
                                   pooling_mode_max_tokens=False)

    model = SentenceTransformer(modules=[word_embedding_model, pooling_model], device=device)
    model.max_seq_length = max_seq_length

    query_embeddings = model.encode(query_list, convert_to_tensor=True)
    key_embeddings = model.encode(example_list, convert_to_tensor=True)

    cosine_scores = util.cos_sim(query_embeddings, key_embeddings)

    for i, result in enumerate(cosine_scores):
        print("Retrieval results for query: {}".format(query_list[i]))
        for j, score in enumerate(result):
            print("{}   (cosine similarity: {:.4f})".format(example_list[j], score))


if __name__ == "__main__":
    keys = [
        "充值未到账",
        "买了会员不能用",
        "无法下载歌曲",
        "下载不了歌曲",
        "下载不了",
        "付费后会员未开通",
        "充值没到账",
    ]

    queries = ["充值未到账"]
    test_1(queries, keys)
    test2(queries, keys)
