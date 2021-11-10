# -*- coding: utf8 -*-
"""
======================================
    Project Name: NLP
    File Name: tokenizers
    Author: czh
    Create Date: 2021/9/28
--------------------------------------
    Change Activity: 
======================================
"""
import re
from typing import List

import stanza
from transformers import BertTokenizerFast


def tokenize(text, vocab, do_lower_case=False) -> List[str]:
    _tokens = []
    for c in text:
        if do_lower_case:
            c = c.lower()
        if c in vocab:
            _tokens.append(c)
        else:
            _tokens.append('[UNK]')
    return _tokens


class ChineseWordTokenizer:
    @staticmethod
    def tokenize(text, ent_list=None, span_list=None, rm_blanks=False):
        """
        :param text:
        :param ent_list: tokenize by entities first
        :param span_list:
        :param rm_blanks:
        :return:
        """
        boundary_ids = set()
        if ent_list is not None and len(ent_list) > 0:
            for ent in ent_list:
                for m in re.finditer(re.escape(ent), text):
                    boundary_ids.add(m.span()[0])
                    boundary_ids.add(m.span()[1])

        if span_list is not None and len(span_list) > 0:
            for sp in span_list:
                boundary_ids = boundary_ids.union(set(sp))

        if len(boundary_ids) > 0:
            split_ids = [0] + sorted(list(boundary_ids)) + [len(text)]
            segs = []
            for idx, split_id in enumerate(split_ids):
                if idx == len(split_ids) - 1:
                    break
                segs.append(text[split_id:split_ids[idx + 1]])
        else:
            segs = [text]

        word_pattern = r"[0-9]+|\[[A-Z]+\]|[a-zA-Z]+|[^0-9a-zA-Z]"
        word_list = []
        for seg in segs:
            word_list.extend(re.findall(word_pattern, seg))

        if rm_blanks:
            word_list = [w for w in word_list if re.sub(r"\s+", "", w) != ""]
        return word_list

    @staticmethod
    def get_tok2char_span_map(word_list):
        text_fr_word_list = ""
        word2char_span = []
        for word in word_list:
            char_span = [len(text_fr_word_list), len(text_fr_word_list) + len(word)]
            text_fr_word_list += word
            word2char_span.append(char_span)
        return word2char_span

    @staticmethod
    def tokenize_plus(text, ent_list=None, span_list=None):
        word_list = ChineseWordTokenizer.tokenize(text, ent_list, span_list)
        res = {
            "word_list": word_list,
            "word2char_span": ChineseWordTokenizer.get_tok2char_span_map(word_list),
        }
        return res


class BertTokenizerAlignedWithStanza(BertTokenizerFast):
    """
    why need this class?
       text: Its favored cities include Boston , Washington , Los Angeles , Seattle , San Francisco and Oakland .
       stanza tokenizer: ['It', 's', 'favored', 'cities', 'include', 'Boston', ',', 'Washington', ',', 'Los',
       'Angeles', ',', 'Seattle', ',', 'San', 'Francisco', 'and', 'Oakland', '.']
       bert tokenizer: ['Its', 'favored', 'cities', 'include', 'Boston', ',', 'Washington', ',', 'Los', 'Angeles',
       ',', 'Seattle', ',', 'San', 'Francisco', 'and', 'Oakland', '.']

       so we need to align bert tokenizer with stanza tokenizer
   """

    def __init__(self, *args, **kwargs):
        super(BertTokenizerAlignedWithStanza, self).__init__(*args, **kwargs)
        self.stanza_language = kwargs["stanza_language"]
        self.stanza_nlp = None

    def get_stanza_nlp(self):
        if self.stanza_nlp is None:
            self.stanza_nlp = stanza.Pipeline(self.stanza_language)
        return self.stanza_nlp

    def tokenize_fr_words(self, words, max_length=None, *args, **kwargs):
        text = " ".join(words)
        tokens = super(BertTokenizerAlignedWithStanza, self).tokenize(text, *args, **kwargs)

        if max_length is not None:
            if max_length > len(tokens):
                tokens.extend(["[PAD]"] * (max_length - len(tokens)))
            else:
                tokens = tokens[:max_length]
        return tokens

    def tokenize(self, text, max_length=None, *args, **kwargs):
        words_by_stanza = [word.text for sent in self.get_stanza_nlp()(text).sentences for word in sent.words]
        return self.tokenize_fr_words(words_by_stanza, max_length=max_length, *args, **kwargs)

    def encode_plus_fr_words(self, words, word2char_span, *args, **kwargs):
        text = " ".join(words)

        new_char_ids2ori_char_ids = []
        for char_sp in word2char_span:
            for char_id in range(char_sp[0], char_sp[1]):
                new_char_ids2ori_char_ids.append(char_id)
            new_char_ids2ori_char_ids.append(-1)  # whitespace = -1

        features = super(BertTokenizerAlignedWithStanza, self).encode_plus(text, *args, **kwargs)

        if "offset_mapping" in features:
            new_offset_mapping = []
            for char_span in features["offset_mapping"]:
                if char_span[1] == 0:
                    new_offset_mapping.append([0, 0])
                    continue
                char_ids = new_char_ids2ori_char_ids[char_span[0]:char_span[1]]
                new_offset_mapping.append([char_ids[0], char_ids[-1] + 1])
            features["offset_mapping"] = new_offset_mapping

        max_length = kwargs["max_length"] if "max_length" in kwargs else None

        features["subword_list"] = self.tokenize_fr_words(words, max_length=max_length)

        return features
