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
import string
from typing import List, Optional

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


class MyTokenizer(BertTokenizerFast):
    SPE = "│"
    LETTERS = string.ascii_letters + string.punctuation + "！=·，；。？《》【】"

    def __init__(self, *args, **kwargs):
        """
        use：MyTokenizer(vocab_file=vocab_file, do_lower_case=True)
        :param args:
        :param kwargs:
        """
        super(MyTokenizer, self).__init__(*args, **kwargs)
        self.letter_dict = {c: i for i, c in enumerate(self.LETTERS)}

    def prepare_text_for_tokenize(self, text: str) -> str:
        """
        主要为了预处理文本。中文间的空格不应该直接剔除，而英文的空格保持原样，此外，替换完后，为了便于恢复原有文本（每个token一致），
        需要保证修改后的文本与原文本长度一致。
        """
        if len(text) <= 2:
            return text
        new_text = text[0]
        for i in range(1, len(text) - 1):
            c = text[i]
            if not c.strip() and not (text[i - 1] in self.letter_dict or text[i + 1] in self.letter_dict):
                new_text += self.SPE
            else:
                new_text += c
        new_text += text[-1]
        new_text = self.pre_tokenize(new_text)
        return new_text

    def pre_tokenize(self, text: str) -> str:
        tokens = []
        for c in text:
            if self.is_chinese_char(c):
                tokens.append(" ")
                tokens.append(c)
                tokens.append(" ")
            # 官方tokenizer会将数字按照subwords解析，但解析后会有offset_mapping和原始的char offset对应不上的情况
            # 例如电话号码
            elif c.isdigit():
                tokens.append(" ")
                tokens.append(c)
                tokens.append(" ")
            else:
                tokens.append(c)
        return "".join(tokens)

    def tokenize(self, text: str, pair: Optional[str] = None, add_special_tokens: bool = False, **kwargs) -> List[str]:
        new_text = self.prepare_text_for_tokenize(text)
        tokens = super(MyTokenizer, self).tokenize(new_text, pair, add_special_tokens, **kwargs)
        return tokens

    def encode_plus_for_me(self, text: str, *args, **kwargs):
        new_text = self.prepare_text_for_tokenize(text)
        features = super(MyTokenizer, self).encode_plus(new_text, *args, **kwargs)

        return features

    def is_chinese_char(self, char: str) -> bool:
        """Checks whether CP is the codepoint of a CJK character."""
        cp = ord(char)
        if (0x4E00 <= cp <= 0x9FFF or
            0x3400 <= cp <= 0x4DBF or
            0x20000 <= cp <= 0x2A6DF or
            0x2A700 <= cp <= 0x2B73F or
            0x2B740 <= cp <= 0x2B81F or
            0x2B820 <= cp <= 0x2CEAF or
            0xF900 <= cp <= 0xFAFF or
            0x2F800 <= cp <= 0x2FA1F) \
                and not self.is_invalid_chinese(char):
            return True

        return False

    @staticmethod
    def is_invalid_chinese(text: str) -> bool:
        """
        判断是不是中文乱码
        :param text:
        :return: False if not Chinese
        :return: False if is Chinese and is valid
        :return: True if is Chinese is invalid
        """
        try:
            text.encode('gb2312')
            return False
        except UnicodeEncodeError:
            return True


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
