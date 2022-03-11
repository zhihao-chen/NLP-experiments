# -*- coding: utf8 -*-
"""
======================================
    Project Name: NLP
    File Name: preprocess
    Author: czh
    Create Date: 2021/11/9
--------------------------------------
    Change Activity: 
======================================
"""
import regex
import unicodedata


def is_chinese_char(char):
    """Checks whether CP is the codepoint of a CJK character."""
    cp = ord(char)
    if ((0x4E00 <= cp <= 0x9FFF) or  #
            (0x3400 <= cp <= 0x4DBF) or  #
            (0x20000 <= cp <= 0x2A6DF) or  #
            (0x2A700 <= cp <= 0x2B73F) or  #
            (0x2B740 <= cp <= 0x2B81F) or  #
            (0x2B820 <= cp <= 0x2CEAF) or
            (0xF900 <= cp <= 0xFAFF) or  #
            (0x2F800 <= cp <= 0x2FA1F)):  #
        return True

    return False


def is_all_alpha(word):
    pattern = r"[a-zA-Z]"
    temp = [not regex.search(pattern, w) for w in word]
    return not any(temp)


def is_not_chinese(sentence):
    """
    是否没有中文，只有英文字符、数字和标点符号
    :param sentence:
    :return:
    """
    pattern = r"[\u4e00-\u9fa5]+"
    res = regex.search(pattern, sentence)
    if res:
        return False
    else:
        return True


def _is_whitespace(char):
    """Checks whether `char` is a whitespace character."""
    # \t, \n, and \r are technically control characters but we treat them
    # as whitespace since they are generally considered as such.
    if char == " " or char == "\t" or char == "\n" or char == "\r":
        return True
    cat = unicodedata.category(char)
    if cat == "Zs":
        return True
    return False


def _is_control(char):
    """Checks whether `char` is a control character."""
    # These are technically control characters but we count them as whitespace
    # characters.
    if char == "\t" or char == "\n" or char == "\r":
        return False
    cat = unicodedata.category(char)
    if cat.startswith("C"):
        return True
    return False


def _is_punctuation(char):
    """Checks whether `char` is a punctuation character."""
    cp = ord(char)
    # We treat all non-letter/number ASCII as punctuation.
    # Characters such as "^", "$", and "`" are not in the Unicode
    # Punctuation class but we treat them as punctuation anyways, for
    # consistency.
    if (33 <= cp <= 47) or (58 <= cp <= 64) or (91 <= cp <= 96) or (123 <= cp <= 126):
        return True
    cat = unicodedata.category(char)
    if cat.startswith("P"):
        return True
    return False


def _is_end_of_word(text):
    """Checks whether the last character in text is one of a punctuation, control or whitespace character."""
    last_char = text[-1]
    return bool(_is_control(last_char) | _is_punctuation(last_char) | _is_whitespace(last_char))


def _is_start_of_word(text):
    """Checks whether the first character in text is one of a punctuation, control or whitespace character."""
    first_char = text[0]
    return bool(_is_control(first_char) | _is_punctuation(first_char) | _is_whitespace(first_char))


def find_head_idx(source, target):
    target_len = len(target)
    for i in range(len(source)):
        if source[i: i + target_len] == target:
            return i
    return -1


class Preprocessor(object):
    def __init__(self, tokenizer, add_special_tokens=True):
        super(Preprocessor, self).__init__()
        self.tokenizer = tokenizer
        self.add_special_tokens = add_special_tokens

    def get_ent2token_spans(self, text, entity_list):
        """实体列表转为token_spans
        Args:
            text (str): 原始文本
            entity_list (list): [(start, end, ent_type),(start, end, ent_type)...]
        """
        ent2token_spans = []

        inputs = self.tokenizer(text, add_special_tokens=self.add_special_tokens, return_offsets_mapping=True)
        token2char_span_mapping = inputs["offset_mapping"]
        text2tokens = self.tokenizer.tokenize(text, add_special_tokens=self.add_special_tokens)

        for ent_span in entity_list:
            ent = text[ent_span[0]:ent_span[1] + 1]
            ent2token = self.tokenizer.tokenize(ent, add_special_tokens=False)

            # 寻找ent的token_span
            # token_start_indexs = [i for i, v in enumerate(text2tokens) if v == ent2token[0]]
            # token_end_indexs = [i for i, v in enumerate(text2tokens) if v == ent2token[-1]]
            # token_start_index = list(filter(lambda x: token2char_span_mapping[x][0] == ent_span[0], token_start_indexs))
            # # token2char_span_mapping[x][-1]-1 减1是因为原始的char_span是闭区间，而token2char_span是开区间
            # token_end_index = list(filter(lambda x: token2char_span_mapping[x][-1] - 1 == ent_span[1],
            #                               token_end_indexs))
            # if len(token_start_index) == 0 or len(token_end_index) == 0:
            #     print(f'[{ent}] 无法对应到 [{text}] 的token_span，已丢弃')
            #     continue
            # token_span = (token_start_index[0], token_end_index[0], ent_span[2])
            # ent2token_spans.append(token_span)
            token_start_index = find_head_idx(text2tokens, ent2token)
            if token_start_index != -1:
                token_end_index = token_start_index + len(ent2token)
                token_span = (token_start_index, token_end_index, ent_span[2])
                ent2token_spans.append(token_span)
            else:
                print(f'[{ent}] 无法对应到 [{text}] 的token_span，已丢弃')
                continue

        return ent2token_spans
