# -*- coding: utf8 -*-
"""
======================================
    Project Name: news_summary
    File Name: utils
    Author: czh
    Create Date: 2021/6/28
--------------------------------------
    Change Activity: 
======================================
"""
import collections
import os
import regex
import unicodedata
from glob import glob

import numpy as np

try:
    import torch
except:
    try:
        import tensorflow as tf
    except Exception as e:
        raise ImportError("no module named torch and tensorflow")


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def check_args(args):
    args.setting_file = os.path.join(args.checkpoint_dir, args.setting_file)
    args.log_file = os.path.join(args.checkpoint_dir, args.log_file)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    with open(args.setting_file, 'wt') as opt_file:
        opt_file.write('------------ Options -------------\n')
        print('------------ Options -------------')
        for k in args.__dict__:
            v = args.__dict__[k]
            opt_file.write('%s: %s\n' % (str(k), str(v)))
            print('%s: %s' % (str(k), str(v)))
        opt_file.write('-------------- End ----------------\n')
        print('------------ End -------------')

    return args


def torch_show_all_params(model, rank=0):
    params = list(model.parameters())
    k = 0
    for i in params:
        t = 1
        for j in i.size():
            t *= j
        k = k + t
    if rank == 0:
        print("Total param num：" + str(k))


# import ipdb
def get_assigment_map_from_checkpoint(tvars, init_checkpoint):
    """Compute the union of the current variables and checkpoint variables."""
    initialized_variable_names = {}
    new_variable_names = set()
    unused_variable_names = set()

    name_to_variable = collections.OrderedDict()
    for var in tvars:
        name = var.name
        m = regex.match("^(.*):\\d+$", name)
        if m is not None:
            name = m.group(1)
        name_to_variable[name] = var

    init_vars = tf.train.list_variables(init_checkpoint)

    assignment_map = collections.OrderedDict()
    for x in init_vars:
        (name, var) = (x[0], x[1])
        if name not in name_to_variable:
            if 'adam' not in name:
                unused_variable_names.add(name)
            continue
        # assignment_map[name] = name
        assignment_map[name] = name_to_variable[name]
        initialized_variable_names[name] = 1
        initialized_variable_names[name + ":0"] = 1

    for name in name_to_variable:
        if name not in initialized_variable_names:
            new_variable_names.add(name)
    return assignment_map, initialized_variable_names, new_variable_names, unused_variable_names


# loading weights
def init_from_checkpoint(init_checkpoint, tvars=None, rank=0):
    if not tvars:
        tvars = tf.compat.v1.trainable_variables()
    assignment_map, initialized_variable_names, new_variable_names, unused_variable_names \
        = get_assigment_map_from_checkpoint(tvars, init_checkpoint)
    tf.compat.v1.train.init_from_checkpoint(init_checkpoint, assignment_map)
    if rank == 0:
        # 显示成功加载的权重
        for t in initialized_variable_names:
            if ":0" not in t:
                print("Loading weights success: " + t)

        # 显示新的参数
        print('New parameters:', new_variable_names)

        # 显示初始化参数中没用到的参数
        print('Unused parameters', unused_variable_names)


def torch_init_model(model, init_checkpoint, delete_module=False):
    state_dict = torch.load(init_checkpoint, map_location='cpu')
    state_dict_new = {}
    # delete module.
    if delete_module:
        for key in state_dict.keys():
            v = state_dict[key]
            state_dict_new[key.replace('module.', '')] = v
        state_dict = state_dict_new
    missing_keys = []
    unexpected_keys = []
    error_msgs = []
    # copy state_dict so _load_from_state_dict can modify it
    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, prefix=''):
        local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})

        module._load_from_state_dict(
            state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')

    load(model, prefix='' if hasattr(model, 'bert') else 'bert.')

    print("missing keys:{}".format(missing_keys))
    print('unexpected keys:{}'.format(unexpected_keys))
    print('error msgs:{}'.format(error_msgs))


def torch_save_model(model, output_dir, scores, max_save_num=1):
    # Save model checkpoint
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
    saved_pths = glob(os.path.join(output_dir, '*.pth'))
    saved_pths.sort()
    while len(saved_pths) >= max_save_num:
        if os.path.exists(saved_pths[0].replace('//', '/')):
            os.remove(saved_pths[0].replace('//', '/'))
            del saved_pths[0]

    save_prex = "checkpoint_score"
    for k in scores:
        save_prex += ('_' + k + '-' + str(scores[k])[:6])
    save_prex += '.pth'

    torch.save(model_to_save.state_dict(),
               os.path.join(output_dir, save_prex))
    print("Saving model checkpoint to %s", output_dir)


def get_char2tok_span(tok2char_span):
    """

    get a map from character level index to token level span
    e.g. "She is singing" -> [
                             [0, 1], [0, 1], [0, 1], # She
                             [-1, -1] # whitespace
                             [1, 2], [1, 2], # is
                             [-1, -1] # whitespace
                             [2, 3], [2, 3], [2, 3], [2, 3], [2, 3], [2, 3], [2, 3] # singing
                             ]

     tok2char_span： a map from token index to character level span
    """

    # get the number of characters
    char_num = None
    for tok_ind in range(len(tok2char_span) - 1, -1, -1):
        if tok2char_span[tok_ind][1] != 0:
            char_num = tok2char_span[tok_ind][1]
            break

    # build a map: char index to token level span
    char2tok_span = [[-1, -1] for _ in range(char_num)]  # 除了空格，其他字符均有对应token
    for tok_ind, char_sp in enumerate(tok2char_span):
        for char_ind in range(char_sp[0], char_sp[1]):
            tok_sp = char2tok_span[char_ind]
            # 因为在bert中，char to tok 也可能出现1对多的情况，比如韩文。
            # 所以char_span的pos1以第一个tok_ind为准，pos2以最后一个tok_ind为准
            if tok_sp[0] == -1:  # 第一次赋值以后不再修改
                tok_sp[0] = tok_ind
            tok_sp[1] = tok_ind + 1  # 每一次都更新
    return char2tok_span


def is_invalid_extr_ent(ent, char_span, text):
    def check_invalid(pat):
        return (char_span[0] - 1 >= 0 and regex.match(pat, text[char_span[0] - 1]) is not None
                and regex.match("^{}+".format(pat), ent) is not None) or \
               (char_span[1] < len(text) and regex.match(pat, text[char_span[1]]) is not None
                and regex.match("{}+$".format(pat), ent) is not None)
    return check_invalid(r"\d") or check_invalid("[A-Za-z]")


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
            (0x2F800 <= cp <= 0x2FA1F)) and not is_invalid_chinese(char):  #
        return True

    return False


def is_all_alpha(word):
    pattern = r"[a-zA-Z]"
    temp = [not regex.search(pattern, w) for w in word]
    return not any(temp)


def is_invalid_chinese(text: str):
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


def is_all_punctuation(word):
    found = True
    for w in word:
        if not _is_punctuation(w):
            found = False
            break
    return found


def is_not_effective_char(char):
    found = True
    if _is_punctuation(char) or is_chinese_char(char) or is_all_alpha(char) or _is_whitespace(char) or char.isdigit():
        found = False
    return found


# 是否为乱码文本，通过乱码阈值控制
def is_garbled_text(text, threshold=0.2):
    is_garbled = False

    invalid_char_num = 0
    text_len = len(text)
    for char in text:
        try:
            if not is_valid_char(char):
                invalid_char_num += 1

        except:  # noqa
            invalid_char_num += 1

        finally:
            if invalid_char_num / text_len >= threshold:
                is_garbled = True
                break

    return is_garbled


def is_number_char(uchar):
    """判断一个unicode是否是数字"""
    if u'\u0030' <= uchar <= u'\u0039':
        return True
    else:
        return False


def is_alphabet_char(uchar):
    """判断一个unicode是否是英文字母"""
    if (u'\u0041' <= uchar <= u'\u005a') or (u'\u0061' <= uchar <= u'\u007a'):
        return True
    else:
        return False


# 不含乱码
def is_valid_char(char):
    try:
        import string
        # 专门添加的解析表格的特殊字符
        other_char = "⊿◣▽▲【】，。、（）-\n\t " + string.punctuation

        if is_chinese_char(char) \
                or is_alphabet_char(char) \
                or is_number_char(char) \
                or char in other_char:

            return True
        else:
            return False
    except:  # noqa
        return False


def find_head_idx(source, target):
    """
    在target字符串中找寻source字符串，返回source字符串在target中的开始索引位置
    :param source:
    :param target:
    :return:
    """
    target_len = len(target)
    for i in range(len(source)):
        if source[i: i + target_len] == target:
            return i
    return -1


def split_short_text(sentence, max_seq_len=510):
    """
    将长文本切分为短文本
    :param sentence:
    :param max_seq_len:
    :return:
    """
    sentences = [line.strip() for line in regex.split(r'[。！？!?]', sentence) if line.strip()]

    lines = []
    text = ""
    if len(sentences) > 1:
        for sent in sentences:
            if not text:
                text = sent
                continue
            if len(text + '。' + sent) <= max_seq_len:
                text = text + '。' + sent
            else:
                lines.append(text)
                text = ""
        if text:
            lines.append(text)
    else:
        for i in range(0, len(sentence), 2):
            if i + max_seq_len < len(sentence):
                text = sentence[i: i+max_seq_len]
                assert len(text) <= max_seq_len
                lines.append(text)
            else:
                break
    return lines


def sequence_padding(inputs, length=None, value=0, seq_dims=1, mode='post'):
    """Numpy函数，将序列padding到同一长度
    """
    if length is None:
        length = np.max([np.shape(x)[:seq_dims] for x in inputs], axis=0)
    elif not hasattr(length, '__getitem__'):
        length = [length]

    slices = [np.s_[:length[i]] for i in range(seq_dims)]
    slices = tuple(slices) if len(slices) > 1 else slices[0]
    pad_width = [(0, 0) for _ in np.shape(inputs[0])]

    outputs = []
    for x in inputs:
        x = x[slices]
        for i in range(seq_dims):
            if mode == 'post':
                pad_width[i] = (0, length[i] - np.shape(x)[i])
            elif mode == 'pre':
                pad_width[i] = (length[i] - np.shape(x)[i], 0)
            else:
                raise ValueError('"mode" argument must be "post" or "pre".')
        x = np.pad(x, pad_width, 'constant', constant_values=value)
        outputs.append(x)

    return np.array(outputs)
