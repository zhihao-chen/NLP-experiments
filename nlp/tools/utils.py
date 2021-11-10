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
import re
from glob import glob

import torch
import tensorflow as tf


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
        m = re.match("^(.*):\\d+$", name)
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
        return (char_span[0] - 1 >= 0 and re.match(pat, text[char_span[0] - 1]) is not None
                and re.match("^{}+".format(pat), ent) is not None) or \
               (char_span[1] < len(text) and re.match(pat, text[char_span[1]]) is not None
                and re.match("{}+$".format(pat), ent) is not None)
    return check_invalid(r"\d") or check_invalid("[A-Za-z]")
