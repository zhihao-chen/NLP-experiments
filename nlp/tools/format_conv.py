# -*- coding: utf8 -*-
"""
======================================
    Project Name: NLP
    File Name: format_conv
    Author: czh
    Create Date: 2021/9/29
--------------------------------------
    Change Activity: 
======================================
"""
import copy

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer


def jaccard_similarity(s1, s2):
    def add_space(s):
        return ' '.join(list(s))

    # 将字中间加入空格
    s1, s2 = add_space(s1), add_space(s2)
    # 转化为TF矩阵
    cv = CountVectorizer(tokenizer=lambda s: s.split())
    corpus = [s1, s2]
    vectors = cv.fit_transform(corpus).toarray()
    # 获取词表内容
    # ret = cv.get_feature_names()
    # print(ret)
    # 求交集
    numerator = np.sum(np.min(vectors, axis=0))
    # 求并集
    denominator = np.sum(np.max(vectors, axis=0))
    # 计算杰卡德系数
    return 1.0 * numerator / denominator


def drop_duplicate_event(event_so_list, sim_entity):
    event_so_list_copy = copy.deepcopy(event_so_list)
    for i in range(len(event_so_list_copy)):

        for j in range(i + 1, len(event_so_list_copy)):
            is_duplicate = True

            s_i_role_set = set()
            for event_s_i, event_o_i in event_so_list_copy[i].items():
                s_i_role_set.add(event_s_i)
            s_j_role_set = set()
            # 如果两个事件没有相同role，就合并，如果有相同role，判断role对应的value是否有重复的，有重复的就合并。
            # TODO 实体里也有些，可以加入
            for event_s_j, event_o_j in event_so_list_copy[j].items():
                s_j_role_set.add(event_s_j)
                event_so_i_value = event_so_list_copy[i].get(event_s_j, [])
                if event_so_i_value:  # 判断某个相同role的value是否有重复的
                    has_duplicate_value = False
                    for entity_i in event_so_i_value:
                        for entity_j in event_o_j:
                            if (entity_i in entity_j) or \
                                    (entity_j in entity_i) or \
                                    jaccard_similarity(entity_i, entity_j) > sim_entity or \
                                    set(entity_i).issubset(entity_j) or \
                                    set(entity_j).issubset(entity_i):
                                has_duplicate_value = True
                                break
                        if has_duplicate_value:
                            break
                    if not has_duplicate_value:
                        is_duplicate = False
                        break
            if is_duplicate:  # 两个结果进行合并
                for event_s_i, event_o_i in event_so_list_copy[i].items():
                    if event_s_i not in event_so_list[j]:
                        event_so_list[j].update({event_s_i: event_o_i})
                    else:
                        for event_o in event_o_i:
                            if event_o not in event_so_list[j][event_s_i]:
                                event_so_list[j][event_s_i].append(event_o)

                event_so_list.remove(event_so_list_copy[i])
                drop_duplicate_event(event_so_list, sim_entity)
                return
            else:
                continue
    return


def merge_events4doc_ee(event_list, sim_entity=0.6):
    new_event_list = []
    event_type_dict = {}
    for event_dict in event_list:
        new_argument_dict = {}
        for argument_dict in event_dict["argument_list"]:
            new_argument_dict.setdefault(argument_dict["type"], []).append(argument_dict["text"])

        event_type_dict.setdefault(event_dict["event_type"], []).append(new_argument_dict)

    for event_type, event_so_list in event_type_dict.items():
        # event_so_list = sorted(event_so_list, key=lambda x: len(x), reverse=False)
        drop_duplicate_event(event_so_list, sim_entity)
        for event_so_dict in event_so_list:
            event_dict = {"event_type": event_type, "argument_list": []}
            for role, arg_list in event_so_dict.items():
                new_arg_dict = {}
                arg_list_sort = sorted(arg_list, key=lambda x: len(x), reverse=True)
                for index, argument in enumerate(arg_list_sort):
                    add_new_arg = True
                    new_arg_dict_copy = copy.deepcopy(new_arg_dict)
                    for new_arg, new_arg_value in new_arg_dict_copy.items():
                        new_arg_value_copy = copy.deepcopy(new_arg_value)
                        new_arg_value_copy.append(new_arg)
                        break_circle = False
                        for temp_arg in new_arg_value_copy:
                            condition = set(argument).issubset(set(temp_arg)) or set(temp_arg).issubset(
                                set(argument))
                            if condition:
                                break_circle = True
                                if len(argument) > len(new_arg):
                                    new_arg_dict[argument] = new_arg_value_copy
                                    new_arg_dict.pop(new_arg)
                                    add_new_arg = False
                                    break
                                else:
                                    new_arg_dict.setdefault(new_arg, []).append(argument)
                                    add_new_arg = False
                                    break

                        if break_circle:
                            break
                    if add_new_arg:
                        new_arg_dict.setdefault(argument, [])
                for new_arg, new_arg_value in new_arg_dict.items():
                    argument_dict = {"type": role, "text": new_arg}
                    if argument_dict not in event_dict["argument_list"]:
                        event_dict["argument_list"].append(argument_dict)
            new_event_list.append(event_dict)
    return new_event_list
