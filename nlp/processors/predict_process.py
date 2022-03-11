# -*- coding: utf8 -*-
"""
======================================
    Project Name: NLP
    File Name: predict_process
    Author: czh
    Create Date: 2022/2/9
--------------------------------------
    Change Activity: 
======================================
"""
# 处理预测结果，例如提取实体和关系

from typing import Dict, List

import torch
import numpy as np


def global_pointer_entity_extract(pred_logits: torch.Tensor,
                                  id2entity: Dict[int, str],
                                  entity_type_names: dict) -> List[List[dict]]:
    batch_size = pred_logits.size(0)
    pred_logits = pred_logits.cpu().numpy()

    pred_list = [[] for i in range(batch_size)]
    for bs, label_id, start, end in zip(*np.where(pred_logits > 0)):
        label = id2entity[label_id]
        label_name = entity_type_names[label]
        res = {'label': label, 'label_name': label_name, 'start': start, 'end': end}
        pred_list[bs].append(res)

    return pred_list
