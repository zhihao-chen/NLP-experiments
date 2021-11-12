# -*- coding: utf8 -*-
"""
======================================
    Project Name: NLP
    File Name: child_tuning_fisher
    Author: czh
    Create Date: 2021/11/11
--------------------------------------
    Change Activity: 
======================================
"""
# 任务相关算法Child-Tuning-D中涉及的Fisher Information Matrix
# https://github.com/alibaba/AliceMind/tree/main/ChildTuning

"""
用法：
from nlp.callback.optimizers.child_tuning_optimizer import ChildTuningAdamW

mode = "ChildTuning-D"  # or "ChildTuning-F"
adam_beta1=0.9
adam_beta2=0.999
adam_epsilon=1e-8

no_decay = ["bias", "LayerNorm.weight"]
optimizer_grouped_parameters = [
    {
        "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
        "weight_decay": self.args.weight_decay,
    },
    {
        "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
        "weight_decay": 0.0,
    },
]
optimizer_cls = ChildTuningAdamW
optimizer_kwargs = {"betas": (adam_beta1, adam_beta2), "eps": adam_epsilon, "lr": learning_rate}
optimizer = optimizer_cls(optimizer_grouped_parameters, mode=mode, **optimizer_kwargs)

if mode == 'ChildTuning-D':
    gradient_mask = calculate_fisher(model, train_dataloader, reserve_p)
    optimizer.set_gradient_mask(gradient_mask)
elif mode == "ChildTuning-F":
    optimizer = optimizer
"""
from tqdm import tqdm

import torch
import numpy as np


def calculate_fisher(model, train_dataloader, reserve_p=1.0, max_grad_norm=1.0, loss_func=None):
    """
    Calculate Fisher Information for different parameters
    遍历训练数据计算Fisher，得到child Network
    :param model:
    :param train_dataloader:
    :param reserve_p:
    :param max_grad_norm:
    :param loss_func: 如果模型输出的是logits，则传入loss_func；如果输出的是loss，则不传loss_func。
    模型输出应该是{"loss":None, "logits": None}或者单个的loss或logits
    :return:
    """
    gradient_mask = dict()
    model.train()

    for name, params in model.named_parameters():
        if 'layer' in name:
            gradient_mask[params] = params.new_zeros(params.size())

    n_l = len(train_dataloader)

    for batch in tqdm(train_dataloader):
        # inputs的参数必须与你定义的模型输入参数一致
        # inputs: {"input_ids":None, "attention_mask": None, "token_type_ids": None, "labels"：None}
        batch = tuple(t.to(model.device) for t in batch)
        inputs = {"input_ids": batch[0], "attention_mask": batch[1],
                  "token_type_ids": batch[2], "labels": batch[3]}
        outputs = model(**inputs)
        if loss_func:
            logits = outputs["logits"] if isinstance(outputs, dict) else outputs
            loss = loss_func(logits)
        else:
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs
        loss.backward()

        for name, params in model.named_parameters():
            if 'layer' in name:
                torch.nn.utils.clip_grad_norm_(params, max_grad_norm)
                gradient_mask[params] += (params.grad ** 2) / n_l
        model.zero_grad()

    print('Calculate Fisher Information')

    # Numpy
    r = None
    for k, v in gradient_mask.items():
        v = v.view(-1).cpu().numpy()
        if r is None:
            r = v
        else:
            r = np.append(r, v)
    polar = np.percentile(r, (1 - reserve_p) * 100)
    for k in gradient_mask:
        gradient_mask[k] = gradient_mask[k] >= polar
    print('Polar => {}'.format(polar))

    # TODO: pytorch: torch.kthvalue

    return gradient_mask

