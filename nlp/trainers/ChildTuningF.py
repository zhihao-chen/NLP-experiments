# -*- coding: utf8 -*-
"""
======================================
    Project Name: NLP
    File Name: ChildTuningF
    Author: czh
    Create Date: 2021/11/11
--------------------------------------
    Change Activity: 
======================================
"""
# https://github.com/alibaba/AliceMind/tree/main/ChildTuning
from transformers import Trainer
from transformers.optimization import get_scheduler

from nlp.callback.optimizers.child_tuning_optimizer import ChildTuningAdamW


class ChildTuningFTrainer(Trainer):
    def __init__(self, **kwargs):
        self.reserve_p = kwargs.pop('reserve_p')
        self.mode = kwargs.pop('mode')
        super().__init__(**kwargs)

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        """
        Setup the optimizer and the learning rate scheduler.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through :obj:`optimizers`, or subclass and override this method in a subclass.
        """
        if self.optimizer is None:
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
            optimizer_kwargs = {"betas": (self.args.adam_beta1, self.args.adam_beta2), "eps": self.args.adam_epsilon,
                                "lr": self.args.learning_rate}
            self.optimizer = optimizer_cls(optimizer_grouped_parameters, reserve_p=self.reserve_p,   # noqa
                                           mode=self.mode, **optimizer_kwargs)

        if self.lr_scheduler is None:
            self.lr_scheduler = get_scheduler(  # noqa
                self.args.lr_scheduler_type,
                self.optimizer,
                num_warmup_steps=self.args.warmup_steps,
                num_training_steps=num_training_steps,
            )
