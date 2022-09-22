#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: czh
@email: zhihao.chen@kuwo.cn
@date: 2022/9/21 14:20
"""
import wandb
from accelerate.tracking import GeneralTracker
from accelerate.logging import get_logger
from typing import Optional

logger = get_logger(__name__)


class CustomWandbTracker(GeneralTracker):
    name = "wandb"
    requires_logging_directory = False

    def __init__(self, run_name: str, **kwargs):
        self.run_name = run_name

        self.run = wandb.init(name=self.run_name, **kwargs)
        logger.info(f"Initialized WandB project {self.run_name}")
        logger.info(
            "Make sure to log any initial configurations with `self.store_init_configuration` before training!"
        )

    @property
    def tracker(self):
        return self.run.run

    def store_init_configuration(self, values: dict):
        """
        Logs `values` as hyperparameters for the run. Should be run at the beginning of your experiment.

        Args:
            values (Dictionary `str` to `bool`, `str`, `float` or `int`):
                Values to be stored as initial hyperparameters as key-value pairs. The values need to have type `bool`,
                `str`, `float`, `int`, or `None`.
        """
        wandb.config.update(values)
        logger.info("Stored initial configuration hyperparameters to WandB")

    def log(self, values: dict, step: Optional[int], **kwargs):
        """
        Logs `values` to the current run.

        Args:
            values (Dictionary `str` to `str`, `float`, `int` or `dict` of `str` to `float`/`int`):
                Values to be logged as key-value pairs. The values need to have type `str`, `float`, `int` or `dict` of
                `str` to `float`/`int`.
            step (`int`, *optional*):
                The run step. If included, the log will be affiliated with this step.
        """
        wandb.log(values, step=step, **kwargs)

    def finish(self):
        """
        Closes `wandb` writer
        """
        self.run.finish()
        logger.info("WandB run closed")
