#!/usr/bin/env python  
# -*- coding:utf-8 _*- 
from enum import Enum


class CaseNotSensitiveEnum(Enum):
    """大小写不敏感"""

    @classmethod
    def _missing_(cls, name):
        for member in cls:
            if member.name.lower() == name.lower():
                return member

    @classmethod
    def choices(cls):
        return [k.value for k in list(cls)]


class RunMode(Enum):
    """
    因为crf的计算loss和infer的逻辑是分开的，为了保证有些操作能够以更优的方式进行：
        1.  训练时，只关注loss，train模式即可
        2.  validation阶段需要打出val_loss和预测的metrics，所以会俩部分都需要。采用eval模式
        3.  模型训练好使用时，其实只需要预测，不需要loss计算，采用infer
    """
    TRAIN = "train"
    INFER = "infer"
    EVAL = "eval"


class DataType(Enum):
    TRAIN = "train"
    EVAL = "dev"
    TEST = "test"


class OptimizerEnum(CaseNotSensitiveEnum):
    AdamW = "AdamW"
    LAMB = "LAMB"
    Adafactor = "Adafactor"
    Adam = "Adam"


class FP16OptLevel(CaseNotSensitiveEnum):
    O1 = "O1"
    O2 = "O2"
    O3 = "O3"
    O4 = "O4"


class MatcherType(CaseNotSensitiveEnum):
    AVG = "avg"
    MIN = "min"
