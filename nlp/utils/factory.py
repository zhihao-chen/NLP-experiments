#!/usr/bin/env python  
# -*- coding:utf-8 _*- 

from dataclasses import asdict
from functools import partial
from typing import Callable, List, Optional

from pydantic import dataclasses


@dataclasses.dataclass
class BaseClass:

    def as_dict(self):
        return asdict(self)


@dataclasses.dataclass
class GoldEntity(BaseClass):
    start_index: int
    end_index: int


@dataclasses.dataclass
class PredEntity(GoldEntity):
    start_prob: float
    end_prob: float


@dataclasses.dataclass
class PredRelation(BaseClass):
    rel: int
    rel_prob: float


@dataclasses.dataclass
class PredTuple(BaseClass):
    rel: int
    rel_prob: float
    ents: List[Optional[PredEntity]]


@dataclasses.dataclass
class GoldTuple(BaseClass):
    rel: int
    ents: List[GoldEntity]


class PartialWrapper:
    """partial的类别封装，主要是将kwarg作为class的属性来处理"""

    def __init__(self, func: Callable, *args, **kwargs):
        assert isinstance(func, Callable)
        self.func = partial(func, *args, **kwargs)
        for k, v in kwargs.items():
            self.__setattr__(k, v)

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)


class PydanticConfig:
    arbitrary_types_allowed = True
