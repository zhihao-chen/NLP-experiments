# -*- coding: utf8 -*-
"""
======================================
    Project Name: NLP
    File Name: errors
    Author: czh
    Create Date: 2022/2/9
--------------------------------------
    Change Activity: 
======================================
"""


class ParseSpanError(Exception):
    pass


class ParseEntityOffsetMappingError(ParseSpanError):
    pass


class EntityNumNotMatchError(ParseSpanError):
    pass
