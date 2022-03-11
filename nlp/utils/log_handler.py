# -*- coding: utf8 -*-
"""
======================================
    Project Name: contact_info_recognize
    File Name: log_handler
    Author: czh
    Create Date: 2021/11/23
--------------------------------------
    Change Activity: 
======================================
"""
import os
from typing import Union
from pathlib import Path
import logging
from logging.handlers import TimedRotatingFileHandler

import socket

from nlp.tools.path import project_root_path

# 日志级别
CRITICAL = 50
FATAL = CRITICAL
ERROR = 40
WARNING = 30
WARN = WARNING
INFO = 20
DEBUG = 10
NOTSET = 0

ROOT_PATH = project_root_path()
LOG_PATH = os.path.join(ROOT_PATH, 'logs')

if not os.path.exists(LOG_PATH):
    os.makedirs(LOG_PATH)


class LogHandler(logging.Logger):
    """
    LogHandler
    """
    def __init__(self, log_path: Union[str, Path] = LOG_PATH,
                 file_name: str = None,
                 level: Union[int, str] = DEBUG,
                 stream: bool = True,
                 file: bool = True):
        """
        :param log_path: 日志文件保存路径
        :param file_name: 日志文件名
        :param level: 日志级别，字符串或者数字
        :param stream: 是否在终端打印日志
        :param file: 是否把日志写入文件
        """
        self.host_name = socket.gethostname()
        self.log_path = log_path
        self.name = file_name
        self.level = level
        logging.Logger.__init__(self, self.name, level=level)

        if stream:
            self.__setStreamHandler__()
        if file:
            self.__setFileHandler__()

    def __setFileHandler__(self, level=None):
        """
        set file handler
        :param level:
        :return:
        """
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)
        file_name = os.path.join(self.log_path, f"{self.name}-{self.host_name}.log")

        msg_format = f"%(asctime)s %(levelname)s {self.host_name} [%(threadName)s] %(filename)s(%(lineno)d): %(message)s"
        # 设置日志回滚, 保存在log目录, 一天保存一个文件, 保留15天
        file_handler = TimedRotatingFileHandler(filename=file_name, when='midnight', backupCount=3, encoding='utf8')
        file_handler.suffix = '%Y%m%d.log'
        if not level:
            file_handler.setLevel(self.level)
        else:
            file_handler.setLevel(level)
        formatter = logging.Formatter(msg_format)

        file_handler.setFormatter(formatter)
        self.file_handler = file_handler
        self.addHandler(file_handler)

    def __setStreamHandler__(self, level=None):
        """
        set stream handler
        :param level:
        :return:
        """
        stream_handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s')
        stream_handler.setFormatter(formatter)
        if not level:
            stream_handler.setLevel(self.level)
        else:
            stream_handler.setLevel(level)
        self.addHandler(stream_handler)

    def reset_name(self, name):
        """
        reset name
        :param name:
        :return:
        """
        self.name = name
        self.removeHandler(self.file_handler)
        self.__setFileHandler__()
