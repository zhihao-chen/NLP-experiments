# -*- coding: utf8 -*-
"""
======================================
    Project Name: NLP
    File Name: path
    Author: czh
    Create Date: 2021/9/10
--------------------------------------
    Change Activity: 
======================================
"""
import inspect
import os
import pathlib
import sys
import types

import regex

FILES_IN_ROOT = ('requirements.txt', 'VERSION', 'conf', 'setup.py', '.idea', '.git')
__ROOT_PATH = None


def expand(path):
    if path is None:
        return None
    return os.path.expanduser(os.path.expandvars(path))


def package_root(module: types.ModuleType):
    """获取module的目录"""
    root, _ = os.path.split(os.path.abspath(module.__file__))
    return root


def project_root_path():
    global __ROOT_PATH
    if __ROOT_PATH is not None:
        return __ROOT_PATH
    path = pathlib.Path(os.getcwd())
    while True:
        if any([path.joinpath(f).exists() for f in FILES_IN_ROOT]):
            __ROOT_PATH = str(path)
            break
        path = path.parent
        if str(path) == '/':
            __ROOT_PATH = os.getcwd()
            break
    return __ROOT_PATH


def root_path():
    return project_root_path()


def home_path():
    return str(pathlib.Path.home())


def whoami():
    caller = inspect.currentframe().f_back
    return caller.f_globals['__name__']


def who_called_me():
    stack = inspect.stack()
    frame_info = stack[2]
    filename = frame_info.filename
    filename = os.path.relpath(filename, root_path())
    name, ext = os.path.splitext(filename)
    name = regex.sub(r'^/', '', name)
    name = regex.sub(r'/', '.', name)
    return name


def function_called_me():
    stack = inspect.stack()
    frame_info = stack[2]
    return frame_info.function


def program_name():
    name = os.environ.get('program_name')
    if name is not None:
        return name
    main_module = sys.modules['__main__']
    if hasattr(main_module, '__file__'):
        return pathlib.Path(main_module.__file__).stem
    return project_name()


def project_name():
    return pathlib.Path(project_root_path() or os.getcwd()).stem


def make_parent_dirs(file_path, exist_ok=True):
    parent = pathlib.Path(file_path).parent
    if parent.exists():
        return
    os.makedirs(parent, exist_ok=exist_ok)


def set_temp_dir(path: str = '~/.temp') -> bool:
    if path is None:
        return False
    path = path.strip()
    if len(path) == 0:
        return False
    if 'TMPDIR' in os.environ:
        return False
    path = expand(path)
    os.makedirs(path, exist_ok=True)
    os.environ['TMPDIR'] = path
    return True


def get_path(*paths):
    if paths is None or len(paths) == 0:
        return ''
    paths = list(filter(None, paths))
    if len(paths) == 0:
        return ''
    if paths[0][0] in ('/', '~', '$'):
        return expand(os.path.join(*paths))
    return expand(os.path.join(project_root_path(), *paths))


def get(*paths):
    return get_path(*paths)
