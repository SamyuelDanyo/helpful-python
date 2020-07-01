#!/usr/bin/env python
#################################################
""" __hp_misc.py
# # Collection of useful miscellaneous Python functions.
#   Part of helpful_python package.
#     - Create dir, remove duplicates, operators.
"""
#################################################
# ###  Author: Samyuel Danyo
# ###  Date: 24/03/2020
# ###  Last Edit: 28/06/2020
#################################################
# coding: utf-8
# ## imports
# Python Standard Library
import operator

# Third Party Imports

# Local Application/Library Specific Imports.
import __hp_validate as validate


#################################################
# Misc
#################################################
OUT_DIR = './out/'
LOG_FILE = OUT_DIR+'HelpfulPython.log'

def remove_dupl_2d(arr):
    """ Remove duplicates in a 2-D list."""
    arr_len = len(arr)
    idx = 0
    unique = set()
    while idx < arr_len:
        if tuple(arr[idx]) in unique:
            del arr[idx]
            arr_len -= 1
            continue
        unique.add(tuple(arr[idx]))
        idx += 1
    return arr

def get_time(src_time):
    """ Get formatted src_time."""
    m, s = divmod(src_time, 60)
    h, m = divmod(m, 60)
    return (h, m, s)

def ops(rule):
    """ Return comparison operators from strings."""
    ops_dict = {'>' : operator.gt,
                '<' : operator.lt,
                '>=': operator.ge,
                '<=': operator.le,
                '=' : operator.eq,
                '==' : operator.eq}
    return ops_dict[rule]

def init_verbose_print(verbose=True, vfunc=print, nvfunc=None):
    """ Initialise verboseprint() if verbose to a specific
        nvfunc or print, else to no printing."""
    global verboseprint
    if verbose:
        verboseprint = vfunc
    else:
        if not nvfunc:
            verboseprint = lambda *a, **k: None
        else:
            verboseprint = nvfunc
    return verboseprint

def log(text):
    """ Logging function."""
    with open(LOG_FILE, 'a') as file:
        file.write(text)
        file.write('\n')

def create_dir(dir_path):
    """ Create directory dir_path if it does not exist."""
    validate.check_python_ver(ver=3.5)
    from pathlib import Path
    Path(dir_path).mkdir(parents=True, exist_ok=True)
