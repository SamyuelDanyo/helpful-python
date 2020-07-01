#!/usr/bin/env python
#################################################
""" helpful_python.py
# # Collection of useful Python functions.
#   Function categories:
#     Dictionary:
#       - Building, sorting, extracting, processing, printing.
#     Number Manipulation:
#      - Hex, binary.
#     Relative Processing:
#      - Sorting, filtering, extraction.
#     Sampling:
#      - Chunking, rebin.
#     Validation:
#      - Check func input, Python ver.
#     Data Handling:
#      - Save, load objects.
#     String Manipulation:
#      - Tokenization, replace multiple.
#     Data Transformation:
#      - Norm, standard, log, binary.
#     Perofrmance Measurment:
#      - Perofrmance metrics, confusion table.
#     Machine Learning:
#      - Convolution, pool output size.
#     Misc:
#      - Create dir, remove duplicates, operators.
"""
#################################################
# ###  Author: Samyuel Danyo
# ###  Date: 24/03/2020
# ###  Last Edit: 28/06/2020
#################################################
# coding: utf-8
# ## imports
# Python Standard Library

# Third Party Imports

# Local Application/Library Specific Imports.
import __hp_data as data
import __hp_dict as dicty
import __hp_machine_learning as ml
import __hp_misc as misc
import __hp_number as number
import __hp_performance as perf
import __hp_relative as relative
import __hp_sample as sample
import __hp_string as string
import __hp_transform as transform
import __hp_validate as validate

#################################################
# Reccomended import: import helpful_python as hp
# Reccomended usage: hp.module.func(input)
# Information about a module: help(hp.module)
# Set output dir: hp.set_out_dir('/path/out/dir')
# Set verboseprint: hp.set_verboseprint(hp.
#                                       misc.
#                                       init_verbose_print(verbose=True,
#                                                          vfunc=print,
#                                                          nvfunc=hp.misc.log)
#################################################

def set_out_dir(out_dir_path='./out'):
    """ Set OUT_DIR constant for all sub-modules."""
    global OUT_DIR
    OUT_DIR = out_dir_path
    data.OUT_DIR = OUT_DIR
    misc.OUT_DIR = OUT_DIR

def set_verboseprint(func=misc.init_verbose_print(verbose=True, vfunc=print, nvfunc=misc.log)):
    """ Set verboseprint() for all sub-modules."""
    global verboseprint
    verboseprint = func
    ml.verboseprint = verboseprint
    transform.verboseprint = verboseprint

set_out_dir()
set_verboseprint()
