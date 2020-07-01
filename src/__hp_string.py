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
import sys
import operator
from collections import Counter  # available in Python 2.7 and newer
from collections import defaultdict  # available in Python 2.5 and newer
from itertools import count as iter_count
import os
import argparse
import pickle
import copy
import time

# Third Party Imports
#   Data Processing
#     Set the seed of the numpy random number generator
import numpy as np # Matrix and vector computation package
from sklearn import model_selection as msel
#   Machine Learning
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
#   Visualisation
# Local Application/Library Specific Imports.

# Set the seed of the numpy random number generator
np.random.seed(seed=1)

#################################################
# String Manipulation
#################################################
def replace_multiple(src_str, old_lst, new):
    """ Replace a list of sub-strings with a new string
        in the source string."""
    # Iterate over the strings to be replaced
    for elem in old_lst:
        # Check if string is in the source string
        if elem in src_str:
            # Replace the string
            src_str = src_str.replace(elem, new)
    return  src_str

def char_split(name, char='.', num=3):
    """ Split name on the first num occurences of char."""
    if name.count(char) < num:
        return os.path.splitext(name)[0]
    return char.join(name.split(char, num)[:num])

def tokenize(text, sep=' '):
    """ Tokenize a sentance, where words are separeted by sep."""
    return([el.strip() for el in text.split(sep)])
