#!/usr/bin/env python
#################################################
""" __hp_transform.py
# # Collection of useful data transformation Python functions.
#     - Norm, standard, log, binary.
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
#   Data Processing
import numpy as np # Matrix and vector computation package

# Local Application/Library Specific Imports.

# Set the seed of the numpy random number generator
np.random.seed(seed=1)

#################################################
# Dataset Transforms
#################################################
verboseprint = print

def standardize(data):
    """ Transform the data to have a mean of zero
        and a standard deviation of 1."""
    verboseprint("INFO:: Standardization in process...")
    data_mean = np.mean(data, axis=0)
    data_var = np.std(data, axis=0)
    verboseprint("INFO:: Standardization finished!")
    return (data-data_mean)/data_var

def normalize(data):
    """ Rescale the data to ∈ [0:1]."""
    verboseprint("INFO:: Normalization in process...")
    data_min = data.min()
    data_max = data.max()
    verboseprint("INFO:: Normalization finished!")
    return (data - data_min)/(data_max - data_min)

def log_transform(data):
    """Transform data to natural_log(data)."""
    verboseprint("INFO:: Log-Transformation done!")
    return np.log(data+0.1)

def binary_transform(data):
    """Transform data to binary based on sign."""
    verboseprint("INFO:: Binarization done!")
    return np.where(data > 0, 1, 0)
