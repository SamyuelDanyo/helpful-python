#!/usr/bin/env python
#################################################
""" __hp_sample.py
# # Collection of useful sampling/binning Python functions.
#     - Chunking, rebin.
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
# Sampling | Binning
#################################################
def chunks(arr, num):
    """ Chunking function.
      Args:
            arr (List, Array): Input to be chunked.
            num (integer): Chunk size.
        Returns:
            chunks of arr. """
    for i in range(0, len(arr), num):
        yield arr[i:i+num]

def resample_ver(arr, new_len):
    """ Resamble (bin) an array on the Y axis.
      Args:
            arr     (List, Array): Input to be resampled.
            new_len (integer)    : New dimension size.
        Returns:
            resampled array with ~new_len elements on Y. """
    chunk_size = int(len(arr)/new_len)
    return np.array([np.sum(chunk[::-1], axis=0) for chunk in chunks(arr, chunk_size)])

def resample_hor(arr, new_len):
    """ Resamble (bin) an array on the X axis.
      Args:
            arr     (List, Array): Input to be resampled.
            new_len (integer)    : New dimension size.
        Returns:
            resampled array with ~new_len elements on X. """
    chunk_size = int(len(np.array(arr).T)/new_len)
    return np.array([np.sum(chunk, axis=0) for chunk in chunks(np.array(arr).T, chunk_size)]).T

def resample(arr, x_len, y_len):
    """ Resamble (bin) a 2-D array.
      Args:
            arr   (List, Array): Input to be resampled.
            x_len (integer): New X-dimension size.
            y_len (integer): New Y-dimension size.
        Returns:
            new_arr (Array): resampled array shape=(~y_len, ~x_len). """
    new_arr = resample_hor(arr, x_len)
    new_arr = resample_ver(new_arr, y_len)
    return new_arr

def rebin(arr, new_shape):
    """ Rebin 2D array arr to shape new_shape by averaging. """
    shape = (new_shape[0], arr.shape[0] // new_shape[0],
             new_shape[1], arr.shape[1] // new_shape[1])
    return arr.reshape(shape).mean(-1).mean(1)
