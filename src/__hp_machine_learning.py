#!/usr/bin/env python
#################################################
""" __hp_machine_learning.py
# # Collection of useful ML Python functions.
#   Part of helpful_python package.
#     - Convolution, pool output size.
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

#################################################
# ## Machine Learning Classifier Helper Functions
#################################################
verboseprint = print

def conv_out(in_size, ksize=5, stride=1, pad=0):
    """ Calculate the output size of convolution.
    # W - Input Width; K - Kernel Size; P - Padding Size; S - Stride Size
    # conv_out = (W - K + 2P) / S + 1"""
    return (in_size - ksize + 2*pad) / stride + 1

def pool_out(in_size, ksize=5, stride=1):
    """ Calculate the output size of pooling.
    # W - Input Width; K - Kernel Size; P - Padding Size; S - Stride Size
    # pool_out = (W - K) / S + 1"""
    return (in_size - ksize) / stride + 1

def calc_conv_out_size(in_size,
                       layers=['conv1', 'pool1', 'conv2', 'pool2'],
                       layers_config=[
                           # [ksize, stride, pad]
                           [5, 1, 0],
                           [2, 2],
                           [5, 1, 0],
                           [2, 2]]):
    """ Calculate the output size of a convolutional network.
    # W - Input Width; K - Kernel Size; P - Padding Size; S - Stride Size
    # conv_out = (W - K + 2P) / S + 1
    # pool_out = (W - K) / S + 1"""
    for layer, config in zip(layers, layers_config):
        if 'conv' in layer:
            K, S, P = config
            out_size = conv_out(in_size, K, S, P)
        elif 'pool' in layer:
            K, S = config
            out_size = pool_out(in_size, K, S)
        else:
            verboseprint("ERROR:: {} is not a valid layer name!!".format(layer))
            return False
        in_size = out_size
    if out_size % 1 != 0:
        verboseprint("ERROR:: Convolution output size={} must be an integer!".format(out_size))
        return False
    return int(out_size)
