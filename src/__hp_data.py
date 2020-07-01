#!/usr/bin/env python
#################################################
""" __hp_data.py
# # Collection of useful data handling Python functions.
#     - Save, load objects.
"""
#################################################
# ###  Author: Samyuel Danyo
# ###  Date: 24/03/2020
# ###  Last Edit: 28/06/2020
#################################################
# coding: utf-8
# ## imports
# Python Standard Library
import pickle

# Third Party Imports
#     Set the seed of the numpy random number generator
import numpy as np # Matrix and vector computation package

# Local Application/Library Specific Imports.

# Set the seed of the numpy random number generator
np.random.seed(seed=1)

#################################################
# Data Handling
#################################################
OUT_DIR = './out/'
def save_obj(obj, name='./obj.pt'):
    """ Save object (array, list...) to filesystem using Pickle."""
    with open(OUT_DIR + name + '.pkl', 'wb') as file:
        pickle.dump(obj, file, pickle.HIGHEST_PROTOCOL)

def load_obj(name='./obj.pt'):
    """ Load preiviously saved object using Pickle."""
    with open(OUT_DIR + name + '.pkl', 'rb') as file:
        return pickle.load(file)

def save_data(data, data_name):
    """ Save data(list of objects) to filesystem using Pickle."""
    print("Saving Data to {} ...".format(OUT_DIR))
    for entry, name in zip(data, data_name):
        save_obj(entry, name)
    print("Data Saved!")

def save_obj_np(obj, name='./obj.npy'):
    """ Save object (array, list...) to filesystem using NumPy."""
    np.save(name, obj)

def load_obj_np(name='./obj.npy', not_arr=False):
    """ Load preiviously saved object using NumPy."""
    if not_arr:
        return np.load(name, allow_pickle='TRUE').item()
    return np.load(name, allow_pickle='TRUE')

def save_data_np(data, data_name):
    """ Save data(list of objects) to filesystem using NumPy."""
    print("Saving Data to {} ...".format(OUT_DIR))
    for entry, name in zip(data, data_name):
        save_obj_np(entry, name)
    print("Data Saved!")
