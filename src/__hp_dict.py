#!/usr/bin/env python
#################################################
""" __hp_dict.py
# # Collection of useful Dictionary (HashMap) Python functions.
#      - Building, sorting, extracting, processing, printing.
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
import pandas as pd

# Local Application/Library Specific Imports.
import __hp_misc as misc

#################################################
# Dictionary
#################################################
def build_dict(keys, vals):
    """ Build a dictionary (HashMap) with keys: vals."""
    new_dict = {}
    for key, val in zip(keys, vals):
        new_dict[key] = val
    return new_dict

def build_sub_dict(keys, full_dict):
    """ Build a sub-dictionary (HashMap) from full-dict with keys."""
    new_dict = {}
    for key in keys:
        new_dict[key] = full_dict[key]
    return new_dict

def sort_dict(my_dict, reverse=True):
    """ Sort dictionary dict by value."""
    return {k: v for k, v in sorted(my_dict.items(), key=lambda item: item[1], reverse=reverse)}

def print_dict(my_dict, callback=print, header=True, keys=None, num=0):
    """ Print dictionary in a nice format."""
    if header:
        callback("            Key | Val            ")
    if keys is None:
        for idx, (key, val) in enumerate(my_dict.items()):
            try:
                callback("{:>15} : {:<15}".format(key, val))
            except Exception:
                callback("{:>15} : {}".format(key, val))
            if idx == num-1:
                return
    else:
        for idx, key in enumerate(keys):
            try:
                callback("{:>15} : {:<15}".format(key, my_dict[key]))
            except Exception:
                callback("{:>15} : {}".format(key, my_dict[key]))
            if idx == num-1:
                return

def get_first_n_items_dict(my_dict, num=None):
    """ Get sub-dictionary of the first num items of d."""
    return {k: my_dict[k] for k in list(my_dict)[:num]}

def get_key_per_value(mydict, rule='>=', thrs=100):
    """ Get dictionary keys based on value threshold."""
    return [key for key, val in mydict.items() if misc.ops(rule)(val, thrs)]

def get_val_sum_per_key(keys, dictionary):
    """ Get dictionary values sum based on keys."""
    return sum([dictionary[key] for key in keys])

def normalise_dict(mean_base_metric_val, metric_dict):
    """ Function for normalising dictionary's values & correlating them to a base metric value.
        Args:
            mean_base_metric_val (Float)     : Value aginst which the metric_dict is relative to.
            metric_dict          (Dictionary): Metric per unique feature.
        Returns:
            metric_dict          (Dictionary): Normalised metric values relative to base metric.
            metric_dict_rate     (Dictionary): Normalised metric values in percentages[%]. """
    total = sum(metric_dict.values(), 0.0)
    # Weighted impact rates (%).
    metric_dict_rate = {k: v / total * 100 for k, v in metric_dict.items()}
    # Weighted impact absolute values.
    metric_dict = {k: v / total * mean_base_metric_val for k, v in metric_dict.items()}
    return (metric_dict, metric_dict_rate)

def extend_dict(my_dict, feat, entry):
    """ Extend a list entry of a dictionary."""
    # check if the key "feat" in dict. if it's in, extend it
    if feat in my_dict:
        my_dict[feat].extend(entry)
    # else new a key
    else:
        my_dict[feat] = entry.copy()

def dict_to_data_frame(my_dict):
    """ Transform dictionary to Pandas data-frame."""
    return pd.DataFrame([my_dict])
