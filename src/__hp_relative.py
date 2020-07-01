#!/usr/bin/env python
#################################################
""" __hp_relative.py
# # Collection of useful relative processing Python functions.
#     - Sorting, filtering, extraction.
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
# Relative Processing
#################################################
def sort_arrays_relatively(sorting_array, *arrays_to_be_sorted):
    """ Function for sorting arrays relative to a sorting array.
        Args:
            sorting_array       (Numpy Array): Array relative to which other arrays will be sorted.
            arrays_to_be_sorted (Tuple[N] of Numpy Array): Arrays to be sorted.
        Returns:
            srted_arrays        (Tuple[N+1] of Numpy Array): Sorted arrays. """
    srted_arrays = ()
    for array in arrays_to_be_sorted:
        srted_arrays += (np.array([x for _, x in sorted(zip(sorting_array, array),
                                                        key=lambda pair: pair[0])]),)
    srted_arrays += (np.array(sorted(sorting_array)),)
    return srted_arrays

def get_metrics_per_base_metric(metric_arrs, base_metric_arr, sort=True):
    """ Helper Function for getting mean metrics values per some base metric.
        !!! Works only on aligned metric_arrs & base_metric_arr.
        Args:
            metric_arrs     (Tuple/List[N] of Numpy Array): Arrays for the metrics to be processed.
                                                            Arbitrary number of arrays can be passed
                                                            but each array's elements need to have a
                                                            corresponding one in base_metric_arr.
            base_metric_arr (Numpy Array): Base metric for the metrics values to be correlated to.
            sort            (Boolean): Unique-base-metric-array to be sorted min->max or reverse.
        Returns:
            metrics_out     (Tuple[N+1] of Numpy Array): Arrays of the mean metrics values per base
                                                         metric & the unique set of base metric
                                                         values (which can be sorted or not). """
    if sort:
        # Get unique values for base metric, sorted min-> max.
        ubase_metric_arr = np.array(sorted(list(set(base_metric_arr))))
    else:
        # Get unique valuess for base metric.
        ubase_metric_arr = np.array(list(set(base_metric_arr)))
    metrics_out = (ubase_metric_arr,)
    # For each metric.
    for metric_arr in metric_arrs:
        # Get mean metric per unique base metric value
        # (mean of all metric_arr elements with the same base metric value).
        metric_out = []
        for val in ubase_metric_arr:
            metric_out.append(np.mean(metric_arr[base_metric_arr == val]))
        metrics_out += (np.array(metric_out),)
    return  metrics_out

def filter_list(src_list, fltr, fltr_src=None, not_in=False):
    """ Filter src_list elemenets for indeces of fltr elements in fltr_src.
        Ex:
            src_list = [1, 2, 3, 4, 5, 6, 7, 8, 9]
            fltr = [a, d]
            fltr_src = [a, b, c, d, e, a, b, c, d]
            return = [1, 4, 6, 9].
        not_in - filters by not in fltr."""
    if fltr_src is None:
        fltr_src = src_list
    if not_in:
        return np.array(src_list)[[entry not in fltr for entry in fltr_src]]
    return np.array(src_list)[[entry in fltr for entry in fltr_src]]
