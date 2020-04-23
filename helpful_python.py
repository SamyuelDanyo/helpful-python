#!/usr/bin/env python
#################################################
""" cbeta.py
# # Conditional Branch Extended Trace Analysis (CBETA) Tool
#   Module contaning neccesary fuinctions for:
#     Visualising correlations between:
#       - CBs' time/space pattern & misprediction impact.
#       - CBs' number of occurrences (count) & misprediction rate.
#       - CBs' frequency of occurrences (frequency) & misprediction rate.
#     Querying into the processed data to guide further insights:
#      - Top misprediction impact CBs' patterns.
#      - Hard-to-predict branches.
#      - Metric correlations.
#      - Misprediction impact of particular branches.
"""
#################################################
# ###  Author: Samyuel Danyo
# ###  Date: 24/03/2020
##################################################
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

# Third Party Imports
#   Data Processing
#     Set the seed of the numpy random number generator
import numpy as np # Matrix and vector computation package

# Local Application/Library Specific Imports.

# Set the seed of the numpy random number generator
np.random.seed(seed=1)

def remove_dupl(arr):
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
    
def check_python_ver(ver=3.7):
    """ Assert minimum Python version."""
    sys_ver = sys.version_info
    assert sys_ver >= (ver//1, ver%1), (
        "Update Python. Versions: Required[{}] | Current[{}]".format(ver, sys_ver))

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

def hex_to_hex(num, short=False):
    """ Return hex number from prefixed(0x) hex number string."""
    num = int(num, 16)
    if short:
        return '%x' % num
    return '0x%x' % num

def to_hex(num, base=10, short=False):
    """ Return hex number from binary, decimal, hex strings/numbers."""
    try:
        if not isinstance(num, str):
            if base == 16:
                num = '{:x}'.format(num)
            elif base == 2:
                num = '{:b}'.format(num)

        if short:
            return '%x' % int('{}'.format(num), base)
        return '0x%x' % int('{}'.format(num), base)
    except Exception:
        return hex_to_hex(num, short)

def ops(rule):
    """ Return comparison operators from strings."""
    ops_dict = {'>' : operator.gt,
                '<' : operator.lt,
                '>=': operator.ge,
                '<=': operator.le,
                '=' : operator.eq,
                '==' : operator.eq}
    return ops_dict[rule]

def get_key_per_value(mydict, rule='>=', thrs=100):
    """ Get dictionary keys based on value threshold."""
    return [key for key, val in mydict.items() if ops(rule)(val, thrs)]

def get_val_sum_per_key(keys, dictionary):
    """ Get dictionary values sum based on keys."""
    return sum([dictionary[key] for key in keys])
    
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

def check(func, arg, msg):
    """ Make sure func(arg) does NOT raise an exception.
        Args:
            func (Function): Function to be cheked:
            arg  (*)       : Argument to func().
            msg  (String)  : Error message ot be printed.
        Returns:
            func(arg) if no exception occurs, None if it does."""
    try:
        arg = func(arg)
        return arg
    except Exception:
        print("ERROR:: "+msg)
        return None

def rcheck(func, arg, msg):
    """ Make sure func(arg) DOES raise an exception.
        Args:
            func (Function): Function to be cheked:
            arg  (*)       : Argument to func().
            msg  (String)  : Error message ot be printed.
        Returns:
            None if exception occurs, arg if it doesn't."""
    try:
        arg = func(arg)
        print("ERROR:: "+msg)
        return None
    except Exception:
        return arg

def save_obj(obj, name):
    with open(OUT_DIR + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open(OUT_DIR + name + '.pkl', 'rb') as f:
        return pickle.load(f)

def save_obj_np(obj, name='./obj.npy'):
    """ Save object (array, list...) to filesystem."""
    np.save(name, obj)

def load_obj_np(name='./obj.npy', not_arr=False):
    """ Load preiviously saved object."""
    if not_arr:
        return np.load(name, allow_pickle='TRUE').item()
    return np.load(name, allow_pickle='TRUE')

def save_data(data, data_name):
    """ Save data(list of objects) to filesystem."""
    print("Saving Data to {} ...".format(OUT_DIR))
    for entry, name in zip(data, data_name):
        save_obj(entry, name)
    print("Data Saved!")
    
global verboseprint
verboseprint = print if opt.verbose else lambda *a, **k: None

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

def least_sig_bits_hex(src, bits, out_t=int, arr=True):
    """ Get the bits least significant bits of a/list of hex number/s.
        Args:
            src (String/List of String): Source hex numbers.
            bits (Integer): Number of least-sig bits.
            out_t (int or hex): Output type. Integer and Hex supported.
        Returns:
            out (out_t/List(out_t)): The least-sig bits for each el in src."""
    # List
    if arr:
        try:
            out = []
            for elem in src:
                # Transform to int
                elem = int(elem, 16)
                # Transform to binary & get least sig bits.
                elem = bin(elem)
                # Get least sig bits (check bits is not more than number).
                elem = elem[-min(bits, len(elem)-2):]
                # Append the number
                if out_t is int or out_t == 'int':
                    out.append(int(elem, 2))
                else:
                    out.append(to_hex(elem, base=2, short=True))
            return out
        except:
            pass
    # Single number
    # Transform to int
    src = int(src, 16)
    # Transform to binary
    src = bin(src)
    # Get least sig bits (check bits is not more than number).
    src = src[-min(bits, len(src)-2):]
    # Return the number
    if out_t is int or out_t == 'int':
        return int(src, 2)
    return cb.to_hex(src, base=2, short=True)

def append_bits(src, append_src, out_t=int):
    """ Append bits to src integer/binary numbers."""
    # List
    try:
        if len(src) != len(append_src):
            print("ERROR:: src and append_src need to have the same lenght!!!")
            return False
        out = []
        for elem, bits in zip(src, append_src):
            if out_t is int or out_t == 'int':
                out.append(int(bin(elem)+'{}'.format(bits), 2))
            else:
                out.append(elem+str(bits))
        return out
    # Single number
    except Exception:
        if out_t is int or out_t == 'int':
            return int(bin(elem)+'{}'.format(bits), 2)
        return elem+str(bits)

def replace_multiple(src_str, old_lst, new):
    """ Replace a list of multiple sub-strings with a new string
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

def __get_perf_matrics(targets, predictions, perc=False):
    """ Private helper Function for calculating performance metrics
        based on 'predictions' compared to 'targets'."""
    CM = metrics.confusion_matrix(targets, predictions, labels=None)
    TN, FN, TP, FP = CM[0][0], CM[1][0], CM[1][1], CM[0][1]

    # Overall Accuracy
    ACC = (TP+TN)/(TP+FP+FN+TN)*(1+perc*99)
    # Overall Error
    ERR = (FP+FN)/(TP+FP+FN+TN)*(1+perc*99)
    # False Positive Error
    FPERR = FP/(TP+FP+FN+TN)*(1+perc*99)
    # False Negative Error
    FNERR = FN/(TP+FP+FN+TN)*(1+perc*99)
    # Precision or positive predictive value [true_pos/pred_pos]
    #   Precision is a good metric, when the cost of False Positive
    #   relative to TP is high.  Weight on prevalence.
    PPV = TP/(TP+FP)*(1+perc*99)
    # Sensitivity, hit rate, recall, or true positive rate [true_pos/actual_pos]
    #   Recall is a good metric, when the cost of FN relative to TP is high.
    TPR = TP/max((TP+FN), 1)*(1+perc*99)
    # F1 Score
    #   F1 Score might be a better measure to use if we need to seek
    #   a balance between Precision and Recall AND there is an uneven class
    #   distribution (large number of Actual Negatives).
    F1 = 2*PPV*TPR/max((PPV+TPR), 1)
    # Negative predictive value [true_neg/pred_neg]
    #   NPV is a good metric, when the cost(FN)/cost(TN) is high. Weight on prevalence.
    NPV = TN/(TN+FN)*(1+perc*99)
    # Specificity or true negative rate [true_neg/actual_neg]
    #   Specificity is a good metric, when the cost of FP relative to TN is high.
    TNR = TN/(TN+FP)*(1+perc*99)
    # Fall out or false positive rate, false alarm ratio [false_pos/actual_neg]
    #   The probability of falsely rejecting the null hypothesis for a particular test.
    #   Good if missing a '0' has high cost. We want to minimize!
    FPR = FP/(FP+TN)*(1+perc*99)
    # False negative rate [false_neg/actual_pos]
    #   The probability of falsely rejecting the alternative hypothesis for a particular test.
    #   Good if missing a '1' has high cost. We want to minimize!
    FNR = FN/max((TP+FN), 1)*(1+perc*99)
    # False discovery rate [false_pos/predicted_pos]
    #   The rate of type I errors (rejecting a true null hypothesis)
    #   when conducting multiple comparisons.
    #   Good if miss-predicting '0' -> '1' or missing a '0' has high cost. We want to minimize!
    FDR = FP/(TP+FP)*(1+perc*99)

    return(ACC, ERR, FPERR, FNERR, PPV, TPR, F1, NPV, TNR, FPR, FNR, FDR)

def get_perf_matrics(targets, predictions, perc=False):
    """ Public helper function for calculating performance metrics
         based on 'predictions' compared to 'targets' as per the confusion table below.
           Confusion Table
              0       1
                  |        T
         0    TN  |  FP    a
           _______|_______ r
                  |        g
         1    FN  |  TP    e
                  |        t
              Prediction
         Args:
             targets (NumPy Array[N])
             predictions (NumPy Array[M][N])
             N - number of samples; M - number of parameter iterations;
         Returns:
             acc   : overall accuracy
             err   : overall error
             fperr : false positives error
             fnerr : false negatives error
             ppv   : precision or positive predictive value [true_pos/pred_pos]
             tpr   : recall, or true positive rate [true_pos/actual_pos]
             npv   : negative predictive value [true_neg/pred_neg]
             tnr   : specificity or true negative rate [true_neg/actual_neg]
             fpr   : fall out or false positive rate [false_pos/actual_neg]
             fnr   : false negative rate [false_neg/actual_pos]
             fdr   : false discovery rate [false_pos/predicted_pos] """
    # Calculation per iteration.
    if np.array(predictions).ndim == 2:
        (acc, err, fperr, fnerr, ppv, tpr,
         f1, npv, tnr, fpr, fnr, fdr) = ([], [], [], [], [], [],
                                         [], [], [], [], [], [])
        for preds_per_iter in predictions:
            (ACC, ERR, FPERR, FNERR, PPV, TPR, F1, NPV, TNR,
             FPR, FNR, FDR) = __get_perf_matrics(targets, preds_per_iter, perc)
            acc.append(ACC)
            err.append(ERR)
            fperr.append(FPERR)
            fnerr.append(FNERR)
            ppv.append(PPV)
            tpr.append(TPR)
            f1.append(F1)
            npv.append(NPV)
            tnr.append(TNR)
            fpr.append(FPR)
            fnr.append(FNR)
            fdr.append(FDR)
    # Calculation for single iteration.
    elif np.array(predictions).ndim == 1:
        (acc, err, fperr, fnerr, ppv, tpr, f1, npv, tnr, fpr,
         fnr, fdr) = __get_perf_matrics(targets, predictions, perc)
    else:
        print("ERROR: PERFORMANCE METRICS AVALIABLE ONLY for PREDICTIONS.ndim in (1,2)")
        return False

    return (acc, err, fperr, fnerr, ppv, tpr, f1, npv, tnr, fpr, fnr, fdr)

def get_accuracy(targets, predictions):
    """ Helper Function for calculating the (%) accuracy of
        'predictions' compared to 'targets'."""
    return (np.abs(targets - predictions) < 1e-10).mean() * 100.0

def get_class_accuracy(predictions, targets):
    """ Helper Function for calculating the (%) accuracy of each class
        by comapring 'predictions' to 'targets'."""
    class_labs = np.unique(targets)
    class_acc = np.zeros(len(class_labs))
    for idx, lab in enumerate(class_labs):
        class_y = predictions[targets == lab]
        class_t = targets[targets == lab]
        class_acc[idx] = get_accuracy(class_t, class_y)
    return (class_labs, class_acc)

def get_accuracy_topn(targets, predictions):
    """ Helper Function for calculating the (%) accuracy
        of topn in 'predictions' compared to 'targets'.
        Args:
            targets (NumPy Array[N])
            predictions (NumPy Array[Nxn]) """
    acc = 0
    for pred, tar in zip(predictions, targets):
        if tar in pred:
            acc += 1
    return acc/len(predictions)*100

def plot_confusion_table(y_true, y_pred, title):
    """ Display a confusion table of targets vs predictions."""
    # Show confusion table
    conf_matrix = metrics.confusion_matrix(y_true, y_pred, labels=None)
    # Plot the confusion table
    class_names = ['${:d}$'.format(x) for x in (0, 1)]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # Show class labels on each axis
    ax.xaxis.tick_top()
    major_ticks = range(0, 2)
    minor_ticks = [x + 0.5 for x in range(0, 2)]
    ax.xaxis.set_ticks(major_ticks, minor=False)
    ax.yaxis.set_ticks(major_ticks, minor=False)
    ax.xaxis.set_ticks(minor_ticks, minor=True)
    ax.yaxis.set_ticks(minor_ticks, minor=True)
    ax.xaxis.set_ticklabels(class_names, minor=False, fontsize=15)
    ax.yaxis.set_ticklabels(class_names, minor=False, fontsize=15)
    # Set plot labels
    ax.yaxis.set_label_position("right")
    ax.set_xlabel('Predicted Label', fontsize=15)
    ax.set_ylabel('True Label', fontsize=15)
    fig.suptitle(title, y=1.03, fontsize=15)
    # Show a grid to seperate digits
    ax.grid(b=True, which=u'minor')
    # Color each grid cell according to the number classes predicted
    ax.imshow(conf_matrix, interpolation='nearest', cmap='binary')
    # Show the number of samples in each cell
    for x in range(conf_matrix.shape[0]):
        for y in range(conf_matrix.shape[1]):
            color = 'w' if x == y else 'k'
            ax.text(x, y, conf_matrix[y, x], ha="center", va="center", color=color)
    plt.show()

