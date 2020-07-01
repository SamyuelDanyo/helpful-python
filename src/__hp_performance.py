#!/usr/bin/env python
#################################################
""" __hp_performance.py
# # Collection of useful performance measuring Python functions.
#      - Perofrmance metrics, confusion table.
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
from sklearn import metrics
#   Visualisation
import matplotlib.pyplot as plt

# Local Application/Library Specific Imports.

# Set the seed of the numpy random number generator
np.random.seed(seed=1)

#################################################
# Performance Measurment
#################################################
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
    PPV = TP/max((TP+FP), 1)*(1+perc*99)
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
    NPV = TN/max((TN+FN), 1)*(1+perc*99)
    # Specificity or true negative rate [true_neg/actual_neg]
    #   Specificity is a good metric, when the cost of FP relative to TN is high.
    TNR = TN/max((TN+FP), 1)*(1+perc*99)
    # Fall out or false positive rate, false alarm ratio [false_pos/actual_neg]
    #   The probability of falsely rejecting the null hypothesis for a particular test.
    #   Good if missing a '0' has high cost. We want to minimize!
    FPR = FP/max((FP+TN), 1)*(1+perc*99)
    # False negative rate [false_neg/actual_pos]
    #   The probability of falsely rejecting the alternative hypothesis for a particular test.
    #   Good if missing a '1' has high cost. We want to minimize!
    FNR = FN/max((TP+FN), 1)*(1+perc*99)
    # False discovery rate [false_pos/predicted_pos]
    #   The rate of type I errors (rejecting a true null hypothesis)
    #   when conducting multiple comparisons.
    #   Good if miss-predicting '0' -> '1' or missing a '0' has high cost. We want to minimize!
    FDR = FP/max((TP+FP), 1)*(1+perc*99)

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
