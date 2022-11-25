#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Probabilistic confusion matrix.

Author:
    Erik Johannes Husom

Created:
    2022-11-25 fredag 11:44:54 

"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn

from scipy.sparse import coo_matrix
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels


def probabilistic_confusion_matrix(
        y_true, 
        y_pred, 
        y_pred_std,
        normalize="true",
    ):

    labels = unique_labels(y_true, y_pred)
    n_labels = labels.size

    cm = coo_matrix(
            (y_pred_std, (y_true, y_pred)),
            shape=(n_labels, n_labels)
    )

    combined_arr = np.stack((y_true, y_pred), axis=1)

    unique_predictions, counts = np.unique(combined_arr, axis=0, return_counts=True)
    u_true = unique_predictions[:,0]
    u_pred = unique_predictions[:,1]

    cm_count = coo_matrix(
            (counts, (u_true, u_pred)),
            shape=(n_labels, n_labels)
    )

    cm = cm / cm_count
    # cm = np.nan_to_num(cm)
    
    df_confusion = pd.DataFrame(cm)

    df_confusion.index.name = "True"
    df_confusion.columns.name = "Pred"
    plt.figure(figsize=(10, 7))
    sn.heatmap(
            df_confusion, 
            cmap="Blues", 
            annot=True, 
            annot_kws={"size": 16},
            # xticklabels=labels,
            # yticklabels=labels,
    )

    plt.show()
    # plt.savefig(PLOTS_PATH / "confusion_matrix.png")


def test_pcm():

    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])
    y_pred = np.array([0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 0])
    y_pred_std = np.array([0.1, 0.1, 0.1, 0.3, 0.1, 0.1, 0.1, 0.4, 0.1, 0.1, 0.2, 0.5])

    probabilistic_confusion_matrix(y_true, y_pred, y_pred_std)


if __name__ == '__main__':

    test_pcm()
