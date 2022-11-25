#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Probabilistic confusion matrix.

Author:
    Erik Johannes Husom

Created:
    2022-11-25 fredag 11:44:54 

"""
import numpy as np
from scipy.sparse import coo_matrix
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels


def probabilistic_confusion_matrix(
        y_true, 
        y_pred, 
        y_pred_std,
        normalize="true",
    ):

    confusion = confusion_matrix(y_true, y_pred, normalize="true")

    # print(confusion)

    labels = unique_labels(y_true, y_pred)
    n_labels = labels.size
    sample_weight = np.ones(y_true.shape[0], dtype=np.int64) * 0.1
    sample_weight = y_pred_std

    cm = coo_matrix(
            (sample_weight, (y_true, y_pred)),
            shape=(n_labels, n_labels)
    )

    # print(cm)
    # cm = cm.toarray()
    # print(cm)

    combined_arr = np.stack((y_true, y_pred), axis=1)
    print(combined_arr)

    u, c = np.unique(combined_arr, axis=0, return_counts=True)
    u_true = u[:,0]
    u_pred = u[:,1]
    print(u_true)
    print(u_pred)
    print(c)

    cm_c = coo_matrix(
            (c, (u_true, u_pred)),
            shape=(n_labels, n_labels)
    )

    cm = cm / cm_c

    print(cm)
    cm = np.nan_to_num(cm)
    
    # normalize="all"
    # normalize = None

    # if normalize == "true":
    #     cm = cm / cm.sum(axis=1, keepdims=True)
    # elif normalize == "pred":
    #     cm = cm / cm.sum(axis=0, keepdims=True)
    # elif normalize == "all":
    #     cm = cm / cm.sum()
    # cm = np.nan_to_num(cm)

    # df_confusion = pd.DataFrame(confusion)

    # df_confusion.index.name = "True"
    # df_confusion.columns.name = "Pred"
    # plt.figure(figsize=(10, 7))
    # sn.heatmap(
    #         df_confusion, 
    #         cmap="Blues", 
    #         annot=True, 
    #         annot_kws={"size": 16},
    #         xticklabels=labels,
    #         yticklabels=labels,
    # )
    # plt.savefig(PLOTS_PATH / "confusion_matrix.png")


def test_pcm():

    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])
    y_pred = np.array([0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 0])
    y_pred_std = np.array([0.1, 0.1, 0.1, 0.3, 0.1, 0.1, 0.1, 0.4, 0.1, 0.1, 0.2, 0.5])

    probabilistic_confusion_matrix(y_true, y_pred, y_pred_std)


if __name__ == '__main__':

    test_pcm()
