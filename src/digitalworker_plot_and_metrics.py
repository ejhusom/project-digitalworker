#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Plot results for DigitalWorker.

Author:
    Erik Johannes Husom

Created:
    2022-11-07 mandag 14:37:06 

"""
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn

from scipy.sparse import coo_matrix
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    explained_variance_score,
    mean_absolute_percentage_error,
    mean_squared_error,
    precision_score,
    r2_score,
)
from sklearn.utils.multiclass import unique_labels

# from config import OUTPUT_FEATURES_PATH
OUTPUT_FEATURES_PATH = "assets/features/output_columns.csv"

def plot_confusion_for_paper(y_test, y_pred):
    """Plotting confusion matrix of a classification model."""

    output_columns = np.array(pd.read_csv(OUTPUT_FEATURES_PATH, index_col=0)).reshape(
        -1
    )

    n_output_cols = len(output_columns)
    indeces = np.arange(0, n_output_cols, 1)

    if len(indeces) == 8:
        labels = [
                "Lie",
                "Kneel",
                "Sit",
                "Stand",
                "Other",
                "Walk",
                "Run",
                "Stairs"
        ]
    elif len(indeces) == 12:
        labels = [
                "Sensor Off",
                "Lie",
                "Kneel",
                "Sit",
                "Stand",
                "Other",
                "Run",
                "Stairs",
                "Cycle",
                "Row",
                "Walk Slow",
                "Walk Fast"
        ]
    elif len(indeces) == 13:
        labels = [
                "Sensor Off",
                "Lie",
                "Kneel",
                "Sit",
                "Stand",
                "Other",
                "Walk",
                "Run",
                "Stairs",
                "Cycle",
                "Row",
                "Walk Slow",
                "Walk Fast"
        ]
    else:
        labels = indeces


    confusion = confusion_matrix(
            y_test, 
            y_pred, 
            normalize="true",
            # labels=labels
    )

    df_confusion = pd.DataFrame(confusion).round(2)

    df_confusion.index.name = "True"
    df_confusion.columns.name = "Pred"
    plt.figure(figsize=(10, 7))
    sn.heatmap(
            df_confusion, 
            cmap="Blues", 
            annot=True, 
            annot_kws={"size": 16},
            xticklabels=labels,
            yticklabels=labels,
    )
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    plt.show()

def calculate_metrics(y_test, y_pred, y_pred_std):

    accuracy = accuracy_score(y_test, y_pred)
    mean_precision = precision_score(y_test, y_pred, average="weighted",
            zero_division=0)
    mean_recall = recall_score(y_test, y_pred, average="weighted")
    mean_f1 = f1_score(y_test, y_pred, average="weighted")
    precision = precision_score(y_test, y_pred, average=None,
            zero_division=0).round(2)
    recall = recall_score(y_test, y_pred, average=None).round(2)
    f1 = f1_score(y_test, y_pred, average=None).round(2)

    confusion = confusion_matrix(
            y_test, 
            y_pred, 
            normalize="true",
    )

    accuracies = confusion.diagonal().round(2).tolist()

    # print(f"precision: {precision}")
    # print(f"recall: {recall}")
    # print(f"F1: {f1}")
    print("Accuracies:")
    print(" & ".join(map(str, accuracies)))
    print("Precision:")
    print(" & ".join(map(str, precision)))
    print("Recall:")
    print(" & ".join(map(str, recall)))
    print("F1:")
    print(" & ".join(map(str, f1)))


    int_labels = unique_labels(y_test, y_pred)
    n_labels = int_labels.size

    cm = coo_matrix(
            (y_pred_std, (y_test, y_pred)),
            shape=(n_labels, n_labels)
    )

    combined_arr = np.stack((y_test, y_pred), axis=1)

    unique_predictions, counts = np.unique(combined_arr, axis=0, return_counts=True)
    u_true = unique_predictions[:,0]
    u_pred = unique_predictions[:,1]

    cm_count = coo_matrix(
            (counts, (u_true, u_pred)),
            shape=(n_labels, n_labels)
    )


    cm = cm / cm_count
    # cm = cm / cm.sum(axis=1, keepdims=True)
    uncertainties = cm.diagonal().round(2).flatten()
    print("Uncertainty:")
    print(" & ".join(map(str, uncertainties)))
    print("Average uncertainty for correct classifications")
    print(uncertainties[~np.isnan(uncertainties)].mean()) 
    print("Average uncertainty for misclassifications")

    # print(np.eye(cm.shape[0],dtype=bool))
    mask = np.eye(cm.shape[0], dtype=bool)
    nondiagonal = cm[~mask]
    # indeces = np.where(~mask)
    # rows = indeces[0]
    # cols = indeces[1]
    # print(cm[rows,cols])
    # unc_misclassifications = nondiagonal.sum()
    unc_misc_wo_nan = nondiagonal[~np.isnan(nondiagonal)]
    print(unc_misc_wo_nan.mean())
    # print


    print(f"Accuracy: {accuracy}")
    print(f"Mean precision: {mean_precision}")
    print(f"Mean recall: {mean_recall}")
    print(f"Mean F1: {mean_f1}")
    
    cm = pd.DataFrame(cm)

    # df_confusion.index.name = "True"
    # df_confusion.columns.name = "Pred"
    plt.figure(figsize=(10, 7))
    sn.heatmap(
            cm, 
            # cmap="Reds", 
            # annot=df_confusion, 
            annot=True,
            # annot_kws={"size": 14},
            # xticklabels=labels,
            # yticklabels=labels,
    )
    plt.tight_layout()
    # plt.savefig(PLOTS_PATH / "probablistic_confusion_matrix.png")
    plt.show()


if __name__ == '__main__': 
    y_pred_file = sys.argv[1] 
    y_test_file = sys.argv[2] 
    y_pred_std_file = sys.argv[3]

    # Read predictions
    y_pred = pd.read_csv(y_pred_file).to_numpy().flatten()
    # print(y_pred)

    # Read true values
    y_test = pd.read_csv(y_test_file, index_col=0).to_numpy()
    # Convert from one-hot encoding back to classes
    y_test = np.argmax(y_test, axis=-1)
    # print(y_test)

    # Read uncertainty
    y_pred_std = pd.read_csv(y_pred_std_file, index_col=0).to_numpy().flatten()

    # plot_confusion_for_paper(y_test, y_pred)
    calculate_metrics(y_test, y_pred, y_pred_std)
