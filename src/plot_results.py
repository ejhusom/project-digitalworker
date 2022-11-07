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

from sklearn.metrics import confusion_matrix

from config import OUTPUT_FEATURES_PATH

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


if __name__ == '__main__': 

    y_pred_file = sys.argv[1]
    y_test_file = sys.argv[2]

    # Read predictions
    y_pred = pd.read_csv(y_pred_file).to_numpy().flatten()
    # print(y_pred)

    # Read true values
    y_test = pd.read_csv(y_test_file, index_col=0).to_numpy()
    # Convert from one-hot encoding back to classes
    y_test = np.argmax(y_test, axis=-1)
    # print(y_test)

    plot_confusion_for_paper(y_test, y_pred)
