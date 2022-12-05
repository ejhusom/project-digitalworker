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
from evaluate import plot_confusion

if __name__ == '__main__': 

    y_pred_file = sys.argv[1]
    y_test_file = sys.argv[2]

    # Read predictions
    y_pred = pd.read_csv(y_pred_file).to_numpy().flatten()

    # Read true values
    y_test = pd.read_csv(y_test_file, index_col=0).to_numpy()
    # Convert from one-hot encoding back to classes
    y_test = np.argmax(y_test, axis=-1)

    plot_confusion(y_test, y_pred)
