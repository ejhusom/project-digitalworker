#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""One-liner describing module.

Author:
    Erik Johannes Husom

Created:
    2021

"""
from config import (
    DATA_PATH,
    INPUT_FEATURES_PATH,
    INTERVALS_PLOT_PATH,
    METRICS_FILE_PATH,
    NON_DL_METHODS,
    OUTPUT_FEATURES_PATH,
    PLOTS_PATH,
    PREDICTION_PLOT_PATH,
    PREDICTIONS_FILE_PATH,
    PREDICTIONS_PATH,
)

# Load parameters
params = yaml.safe_load(open("params.yaml"))["evaluate"]
params_train = yaml.safe_load(open("params.yaml"))["train"]
params_split = yaml.safe_load(open("params.yaml"))["split"]
classification = yaml.safe_load(open("params.yaml"))["clean"]["classification"]
onehot_encode_target = yaml.safe_load(open("params.yaml"))["clean"][
    "onehot_encode_target"
]
dropout_uncertainty_estimation = params["dropout_uncertainty_estimation"]
uncertainty_estimation_sampling_size = params["uncertainty_estimation_sampling_size"]
show_inputs = params["show_inputs"]
learning_method = params_train["learning_method"]

test = np.load("assets/combined/test.npz")
X_test = test["X"]
y_test = test["y"]
y_pred_std = None

model = models.load_model("assets/models/model.h5")

predictions = []

for i in range(uncertainty_estimation_sampling_size):
    predictions.append(model(X_test, training=True))

predictions = np.stack(predictions, -1)
mean = np.mean(predictions, axis=-1)
std = - 1.0 * np.sum(mean * np.log(mean + 1e-15), axis=-1)

y_pred = mean
y_pred_std = std
pd.DataFrame(y_pred_std).to_csv(PREDICTIONS_PATH /
        "predictions_uncertainty.csv")

y_pred = np.argmax(y_pred, axis=-1)


