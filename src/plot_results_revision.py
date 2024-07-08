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

from config import OUTPUT_FEATURES_PATH, PLOTS_PATH

def plot_results_2():

    # Assuming the results are stored in CSV format or similar
    results = {
        "sensors": ["arm", "arm-trunk", "arm-trunk-thigh", "arm-trunk-thigh-calf", "arm-trunk-thigh-calf-hip"],
        "accuracy": [0.726, 0.801, 0.853, 0.889, 0.92],
        "precision": [0.699, 0.735, 0.781, 0.813, 0.84],
        "recall": [0.727, 0.801, 0.853, 0.889, 0.92],
        "f1": [0.700, 0.739, 0.787, 0.821, 0.85],
        "uncertainty": [1.22, 1.10, 0.95, 0.80, 0.67]
    }

    df = pd.DataFrame(results)

    # Plot accuracy, precision, recall, and F1-score
    metrics = ["accuracy", "precision", "recall", "f1"]
    df.plot(x="sensors", y=metrics, kind="bar", figsize=(10, 6))
    plt.title("Model Performance Metrics Across Different Sensor Configurations")
    plt.ylabel("Score")
    plt.xlabel("Sensor Configuration")
    plt.legend(title="Metrics")
    plt.show()

    # Plot uncertainty
    plt.figure(figsize=(10, 6))
    plt.plot(df["sensors"], df["uncertainty"], marker='o', linestyle='-', color='r')
    plt.title("Uncertainty Across Different Sensor Configurations")
    plt.ylabel("Uncertainty")
    plt.xlabel("Sensor Configuration")
    plt.show()

    # Box plot for uncertainty
    uncertainties = {
        "correct_classifications": [1.005, 0.95, 0.87, 0.80, 0.67],
        "misclassifications": [1.22, 1.15, 1.10, 1.05, 1.00]
    }
    df_uncertainty = pd.DataFrame(uncertainties)

    df_uncertainty.plot(kind='box', figsize=(8, 6))
    plt.title("Uncertainty Distribution for Correct Classifications vs. Misclassifications")
    plt.ylabel("Uncertainty")
    plt.xlabel("Classification Type")
    plt.xticks([1, 2], ["Correct", "Misclassifications"])
    plt.show()

def accuracy_across_classes():
    # Data for accuracies of different classes
    data = {
        "Feature set": ["Arm, trunk, thigh, calf, hip", "Arm, trunk, thigh, calf", "Arm, trunk, thigh", "Arm, trunk", "Arm"],
        "Lie": [0.99, 0.99, 0.99, 0.98, 0.96],
        "Kneel": [0.92, 0.86, 0.00, 0.00, 0.00],
        "Sit": [0.98, 0.98, 0.99, 0.93, 0.83],
        "Stand": [0.89, 0.91, 0.84, 0.47, 0.40],
        "Other": [0.51, 0.43, 0.45, 0.22, 0.16],
        "Walk": [0.76, 0.71, 0.78, 0.26, 0.15],
        "Run": [0.73, 0.63, 0.69, 0.54, 0.50],
        "Stairs": [0.24, 0.15, 0.28, 0.00, 0.00]
    }

    # Convert the data into a DataFrame
    df = pd.DataFrame(data)

    # Set the 'Feature set' as the index
    df.set_index('Feature set', inplace=True)

    # Reverse the order of the feature sets
    df = df.iloc[::-1]

    # Plotting the data
    df.plot(kind="bar", figsize=(8, 6), colormap='viridis')
    # plt.title("Accuracy of Different Classes Across Sensor Configurations")
    plt.xlabel("Feature set")
    plt.ylabel("Accuracy")
    plt.legend(title="Posture classes", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    plt.savefig(PLOTS_PATH / "accuracy_across_classes.pdf")

    # Show the plot
    plt.show()

def uncertainty():
    # Data
    data = {
        'Set': [
            'Arm, trunk, thigh, calf, hip',
            'Arm, trunk, thigh, calf',
            'Arm, trunk, thigh',
            'Arm, trunk',
            'Arm'
        ],
        'Accuracy': [0.92, 0.92, 0.91, 0.79, 0.73],
        'Uncertainty (correct)': [6.44, 7.19, 6.08, 9.61, 11.80],
        'Uncertainty (misclassifications)': [11.55, 12.65, 11.33, 13.08, 14.34]
    }

    # Creating DataFrame
    df = pd.DataFrame(data)

    # Reversing the order for plotting
    df = df.iloc[::-1]

    # Plot
    fig, ax1 = plt.subplots(figsize=(7, 6))

    # Bar plot for accuracy
    bars = ax1.bar(df['Set'], df['Accuracy'], color='skyblue', label='Accuracy')
    ax1.set_ylabel('Accuracy')
    ax1.set_xlabel('Feature Set')
    ax1.set_title('Accuracy and Uncertainty for Different Feature Sets')
    ax1.set_ylim(0, 1)

    # Line plot for uncertainty
    ax2 = ax1.twinx()
    ax2.plot(df['Set'], df['Uncertainty (correct)'], color='green', marker='o', label='Uncertainty (correct)')
    ax2.plot(df['Set'], df['Uncertainty (misclassifications)'], color='red', marker='o', label='Uncertainty (misclassifications)')
    ax2.set_ylabel('Uncertainty')

    # Combining legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower left')

    # Rotate x-ticks for better readability
    ax1.set_xticklabels(df['Set'], rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig(PLOTS_PATH / "uncertainty.pdf")
    plt.show()


if __name__ == '__main__': 

#     y_pred_file = sys.argv[1]
#     y_test_file = sys.argv[2]

#     # Read predictions
#     y_pred = pd.read_csv(y_pred_file).to_numpy().flatten()

#     # Read true values
#     y_test = pd.read_csv(y_test_file, index_col=0).to_numpy()
#     # Convert from one-hot encoding back to classes
#     y_test = np.argmax(y_test, axis=-1)

#     plot_confusion(y_test, y_pred)

    # accuracy_across_classes()
    uncertainty()
    # plot_results_2()
