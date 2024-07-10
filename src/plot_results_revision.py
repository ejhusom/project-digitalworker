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

import os


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
    df.plot(kind="bar", figsize=(8, 4), colormap='viridis')
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
    fig, ax1 = plt.subplots(figsize=(7, 5))

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

def generate_uncertainty_plots(y_true, y_predicted, y_predicted_std):
    class_names = [
            "Lie",
            "Kneel",
            "Sit",
            "Stand",
            "Other",
            "Walk",
            "Run",
            "Stairs"
    ]
    # Create a DataFrame for easier manipulation and plotting
    data = {
        'True': [class_names[i] for i in y_true],
        'Predicted': [class_names[i] for i in y_predicted],
        'Uncertainty': y_predicted_std,
        'Correct': y_true == y_predicted
    }

    df = pd.DataFrame(data)

    # Plot 1: Uncertainty Distribution Plot
    plt.figure(figsize=(12, 6))
    sn.boxplot(x='True', y='Uncertainty', data=df)
    plt.title('Uncertainty Distribution for Each Class')
    plt.xlabel('Class')
    plt.ylabel('Uncertainty')
    plt.xticks(rotation=45)
    plt.show()

    # Plot 2: Uncertainty vs. Accuracy Scatter Plot
    plt.figure(figsize=(12, 6))
    sn.scatterplot(x='True', y='Uncertainty', hue='Correct', data=df, palette={True: 'green', False: 'red'})
    plt.title('Uncertainty vs. Accuracy')
    plt.xlabel('Class')
    plt.ylabel('Uncertainty')
    plt.legend(title='Correct Prediction')
    plt.xticks(rotation=45)
    plt.show()

    # Plot 3: Uncertainty Heatmap
    uncertainty_mean = df.groupby('True')['Uncertainty'].mean().reset_index()
    uncertainty_mean = uncertainty_mean.set_index('True').T  # Transpose for heatmap
    
    plt.figure(figsize=(10, 6))
    sn.heatmap(uncertainty_mean, annot=True, cmap='viridis', cbar_kws={'label': 'Uncertainty'})
    plt.title('Average Uncertainty for Each Class')
    plt.xlabel('Class')
    plt.ylabel('Average Uncertainty')
    plt.xticks(rotation=45)
    plt.show()


def uncertainty_detailed():
    y_test_file = "assets/predictions/true_values.csv"
    y_pred_file = "assets/predictions/predictions.csv"
    y_pred_std_file = "assets/predictions/predictions_uncertainty.csv"

    # Read true values
    y_test = pd.read_csv(y_test_file, index_col=0).to_numpy()
    # Convert from one-hot encoding back to classes
    y_test = np.argmax(y_test, axis=-1)

    # Read predictions
    y_pred = pd.read_csv(y_pred_file).to_numpy().flatten()
    # print(y_pred)

    # Read uncertainty
    y_pred_std = pd.read_csv(y_pred_std_file, index_col=0).to_numpy()
    y_pred_std = np.mean(y_pred_std, axis=1)

    confusion = confusion_matrix(
            y_test, 
            y_pred, 
            normalize="true",
    )

    accuracies = confusion.diagonal().round(2).tolist()

    print("Accuracies:")
    print(" & ".join(map(str, accuracies)))

    int_labels = unique_labels(y_test, y_pred)
    n_labels = int_labels.size
    print(n_labels)
    print(y_test.shape)
    print(y_pred.shape)
    print(y_pred_std.shape)

    generate_uncertainty_plots(y_test, y_pred, y_pred_std)


def uncertainty_final():

    # Define the directories and file names
    directories = ['arm', 'arm_trunk', 'arm_trunk_thigh', 'arm_trunk_thigh_calf', 'arm_trunk_thigh_calf_hip']
    feature_sets = ['Arm', 'Arm, trunk', 'Arm, trunk, thigh', 'Arm, trunk, thigh, calf', 'Arm, trunk, thigh, calf, hip']

    for i in range(len(directories)):
        directories[i] = "assets/results/" + directories[i]

    # Mapping posture integers to class names
    class_names = ['Lie', 'Kneel', 'Sit', 'Stand', 'Other', 'Walk', 'Run', 'Stairs']

    # Initialize a list to collect all data
    all_data = []

    # Read data from each directory
    for feature_set, directory in zip(feature_sets, directories):
        y_test_file = os.path.join(directory, 'true_values.csv')
        y_pred_file = os.path.join(directory, 'predictions.csv')
        y_pred_std_file = os.path.join(directory, 'predictions_uncertainty.csv')

        # Read true values
        y_test = pd.read_csv(y_test_file, index_col=0).to_numpy()
        true_values = np.argmax(y_test, axis=-1)

        # Read predictions
        y_pred = pd.read_csv(y_pred_file).to_numpy().flatten()

        # Read uncertainty
        y_pred_std = pd.read_csv(y_pred_std_file, index_col=0).to_numpy()
        y_pred_std = np.mean(y_pred_std, axis=1)

        # Create a DataFrame for easier manipulation and plotting
        data = {
            'FeatureSet': feature_set,
            'True': [class_names[i] for i in true_values],
            'Predicted': [class_names[i] for i in y_pred],
            'Uncertainty': y_pred_std,
            'Correct': true_values == y_pred
        }
        df = pd.DataFrame(data)
        all_data.append(df)

    # Concatenate all data into a single DataFrame
    df_all = pd.concat(all_data)

    # Ensure the 'True' column is treated as categorical with the correct order
    df_all['True'] = pd.Categorical(df_all['True'], categories=class_names, ordered=True)

    # Group and calculate mean uncertainty
    grouped = df_all.groupby(['FeatureSet', 'True', 'Correct'])['Uncertainty'].mean().reset_index()

    # Pivot the data for better plotting
    pivot_correct = grouped[grouped['Correct'] == True].pivot(index='True', columns='FeatureSet', values='Uncertainty')
    pivot_incorrect = grouped[grouped['Correct'] == False].pivot(index='True', columns='FeatureSet', values='Uncertainty')

    # Plot the results
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 4.5))#, sharey=True)

    sn.heatmap(pivot_correct, annot=True, cmap='Reds', ax=axes[0], cbar_kws={'label': 'Uncertainty'})
    axes[0].set_title('Average Uncertainty for Correct Predictions')
    axes[0].set_xlabel('Feature Set')
    axes[0].set_ylabel('Class')
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45, ha='right')
    # axes[0].set_yticklabels(axes[0].get_yticklabels(), rotation=45, va='top')
    axes[0].set_yticklabels(axes[0].get_yticklabels(), rotation=0)

    sn.heatmap(pivot_incorrect, annot=True, cmap='Reds', ax=axes[1], cbar_kws={'label': 'Uncertainty'})
    axes[1].set_title('Average Uncertainty for Misclassifications')
    axes[1].set_xlabel('Feature Set')
    axes[1].set_ylabel('Class')
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45, ha='right')
    # axes[1].set_yticklabels(axes[1].get_yticklabels(), rotation=45, va='top')
    axes[1].set_yticklabels(axes[1].get_yticklabels(), rotation=0)

    plt.tight_layout()
    plt.savefig(PLOTS_PATH / "uncertainty_heatmap_all_results.pdf")
    # plt.show()

    # Pivot the data for better printing
    pivot_correct = grouped[grouped['Correct'] == True].pivot(index='True', columns='FeatureSet', values='Uncertainty')
    pivot_incorrect = grouped[grouped['Correct'] == False].pivot(index='True', columns='FeatureSet', values='Uncertainty')

    # Print the results as tables
    print("Average Uncertainty for Correct Predictions")
    print(pivot_correct)

    print("\nAverage Uncertainty for Misclassifications")
    print(pivot_incorrect)


if __name__ == '__main__': 
    # Testing functions:
    # uncertainty_detailed()
    # plot_results_2()

    # Final functions:
    # accuracy_across_classes()
    # uncertainty()
    uncertainty_final()
