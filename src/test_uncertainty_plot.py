import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Sample data arrays (Replace these with your actual data)
feature_sets = ['Arm, trunk, thigh, calf, hip', 'Arm, trunk, thigh, calf', 'Arm, trunk, thigh', 'Arm, trunk', 'Arm']
num_samples = 100
y_true = np.random.randint(0, 8, size=num_samples)
y_predicted = np.random.randint(0, 8, size=num_samples)
y_predicted_std = np.random.rand(num_samples)
feature_set = np.random.choice(feature_sets, num_samples)

# Mapping posture integers to class names
class_names = ['Lie', 'Kneel', 'Sit', 'Stand', 'Other', 'Walk', 'Run', 'Stairs']

# Create a DataFrame for easier manipulation and plotting
data = {
    'FeatureSet': feature_set,
    'True': [class_names[i] for i in y_true],
    'Predicted': [class_names[i] for i in y_predicted],
    'Uncertainty': y_predicted_std,
    'Correct': y_true == y_predicted
}

df = pd.DataFrame(data)

# Ensure the 'True' column is treated as categorical with the correct order
df['True'] = pd.Categorical(df['True'], categories=class_names, ordered=True)

# Group and calculate mean uncertainty
grouped = df.groupby(['FeatureSet', 'True', 'Correct'])['Uncertainty'].mean().reset_index()

# Pivot the data for better plotting
pivot_correct = grouped[grouped['Correct'] == True].pivot(index='True', columns='FeatureSet', values='Uncertainty')
pivot_incorrect = grouped[grouped['Correct'] == False].pivot(index='True', columns='FeatureSet', values='Uncertainty')

# Plot the results
fig, axes = plt.subplots(2, 1, figsize=(15, 12), sharex=True)

sns.heatmap(pivot_correct, annot=True, cmap='viridis', ax=axes[0], cbar_kws={'label': 'Uncertainty'})
axes[0].set_title('Average Uncertainty for Correct Predictions')
axes[0].set_xlabel('Feature Set')
axes[0].set_ylabel('Class')
axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45)

sns.heatmap(pivot_incorrect, annot=True, cmap='viridis', ax=axes[1], cbar_kws={'label': 'Uncertainty'})
axes[1].set_title('Average Uncertainty for Misclassifications')
axes[1].set_xlabel('Feature Set')
axes[1].set_ylabel('Class')
axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45)

plt.tight_layout()
plt.show()

