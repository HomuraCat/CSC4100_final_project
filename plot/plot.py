import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read CSV data from file
csv_file = "../Qwen2.5-7b/model/results/pronoun_probabilities.csv"
try:
    df = pd.read_csv(csv_file)
except FileNotFoundError:
    print(f"Error: File '{csv_file}' not found. Please provide the correct file path.")
    exit()

# Calculate mean, standard deviation, and quartiles
means = [df['male_prob'].mean(), df['female_prob'].mean()]
stds = [df['male_prob'].std(), df['female_prob'].std()]
male_quartiles = [np.percentile(df['male_prob'], 25), np.percentile(df['male_prob'], 50), np.percentile(df['male_prob'], 75)]
female_quartiles = [np.percentile(df['female_prob'], 25), np.percentile(df['female_prob'], 50), np.percentile(df['female_prob'], 75)]
categories = ['Male', 'Female']

# Create bar chart with error bars
plt.figure(figsize=(8, 6))
bars = plt.bar(categories, means, yerr=stds, capsize=5, color=['skyblue', 'lightcoral'], edgecolor='black')
plt.xlabel('Gender')
plt.ylabel('Probability')
#plt.title('Distribution of Male and Female Probabilities after fine-tuning Qwen-7B')
plt.title('Distribution of Male and Female Probabilities in Qwen-7B')
plt.ylim(0, 1.0)  # Set y-axis limit for better visualization
plt.grid(True, axis='y', linestyle='--', alpha=0.7)

# Add quartile markers (dots) on each bar
for i, (bar, quartiles) in enumerate(zip(bars, [male_quartiles, female_quartiles])):
    x = bar.get_x() + bar.get_width() / 2  # Center of the bar
    # Plot 25%, 50%, 75% quartiles as dots
    #plt.plot(x, quartiles[0], marker='o', color='blue', markersize=8, label='25th Percentile' if i == 0 else "")
    plt.plot(x, quartiles[1], marker='o', color='green', markersize=8, label='50th Percentile (Median)' if i == 0 else "")
    #plt.plot(x, quartiles[2], marker='o', color='red', markersize=8, label='75th Percentile' if i == 0 else "")
    # Add mean label on top of the bar
    plt.text(x, means[i] + 0.05, f'Mean: {means[i]:.3f}', ha='center', va='bottom', fontsize=10)

# Add legend outside the plot on the right
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

# Adjust layout to prevent clipping of legend
plt.tight_layout()

# Save the figure
plt.savefig('barchart_with_quartiles_and_mean.png', dpi=300, bbox_inches='tight')
plt.close()

# Print statistics for reference
print(f"Male Probability: Mean = {means[0]:.4f}, Std = {stds[0]:.4f}, Q1 = {male_quartiles[0]:.4f}, Median = {male_quartiles[1]:.4f}, Q3 = {male_quartiles[2]:.4f}")
print(f"Female Probability: Mean = {means[1]:.4f}, Std = {stds[1]:.4f}, Q1 = {female_quartiles[0]:.4f}, Median = {female_quartiles[1]:.4f}, Q3 = {female_quartiles[2]:.4f}")
print("Figure saved as 'barchart_with_quartiles_and_mean.png'")