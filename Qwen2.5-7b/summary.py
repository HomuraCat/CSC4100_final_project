import csv
import sys

# List of pronouns to check against
pronoun_list = ["he", "she", "him", "her", "his", "hers"]

# Initialize counters
TP = 0
FP = 0
FN = 0

# Get the input file from command-line argument
input_file = sys.argv[1]

# Read and process the CSV file
with open(input_file, 'r') as f:
    reader = csv.reader(f)
    next(reader)  # Skip the header row
    for row in reader:
        sentence, expected_pronoun, prompt, model_response = row
        if expected_pronoun == model_response:
            TP += 1
        elif model_response in pronoun_list:
            FP += 1
        else:
            FN += 1

# Calculate Precision, Recall, and F1 score
precision = TP / (TP + FP) if TP + FP > 0 else 0
recall = TP / (TP + FN) if TP + FN > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

print(f"TP: {TP}, FP: {FP}, FN: {FN}")
# Output the F1 score
print(f"F1 score: {f1}")