import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, mean_squared_error, roc_curve, auc
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Load the CSV file
file_path = 'output.csv'
data = pd.read_csv(file_path)

# Extract relevant columns
labels = data['Label']
probabilities = data['Probability']

# Encode labels as categorical integer values
label_categories, label_encoded = np.unique(labels, return_inverse=True)

# Calculate F1, Precision, Recall (macro and micro), MSE, and RMSE
threshold = 0.05
binary_predictions = (probabilities >= threshold).astype(int)
binary_labels = (label_encoded > 0).astype(int)

# Calculate metrics with zero_division parameter
precision_macro = precision_score(binary_labels, binary_predictions, average='macro', zero_division=0)
recall_macro = recall_score(binary_labels, binary_predictions, average='macro', zero_division=0)
f1_macro = f1_score(binary_labels, binary_predictions, average='macro', zero_division=0)

precision_micro = precision_score(binary_labels, binary_predictions, average='micro', zero_division=0)
recall_micro = recall_score(binary_labels, binary_predictions, average='micro', zero_division=0)
f1_micro = f1_score(binary_labels, binary_predictions, average='micro', zero_division=0)

mse = mean_squared_error(binary_labels, probabilities)
rmse = np.sqrt(mse)

# Print the calculated metrics
print(f"Precision (Macro): {precision_macro}")
print(f"Recall (Macro): {recall_macro}")
print(f"F1 Score (Macro): {f1_macro}")
print(f"Precision (Micro): {precision_micro}")
print(f"Recall (Micro): {recall_micro}")
print(f"F1 Score (Micro): {f1_micro}")
print(f"MSE: {mse}")
print(f"RMSE: {rmse}")

# Confusion Matrix
conf_matrix = confusion_matrix(binary_labels, binary_predictions)
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["Negative", "Positive"], yticklabels=["Negative", "Positive"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

# Bar Plot - Label Distribution
plt.figure(figsize=(10, 6))
label_counts = labels.value_counts()
label_counts.plot(kind="bar", color="skyblue")
plt.title("Label Distribution")
plt.xlabel("Label")
plt.ylabel("Count")
plt.xticks(rotation=45, ha="right")
plt.show()

# Pie Chart - Label Distribution
plt.figure(figsize=(8, 8))
label_counts.plot(kind="pie", autopct="%1.1f%%", startangle=140, colors=plt.cm.Paired.colors)
plt.title("Label Distribution")
plt.ylabel("")
plt.show()

# Probability Distribution
plt.figure(figsize=(10, 6))
plt.hist(probabilities, bins=20, color="lightgreen", edgecolor="black")
plt.title("Probability Distribution")
plt.xlabel("Probability")
plt.ylabel("Frequency")
plt.show()


fpr, tpr, thresholds = roc_curve(binary_labels, probabilities)
roc_auc = auc(fpr, tpr)


plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()
