import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, recall_score, precision_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pickle
import logging

# Configure logger
## Your code here
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
h = logging.StreamHandler()
f = logging.Formatter("%(levelname)s@%(asctime)s: %(message)s", datefmt="%H:%M:%S")
h.setFormatter(f)
logger.addHandler(h)

# Read in test data
## Your code here
# df = pd.read_csv("framingham.csv", index_col=False)
df = pd.read_csv("test.csv", index_col=0)
Y_test = df.values[:, -1]  # Getting value TenYearCHD
X_test = df.values[:, :-1]  # Getting the values of other columns

# Read in model
file = open("classifier.pkl", "rb")
trained_pickle = pickle.load(file)
logreg = trained_pickle["logreg"]  # Retrieve the model saved as logreg
scaler = trained_pickle["scaler"]  # Retrieve the scaler

# Scale X
## Your code here
# transform the test dataset
X_test_scaled = scaler.transform(X_test)

# Fit X_test_scaled with the model already saved to pickle
logreg.fit(X_test_scaled, Y_test)

# Check accuracy on test data
## Your code here
test_accuracy = logreg.score(X_test_scaled, Y_test)
print(f"Test Accuracy = {test_accuracy}")


# Show confusion matrix
## Your code here
y_pred = logreg.predict(X_test_scaled)

# Generate the confusion matrix
cm = confusion_matrix(Y_test, y_pred)
print(f"Confusion matrix = \n{cm}")


# Try a bunch of thresholds
threshold = 0.0
best_f1 = -1.0
thresholds = []
recall_scores = []
precision_scores = []
f1_scores = []
best_threshold = 0
while threshold <= 1.0:
    thresholds.append(threshold)
    ## Your code here
    y_pred_new_threshold = (
        logreg.predict_proba(X_test_scaled)[:, 1] >= threshold
    ).astype(int)
    precision = precision_score(Y_test, y_pred_new_threshold)
    recall = recall_score(Y_test, y_pred_new_threshold)
    # checking if the denominator will be 0 before deviding to find the F1 to avoid 'nan' output
    numerator = 2 * (precision * recall)
    denominator = precision + recall
    if denominator > 0:
        f1 = numerator / denominator
    else:
        f1 = numerator / 1
    # f1 = (2 * (precision * recall)) / (precision + recall )

    # accuracy = accuracy_score(Y_test, y_pred_new_threshold)
    # accuracy = logreg.score(X_test.reshape(-1, 1), y_pred_new_threshold)

    # Generate the confusion matrix
    # Using the confusion matrix to find accuracy
    cm = confusion_matrix(Y_test, y_pred_new_threshold)
    true_negative = cm[1][1]
    false_positive = cm[0][1]
    false_negative = cm[1][0]
    true_positive = cm[0][0]

    sum_all_elements_of_confusion_matrix = np.concatenate(cm).sum()

    accuracy = (true_positive + true_negative) / sum_all_elements_of_confusion_matrix

    recall_scores.append(recall)
    precision_scores.append(precision)
    f1_scores.append(f1)
    # getting the best F1
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold
    else:
        best_f1 = best_f1
        best_threshold

    logger.info(
        f"Threshold={threshold:.3f} Accuracy={accuracy:.3f} Recall={recall:.2f} Precision={precision:.2f} F1 = {f1:.3f}"
    )
    threshold += 0.02

# Make a confusion matrix for the best threshold
## Your code here
print(f"best_f1_score = {best_f1:.3f}")
print(f"best_threshold = {best_threshold:.3f}")
# Make a confusion matrix for the best threshold
## Your code here
y_pred_best_threshold = (
    logreg.predict_proba(X_test_scaled)[:, 1] >= best_threshold
).astype(int)

cm = confusion_matrix(Y_test, y_pred_best_threshold)

print(f"Confusion matrix Using Best Threshold = \n{cm}")

# Plot recall, precision, and F1 vs Threshold
fig, ax = plt.subplots()
ax.plot(thresholds, recall_scores, "b", label="Recall")
ax.plot(thresholds, precision_scores, "g", label="Precision", color="g")
ax.plot(thresholds, f1_scores, "r", label="F1", color="r")
ax.vlines(best_threshold, 0, 1, "r", linewidth=0.5, linestyle="dashed")
ax.set_xlabel("Threshold")
ax.legend()
fig.savefig("threshold.png")
