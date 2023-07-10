import pandas as pd
from sklearn.linear_model import LogisticRegression
import pickle
from sklearn.preprocessing import StandardScaler

# Read the training data
print("Reading input...")
## Your code here
df = pd.read_csv("train.csv", index_col=0)

Y = df.values[:, -1]  # Getting value TenYearCHD
X = df.values[:, :-1]  # Getting the values of other columns

print("Scaling...")
## Your code here
# Standardize the X column.
scaler = StandardScaler().fit(X)
X_scaled = scaler.transform(X)

print("Fitting...")
## Your code here
logreg = LogisticRegression()
logreg.fit(X_scaled, Y)

# Get the accuracy on the training data
## Your code here
train_accuracy = logreg.score(X_scaled, Y)
print(f"Training accuracy = {train_accuracy}")

# Write out the scaler and logisticregression objects into a pickle file
pickle_path = "classifier.pkl"
print(f"Writing scaling and logistic regression model to {pickle_path}...")
## Your code here
scaler_model = {}  # Creating a dictionary to hold the scaler and the model(logreg)
scaler_model["logreg"] = logreg
scaler_model["scaler"] = scaler
pickle.dump(scaler_model, open(pickle_path, "wb"))  # Save the model to pickle file
