# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import the dataset
data = pd.read_csv("Cancer_Data.csv")

# Data Editing: Remove unnecessary columns
data.drop(["id", "Unnamed: 32"], axis=1, inplace=True)

# Data Visualization
M = data[data.diagnosis == "M"]  # Malignant tumors
B = data[data.diagnosis == "B"]  # Benign tumors

# Create a scatter plot to visualize the data
plt.scatter(M.radius_mean, M.texture_mean, color="red", label="Malignant", alpha=0.3)
plt.scatter(B.radius_mean, B.texture_mean, color="green", label="Benign", alpha=0.3)
plt.xlabel("radius_mean")
plt.ylabel("texture_mean")
plt.legend()
plt.show()

# Convert the diagnosis column to binary values (1 for M, 0 for B)
data.diagnosis = [1 if each == "M" else 0 for each in data.diagnosis]
y = data.diagnosis.values
x_data = data.drop(["diagnosis"], axis=1)

# Data Normalization
x = (x_data - np.min(x_data)) / (np.max(x_data) - np.min(x_data))

# Train-Test Split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)

# Support Vector Machine (SVM)
from sklearn.svm import SVC
svm = SVC(random_state=1)
svm.fit(x_train, y_train)

# Calculate and print the accuracy of the SVM algorithm on the test data
print("Accuracy of the SVM algorithm:", svm.score(x_test, y_test))
