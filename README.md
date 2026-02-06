# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Read Placement_Data.csv, convert categorical columns (gender, status) into numeric values and select Gender and Salary as input features.
2. Divide data into training and testing sets and train the Logistic Regression model using scikit-learn.
3. Predict the output for test data and calculate accuracy score and confusion matrix.
4. Generate a heatmap of the confusion matrix to analyze model performance.

## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: Denesh Raj Balaji Rao
RegisterNumber:  25005647
*/
```
```
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load Dataset
data = pd.read_csv("Placement_Data.csv")

# ---- DATA PREPROCESSING ----

# Convert gender: M -> 1, F -> 0
data['gender'] = data['gender'].map({'M': 1, 'F': 0})

# Convert status to numeric: Placed -> 1, Not Placed -> 0
data['status'] = data['status'].map({'Placed': 1, 'Not Placed': 0})

# Ensure salary is numeric
data['salary'] = pd.to_numeric(data['salary'], errors='coerce')

# Fill missing salary with 0 (for not placed students)
data['salary'] = data['salary'].fillna(0)

# Selecting features and target
X = data[['gender', 'salary']].values.astype(float)
y = data['status'].values.astype(float)

print("Dataset Shape:", X.shape)


# ---- LOGISTIC REGRESSION USING GRADIENT DESCENT ----

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def logistic_regression_gd(X, y, lr, epochs):
    m, n = X.shape
    w = np.zeros(n)
    b = 0

    for i in range(epochs):
        z = np.dot(X, w) + b
        y_pred = sigmoid(z)

        dw = (1/m) * np.dot(X.T, (y_pred - y))
        db = (1/m) * np.sum(y_pred - y)

        w = w - lr * dw
        b = b - lr * db

    return w, b


learning_rate = 0.000001
epochs = 1000

w, b = logistic_regression_gd(X, y, learning_rate, epochs)

print("\n------ Gradient Descent Implementation ------")
print("Weights:", w)
print("Bias:", b)


# ---- LOGISTIC REGRESSION USING SCIKIT-LEARN ----

X1 = data[['gender', 'salary']]
y1 = data['status']

X_train, X_test, y_train, y_test = train_test_split(X1, y1, test_size=0.2)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("\n------ Scikit-Learn Implementation ------")
print("Accuracy:", accuracy)
print("Confusion Matrix:\n", cm)

plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")

plt.title("Confusion Matrix Heatmap")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.show()
```

## Output:
<img width="560" height="593" alt="image" src="https://github.com/user-attachments/assets/4f8e804f-c6e6-4f51-a90a-b96bc79d1e41" />



## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

