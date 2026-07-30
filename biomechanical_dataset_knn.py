# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 16:18:23 2026

@author: osemi
"""

#%% Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#%% Creating the dataframe
df = pd.read_csv("column_2C_weka.csv")

#%% Class Rename
A = df[df['class'] == "Abnormal"]
N = df[df['class'] == "Normal"]

#%% Scatter Plot
plt.figure(figsize=(8,5))
plt.scatter(A.pelvic_radius, A.degree_spondylolisthesis, color = "red", label = "Abnormal", alpha = 0.3)
plt.scatter(N.pelvic_radius, N.degree_spondylolisthesis, color = "green", label = "Normal", alpha = 0.3)
plt.xlabel("Pelvic Radius")
plt.ylabel("Degree Spondylolisthesis")
plt.legend()
plt.title("Pelvic Radius & Spondylolisthesis")
plt.show() 

#%% Most important thing is normalization for KNN.
df['class'] = [1 if each == 'Abnormal' else 0 for each in df['class']]
y = df['class'].values
X_data = df.drop(['class'], axis = 1)

X = (X_data - np.min(X_data)) / (np.max(X_data) - np.min(X_data))

#%% Train / Test Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)

#%% KNN Algorithm
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, y_train)
prediction = knn.predict(X_test)
print("{} nn score: {}".format(3, knn.score(X_test, y_test)))

#%% Finding the optimal k value
score_list = []
for each in range(1, 15):
    knn2 = KNeighborsClassifier(n_neighbors = each)
    knn2.fit(X_train, y_train)
    score_list.append(knn2.score(X_test, y_test))
plt.figure(figsize=(8, 5))
plt.clf()
plt.plot(range(1, 15), score_list, color="blue", marker="o")
plt.xlabel("K Values")
plt.ylabel("Accuracy")
plt.title("KNN Accuracy vs K Value")
plt.grid(True)
plt.show()

#%% Confusion Matrix & Classification Report (Selected K = 9)
from sklearn.metrics import confusion_matrix, classification_report

knn_best = KNeighborsClassifier(n_neighbors=9)
knn_best.fit(X_train, y_train)
y_pred = knn_best.predict(X_test)

# Confusion Matrix Plot
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 5))
plt.clf()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
            xticklabels=["Normal (0)", "Abnormal (1)"], 
            yticklabels=["Normal (0)", "Abnormal (1)"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("KNN (K=9) Confusion Matrix")
plt.show()

# Detailed Classification Report
print("\n--- Classification Report (K=9) ---")
print(classification_report(y_test, y_pred, target_names=["Normal (0)", "Abnormal (1)"]))