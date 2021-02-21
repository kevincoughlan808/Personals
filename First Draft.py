# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 19:15:23 2021

@author: Kevin
"""
##Libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
sns.set_palette('husl')

##EDA

dataset = pd.read_csv('C:/Users/Kevin/OneDrive/Documents/Coding/Personal Project/Project 1 - Iris/Data/iris.data', header = None)
dataset.columns = names = ["Sepal Length","Sepal Width","Petal Length","Petal Width","Iris Type"]

dataset.head()
dataset.describe()
dataset['Iris Type'].value_counts()

viz = sns.pairplot(dataset, hue='Iris Type', markers = '+')
plt.show()

g = sns.violinplot(y='Iris Type', x='Sepal Length', data=dataset, inner='quartile')
g = sns.violinplot(y='Iris Type', x='Sepal Width', data=dataset, inner='quartile')
g = sns.violinplot(y='Iris Type', x='Petal Length', data=dataset, inner='quartile')
g = sns.violinplot(y='Iris Type', x='Petal Width', data=dataset, inner='quartile')



## Data Preprocessing

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
print(X)
print(y)

# Encoding the Dependent Variable
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)
print(y)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)
print(X_train)
print(X_test)
print(y_train)
print(y_test)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train[:, 0:4] = sc.fit_transform(X_train[:, 0:4])
X_test[:, 0:4] = sc.transform(X_test[:, 0:4])
print(X_train)
print(X_test)

# Training Naive Bayes model on the training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Making the confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)

# Naive Bayes achieves an accuracy of 96.67%

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 5)
print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))

# Cross-validation yields a 94.17% accuracy with Stndard deviation of 3.33%











