# Lab 10 - Support Vector Machines (SVM)
# ARTI308 - Machine Learning

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

iris = pd.read_csv('iris.csv')
print(iris.head())
print(iris['species'].value_counts())

# Pairplot
sns.pairplot(iris, hue='species')
plt.show()

# KDE plot for Setosa
setosa = iris[iris['species'] == 'setosa']
sns.kdeplot(data=setosa, x='sepal_length', y='sepal_width', fill=True)
plt.title('Setosa: Sepal Length and Width')
plt.show()

# Train and test
from sklearn.model_selection import train_test_split

X = iris.drop('species', axis=1)
y = iris['species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)
print('Training data:', len(X_train))
print('Testing data:', len(X_test))

# SVM model
from sklearn.svm import SVC
model = SVC()
model.fit(X_train, y_train)

# Evaluation
from sklearn.metrics import confusion_matrix, classification_report
predictions = model.predict(X_test)
print('Confusion Matrix:')
print(confusion_matrix(y_test, predictions))
print('Classification Report:')
print(classification_report(y_test, predictions))

# GridSearchCV
from sklearn.model_selection import GridSearchCV
param_grid = {'C': [0.1, 1], 'gamma': [0.1, 0.01], 'kernel': ['rbf']}
grid = GridSearchCV(SVC(), param_grid, cv=2)
grid.fit(X_train, y_train)
print('Best Parameters:', grid.best_params_)

# Grid evaluation
grid_predictions = grid.predict(X_test)
print('Grid Confusion Matrix:')
print(confusion_matrix(y_test, grid_predictions))
print('Grid Classification Report:')
print(classification_report(y_test, grid_predictions))
