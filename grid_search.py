#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 13:35:49 2019

Author: David O'Callaghan
"""
import itertools
import numpy as np

from sklearn.model_selection import KFold

from format_hazelnut_data import format_hazelnut_data
from logistic_regression import LogisticRegression
from classifier_utils import train_test_split
from classifier_utils import ZNormaliser

# Parameter space for grid search (27 permutations)
learning_rates = [0.01, 0.1, 1.0]
tols = [1e-7, 1e-6, 1e-5]
l2_strengths = [0.1, 1.0, 10]

# Load the data into a dataframe
hazelnut_data = format_hazelnut_data("hazelnuts.txt")

# Create feature matrix and target vector
X, y = hazelnut_data.iloc[:,:-1].values, hazelnut_data.iloc[:,-1].values

# Split the data into training and test
X_train, X_test, y_train, y_test = train_test_split(X, y)

# Z Normalise the data
zn = ZNormaliser()
X_train = zn.fit_transform(X_train)
X_test = zn.transform(X_test)

# 5 fold cross-validation
cv = KFold(n_splits=5)

results = []
i = 1
# For each permutation of the parameters
for learning_rate, tol, l2_strength in itertools.product(learning_rates, tols, l2_strengths):
    print(f'{i} of 27 permutations...')
    cv = KFold(n_splits=5)
    clf = LogisticRegression(learning_rate=learning_rate, tol=tol, l2_strength=l2_strength, max_iter=500)
    
    scores = [] # to store score from each split
    
    # Compute cross validation score
    for train_index, validation_index in cv.split(X_train):
        #train
        clf.fit(X_train[train_index, :], y_train[train_index])
        #score
        scores.append(clf.score(X_train[validation_index, :], y_train[validation_index]))
    
    # Store the results
    cv_score = np.mean(scores)
    params = {'learning_rate': learning_rate,
              'tol': tol,
              'l2_strength': l2_strength,
              'cv_score': cv_score}
    
    results.append(params)
    i += 1

# Print the parameters that gave the best CV score
best_score = 0
for result in results:
    if result['cv_score'] > best_score:
        best_score = result['cv_score']
        best_params = result
print('\nTuned parameters and Cross-Validation score:')    
print(best_params)
