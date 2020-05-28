#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 13:36:53 2019

Author: David O'Callaghan
"""

import numpy as np

class ZNormaliser:
    """
    This class is for the Z normalisation
    of a numpy array X with an attribute in
    each column and a sample in each row.
    """
    
    def fit(self, X):
        """
        Compute the mean and standard
        deviation in for each attribute
        in X.
        """
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        
    def transform(self, X):
        """
        Z normalises X using the mean and
        standard deviation found in the fit()
        method.
        """
        return (X - self.mean) / self.std
    def fit_transform(self, X):
        """
        For convenience, this method calls
        the fit() method on X and then returns
        the array from the transform method.
        """
        self.fit(X)
        return self.transform(X)
    
def train_test_split(X, y, split=(1/3)):
    """
    This function splits the feature matrix X
    and the target vector y into training and 
    test data. The fraction provided in 'split'
    is used to compute the number of test samples.
    The data is shuffled before being split.
    """
    n = np.size(X, axis=0)
    n_test = round(split * n) # Compute the number of samples for testing
    n_train = n - n_test # Compute the number of samples for training

    indices = np.arange(n) # Create an array for the indices
    np.random.shuffle(indices) # Shuffle the array to create a random split
    train_indices = indices[:n_train]
    test_indices = indices[n_train:]

    # Assign the training and test data
    X_train = X[train_indices, :]
    X_test = X[test_indices, :]
    y_train = y[train_indices]
    y_test = y[test_indices]

    return X_train, X_test, y_train, y_test     
