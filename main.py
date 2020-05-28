#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 13:30:15 2019

Author: David O'Callaghan
"""
import numpy as np

from format_hazelnut_data import format_hazelnut_data
from logistic_regression import LogisticRegression
from classifier_utils import train_test_split
from classifier_utils import ZNormaliser

# For plotting
import matplotlib.pyplot as plt
from plot_confusion_matrix import plot_confusion_matrix

# Reference Classifier
from sklearn.linear_model import LogisticRegression as LogisticRegressionSKLearn

if __name__ == "__main__":
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
    
    # Parameters for both custom and Sci-kit Learn models (from grid_search.py)
    learning_rate_ = 1.0
    tol_ = 1e-7
    max_iter_ = 500
    l2_strength_ = 0.1
    
    # Create a LogisticRegression object
    clf1 = LogisticRegression(learning_rate=learning_rate_, tol=tol_,
                             max_iter=max_iter_, l2_strength=l2_strength_,
                             verbose=True)
    clf2 = LogisticRegressionSKLearn(solver='lbfgs', tol=tol_, 
                                         penalty='l2', C=1/l2_strength_, 
                                         max_iter=max_iter_, multi_class='ovr')
    
    # Train the model on the training data
    clf1.fit(X_train, y_train)
    clf2.fit(X_train, y_train)
    
    # Record the accuracy of the predictions on the test data
    y_pred1 = clf1.predict(X_test)
    score1 = np.round(100*clf1.score(X_test, y_test), 4)
    print(f'Accuracy: {score1}%')
    
    y_pred2 = clf2.predict(X_test)
    score2 = np.round(100*clf2.score(X_test, y_test), 4)
    print(f'Accuracy: {score2}%')
    
    ##-----------------------------------------------------------------------##
    ##---------------------------- Plotting ---------------------------------##
    ##-----------------------------------------------------------------------##
    fig, axes = plt.subplots(nrows=1, ncols=3)
    for i in range(3):
        axes[i].plot(list(range(1,len(clf1.cost[i])+1)), clf1.cost[i], linewidth=3)
        axes[i].grid(True)
        axes[i].set_title(f'Gradient Descent - {clf1.target_mapping[i]}')
        axes[i].set_xlabel('Number of Iterations')
            
    axes[0].set_ylabel('Cost $J(w)$')   
    fig.set_figheight(5)
    fig.set_figwidth(15)
    plt.show()
    
    # Plot non-normalized confusion matrix
    class_names = np.unique(y)
    plot_confusion_matrix(y_test, y_pred1, classes=class_names,
                          title='Confusion matrix (Student Implementation)')
    plt.show()
    
    # Plot non-normalized confusion matrix
    class_names = np.unique(y)
    plot_confusion_matrix(y_test, y_pred2, classes=class_names,
                          title='Confusion matrix (Sci-kit Learn Implementation)')
    plt.show()
    ##-----------------------------------------------------------------------##
    ##-----------------------------------------------------------------------##
    
    scores = np.zeros((10, 2))
    for i in range(10):
        print(f'\n\n~~~ Train-test split: {i+1} ~~~\n')
        # Split the data into training and test
        X_train, X_test, y_train, y_test = train_test_split(X, y)
        zn = ZNormaliser()
        X_train = zn.fit_transform(X_train)
        X_test = zn.transform(X_test)
        
        ##-------------------------------------------------------------------##
        ##-------------- Custom Logistic Regression Classifier --------------##
        ##-------------------------------------------------------------------##
        print('Training Custom Classifier.....')
        # Create a LogisticRegression object
        clf1 = LogisticRegression(learning_rate=learning_rate_, tol=tol_, 
                                  max_iter=max_iter_, l2_strength=l2_strength_)
        
        # Train the model on the training data
        clf1.fit(X_train, y_train)
        
        # Record the accuracy of the predictions on the test data
        scores[i,0] = clf1.score(X_test, y_test)
        ##-------------------------------------------------------------------##
        ##-------------------------------------------------------------------##
        
        ##-------------------------------------------------------------------##
        ##-------------- SKLearn Logistic Regression Classifier -------------##
        ##-------------------------------------------------------------------##
        print('\nTraining Sci-Kit Learn Classifier.....')
        # Create a LogisticRegression object #liblinear #lbfgs
        clf2 = LogisticRegressionSKLearn(solver='lbfgs', tol=tol_, 
                                         penalty='l2', C=1/l2_strength_, 
                                         max_iter=max_iter_, multi_class='ovr')
        
        # Train the model on the training data
        clf2.fit(X_train, y_train)
        
        # Make predictions on the test data
        y_pred = clf2.predict(X_test)
        
        # Record the accuracy of the predictions
        scores[i,1] = clf2.score(X_test, y_test)
        ##-------------------------------------------------------------------##
        ##-------------------------------------------------------------------##
    
    # Print results
    mean_scores = np.mean(scores, axis=0)

    print('\n\n~~~~~~ Results for 10 train-test splits ~~~~~~')
    print('\tStudent LR\tSci-kit Learn LR')
    for score in np.round(100*scores, 3):
        print(f'\t{score[0]}%\t\t{score[1]}%')
    print(f'\nMean:\t{np.round(100*mean_scores[0], 4)}%\t\t{np.round(100*mean_scores[1], 4)}%')
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
