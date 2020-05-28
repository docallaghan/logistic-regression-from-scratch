----------------------------------------------------------
                CT4101 Machine Learning
                      Assignment 2
                
            Student Name: David O'Callaghan
                  Student ID: 19233706
                
                   November 10, 2019
-----------------------------------------------------------

This folder contains the Python code for the implementation
and evaluation of the Logistic Regression (LR) algorithm 
for classifying samples in the 'Hazelnuts' dataset.

Files:
main.py (THE SCRIPT TO RUN) -
  This is the script written to test and evaluate the LR
  implementation. It makes use of all other files below
  except for grid_search.py
  Run command: <path to python>/python3 hazelnuts.py

logistic_regression.py (THE LR IMPLEMENTATION) -
  This contains the implementation of the LR algorithm in
  a Python class called LogisticRegression

classifier_utils.py - 
  This file contains APIs for normalising data and
  splitting data into training and test sets

format_hazelnut_data.py
  This file contains a function to format the data in
  hazelnuts.txt into a pandas dataframe

hazelnuts.txt
  This is the data-set provided for the assignment

plot_confusion_matrix.py
  This contains code from https://scikit-learn.org/ to
  generate a plot for a confusion matrix

grid_search.py
  This is another script that was kept separate since it
  takes a while to run (about 5 minutes). It performs a
  grid search on a set of parameters for 
  LogisticRegression and finds the ones that give the
  highest cross-validation score
  Run command: <path to python>/python3 grid_search.py
