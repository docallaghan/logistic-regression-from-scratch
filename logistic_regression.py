#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 13:34:44 2019

Author: David O'Callaghan
"""
import numpy as np

class LogisticRegression:
    """
    The LogisticRegression class is written for a machine learning 
    classification problem. It is written to work with data in numpy arrays
    and can work for traditional binary classification problems or multiclass
    problems using the One Vs the Rest (OVR) scheme.
    """ 

    def __init__(self, learning_rate=0.1, tol=0.0001, max_iter=1000, 
                 l2_strength=10.0, rand=False, verbose=False):

        self.learning_rate = learning_rate  # alpha in the gradient descent formula
        self.tol = tol # The cost tolerance to stop gradient descent
        self.max_iter = max_iter # The max number of iterations in gradient descent
        self.l2_strength = l2_strength # The value for lambda in l2 regularisation
        self.rand = rand # A boolean to set the initial weights to random values
        self.verbose = verbose # A boolean to monitor the cost function

        self.initial_weights_set = False # Allow user to define own weights
        self.model_trained = False # To ensure the model is trained before predictions are made
        self.cost = [] # To store the cost during each iteration of gradient descent
    
    def __str__(self):
        return f'LogisticRegression(learning_rate={self.learning_rate}, \
tol={self.tol}, max_iter={self.max_iter}, l2_strength={self.l2_strength}, \
rand={self.rand}, verbose={self.verbose})'
        
    def fit(self, X, y):
        """
        Fit the model to the data in X with targets y.

        Inputs:  X - A 2-D numpy array containing data with a column for each
                 attribute and a row for each sample - The feature matrix
                 y - A 1-D numpy array containing the label for each sample
                 in X. y can be categorical.

        Returns: None
        """
        X = self.__insert_x0(X) # Add the x0 column

        self.num_samples_training, self.num_attributes = np.shape(X) 
        self.num_classes = np.size(np.unique(y), axis=0) # Compute the number of classes in y

        if not self.initial_weights_set:
            self.__initialise_weights() # Set initial values for weights

        if self.num_classes <= 2: # If a binary classification problem
            self.w = self.__update_weights(X, y, self.w)
        else: # Otherwise multiclass
            y_ovr = self.__get_ovr_targets(y) # Create OVR targets
            for index, target in enumerate(self.targets): # Fit weights for each target vector in the OVR scheme
                if self.verbose:
                    print(f'~~~~~~~~~OVR: {self.target_mapping[index]}~~~~~~~~~')
                self.w[:,index] = self.__update_weights(X, y_ovr[:,index], self.w[:,index])
                
        self.model_trained = True

    def predict_proba(self, X):
        """
        This function computes the probabilities for each sample in the 
        feature matrix X.

        Inputs:  X - 2-D numpy array in the form of a feature matrix to make
                 predictions on

        Returns: y_prob - A numpy array containing the predicted probabilities
                 Size 1-D if a binary classification problem. If multiclass it
                 is has self.num_classes number of columns.
        """
        if not self.model_trained: # Ensure the model is trained
            print('Model needs to be trained first. Call the fit() method')
            return

        X = self.__insert_x0(X) # Insert column of ones for bias / intercept

        self.num_samples_testing = np.size(X, axis=0)
        if self.num_classes <= 2:
            z = self.__net_input(X, self.w)
            y_prob = self.__logistic(z)
        else: # If multiclass
            y_prob = np.zeros((self.num_samples_testing, self.num_classes))
            for index, target in enumerate(self.targets): # For each class
                z = self.__net_input(X, self.w[:,index]) 
                y_prob[:,index] = self.__logistic(z)
        return y_prob
    
    def predict(self, X, threshold=0.5):
        """
        For binary classification, this function computes the predicted class
        of each sample in X based on the threshold. If a multiclass problem
        the class with the highest predicted probabilty is taken as the
        predicted class.

        Inputs:  X - 2-D numpy array in the form of a feature matrix to make
                 predictions on.
                 threshold - (Default 0.5) The threshold for binary
                 classification.

        Returns: y_pred - 1-D numpy array containing the predicted class for
                 each sample in X.
        """
        if not self.model_trained:
            print('Model needs to be trained first. Call the fit() method')
            return

        y_prob = self.predict_proba(X)
        if self.num_classes <= 2: 
            y_pred = np.where(y_prob > threshold, 1, 0) # Apply threshold
        else:
            y_pred = self.__compute_ovr_pred(y_prob) # Assign most probable class
        return y_pred
    
    def score(self, X_test, y_test):
        """
        Compute the accuracy of prediction made for the samples in X_test
        compared to the true values in y_test.

        Inputs : X_test - 2-D numpy array in the form of a feature matrix to make
                 predictions on.
                 y_test - The true categories for each sample in X_test

        Returns: The percentage of correct predictions (between 0 and 1)
        """
        if not self.model_trained:
            print('Model needs to be trained first. Call the fit() method')
            return

        y_pred = self.predict(X_test) # Make predictions on X_test
        count = 0 # To store number of correct predictions

        for test, prediction in zip(y_test, y_pred):
            if prediction == test: # Check if correct
                count += 1

        return count / np.size(y_test) # Return percentage correct
    
    def __initialise_weights(self):
        """
        Initialise the weights of the model to either random
        values or zeros. Depends on value of self.rand

        Inputs:  None

        Returns: None
        """
        if self.num_classes <= 2:
            if self.rand:
                self.w = np.random.random((self.num_attributes,))
            else:
                self.w = np.zeros((self.num_attributes,))
        else: # For multiclass classification
            if self.rand:
                self.w = np.random.random((self.num_attributes, self.num_classes))
            else:
                self.w = np.zeros((self.num_attributes, self.num_classes))
        # For error handling
        self.initial_weights_set = True
        
    def __update_weights(self, X, y, w):
        """
        Update the weights of the model using the Logistic Regression weight
        update formula self.max_iter times.

        Inputs:  X - A 2-D numpy array containing data with a column for each
                 attribute and a row for each sample - The feature matrix
                 y - A 1-D numpy array containing the label for each sample
                 in X.
                 w - A 1-D numpy array containing the initialised weight to be 
                 applied to each attribute.

        Returns: w - A 1-D numpy array containing the "learned" weight to be 
                 applied to each attribute.
        """
        cost = []
        i = 0
        while i < self.max_iter:
            z = self.__net_input(X, w)
            h_z = self.__logistic(z)
            cost.append(self.__cost_function(h_z, y, w))
            
            if self.verbose and i % 100 == 0: # To monitor cost
                print(f'Cost: {cost[-1]}')
            
            # Gradient descent formula
            for j in range(self.num_attributes):
                if j == 0:
                    w[j] -= self.learning_rate * np.mean((h_z - y) * X[:,j])  
                else:
                    l2_reg_term = self.l2_strength * w[j] / self.num_samples_training
                    w[j] -= self.learning_rate * (np.mean((h_z - y) * X[:,j]) + l2_reg_term)

            i += 1
            if i >= 2: # Make sure at least 2 iterations are done
                if np.abs(cost[-1] - cost[-2]) <= self.tol:
                    if self.verbose:
                        print(f'Met tolerance condition after {i} iterations')
                    break
                if i == self.max_iter and self.verbose:
                    print(f'Hit maximum number of iterations: {i}')

        # To enable plotting of cost function
        self.cost.append(cost)
        return w
    
    def __cost_function(self, h_z, y, w):
        """
        Compute the cost to monitor gradient descent.

        Inputs:  h_z - The output from the logistic function.
                 y - The target vector.

        Returns: J_w - The computed cost for the predictions using
                 the log-liklihood.
                 
        """
        # Log loss function formula for LR
        l2_reg_term = (self.l2_strength / (2*self.num_samples_training)) * np.sum(w[1:]**2)
        J_w = -np.mean(y*np.log(h_z) + (1-y)*np.log(1-h_z)) + l2_reg_term
        return J_w
                    
    
    def __net_input(self, X, w):
        """
        Compute the net input of the model. That is the sum
        of the products of each attribute value and and the 
        associated weight for each sample.

        Inputs:  X - A matrix of the attributes values for
                 each samples. It is assumed that the samples
                 are rows and the attributes are columns.
                 w - The weights of the model.

        Returns: z - The net input of the attrubutes and the
                 weights for each sample
        """
        num_samples, num_attributes = np.shape(X)

        z = np.zeros((num_samples,))
        # Dot product for each sample
        for i in range(num_samples):
            z[i] = np.dot(X[i,:], w)
        return z

    def __logistic(self, z):
        """
        This function acts as the logistic or sigmoid function
        for the model.

        Inputs:  z - This can be a scalar or 1-D numpy array
                 that is computed as the net input for the 
                 Logistic Regression model.

        Returns: h_z - The output of the logistic function.
                 This will have the same dimentions as the
                 input z.
        """
        # Formula for logistic function
        h_z = 1 / (1 + np.exp(-z))
        return h_z
   
    def __get_ovr_targets(self, y):
        """
        This function takes a 1-D numpy array with 3 or more
        unique categories and creates an 2-D numpy array with
        One Vs the Rest (OVR) elements (one-hot encoded). A
        dictionary is also created containing the mapping of
        category to column number in y_ovr (and vice-versa).

        Inputs:  y - 1-D numpy array (the target vector) with 3
                 or more unique categories.

        Returns: y_ovr - 2-D numpy array with the same number of
                 rows as y and the number of unique categories as
                 the number of columns. The values are one-hot
                 encoded for the OVR scheme.
        """
        num_samples = np.size(y, axis=0)
        self.targets = np.sort(np.unique(y)) # Get the names of the classes and sort
                
        y_ovr = np.zeros((num_samples, self.num_classes))
        self.target_mapping = dict() # This will contain the mapping for the target vectors

        for index, target in enumerate(self.targets):
            y_ovr[:,index] = np.where(y == target, 1, 0) # One vs the rest
            self.target_mapping[target] = index # class to number
            self.target_mapping[index] = target # number to class
        return y_ovr
    
    def __compute_ovr_pred(self, y_prob):
        """
        This function takes a 2-D numpy array of predicted
        probabilities from the OVR scheme. The max probability
        in each row is computed and this is taken to be the 
        predicted category. The mapping created in
        __get_ovr_targets() is used to decipher which column
        corresponds to which category.

        Inputs:  y_prob - 2-D numpy array of predicted
                 probabilities using the OVR scheme.

        Returns: y_pred - 1-D numpy array containing the
                 predicted categories.
        """
        y_pred = []
        for i in range(self.num_samples_testing):
            # Find the class with the max prob
            pred_class = np.argmax(y_prob[i,:])
            # Assign label with the saved mapping
            y_pred.append(self.target_mapping[pred_class])
        
        # Make the same data type of the original target vector
        y_pred = np.array(y_pred, dtype=object)
        return y_pred
    
    def __insert_x0(self, X):
        """
        This function inserts a vector of ones into X as 
        the first column. This is used as the bias or intercept
        for Logistic Regression.

        Inputs:  X - A 2-D numpy array containing data with a 
                 column for each attribute and a row for each 
                 sample - The feature matrix.

        Returns: The original input X with an extra column of
                 1's inserted in the first position.
        """
        X0 = np.ones((np.size(X, axis=0), 1))
        return np.hstack((X0, X))
        
