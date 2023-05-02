import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from pathlib import Path

def accuracy(y, pred):
    return np.sum(y == pred) / float(len(y))


def parse_spambase_data(filename):
    """ Given a filename return X and Y numpy arrays

    X is of size number of rows x num_features
    Y is an array of size the number of rows
    Y is the last element of each row. (Convert 0 to -1)
    """
    ### BEGIN SOLUTION
    df = pd.read_csv(filename, header=None)
    df = df.replace(to_replace=0, value=-1)
    X = df.drop(df.columns[-1], axis=1, inplace=False)
    X = X.values
    Y = df.iloc[:, -1].values
    ### END SOLUTION
    return X, Y


def adaboost(X, Y, num_iter, max_depth=1):
    """Given an numpy matrix X, a array y and num_iter return trees and weights 
   
    Input: X, y, num_iter
    Outputs: array of trees from DecisionTreeClassifier
             trees_weights array of floats
    Assumes y is {-1, 1}
    """
    trees = []
    trees_weights = [] 
    N, _ = X.shape
    d = np.ones(N) / N
    ### BEGIN SOLUTION
    weights = d
    for i in range(num_iter): 
        clf = DecisionTreeClassifier(max_depth=max_depth)
        clf = clf.fit(X, Y, sample_weight=weights)
        trees.append(clf)
        y_hat = clf.predict(X)
        y_not_equal = (Y != y_hat)
        err = (weights * y_not_equal).sum() 
        alpha = np.log((1-err)/err)
        trees_weights.append(alpha)
        weights = weights * (y_not_equal * (1-err)/err)
    ### END SOLUTION
    return trees, trees_weights


def adaboost_predict(X, trees, trees_weights):
    """Given X, trees and weights predict Y
    """
    # X input, y output
    N, _ =  X.shape
    y = np.zeros(N)
    ### BEGIN SOLUTION
    y = np.zeros(X.shape[0])
    for tree, weight in zip(trees, trees_weights):
        tree_predictions = tree.predict(X)
        y += weight * tree_predictions      
    y = np.sign(y)
    ### END SOLUTION
    return y
