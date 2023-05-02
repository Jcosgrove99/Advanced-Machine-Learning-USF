import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

def load_dataset(path="data/rent-ideal.csv"):
    dataset = np.loadtxt(path, delimiter=",", skiprows=1)
    y = dataset[:, -1]
    X = dataset[:, 0:- 1]
    return X, y

def gradient_boosting_mse(X, y, num_iter, max_depth=1, nu=0.1):
    """Given predictors X, an array y, and num_iter (big M in the sum)
    
    Set random_state = 0 in your DecisionTreeRegressor() trees.

    Return the y_mean and trees 
   
    Input: X, y, num_iter
           max_depth
           nu (shrinkage parameter)

    Outputs:y_mean, array of trees from DecisionTreeRegressor 
    """
    trees = []
    N, _ = X.shape
    y_mean = np.mean(y)
    fm = y_mean
    ### BEGIN SOLUTION
    for i in range(num_iter): 
        r = y - fm
        reg = DecisionTreeRegressor(random_state=0,max_depth=max_depth)
        Tm = reg.fit(X,r)
        trees.append(Tm)
        update = Tm.predict(X) 
        fm += nu * update 

    ### END SOLUTION
    return y_mean, trees  

def gradient_boosting_predict(X, trees, y_mean,  nu=0.1):
    """
    Given X, trees, y_mean predict y_hat
    """
    ### BEGIN SOLUTION
    y_hat = np.full(X.shape[0], y_mean)
    for i in trees: 
        y_hat += nu * i.predict(X)
    ### END SOLUTION
    return y_hat

