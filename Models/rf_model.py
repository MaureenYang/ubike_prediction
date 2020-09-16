# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 20:56:50 2019

@author: Mandy
"""

from sklearn.datasets import load_boston
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import PredefinedSplit
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re


def index_splitter(N, fold):
    index_split = []
    test_num = int(N/fold)
    train_num = N-test_num

    for i in range(0,train_num):
        index_split.append(-1)

    for i in range(train_num,N):
        index_split.append(0)

    return index_split


Trained_Paras = {'bootstrap': True,
'max_depth': 11,
'max_features': 'auto',
'min_samples_leaf': 4,
'min_samples_split': 3,
'n_estimators': 1379}

# Number of trees in random forest
def rf(X, Y, kfold=3, feature_set=None):
    arr = index_splitter(N = len(X), fold = kfold)
    ps = PredefinedSplit(arr)

    for train, test in ps.split():
        train_index = train
        test_index = test


    train_X, train_y = X.values[train_index,:], Y.values[train_index]
    test_X, test_y = X.values[test_index,:], Y.values[test_index]
    arr = index_splitter(N = len(train_X), fold = kfold)
    ps2 = PredefinedSplit(arr)

    rf = RandomForestRegressor(random_state = 42)

    from pprint import pprint

    # Look at parameters used by our current forest
    print('Parameters currently in use:\n')
    pprint(rf.get_params())

    # Use the random grid to search for best hyperparameters
    # First create the base model to tune

    ## Grid search of parameters, using 3 fold cross validation based on Random search
    from sklearn.model_selection import GridSearchCV

    # Number of trees in random forest
    #n_estimators = [Trained_Paras["n_estimators"]]
    n_estimators = [Trained_Paras["n_estimators"]]

    # Number of features to consider at every split
    max_features = [Trained_Paras["max_features"]]

    # Maximum number of levels in tree
    max_depth = [int(x) for x in range(Trained_Paras["max_depth"]-10,Trained_Paras["max_depth"]+10,2)]
    #max_depth = []
    #max_depth.append(Trained_Paras["max_depth"])

    # Minimum number of samples required to split a node
    min_samples_split = [Trained_Paras["min_samples_split"]]
    for x in range(Trained_Paras["min_samples_split"]-2,Trained_Paras["min_samples_split"]+3,2):
        if x>1:
            min_samples_split.append(int(x))

    # Minimum number of samples required at each leaf node
    min_samples_leaf = [Trained_Paras["min_samples_leaf"]]
    for x in range(Trained_Paras["min_samples_leaf"]-1,Trained_Paras["min_samples_leaf"]+1,1):
        if x>0:
            min_samples_leaf.append(int(x))

    # Method of selecting samples for training each tree
    bootstrap = [Trained_Paras["bootstrap"]]

    # Create the random grid
    grid_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}

    rf_grid = GridSearchCV(estimator=rf, param_grid=grid_grid, scoring='neg_mean_squared_error',
                                  cv = ps2.split(), verbose=2, n_jobs=-1)
    # Fit the grid search model
    rf_grid.fit(train_X, train_y)
    BestPara_grid = rf_grid.best_params_

    pprint(rf_grid.best_params_)
    cv_results_grid = rf_grid.cv_results_


    #prediction

    predict_y_grid=rf_grid.predict(test_X)
    # Performance metrics
    from sklearn.metrics import mean_squared_log_error
    from sklearn.metrics import mean_squared_error

    errors_Grid_CV = (mean_squared_error(predict_y_grid,test_y))#,squared = False))
    print(errors_Grid_CV)



    return rf_grid.predict, rf_grid.best_estimator_
