# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 20:56:50 2019

@author: Mandy

"""

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import PredefinedSplit
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import mean_squared_error

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd



def index_splitter(N, fold):
    index_split = []
    test_num = int(N/fold)
    train_num = N-test_num

    for i in range(0,train_num):
        index_split.append(-1)

    for i in range(train_num,N):
        index_split.append(0)

    return index_split


# Number of trees in random forest
def gb(X, Y, kfold=3, feature_set=None):
    arr = index_splitter(N = len(X), fold = kfold)
    ps = PredefinedSplit(arr)

    for train, test in ps.split():
        train_index = train
        test_index = test

    gb = GradientBoostingRegressor(random_state = 42)
    train_X, train_y = X.values[train_index,:], Y.values[train_index]
    test_X, test_y = X.values[test_index,:], Y.values[test_index]
    arr = index_splitter(N = len(train_X), fold = kfold)
    ps2 = PredefinedSplit(arr)

    # Create the random grid
    BestPara_random = {
            'learning_rate' : 0.1,
            'n_estimators': 100,
                   'max_features': 'auto',
                   'max_depth': 5,
                   'min_samples_split': 8,
                   'min_samples_leaf': 4,
                   }

    ## Grid search of parameters, using 3 fold cross validation based on Random search
    from sklearn.model_selection import GridSearchCV

    lr = [BestPara_random['learning_rate']]            # Number of trees in random forest
    n_estimators = [BestPara_random["n_estimators"]]   # Number of features to consider at every split
    max_features = [BestPara_random["max_features"]]   # Maximum number of levels in tree

    max_depth = [int(x) for x in range(BestPara_random["max_depth"]-10,BestPara_random["max_depth"]+10,2)]
    max_depth = [BestPara_random["max_depth"]]

    # Minimum number of samples required to split a node
    min_samples_split = [BestPara_random["min_samples_split"]]

    # Minimum number of samples required at each leaf node
    min_samples_leaf = [BestPara_random["min_samples_leaf"]]

    # Create the random grid
    grid_grid = {'learning_rate' : lr,
                 'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   }


    gb_grid = GridSearchCV(estimator=gb, param_grid=grid_grid, scoring='neg_mean_squared_error',
                                  cv = ps2.split(), verbose=2, n_jobs=-1)

    gb_grid.fit(train_X, train_y)
    predict_y_base=gb_grid.predict(test_X)

    errors_baseline = (mean_squared_error(predict_y_base,test_y))#,squared = False))
    print('RMSE', errors_baseline)

    if False:
        x_axis = range(3)
        plt.bar(x_axis, results)
        plt.xticks(x_axis, ('GridSearchCV','RandomizedSearchCV', 'Baseline'))
        plt.show()

        #feature importance
        num_feature = len(gb_grid.best_estimator_.feature_importances_)
        plt.figure(figsize=(24,6))
        plt.bar(range(0,num_feature*4,4),gb_grid.best_estimator_.feature_importances_)

        label_name = X.keys()

        plt.xticks(range(0,num_feature*4,4), label_name)
        plt.title("Feature Importances"+",kfold="+str(kfold))
        plt.show()

        fig=plt.figure(figsize=(20,8))
        ax = fig.gca()
        x_label = range(0,len(predict_y_grid))
        plt.title("kfold="+str(kfold))
        ax.plot(x_label, predict_y_grid, 'r--', label = "predict")
        ax.plot(x_label, test_y, label = "ground_truth")
        ax.set_ylim(0, 200)
        ax.legend()
        plt.show()

    return gb_grid.predict,gb_grid
