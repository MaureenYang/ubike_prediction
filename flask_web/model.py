import numpy as np
#import matplotlib.pyplot as plt
import pandas as pd
import pickle
from sklearn.model_selection import PredefinedSplit
from sklearn.linear_model import LinearRegression


if True:
    df = pd.read_csv('src_csv/new_data_sno1_predict.csv')

    def index_splitter(N, fold):
        index_split = []
        test_num = int(N/fold)
        train_num = N-test_num

        for i in range(0,train_num):
            index_split.append(-1)

        for i in range(train_num,N):
            index_split.append(0)

        return index_split

    # preprocessing
    df = df.dropna()
    df['time'] = pd.to_datetime(df['time'], format='%Y-%m-%d %H:%M:%S', errors='ignore')
    df = df.set_index(pd.DatetimeIndex(df['time']))
    X = df.drop(columns = [df.keys()[0],'sbi','bemp','time'])
    Y = df['bemp']


    # Data Splitter
    arr = index_splitter(N=len(X), fold=3)
    ps = PredefinedSplit(arr)

    for train, test in ps.split():
        train_index = train
        test_index = test

    train_X, train_y = X.iloc[train_index,:], Y.iloc[train_index]
    test_X, test_y = X.iloc[test_index,:], Y.iloc[test_index]

if False:
    regressor = LinearRegression()
    regressor.fit(train_X, train_y)
    pickle.dump(regressor, open('model.pkl','wb'))

model = pickle.load(open('lasso_new.pkl','rb'))

print(model.predict(test_Xn))
