# -*- coding: utf-8 -*-
# @Time    : 2020/6/7 17:09
# @Author  : Equator
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LinearRegression, Lasso, ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR

from code.preprocessing.data_scan import data_split
from code.visual.algorithm_comparison import box_plot

num_folds = 10
seed = 7
scoring = 'neg_mean_squared_error'


def normalize(train_x, train_y):
    pipelines = {}
    pipelines['ScalerLR'] = Pipeline([('Scaler', StandardScaler()), ('LR', LinearRegression())])
    pipelines['ScalerLASSO'] = Pipeline([('Scaler', StandardScaler()), ('LASSO', Lasso())])
    pipelines['ScalerEN'] = Pipeline([('Scaler', StandardScaler()), ('EN', ElasticNet())])
    pipelines['ScalerKNN'] = Pipeline([('Scaler', StandardScaler()), ('KNN', KNeighborsRegressor())])
    pipelines['ScalerCART'] = Pipeline([('Scaler', StandardScaler()), ('LR', DecisionTreeRegressor())])
    pipelines['ScalerSVM'] = Pipeline([('Scaler', StandardScaler()), ('SVM', SVR())])
    results = []
    for key in pipelines:
        fold = KFold(n_splits=num_folds, random_state=seed, shuffle=True)
        result = cross_val_score(pipelines[key], train_x, train_y, cv=fold, scoring=scoring)
        results.append(result)
        print('%s %f (%f)' % (key, result.mean(), result.std()))
    # 箱线图
    box_plot(results, pipelines.keys())


if __name__ == '__main__':
    train_x, test_x, train_y, test_y = data_split()
    normalize(train_x, train_y)
