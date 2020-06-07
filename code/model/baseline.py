# -*- coding: utf-8 -*-
# @Time    : 2020/6/7 11:18
# @Author  : Equator
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LinearRegression, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from code.preprocessing.data_scan import data_split
from code.visual.algorithm_comparison import box_plot

num_folds = 10
seed = 7
scoring = 'neg_mean_squared_error'


def baseline(train_x, train_y):
    models = {}
    models['LR'] = LinearRegression()
    models['Lasso'] = Lasso()
    models['EN'] = ElasticNet()
    models['KNN'] = KNeighborsRegressor()
    models['CART'] = DecisionTreeRegressor()
    models['SVM'] = SVR()
    results = []
    for key in models:
        fold = KFold(n_splits=num_folds, random_state=seed, shuffle=True)
        result = cross_val_score(models[key], train_x, train_y, cv=fold, scoring=scoring)
        results.append(result)
        print('%s %f (%f)' % (key, result.mean(), result.std()))
    # 箱线图
    box_plot(results, models.keys())


if __name__ == '__main__':
    train_x, test_x, train_y, test_y = data_split()
    baseline(train_x, train_y)
