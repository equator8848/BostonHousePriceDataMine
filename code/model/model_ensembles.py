# -*- coding: utf-8 -*-
# @Time    : 2020/6/7 19:38
# @Author  : Equator
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, cross_val_score

from code.preprocessing.data_scan import data_split
from code.visual.algorithm_comparison import box_plot

num_folds = 10
seed = 7
scoring = 'neg_mean_squared_error'


def ensembles(train_x, train_y):
    ensembles = {}
    ensembles['ScalerAdaBoost'] = Pipeline([('Scaler', StandardScaler()), ('AdaBoost', AdaBoostRegressor())])
    ensembles['ScalerAdaBoostKNN'] = Pipeline([('Scaler', StandardScaler()),
                                               ('AdaBoostKNN',
                                                AdaBoostRegressor(base_estimator=KNeighborsRegressor(n_neighbors=1)))])
    ensembles['ScalerAdaBoostLR'] = Pipeline(
        [('Scaler', StandardScaler()), ('AdaBoostLR', AdaBoostRegressor(LinearRegression()))])
    ensembles['ScalerAdaBoostRFR'] = Pipeline([('Scaler', StandardScaler()), ('RFR', RandomForestRegressor())])
    ensembles['ScalerAdaBoostETR'] = Pipeline([('Scaler', StandardScaler()), ('ETR', ExtraTreesRegressor())])
    ensembles['ScalerAdaBoostGBR'] = Pipeline([('Scaler', StandardScaler()), ('GBR', GradientBoostingRegressor())])
    results = []
    for key in ensembles:
        fold = KFold(n_splits=num_folds, random_state=seed, shuffle=True)
        result = cross_val_score(ensembles[key], train_x, train_y, cv=fold, scoring=scoring)
        results.append(result)
        print('%s %f (%f)' % (key, result.mean(), result.std()))
    box_plot(results, ensembles.keys())


if __name__ == '__main__':
    train_x, test_x, train_y, test_y = data_split()
    ensembles(train_x, train_y)

