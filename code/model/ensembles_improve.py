# -*- coding: utf-8 -*-
# @Time    : 2020/6/7 20:17
# @Author  : Equator
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor

from code.preprocessing.data_scan import data_split

num_folds = 10
seed = 7
scoring = 'neg_mean_squared_error'


def GBR_ensembles_improve(train_x, train_y):
    scaler = StandardScaler().fit(train_x)
    rescaledX = scaler.transform(train_x)
    param_gird = {'n_estimators': [10, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900]}
    model = GradientBoostingRegressor()
    fold = KFold(n_splits=num_folds, random_state=seed, shuffle=True)
    grid = GridSearchCV(model, param_gird, scoring=scoring, cv=fold)
    grid_result = grid.fit(X=rescaledX, y=train_y)
    cv_result = zip(grid_result.cv_results_['mean_test_score'], grid_result.cv_results_['std_test_score'],
                    grid_result.cv_results_['params'])
    for mean, std, param in cv_result:
        print("%f (%f) with %r" % (mean, std, param))
    print('最优：%s 使用 %s' % (grid_result.best_score_, grid_result.best_params_))


def ETR_ensembles_improve(train_x, train_y):
    scaler = StandardScaler().fit(train_x)
    rescaledX = scaler.transform(train_x)
    param_gird = {'n_estimators': [5, 10, 30, 40, 50, 60, 70, 80, 90, 100]}
    model = ExtraTreesRegressor()
    fold = KFold(n_splits=num_folds, random_state=seed, shuffle=True)
    grid = GridSearchCV(model, param_gird, scoring=scoring, cv=fold)
    grid_result = grid.fit(X=rescaledX, y=train_y)
    cv_result = zip(grid_result.cv_results_['mean_test_score'], grid_result.cv_results_['std_test_score'],
                    grid_result.cv_results_['params'])
    for mean, std, param in cv_result:
        print("%f (%f) with %r" % (mean, std, param))
    print('最优：%s 使用 %s' % (grid_result.best_score_, grid_result.best_params_))


if __name__ == '__main__':
    train_x, test_x, train_y, test_y = data_split()
    GBR_ensembles_improve(train_x, train_y)
    ETR_ensembles_improve(train_x, train_y)
