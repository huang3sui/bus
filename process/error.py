# -*- coding: utf-8 -*-
"""
# @Time    : 2023/12/27 17:04
# @Author  : Huang
# @File    : error.py
# @Software: PyCharm 
# @Comment : 
"""
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import  make_scorer, mean_absolute_error, mean_squared_error
from sklearn.model_selection import learning_curve

# MAE
def mean_absolute_error(true, pred):
    abs_error = np.abs(true - pred)
    sum_abs_error = np.sum(abs_error)
    mae_loss = sum_abs_error / true.size
    return mae_loss

# MAPE
def mean_absolute_percentage_error(true, pred):
    abs_error = (np.abs(true - pred)) / true
    sum_abs_error = np.sum(abs_error)
    mape_loss = (sum_abs_error / true.size) * 100
    return mape_loss

# MSE
def mean_squared_error(true, pred):
    squared_error = np.square(true - pred)
    sum_squared_error = np.sum(squared_error)
    mse_loss = sum_squared_error / true.size
    return mse_loss

def plot_learning_curve(estimator, pic_path, title, X, y, ylim=None, cv=None,n_jobs=-1, train_size=np.linspace(.1, 1.0, 5 )):
    plt.figure()
    # plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel('Training example')
    plt.ylabel('loss')
    # train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_size, scoring=make_scorer(mean_squared_error))

    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_size)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()#区域
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1,
                     color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color='r',
             label="Training score")
    plt.plot(train_sizes, test_scores_mean,'o-',color="g",
             label="Cross-validation score")
    plt.ylabel('$y^2$', )
    plt.legend(loc="best")
    plt.savefig(pic_path + title +'.png', dpi=400,
                bbox_inches='tight')
    plt.show()
    return plt