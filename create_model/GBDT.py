# -*- coding: utf-8 -*-
"""
# @Time    : 2022/12/30 20:20
# @Author  : Huang
# @File    : GBDT.py
# @Software: PyCharm 
# @Comment : 
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV,  ShuffleSplit, cross_val_score
from matplotlib import pyplot as plt
from process.error import plot_learning_curve
from process.error import mean_squared_error, mean_absolute_percentage_error, mean_absolute_error
import warnings
warnings.filterwarnings("ignore")

line_x_df = pd.read_csv(r'E:\公交客流预测\data\11路公交各时段客流量_scale.csv', encoding= 'gbk')
target = line_x_df['population']
data = line_x_df.drop(['deal_time', 'population', 'date'], axis=1)
feature_name = data.columns.values
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.1)
clf = GradientBoostingRegressor()

parameters = {
                'learning_rate': [0.001, 0.01, 0.03, 0.07, 0.1],
                'n_estimators': [100, 300, 350, 370, 500],
                'max_depth': [3, 5, 7, 10],
            }
grid_search = GridSearchCV(clf, parameters,n_jobs = -1)
from time import time
t0 = time()
grid_search.fit(X_train, y_train)
best_parameters = grid_search.best_params_
print('{:-^50}'.format('GridSearch网格搜索多参数调优'), '\n最优参数：', grid_search.best_params_, '\n最优得分',
      grid_search.best_score_)
print('grid_searching takes %0.3fs' % (time() - t0))

clf.set_params(learning_rate =best_parameters['learning_rate'], n_estimators = best_parameters['n_estimators'], max_depth = best_parameters['max_depth'])
clf.fit(X_train, y_train)

path = 'E:\公交客流预测\model_train_pic\GBDT_各时段客流量'
plot_learning_curve(clf, path, 'Liner_model', X_train, y_train, cv=10, n_jobs=-1)

y_pred = clf.predict(X_test)

# 客流图
def plot_yred(y_pred, y_test, path):
    plt.plot(y_pred, 'o--',lw=1,label='y_pred')
    plt.scatter(np.arange(len(y_test)), y_test, edgecolor="black", c="darkorange", label="y_test")
    plt.legend(loc="best")
    plt.savefig(path + '\线路11_GBDT_各时段客流预测.png', dpi=400,
                bbox_inches='tight')
    plt.show()
plot_yred(y_pred, y_test, path)

mae = mean_absolute_error(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
print('平均绝对误差 MAE：', mae, '\n平均绝对百分比误差 MAPE：', mape, '\n均方误差 MSE：', mse)
# 特征重要性评估

importances = clf.feature_importances_
importances_df = pd.DataFrame()
importances_df['特征名称'] = feature_name
importances_df['特征重要性'] = importances
importances_df = importances_df.sort_values('特征重要性', ascending=False)
print(importances_df)

def plot_feature(col, feature_imporance, pic_path):
    plt.figure(figsize=(12, 5))
    plt.bar(col, feature_imporance)
    plt.plot(feature_imporance, 'ro-')
    plt.xticks(rotation=30)
    plt.ylim([min(feature_imporance[:-1]), max(feature_imporance[:-1]) + 0.05])
    plt.xlabel("feature name", fontdict={"fontsize": 12})
    plt.ylabel("importance coff", fontdict={"fontsize": 12})
    # plt.title("Feature importance bar chart", fontdict={"fontsize": 14})
    plt.grid()
    plt.savefig(pic_path + '\线路11_GBDT_特征重要性.png', dpi=400,
                bbox_inches='tight')
    plt.show()
plot_feature(feature_name, importances, path)




