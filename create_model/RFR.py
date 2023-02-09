# -*- coding: utf-8 -*-
"""
# @Time    : 2023/2/6 7:24
# @Author  : Huang
# @File    : RFR.py
# @Software: PyCharm 
# @Comment : 
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings("ignore")

plt.rcParams['font.sans-serif'] = ['simsun']  # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题


line_x_df = pd.read_csv(r'E:\公交客流预测\data\6路公交各时段客流量.csv', encoding= 'gbk')
target = line_x_df['flow']
data = line_x_df.drop(['flow', 'date', 'deal_time'], axis=1)
feature_name = data.columns.values
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2)
# test_size = 7 * 16
# X_train, X_test, y_train, y_test = data.iloc[:-test_size, :], data.iloc[-test_size:,:],target[:-test_size], target[-test_size:]
#使用随机匹配择优，此处参数设置与前面相同
n_estimators_range=[int(x) for x in np.linspace(start=100,stop=1200,num=25)]# 100 1200 12
max_depth_range=[int(x) for x in np.linspace(2,15,num=6)] #5  30  60
# max_depth_range.append(None)
max_features_range=['auto','sqrt','log2']
min_samples_split_range=[2,5,7,9]
min_samples_leaf_range=[1, 3,5,7,9,13]   #1 2 5 10

rfr_hp_range={'n_estimators':n_estimators_range,
                        'max_features':max_features_range,
                        'max_depth':max_depth_range,
                        'min_samples_split':min_samples_split_range,
                        'min_samples_leaf':min_samples_leaf_range
                        }
print(rfr_hp_range)



clf=RandomForestRegressor(oob_score=True)
random_search=RandomizedSearchCV(estimator=clf,
                                param_distributions=rfr_hp_range,
                                # n_iter=200,
                                n_iter=50,
                                n_jobs=-1,
                                cv=10,
                                verbose=1,
                                )
random_search.fit(X_train, y_train)

best_hp_base=random_search.best_params_
print(best_hp_base)
best_n_estimators = best_hp_base['n_estimators']
best_min_samples_split = best_hp_base['min_samples_split']
best_max_depth = best_hp_base['max_depth']
best_max_features = best_hp_base['max_features']
best_min_samples_leaf = best_hp_base['min_samples_leaf']

best_n_estimators_index = rfr_hp_range['n_estimators'].index(best_n_estimators)

if best_n_estimators_index == 0:
    best_n_estimators_b = rfr_hp_range['n_estimators'][best_n_estimators_index + 1]
    best_n_estimators_f = best_n_estimators - (best_n_estimators_b - best_n_estimators)
elif best_n_estimators_index == int(len(rfr_hp_range['n_estimators']) - 1):
    best_n_estimators_f = rfr_hp_range['n_estimators'][best_n_estimators_index - 1]
    best_n_estimators_b = best_n_estimators + (best_n_estimators - best_n_estimators_f)
else:
    best_n_estimators_f, best_n_estimators_b = rfr_hp_range['n_estimators'][best_n_estimators_index - 1],rfr_hp_range['n_estimators'][best_n_estimators_index + 1]

new_n_nestimators = []
new_n_nestimators.append(best_n_estimators_f)
new_n_nestimators.append(best_n_estimators_b)
new_n_nestimators.append(best_n_estimators)

f_d = abs(best_n_estimators - best_n_estimators_f)
b_d = abs(best_n_estimators - best_n_estimators_b)

for i in [0.2, 0.4, 0.6, 0.8, 0.25, 0.75]:
    new_n_nestimators.append(int(best_n_estimators_f + (f_d * i)))
    new_n_nestimators.append(int(best_n_estimators + (b_d * i)))

#使用网格搜索，在随机匹配择优的基础上选取临近的超参数进行遍历匹配择优
rfr_hp_range_base={'n_estimators':new_n_nestimators,
                'min_samples_split':[best_min_samples_split,best_min_samples_split+3,best_min_samples_split+5,best_min_samples_split+7],
                'max_depth':[best_max_depth,best_max_depth+3,best_max_depth+2,best_max_depth+1]
                }

print(rfr_hp_range_base)
clf_base=RandomForestRegressor(oob_score=True)
grid_search=GridSearchCV(estimator=clf_base,
                          param_grid=rfr_hp_range_base,
                          cv=10,
                          verbose=1,
                          n_jobs=-1)
grid_search.fit(X_train,y_train)

best_hp=grid_search.best_params_
print(best_hp)
best_n_estimators = best_hp['n_estimators']
best_min_samples_split = best_hp['min_samples_split']
best_max_depth = best_hp['max_depth']

clf_best = RandomForestRegressor(n_estimators=best_n_estimators,min_samples_split=best_min_samples_split,min_samples_leaf=best_min_samples_leaf,max_depth=best_max_depth, oob_score=True)
clf_best.fit(X_train,y_train)
print(clf_best.score(X_test,y_test))

from process.error import plot_learning_curve
from process.error import mean_squared_error, mean_absolute_percentage_error, mean_absolute_error

path = 'E:\公交客流预测\model_train_pic\RF'
plot_learning_curve(clf_best, path, 'RF_model学习曲线', X_train, y_train, cv=10, n_jobs=-1)

y_pred = clf_best.predict(X_test)

# 客流图
def plot_yred(y_pred, y_test, path):
    plt.plot(y_pred, 'o--',lw=1,label='y_pred')
    plt.scatter(np.arange(len(y_test)), y_test, edgecolor="black", c="darkorange", label="y_test")
    plt.legend(loc="best")
    plt.savefig(path + '\RF客流预测.png', dpi=400,
                bbox_inches='tight')
    plt.show()
plot_yred(y_pred, y_test, path)
train_pred = clf_best.predict(X_train)
plot_yred(train_pred, y_train, path)

mae = mean_absolute_error(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
print('平均绝对误差 MAE：', mae, '\n平均绝对百分比误差 MAPE：', mape, '\n均方误差 MSE：', mse)
# 特征重要性评估

importances = clf_best.feature_importances_
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
    plt.xlabel("feature name", fontdict={"fontsize": 10.5})
    plt.ylabel("importance coff", fontdict={"fontsize": 12})
    # plt.title("Feature importance bar chart", fontdict={"fontsize": 14})
    plt.grid()
    plt.savefig(pic_path + '\RF_特征重要性.png', dpi=400,
                bbox_inches='tight')
    plt.show()
plot_feature(feature_name, importances, path)






Y1_pred = clf_best.predict(X_test)

from sklearn import metrics
import scipy.stats as stats
random_forest_pearson_r=stats.pearsonr(y_test,Y1_pred)
random_forest_R2=metrics.r2_score(y_test,Y1_pred)
print(random_forest_R2)
random_forest_RMSE=metrics.mean_squared_error(y_test.values,Y1_pred)**0.5
MSE = metrics.mean_squared_error(y_test.values,Y1_pred)
print('Pearson correlation coefficient is {0}, and RMSE is {1}.'.format(random_forest_pearson_r[0],
                                                                        random_forest_RMSE))
print(MSE)
#使用测试集数据得到预测值

