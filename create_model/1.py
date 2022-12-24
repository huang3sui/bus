import pandas as pd
from sklearn.model_selection import train_test_split
import joblib
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.tree import DecisionTreeRegressor,export_graphviz
import pydotplus
from sklearn import ensemble
import warnings

warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, cross_val_score

line_6_df = pd.read_csv(r'E:\公交客流预测\data\6路公交日客流量.csv', encoding='gbk')
line_6_df = line_6_df.dropna()
target = line_6_df['population']
data = line_6_df.drop(['date', 'population'], axis=1)
print(data.columns)
clf_DT = DecisionTreeRegressor()
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=7)
scores_3 = cross_val_score(clf_DT, X_train, y_train, cv = 3)
scores_5_mse = cross_val_score(clf_DT, X_train, y_train,scoring="neg_mean_squared_error", cv = 5)
scores_5_mae = cross_val_score(clf_DT, X_train, y_train,scoring="neg_mean_absolute_error", cv = 5)
scores_5_r2 = cross_val_score(clf_DT, X_train, y_train,scoring="r2", cv = 5)
print('{:-<50}\nMSE:'.format('5折'), scores_5_mse.mean(),'\nMae:', scores_5_mae.mean(),'\nR2:',scores_5_r2.mean())
scores_10_mse = cross_val_score(clf_DT, X_train, y_train,scoring="neg_mean_squared_error", cv = 10)
scores_10_mae = cross_val_score(clf_DT, X_train, y_train,scoring="neg_mean_absolute_error", cv = 10)
scores_10_r2 = cross_val_score(clf_DT, X_train, y_train,scoring="r2", cv = 10)
print('{:-<50}\nMSE:'.format('10折'), scores_10_mse.mean(), '\nMae:', scores_10_mae.mean(), '\nR2:',scores_10_r2.mean())
'''
parameters = {'splitter': ('best', 'random'),
              'max_depth': range(2, 20, 1),
              'criterion': ['friedman_mse', 'mse', 'mae'],
              'min_samples_split': range(5, 20, 3),
              'min_samples_leaf': range(5, 20, 3)}
grid_search = GridSearchCV(clf_DT, parameters, n_jobs = -1, cv=10)
grid_search.fit(X_train, y_train)
best_param = grid_search.best_params_
print('{:-^50}'.format('GridSearch网格搜索多参数调优'), '\n最优参数：', grid_search.best_params_, '\n最优得分',
      grid_search.best_score_)

best_clf_DT = DecisionTreeRegressor(criterion = best_param['criterion'], splitter = best_param['splitter'], max_depth = best_param['max_depth'],
                                    min_samples_split = best_param['min_samples_split'], min_samples_leaf = best_param['min_samples_leaf'])
'''

r2_best, best_model, best_depth, best_feature_importance = 0, '', 0, []
clf = None
test_r2, train_r2 = [], []
depth = np.arange(2, 15)
for d in depth:
    clf = DecisionTreeRegressor(criterion = 'mae', splitter='random', max_depth=d, min_samples_split= 14, min_samples_leaf=8)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    r2_test = clf.score(X_test, y_test)
    r2_train = clf.score(X_train, y_train)
    test_r2.append(r2_test)
    train_r2.append(r2_train)
    if r2_best < r2_test:
        r2_best, best_model, best_depth, best_feature_importance = r2_best, clf, d, clf.feature_importances_
    plt.plot(y_pred, lw=1, label='depth=%d, r2=%.5f' % (d, r2_test))
print(best_feature_importance)
plt.legend(loc = 'upper right')
plt.xlabel("Sample observation number", fontdict = {"fontsize": 12})
plt.ylabel("y values", fontdict = {"fontsize": 12})
plt.title("Decision Tree Regression of different depth",fontdict = {"fontsize": 14})
plt.grid()
plt.show()

plt.figure(figsize=(12, 5))
plt.plot(depth, train_r2, 'ro-', label='Train r2')
plt.plot(depth, test_r2, 'bs-', label='Test r2')
plt.legend(fontsize=12)
plt.xlabel("depth", fontdict={"fontsize": 12})
plt.xlabel("r2", fontdict={"fontsize": 12})
plt.title("Learning Curve of Decision Tree Regression", fontdict={"fontsize": 14})
plt.grid()
plt.show()

plt.figure(figsize = (12, 5))
print(len(data.columns.values),len(best_feature_importance))
plt.bar(data.columns.values, best_feature_importance)
plt.plot(best_feature_importance, 'ro-')
plt.xticks(rotation = 30)
plt.ylim([min(best_feature_importance[:-1]), max(best_feature_importance[:-1]) + 0.05])
plt.xlabel("feature name", fontdict = {"fontsize": 12})
plt.ylabel("importance coff", fontdict = {"fontsize": 12})
plt.title("Feature importance bar chart",fontdict = {"fontsize": 14})
plt.grid()
plt.show()

feature_names = data.columns.values.tolist()
feature_names.append('bias')
import os
# 用来输出可视化图
os.environ['PATH'] = os.pathsep + r'D:\ProgramData\Anaconda\Library\bin'
dot_data = export_graphviz(best_model, out_file = None
                           , feature_names = data.columns.values  # 特征名称
                           , filled=True # 填充颜色
                           , rounded=True  # 边框变为圆弧
                           , special_characters=True # 显示特殊字符
                            , proportion=True  # 呈现百分数
                          )
# 输出PDF格式
graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_pdf("{}.pdf".format('决策树'))

# 输出为PNG格式
with open("{}.png".format('决策树'), "wb") as file:
    file.write(graph.create_png())

