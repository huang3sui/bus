# -*- coding: utf-8 -*-
"""
# @Time    : 2022/12/22 9:42
# @Author  : Huang
# @File    : DecisionTree.py
# @Software: PyCharm 
# @Comment : 
"""
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor,export_graphviz
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, cross_val_score
import pydotplus
import joblib
import warnings
warnings.filterwarnings("ignore")

def data(path):
    line_x_df = pd.read_csv(path, encoding='gbk')
    line_x_df = line_x_df.dropna()
    target = line_x_df['population']
    #data = line_x_df.drop(['date', 'population'], axis=1)#日客流
    data = line_x_df.drop(['date', 'population', 'deal_time'], axis=1)#各时段客流量
    feature_name = data.columns.values
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=7)
    return X_train, X_test, y_train, y_test, feature_name

def cross_score(X_train, y_train):
    clf_DT = DecisionTreeRegressor()
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

def fit_DT_tree(X_train, y_train, X_test, y_test, col, func, line, pic_path, model_path):
    best_score, best_model, best_depth, best_feature_importance = 0, '', 0, []
    clf = None
    test_score, train_score = [], []
    depth = np.arange(2, 15)
    for d in depth:
        clf = DecisionTreeRegressor(criterion = func, splitter='random', max_depth=d, min_samples_split= 14, min_samples_leaf=8)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        score_test = clf.score(X_test, y_test)
        score_train = clf.score(X_train, y_train)
        test_score.append(score_test)
        train_score.append(score_train)
        if best_score < score_test:
            best_score, best_model, best_depth, best_feature_importance = score_test, clf, d, clf.feature_importances_
        plt.plot(y_pred, lw=1, label='depth=%d, r2=%.5f' % (d, score_test))
    plt.scatter(np.arange(7),y_test, edgecolor="black", c="darkorange", label="y_test")

    plt.legend(loc = 'upper right')
    plt.xlabel("Sample observation number", fontdict = {"fontsize": 12})
    plt.ylabel("score", fontdict = {"fontsize": 12})
    #plt.title("Decision Tree Regression of different depth",fontdict = {"fontsize": 14})
    plt.grid()
    plt.savefig(pic_path + '\{}_{}路公交决策树模型评估图.png'.format(func, line), dpi=400,
                bbox_inches='tight')
    plt.show()
    # 学习曲线
    plot_learn_curve(depth, train_score, test_score, func, line, pic_path)
    plot_feature(col, best_feature_importance, func, line, pic_path)
    plot_tree(col, best_model, func, line, pic_path)
    # 模型保存
    joblib.dump(best_model, model_path + '\line_{}_{}_DT.model'.format(line, func))

def plot_learn_curve(depth, train_score, test_score, func, line, pic_path):
    plt.figure(figsize=(12, 5))
    plt.plot(depth, train_score, 'ro-', label='Train r2')
    plt.plot(depth, test_score, 'bs-', label='Test r2')
    plt.legend(fontsize=12)
    plt.xlabel("depth", fontdict={"fontsize": 12})
    plt.xlabel("score", fontdict={"fontsize": 12})
    #plt.title("Learning Curve of Decision Tree Regression", fontdict={"fontsize": 14})
    plt.savefig(pic_path + '\{}_{}路公交决策树学习曲线图.png'.format(func,line), dpi=400, bbox_inches='tight')
    plt.grid()
    plt.show()

def plot_feature(col, feature_imporance, func, line, pic_path):
    plt.figure(figsize = (12, 5))
    plt.bar(col, feature_imporance)
    plt.plot(feature_imporance, 'ro-')
    plt.xticks(rotation = 30)
    plt.ylim([min(feature_imporance[:-1]), max(feature_imporance[:-1]) + 0.05])
    plt.xlabel("feature name", fontdict = {"fontsize": 12})
    plt.ylabel("importance coff", fontdict = {"fontsize": 12})
    plt.title("Feature importance bar chart",fontdict = {"fontsize": 14})
    plt.grid()
    plt.savefig(pic_path + '\{}_{}路公交决策树特征重要性.png'.format(func, line), dpi=400,
                bbox_inches='tight')
    plt.show()

def plot_tree(col, model, func, line, pic_path):
    feature_names = col.tolist()
    feature_names.append('bias')
    # 用来输出可视化图
    os.environ['PATH'] = os.pathsep + r'D:\ProgramData\Anaconda\Library\bin'
    dot_data = export_graphviz(model, out_file = None
                               , feature_names = col  # 特征名称
                               , filled=True # 填充颜色
                               , rounded=True  # 边框变为圆弧
                               , special_characters=True # 显示特殊字符
                                , proportion=True  # 呈现百分数
                              )
    # 输出PDF格式
    graph = pydotplus.graph_from_dot_data(dot_data)
    graph.write_pdf(pic_path + '\{}_{}路公交决策树.pdf'.format(func, line))

    # 输出为PNG格式
    with open(pic_path + '\{}_{}路公交决策树.png'.format(func, line), "wb") as file:
        file.write(graph.create_png())

if __name__ == '__main__':
    criterions = ['friedman_mse', 'mse', 'mae']
    pic_path = 'E:\公交客流预测\model_train_pic\决策树_日客流量'
    model_path = 'E:\公交客流预测\model\决策树_日客流量'
    day_pic_path = 'E:\公交客流预测\model_train_pic\决策树_各时段客流量'
    day_model_path = 'E:\公交客流预测\model\决策树_各时段客流量'

    # 线路6
    '''    line_6_path = 'E:\公交客流预测\data\\6路公交日客流量.csv'
    line_6_data = data(line_6_path)
    line_11_path = 'E:\公交客流预测\data\\11路公交日客流量.csv'
    line_11_data = data(line_11_path)
    '''
    day_line_6_path = 'E:\公交客流预测\data\\6路公交各时段客流量.csv'
    day_line_6_data = data(day_line_6_path)
    day_line_11_path = 'E:\公交客流预测\data\\11路公交各时段客流量.csv'
    day_line_11_data = data(day_line_11_path)

    # 交叉验证
    #print('日客流量')
    #cross_score(line_6_data[0], line_6_data[2])
    #cross_score(line_11_data[0], line_11_data[2])
    print('各时段客流量')
    cross_score(day_line_6_data[0], day_line_6_data[2])
    cross_score(day_line_11_data[0], day_line_11_data[2])
    # 决策树
    for func in criterions:
        #fit_DT_tree(line_6_data[0], line_6_data[2], line_6_data[1], line_6_data[3], line_6_data[4], func, '6', pic_path, model_path)
        #fit_DT_tree(line_11_data[0], line_11_data[2], line_11_data[1], line_11_data[3], line_11_data[4], func, '11',pic_path, model_path)
        #fit_DT_tree(day_line_6_data[0], day_line_6_data[2], day_line_6_data[1], day_line_6_data[3], day_line_6_data[4], func, '6', day_pic_path, day_model_path)
        fit_DT_tree(day_line_11_data[0], day_line_11_data[2], day_line_11_data[1], day_line_11_data[3], day_line_11_data[4], func, '11',day_pic_path, day_model_path)


"""#划分测试集和训练集
#网络搜索
# min_samples_split:分裂内部节点需要的最少样例数.int(具体数目),float(数目的百分比)
# n_estimators:森林中数的个数。这个属性是典型的模型表现与模型效率成反比的影响因子,即便如此,你还是应该尽可能提高这个数字,以让你的模型更准确更稳定。
X_train, X_test, y_train, y_test = train_test_split(feature, target, test_size=7)

def GridSearch():
    parameters = {'splitter':('best', 'random'),'max_depth': range(2,10,1), 'criterion':['friedman_mse', 'mse','mae'],'min_samples_split':range(5,50,5),'min_samples_leaf':range(5,20,1)}
    model = DecisionTreeRegressor()
    grid_search = GridSearchCV(model, parameters, cv=5)
    grid_search.fit(X_train, y_train)
    print('{:-^50}'.format('GridSearch网格搜索多参数调优'),'\n最优参数：',grid_search.best_params_,'\n最优得分',grid_search.best_score_)
    return grid_search.best_params_

def createTree(splitter, depth,criter_name,samples_split,samples_leaf,ccp=0):
    # 决策树
    clf = DecisionTreeRegressor(splitter = splitter,max_depth =depth,criterion = criter_name,min_samples_split= samples_split,min_samples_leaf= samples_leaf,ccp_alpha=ccp)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    score = clf.score(X_test, y_test)    #模型正确度
    #score = dtr.score(data_test, target_test)
    print('{:-^50}'.format('决策树'),'---------\n模型准确度',score)
    return y_pred, clf

def count_precision(Y_test,Y_predict):
    Y_test = list(Y_test)
    Y_predict = list(Y_predict)
    deviations = 0
    precision = 0
    for i in range(len(Y_test)):
        cur_dev = abs(Y_test[i]-Y_predict[i])/Y_test[i]
        deviations = deviations + cur_dev
        if(cur_dev==0): #可以不要
            cur_pre = 10
        elif(cur_dev>0.3): #此项必须
            cur_pre = 0
        else:
            #cur_pre = 10*(1-cur_dev**3/(0.3**3))
            cur_pre = 10*(1-cur_dev**0.3/(0.3**0.3))
        precision = precision + cur_pre
    precision = precision / (10*len(Y_test))
    print('总偏差：',deviations, '\n平均精度：',precision)

Search_model = GridSearch()
# (Search_model['splitter'], Search_model['max_depth'], Search_model['criterion'], Search_model['min_samples_split'], Search_model['min_samples_leaf'])
b = createTree(Search_model['splitter'], Search_model['max_depth'], Search_model['criterion'], Search_model['min_samples_split'], Search_model['min_samples_leaf'])

count_precision(y_test,b[0])
print('MSE: %6.2f' % metrics.mean_squared_error(y_test, b[0]))

print('RMSE: %6.2f' % np.sqrt(metrics.mean_squared_error(y_test, b[0])))

print('MAE: %6.2f' % metrics.mean_absolute_error(y_test, b[0]))
X = np.arange(1,len(X_test)+1)
plt.plot(X,y_test,label='true',c='b',)

plt.plot(X,b[0],label='predict',c='r')
plt.legend()
plt.show()


# 特征重要性
a = pd.Series(b[1].feature_importances_, index=line_6_df.columns[2:]).sort_values(ascending=False)
print(a)

"""



'''
X_train, X_test, y_train, y_test = train_test_split(feature, target, test_size=0.2)
print(X_train.shape,y_train.shape, X_test.shape, y_test.shape)
clf_DT = DecisionTreeRegressor()
X_train, X_test, y_train, y_test = train_test_split(feature, target, test_size=0.2, random_state=123)



clf_DT.fit(X_train, y_train)
y_pred = clf_DT.predict(X_test)


def count_precision(Y_test,Y_predict):
    Y_test = list(Y_test)
    Y_predict = list(Y_predict)
    deviations = 0
    precision = 0
    for i in range(len(Y_test)):
        cur_dev = abs(Y_test[i]-Y_predict[i])/Y_test[i]
        deviations = deviations + cur_dev
        if(cur_dev==0): #可以不要
            cur_pre = 10
        elif(cur_dev>0.3): #此项必须
            cur_pre = 0
        else:
            #cur_pre = 10*(1-cur_dev**3/(0.3**3))
            cur_pre = 10*(1-cur_dev**0.3/(0.3**0.3))
        precision = precision + cur_pre
    precision = precision / (10*len(Y_test))
    print(deviations,precision)

for i in np.arange(1,1000):
    y_pred = y_pred + clf_DT.predict(X_test)
y_pred = y_pred/1000

count_precision(y_test,y_pred)
X = np.arange(1,len(X_test)+1)
plt.plot(X,y_test,label='true',c='b')

plt.plot(X,y_pred,label='predict',c='r')
plt.show()
'''


