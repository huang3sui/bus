# -*- coding: utf-8 -*-
"""
# @Time    : 2022/12/12 16:22
# @Author  : Huang
# @File    : LSTM.py
# @Software: PyCharm 
# @Comment : 
"""

from math import sqrt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.utils import plot_model
import os
import joblib

plt.rcParams['font.sans-serif'] = ['FangSong']  # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题


"""
本文是LSTM多元预测
用3个步长的数据预测1个步长的数据
包含：
对数据进行缩放，缩放格式为n行*8列，因为数据没有季节性，所以不做差分
对枚举列（风向）进行数字编码
构造3->1的监督学习数据
构造网络开始预测
将预测结果重新拼接为n行*8列数据
数据逆缩放，求RSME误差
"""

# 转换成监督数据，n+1列数据，n->1，n组预测一组
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    # 将3组输入数据依次向下移动3，2，1行，将数据加入cols列表（技巧：(n_in, 0, -1)中的-1指倒序循环，步长为1）
    for i in range(n_in, 0, -1):
    	cols.append(df.shift(i))
    	names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    # 将一组输出数据加入cols列表（技巧：其中i=0）
    for i in range(0, n_out):
    	cols.append(df.shift(-i))
    	if i == 0:
    		names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
    	else:
    		names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # cols列表(list)中现在有四块经过下移后的数据(即：df(-3),df(-2),df(-1),df)，将四块数据按列 并排合并
    agg = pd.concat(cols, axis=1)
    # 给合并后的数据添加列名
    agg.columns = names
    print(agg)
    # 删除NaN值列
    if dropnan:
    	agg.dropna(inplace=True)
    return agg

# load dataset
def load_data(path, seq, n_train):
    df = pd.read_csv(path, parse_dates=["date"], index_col=[0])
    data = df.pop('population')
    df.insert(0, 'population', data)
    values = df.values
    values = values.astype('float32')

    # 标准化/放缩 特征值在（0,1）之间
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)
    # 特征值数
    n_features = df.shape[1]
    # 构造一个3->1的监督学习型数据
    reframed = series_to_supervised(scaled, seq, 1)
    print(reframed.shape)

    # split into train and test sets
    values = reframed.values

    # 用12月的数据来训练
    train = values[:-n_train, :]
    test = values[-n_train:, :]
    # split into input and outputs
    n_obs = seq * n_features
    # 有32=(4*8)列数据，取前24=(3*8) 列作为X，倒数第8列=(第25列)作为Y
    X_train, y_train = train[:, :n_obs], train[:, -n_features]
    X_test, y_test = test[:, :n_obs], test[:, -n_features]
    print(X_train.shape, len(X_train), y_train.shape)
    # 将数据转换为3D输入，timesteps=3，3条数据预测1条 [samples, timesteps, features]
    X_train = X_train.reshape((X_train.shape[0], seq, n_features))
    X_test = X_test.reshape((X_test.shape[0], seq, n_features))
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
    return X_train, X_test, y_train, y_test, n_features, scaler

def fit_LSTM(X_train, X_test, y_train, y_test, epochs, model_path, model_name, pic_path, title):
    # 设计网络
    model = Sequential()
    model.add(LSTM(50, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mae', optimizer='adam')
    # 拟合网络
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=32, validation_data=(X_test, y_test), verbose=2, shuffle=False)
    model.summary()
    print(model.summary())
    os.environ["PATH"] += os.pathsep + r'C:\Program Files (x86)\Graphviz 2.28\bin'
    plot_model(model,'modle.png', show_shapes=True,dpi=400)

    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.savefig(pic_path + title +'.png', dpi=400,
                bbox_inches='tight')

    plt.show()

    joblib.dump(model, model_path + model_name + '.model')
    return model

def pred_flow(model, seq, n_features, X_test, y_test, scaler, pic_path, title):
    # 执行预测
    yhat = model.predict(X_test)
    # 将数据格式化成 n行 * 24列
    X_test = X_test.reshape((X_test.shape[0], seq*n_features))
    # 将预测列据和后7列数据拼接，因后续逆缩放时，数据形状要符合 n行*8列 的要求
    inv_yhat = np.concatenate((yhat, X_test[:, -(n_features -1):]), axis=1)
    # 对拼接好的数据进行逆缩放
    inv_yhat = scaler.inverse_transform(inv_yhat)
    inv_yhat = inv_yhat[:,0]

    y_test = y_test.reshape((len(y_test), 1))
    # 将真实列据和后7列数据拼接，因后续逆缩放时，数据形状要符合 n行*8列 的要求
    inv_y = np.concatenate((y_test, X_test[:, -(n_features -1):]), axis=1)
    # 对拼接好的数据进行逆缩放
    inv_y = scaler.inverse_transform(inv_y)
    inv_y = inv_y[:,0]

    # 计算RMSE误差值
    rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
    print('Test RMSE: %.3f' % rmse)


    plt.plot(inv_y, label='true')
    plt.plot(inv_yhat, label='pred')
    plt.legend(loc="best")
    plt.savefig(pic_path + title +'.png', dpi=400,
                bbox_inches='tight')

    plt.legend(loc = 'best')
    plt.show()

if __name__ == '__main__':
    for i in ['6', '11']:
        day_line_x_path = "E:\公交客流预测\data\{}路公交日客流量.csv".format(i)
        model_path = 'E:\公交客流预测\model\LSTM\LSTM_'
        pic_path = 'E:\公交客流预测\model_train_pic\LSTM\LSTM_'
        day_seq = 7
        day_line_x_data = load_data(day_line_x_path, day_seq, 31)
        X_train, X_test, y_train, y_test = day_line_x_data[0], day_line_x_data[1], day_line_x_data[2], day_line_x_data[3]
        n_features, scaler = day_line_x_data[-2], day_line_x_data[-1]
        model = fit_LSTM(X_train, X_test, y_train, y_test, 100, model_path, '7days_line_{}'.format(i), pic_path, '7days_line_{}_学习曲线'.format(i))
        pred_flow(model, day_seq, n_features, X_test, y_test, scaler, pic_path, '7days_line_{}_客流预测'.format(i))



# # load dataset
#
# dataset=pd.read_csv(r"E:\公交客流预测\data\6路公交日客流量.csv",parse_dates=["date"], index_col=[0])
# data = dataset.pop('population')
# dataset.insert(0, 'population', data)
# # dataset = read_csv('pollution.csv', header=0, index_col=0)
# values = dataset.values
# values = values.astype('float32')
# # 标准化/放缩 特征值在（0,1）之间
# scaler = MinMaxScaler(feature_range=(0, 1))
# scaled = scaler.fit_transform(values)
# # 用3小时数据预测一小时数据，13个特征值
# n_hours = 7
# n_features = dataset.shape[1]
# # 构造一个3->1的监督学习型数据
# reframed = series_to_supervised(scaled, n_hours, 1)
# print(reframed.shape)
#
# # split into train and test sets
# values = reframed.values
# # 用12月的数据来训练
# n_train_hours = 31
# train = values[:-n_train_hours, :]
# test = values[-n_train_hours:, :]
# # split into input and outputs
# n_obs = n_hours * n_features
# # 有32=(4*8)列数据，取前24=(3*8) 列作为X，倒数第8列=(第25列)作为Y
# train_X, train_y = train[:, :n_obs], train[:, -n_features]
# test_X, test_y = test[:, :n_obs], test[:, -n_features]
# print(train_X.shape, len(train_X), train_y.shape)
# # 将数据转换为3D输入，timesteps=3，3条数据预测1条 [samples, timesteps, features]
# train_X = train_X.reshape((train_X.shape[0], n_hours, n_features))
# test_X = test_X.reshape((test_X.shape[0], n_hours, n_features))
# print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
#
# # 设计网络
# model = Sequential()
# model.add(LSTM(32, input_shape=(train_X.shape[1], train_X.shape[2])))
# model.add(Dense(1))
# model.compile(loss='mae', optimizer='adam')
# # 拟合网络
# history = model.fit(train_X, train_y, epochs=100, batch_size=32, validation_data=(test_X, test_y), verbose=2, shuffle=False)
#
#
# model.summary()
# print(model.summary())
# os.environ["PATH"] += os.pathsep + r'C:\Program Files (x86)\Graphviz 2.28\bin'
# plot_model(model,'modle.png', show_shapes=True)
#
# # plot history
# plt.plot(history.history['loss'], label='train')
# plt.plot(history.history['val_loss'], label='test')
# plt.legend()
# plt.show()
#
# # 执行预测
# yhat = model.predict(test_X)
# # 将数据格式化成 n行 * 24列
# test_X = test_X.reshape((test_X.shape[0], n_hours*n_features))
# # 将预测列据和后7列数据拼接，因后续逆缩放时，数据形状要符合 n行*8列 的要求
# inv_yhat = concatenate((yhat, test_X[:, -(n_features -1):]), axis=1)
# # 对拼接好的数据进行逆缩放
# inv_yhat = scaler.inverse_transform(inv_yhat)
# inv_yhat = inv_yhat[:,0]
#
# test_y = test_y.reshape((len(test_y), 1))
# # 将真实列据和后7列数据拼接，因后续逆缩放时，数据形状要符合 n行*8列 的要求
# inv_y = concatenate((test_y, test_X[:, -(n_features -1):]), axis=1)
# # 对拼接好的数据进行逆缩放
# inv_y = scaler.inverse_transform(inv_y)
# inv_y = inv_y[:,0]
#
# # 计算RMSE误差值
# rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
# print('Test RMSE: %.3f' % rmse)
#
#
# plt.plot(inv_y, label='true')
# plt.plot(inv_yhat, label='pred')
# plt.legend()
# plt.show()




# line_x_df = pd.read_csv(r'E:\公交客流预测\data\6路公交日客流量.csv',parse_dates=["date"], index_col=[0])
#
# #增加前一天
# line_x_df['pre_date_flow'] = line_x_df.loc[:,['population']].shift(1)
# #5日，10日移动平均
# line_x_df['MA5'] = line_x_df['population'].rolling(5).mean()
# line_x_df['MA10'] = line_x_df['population'].rolling(10).mean()
#
# print(line_x_df.columns)
# # print(line_x_df)
#
# line_x_df.dropna(inplace=True)
# print(line_x_df)
#
# def sliding_windows(data, seq_length):
#     x = []
#     y = []
#
#     for i in range(len(data)-seq_length-1):
#         _x = data[i:(i+seq_length)]
#         _y = data[i+seq_length]
#         x.append(_x)
#         y.append(_y)
#
#     return np.array(x),np.array(y)
#
# sc = MinMaxScaler()
# training_data = sc.fit_transform(line_x_df)
#
# seq_length = 4
# x, y = sliding_windows(training_data, seq_length)
#
# train_size = int(len(y) * 0.67)
# test_size = len(y) - train_size
#
# dataX = Variable(torch.Tensor(np.array(x)))
# dataY = Variable(torch.Tensor(np.array(y)))
#
# trainX = Variable(torch.Tensor(np.array(x[0:train_size])))
# trainY = Variable(torch.Tensor(np.array(y[0:train_size])))
#
# testX = Variable(torch.Tensor(np.array(x[train_size:len(x)])))
# testY = Variable(torch.Tensor(np.array(y[train_size:len(y)])))
#
#
#
# # scaler = MinMaxScaler()
# # for col in line_x_df.columns:
# #     line_x_df[col] = scaler.fit_transform(line_x_df[col].values.reshape(-1, 1))
# #
# # total_len = line_x_df.shape[0]
# # print("df.shape=",line_x_df.shape)
# # print("df_len=", total_len)
# #
# #
# # # LSTM模型
# input_dim = line_x_df.shape[1]    # 数据的特征数
# hidden_dim = 32                           # 隐藏层的神经元个数
# num_layers = 2                            # LSTM的层数
# output_dim = 1                            # 预测值的特征数
#
# class LSTM(nn.Module):
#     def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
#         super(LSTM, self).__init__()
#         # Hidden dimensions
#         self.hidden_dim = hidden_dim
#
#         # Number of hidden layers
#         self.num_layers = num_layers
#
#         # 建立LSTM
#         # batch_first = True 使input和output的tensor形状变为(batch_dim, seq_dim,, feature_dim)
#         self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first = True)
#         # Readout layer 在LSTM后再加一个全连接层，因为是回归问题，所以不能在线性层后加激活函数
#         self.fc = nn.Linear(hidden_dim, output_dim)
#
#     def forward(self, x):
#         # Initialize hidden state with zero
#         h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
#         # x.size(0)就是batch_size
#         # Initialize cell state
#         c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
#
#         # One time step
#         # We need to detach as we are doing truncated backpropagation through time (BPTT)
#         # If we don't, we'll backprop all the way to the start even after going through another batch
#         out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
#
#         out = self.fc(out)
#
#         return out
#
# # 创建两个列表，用来存储数据的特征和标签
# data_feat, data_target = [], []
#
# # 设每条数据序列有20组数据
# seq = 5
#
# for index in range(len(line_x_df - seq)):
#     # 构建特征集
#     data_feat.append(line_x_df.drop('population', axis= 1)[index : index + seq].values)
#     # 构建target集
#     data_target.append(line_x_df['population'][index : index + seq])
#
# print('data_feat',data_feat)
#
# # 将特征集和目标集整理成numpy数组
# data_feat = np.array(data_feat)
# data_target = np.array(data_target)
#
# # 按照8:2划分训练集和测试集
# test_size = int(np.round(0.2 * line_x_df.shape[0]))
# train_size = data_feat.shape[0] - test_size
#
# seq = 10
# x, y =[], []
# for i in range(total_len-seq):
#
#
#
# X_train = torch.from_numpy(data_feat[:train_size].reshape(-1,seq,input_dim)).type(torch.Tensor)
# # 这里第一个维度自动确定，我们认为其为batch_size，因为在LSTM类的定义中，设置了batch_first=True
# X_test  = torch.from_numpy(data_feat[train_size:].reshape(-1,seq,input_dim)).type(torch.Tensor)
# y_train = torch.from_numpy(data_target[:train_size].reshape(-1,seq,1)).type(torch.Tensor)
# y_test  = torch.from_numpy(data_target[train_size:].reshape(-1,seq,1)).type(torch.Tensor)
#
# print('x_train.shape = ',X_train.shape)
# print('y_train.shape = ',y_train.shape)
# print('x_test.shape = ',X_test.shape)
# print('y_test.shape = ',y_test.shape)
#
# # 设置GPU加速
# # device = torch.device('cuda')
#
# # LSTM模型实例化
# 实例化模型

# model = LSTM(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers)
#
# # 定义优化器和损失函数
# optimiser = torch.optim.Adam(model.parameters(), lr=0.01) # 使用Adam优化算法
# loss_fn = torch.nn.MSELoss(size_average=True)             # 使用均方差作为损失函数
#
# # 设定数据遍历次数
# num_epochs = 100
#
# # 打印模型结构
# print(model)
#
#
# # 打印模型各层的参数尺寸
# for i in range(len(list(model.parameters()))):
#     print(list(model.parameters())[i].size())
#
# # train model
# hist = np.zeros(num_epochs)
# for t in range(num_epochs):
#     # Initialise hidden state
#     # Don't do this if you want your LSTM to be stateful
#     # model.hidden = model.init_hidden()
#
#     # Forward pass
#     y_train_pred = model(trainX)
#
#     loss = loss_fn(y_train_pred, trainY)
#     if t % 10 == 0 and t != 0:  # 每训练十次，打印一次均方差
#         print("Epoch ", t, "MSE: ", loss.item())
#     hist[t] = loss.item()
#
#     # Zero out gradient, else they will accumulate between epochs 将梯度归零
#     optimiser.zero_grad()
#
#     # Backward pass
#     loss.backward()
#
#     # Update parameters
#     optimiser.step()

# 计算训练得到的模型在训练集上的均方差
# y_train_pred = model(trainX)
# loss_fn(y_train_pred, trainY).item()
#
# # make predictions
# y_test_pred = model(testX)
# loss_fn(y_test_pred, testY).item()
#
# "训练集效果图"
# # 无论是真实值，还是模型的输出值，它们的维度均为（batch_size, seq, 1），seq=20
# # 我们的目的是用前20天的数据预测今天的股价，所以我们只需要每个数据序列中第20天的标签即可
# # 因为前面用了使用DataFrame中shift方法，所以第20天的标签，实际上就是第21天的股价
# pred_value = y_train_pred.detach().numpy()[:,-1,0]
# true_value = trainY.detach().numpy()[:,-1,0]
#
# plt.plot(pred_value, label="Preds")    # 预测值
# plt.plot(true_value, label="Data")    # 真实值
# plt.legend()
# plt.show()
#
#
#
#
#
#
#








"""

corr = X.corr()

plt.figure(figsize=(10,6))
ax = sns.heatmap(corr, xticklabels=corr.columns,yticklabels=corr.columns, linewidths=0.2, cmap="RdYlGn",annot=True)
plt.title("变量间相关系数")
plt.show()

weekday_dum = pd.get_dummies(line_x_df['weekday'], prefix='weekday')
weather_d_dum = pd.get_dummies(line_x_df['weather_d'], prefix='weather_d')
weather_n_dum = pd.get_dummies(line_x_df['weather_n'], prefix='weather_n')
temperature_l_dum = pd.get_dummies(line_x_df['temperature_l'], prefix='temperature_l')
temperature_h_dum = pd.get_dummies(line_x_df['temperature_h'], prefix='temperature_h')
wind_d_dum = pd.get_dummies(line_x_df['wind_d'], prefix='wind_d')
wind_n_dum = pd.get_dummies(line_x_df['wind_n'], prefix='wind_n')


line_x_dum = pd.concat([line_x_df, weekday_dum, weather_d_dum, weather_n_dum, temperature_l_dum, temperature_h_dum, wind_d_dum, wind_n_dum], axis=1)
print(line_x_dum)
line_x_dum.drop(['weekday', 'weather_d', 'weather_n', 'temperature_l', 'temperature_h', 'wind_d', 'wind_n'], axis=1, inplace=True)

print(line_x_dum.columns)


plt.figure(figsize=(20,4))
line_x_dum.corr()['population'].sort_values(ascending=False).plot(kind='bar')
plt.title('人流与其他变量相关性')

plt.show()
"""







#
# line_x_df.dropna(inplace=True)
# X = line_x_df.drop(['population'], axis=1)
# Y = line_x_df['population']
# corr = X.corr()
#
# """
# plt.figure(figsize=(10,6))
# ax = sns.heatmap(corr, xticklabels=corr.columns,yticklabels=corr.columns, linewidths=0.2, cmap="RdYlGn",annot=True)
# plt.title("变量间相关系数")
# plt.show()
#
# weekday_dum = pd.get_dummies(line_x_df['weekday'], prefix='weekday')
# weather_d_dum = pd.get_dummies(line_x_df['weather_d'], prefix='weather_d')
# weather_n_dum = pd.get_dummies(line_x_df['weather_n'], prefix='weather_n')
# temperature_l_dum = pd.get_dummies(line_x_df['temperature_l'], prefix='temperature_l')
# temperature_h_dum = pd.get_dummies(line_x_df['temperature_h'], prefix='temperature_h')
# wind_d_dum = pd.get_dummies(line_x_df['wind_d'], prefix='wind_d')
# wind_n_dum = pd.get_dummies(line_x_df['wind_n'], prefix='wind_n')
#
#
# line_x_dum = pd.concat([line_x_df, weekday_dum, weather_d_dum, weather_n_dum, temperature_l_dum, temperature_h_dum, wind_d_dum, wind_n_dum], axis=1)
# print(line_x_dum)
# line_x_dum.drop(['weekday', 'weather_d', 'weather_n', 'temperature_l', 'temperature_h', 'wind_d', 'wind_n'], axis=1, inplace=True)
#
# print(line_x_dum.columns)
#
#
# plt.figure(figsize=(20,4))
# line_x_dum.drop('date',axis = 1).corr()['population'].sort_values(ascending=False).plot(kind='bar')
# plt.title('人流与其他变量相关性')
# plt.savefig('D:\桌面\\1.png', dpi=400, bbox_inches='tight')
#
# plt.show()"""
# x = np.array(X)
# y = np.array(Y)
# scaler_x = MinMaxScaler()
# scaler_y = MinMaxScaler()
# x = scaler_x.fit_transform(x)#自变量和因变量分别归一化
# y = scaler_y.fit_transform(np.reshape(y,(len(y),1)))
# x_length = x.shape[0]
# split = int(x_length*0.8)
# x_train, x_test = x[:split], x[split:]
# y_train, y_test = y[:split], y[split:]
#
#
# x_train = np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))
# x_test = np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))
#
# # LSTM
# model = Sequential()
# model.add(LSTM(32, input_dim=1))
# model.add(Dense(1))
# model.compile(loss='mean_squared_error', optimizer='adam')
# model.fit(x_train, y_train, epochs=100, batch_size=1, verbose=2)
# from keras.utils import plot_model
# model.summary()
# print(model.summary())
# import  os
# os.environ["PATH"] += os.pathsep + r'C:\Program Files (x86)\Graphviz 2.28\bin'
# plot_model(model,'modle.png', show_shapes=True)
#
# # plot_model(model, "model.png")
# # 预测
# predict = model.predict(x_test)
# plt.figure(figsize=(15,6))
# plt.title('预测结果图')
# y_test = scaler_y.inverse_transform(np.reshape(y_test,(len(y_test),1)))
# predict = scaler_y.inverse_transform(predict)
# plt.plot(y_test.ravel(),label='真实值')
# plt.plot(predict,label='预测值')
# plt.xticks([])
# plt.legend()
# plt.show()



