# -*- coding: utf-8 -*-
"""
# @Time    : 2023/2/8 14:42
# @Author  : Huang
# @File    : 相关性分析.py
# @Software: PyCharm 
# @Comment : 
"""

import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

plt.rcParams['font.sans-serif'] = ['simsun']  # 指定默认字体
plt.rcParams['font.size'] = 10.5
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
plt.rcParams['savefig.dpi'] = 400

for line in ['6', '11']:
    df = pd.read_csv(r'E:\公交客流预测\data\{}路公交各时段客流量.csv'.format(line), parse_dates=["deal_time"])
    # print(df.columns)
    feature = df.drop(['deal_time', 'date'], axis=1)
    # # 获取所有特征变量
    feature_weather = df.drop(['deal_time', 'date', 'weekday', 'is_holiday', 'time_type'],axis=1)
    print(feature_weather)
    # 得到相关性矩阵
    corr = feature_weather.corr()
    # 特征矩阵热力图可视化
    plt.figure(figsize=(25,15))
    ax = sns.heatmap(corr, xticklabels=corr.columns,yticklabels=corr.columns, linewidths=0.2, cmap="RdYlGn",annot=True)
    plt.title("变量间相关系数")
    plt.xticks(rotation=20)
    plt.yticks(rotation=20)
    plt.savefig('E:\公交客流预测\EDA图象\线路{}变量间相关性热力图_天气情况.png'.format(line), bbox_inches='tight')
    plt.show()

    # 特征相关性分析，是为了查看特征之间是否存在多重共线性
    # # 如果有多重共线性的话，就要对相关性特别高的特征进行有选择的删除
    # # 从热力图的结果来看，MA5和MA10的相关性是最高的，但也可以接受，不需要对特征进行删除
    # #
    # # 目标相关性分析
    df_onehot = feature

    weekday_dum = pd.get_dummies(df_onehot['weekday'], prefix='weekday')
    weather_d_dum = pd.get_dummies(df_onehot['weather_d'], prefix='weather_d')
    weather_n_dum = pd.get_dummies(df_onehot['weather_n'], prefix='weather_n')
    temperature_l_dum = pd.get_dummies(df_onehot['temperature_l'], prefix='temperature_l')
    temperature_h_dum = pd.get_dummies(df_onehot['temperature_h'], prefix='temperature_h')
    wind_d_dum = pd.get_dummies(df_onehot['wind_d'], prefix='wind_d')
    wind_n_dum = pd.get_dummies(df_onehot['wind_n'], prefix='wind_n')
    is_holiday_dum = pd.get_dummies(df_onehot['is_holiday'], prefix='is_holiday')
    time_type_dum = pd.get_dummies(df_onehot['time_type'], prefix='time_type')
    # air_dum = pd.get_dummies(df_onehot['air'], prefix='air')

    df_onehot = pd.concat([df_onehot, weekday_dum, weather_d_dum, weather_n_dum, temperature_l_dum, temperature_h_dum, wind_d_dum, wind_n_dum, time_type_dum, is_holiday_dum], axis=1)
    # print(line_x_dum)
    df_onehot.drop(['weekday', 'weather_d', 'weather_n', 'temperature_l', 'temperature_h', 'wind_d', 'wind_n', 'time_type', 'is_holiday'], axis=1, inplace=True)
    #
    # print(line_x_dum.columns)
    # # 可视化展示
    plt.figure(figsize=(20,5.5))
    df_onehot.corr()['flow'].sort_values(ascending=False).plot(kind='bar')
    # plt.title('人流与其他变量相关性')
    # plt.xticks(rotation=60)
    plt.savefig('E:\公交客流预测\EDA图象\线路{}人流与其他变量相关性.png'.format(line), bbox_inches='tight')
    #
    plt.show()
    col_h = []
    col_d = []
    col_w = []
    for i in range(1, 16):
        h_col = 'M_{}h'.format(i)
        df[h_col] = df.loc[:,['flow']].shift(i)
        col_h.append(h_col)

        day_seq = 15 + i
        day_col = 'M_D{}h'.format(i)
        df[day_col] = df.loc[:,['flow']].shift(day_seq)
        col_d.append(day_col)

        week_seq = 111 + 1
        week_col = 'M_W{}h'.format(i)
        df[week_col] = df.loc[:,['flow']].shift(week_seq)
        col_w.append(week_col)
    # print(len(col_h), len(col_d), len(col_w))
    df.to_csv('E:\公交客流预测\data\{}路公交各时段客流量_Move.csv'.format(line),encoding = 'gbk', index = False)



    corr_df = pd.DataFrame()
    corr_df['Move_hours'] = col_h
    # print(corr_df)
    corr_h = []
    for i in col_h:
        corr_h.append(round(df['flow'].corr(df[i]), 2))
    print(len(corr_h),corr_h)
    corr_df['Hour'] = corr_h

    corr_d = []
    for i in col_d:
        corr_d.append(round(df['flow'].corr(df[i]), 2))
    print(len(corr_d),corr_d)
    corr_df['Day'] = corr_d

    corr_w = []
    for i in col_w:
        corr_w.append(round(df['flow'].corr(df[i]), 2))
    # print(len(corr_w),corr_w)
    corr_df['Week'] = corr_w
    # print(corr_df.columns)
    corr_df = corr_df.rename(columns={"0" : "Move_hours"})
    # print(corr_df)
    corr_df.to_csv('E:\公交客流预测\data\线路{}历史时刻与客流相关性.csv'.format(line),encoding = 'gbk', index = False)
    plt.plot(corr_df['Move_hours'].tolist(), corr_df['Hour'].tolist(), 'o--', label="Hour")
    plt.plot(corr_df['Move_hours'].tolist(), corr_df['Day'].tolist(), 'v:', label="Day")
    plt.plot(corr_df['Move_hours'].tolist(), corr_df['Week'].tolist(), '*-.', label="Week")
    plt.ylabel('相关性')
    plt.legend(loc="best")
    plt.savefig('E:\公交客流预测\EDA图象\\线路{}历史时刻与客流相关性.png'.format(line), bbox_inches='tight')
    plt.show()





