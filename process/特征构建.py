# -*- coding: utf-8 -*-
"""
# @Time    : 2022/12/20 16:14
# @Author  : Huang
# @File    : 特征构建.py
# @Software: PyCharm 
# @Comment : 
"""

import pandas as pd
import numpy as np
import numpy as np


def add_peak_type(date_type, h):
    h = pd.to_datetime(h)
    if date_type == 0 and h.hour in np.arange(7, 10):  # 工作日早高峰
        return 0
    elif date_type == 0 and h.hour in np.arange(16, 19):  # 工作日晚高峰
        return 1
    elif date_type == 1 and h.hour in (np.arange(8, 9) and np.arange(14, 18)):  # 节假日高峰
        return 2
    else:
        return 3

def add_holiday(date, w):
    w2h = ['2014-09-08', '2014-10-01', '2014-10-02', '2014-10-03', '2014-10-06', '2014-10-07']
    h2w = ['2014-09-28', '2014-10-11']
    # w = w.astype(int)
    if date in w2h:
        return 1
    elif date not in h2w and (w >= 6):
        return 1
    else:
        return 0



weather_df = pd.read_csv(r'E:\公交客流预测\data\天气情况.csv')

weather_df['date'] = pd.to_datetime(weather_df['date'])

for i in [6, 11]:
    line_x_df = pd.read_csv(r'E:\公交客流预测\data\{}路公交刷卡记录.csv'.format(i), encoding = 'gbk')
    line_x_df['deal_time'] = pd.to_datetime(line_x_df['deal_time'])
    line_x_df['weekday'] = line_x_df['deal_time'].dt.weekday + 1

    line_x_df['is_holiday'] = line_x_df.apply(lambda x: add_holiday(x['date'], x['weekday']), axis=1)
    line_x_df['time_type'] = line_x_df.apply(lambda x: add_peak_type(x['is_holiday'], x['deal_time']), axis=1)

    line_x_day_population = line_x_df.groupby(['date', 'weekday', 'is_holiday'])['card_id'].count().reset_index()
    #print(line_x_df['date'])
    #print(line_x_day_population)
    #print(line_x_day_population['date'])
    # 各时段客流量
    line_x_hour_population = line_x_df.groupby(['deal_time', 'weekday', 'is_holiday', 'time_type'])['card_id'].count().reset_index()
    line_x_hour_population['deal_time'] = pd.to_datetime(line_x_hour_population['deal_time'])
    line_x_hour_population['date'] = line_x_hour_population['deal_time'].dt.date
    line_x_hour_population['date'] = pd.to_datetime(line_x_hour_population['date'])

    line_x_hour_feature_df =pd.merge(line_x_hour_population, weather_df[weather_df['date'].dt.year<2015], on = 'date', how = 'outer')
    line_x_hour_feature_df = line_x_hour_feature_df.rename(columns={"card_id" : "population"})
    line_x_hour_feature_df = line_x_hour_feature_df.sort_values(by = 'deal_time')
    line_x_hour_feature_df.to_csv('E:\公交客流预测\data\\{}路公交各时段客流量.csv'.format(i), encoding = 'gbk', index = False)

    line_x_day_population['date'] = pd.to_datetime(line_x_day_population['date'])
    line_x_feature_df = pd.merge(line_x_day_population, weather_df[weather_df['date'].dt.year<2015], on = 'date', how = 'outer')
    line_x_feature_df = line_x_feature_df.rename(columns={"card_id" : "population"})
    # 交易时间排序
    line_x_feature_df = line_x_feature_df.sort_values(by='date')
    print(line_x_feature_df)
    line_x_feature_df.to_csv('E:\公交客流预测\data\\{}路公交日客流量.csv'.format(i), encoding = 'gbk', index = False)