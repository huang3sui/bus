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

weather_df = pd.read_csv(r'E:\公交客流预测\data\天气情况.csv', parse_dates=["date"])

weather_df['date'] = pd.to_datetime(weather_df['date'])

for i in [6, 11]:
    line_x_df = pd.read_csv(r'E:\公交客流预测\data\{}路公交刷卡记录.csv'.format(i), parse_dates=["deal_time"], encoding = 'gbk')
    # line_x_df['deal_time'] = pd.to_datetime(line_x_df['deal_time'])
    line_x_df['weekday'] = line_x_df['deal_time'].dt.weekday + 1

    line_x_df['is_holiday'] = line_x_df.apply(lambda x: add_holiday(x['date'], x['weekday']), axis=1)
    line_x_df['time_type'] = line_x_df.apply(lambda x: add_peak_type(x['is_holiday'], x['deal_time']), axis=1)
    # 日客流量
    line_x_day_flow = line_x_df.groupby(['date', 'weekday', 'is_holiday'])['card_id'].count().reset_index()
    #print(line_x_df['date'])
    #print(line_x_day_flow)
    #print(line_x_day_flow['date'])
    # 各时段客流量
    line_x_hour_flow = line_x_df.groupby(['deal_time', 'weekday', 'is_holiday', 'time_type'])['card_id'].count().reset_index()
    # line_x_hour_flow['deal_time'] = pd.to_datetime(line_x_hour_flow['deal_time'])
    line_x_hour_flow['date'] = line_x_hour_flow['deal_time'].dt.date
    line_x_hour_flow['date'] = pd.to_datetime(line_x_hour_flow['date'])

    line_x_hour_feature_df =pd.merge(line_x_hour_flow, weather_df[weather_df['date'].dt.year<2015], on = 'date', how = 'outer')
    air = pd.read_csv(r'E:\公交客流预测\data\\air.csv', parse_dates=["date"], encoding='gbk')
    line_x_hour_feature_df = pd.merge(line_x_hour_feature_df, air, on = 'date', how = 'outer')
    line_x_hour_feature_df = line_x_hour_feature_df.rename(columns={"card_id" : "flow"})
    line_x_hour_feature_df = line_x_hour_feature_df.sort_values(by = 'deal_time')
    line_x_hour_feature_df.dropna(inplace = True)
    line_x_hour_feature_df = line_x_hour_feature_df[(line_x_hour_feature_df['date'] >= '2014-08-20')]



    line_x_hour_feature_df.to_csv('E:\公交客流预测\data\\{}路公交各时段客流量.csv'.format(i), encoding = 'gbk', index = False)

    line_x_day_flow['date'] = pd.to_datetime(line_x_day_flow['date'])
    line_x_feature_df = pd.merge(line_x_day_flow, weather_df[weather_df['date'].dt.year<2015], on = 'date', how = 'outer')
    line_x_feature_df = pd.merge(line_x_feature_df, air, on = 'date', how = 'outer')

    line_x_feature_df = line_x_feature_df.rename(columns={"card_id" : "flow"})
    line_x_feature_df.dropna(inplace=True)
    # 交易时间排序
    line_x_feature_df = line_x_feature_df.sort_values(by='date')
    print(line_x_feature_df)
    line_x_feature_df.to_csv('E:\公交客流预测\data\\{}路公交日客流量.csv'.format(i), encoding = 'gbk', index = False)