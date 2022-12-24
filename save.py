# -*- coding: utf-8 -*-
"""
# @Time    : 2022/12/23 21:00
# @Author  : Huang
# @File    : save.py
# @Software: PyCharm 
# @Comment : 
"""
import pandas as pd
import numpy as np

line_x_df = pd.read_csv(r'E:\公交客流预测\data\6路公交刷卡记录.csv'.format(6), encoding='gbk')
line_x_df['deal_time'] = pd.to_datetime(line_x_df['deal_time'])

line_x_hour_population = line_x_df.groupby(['deal_time'])['card_id'].count().reset_index()
def split_time(A):
    A = pd.to_datetime(A)
    if A.hour in np.arange(6,10):
        return 0
    elif A.hour in np.arange(16,19):
        return 1
    else:
        return 2

line_x_hour_population['time_type']= line_x_hour_population['deal_time'].apply(lambda x: split_time(x))
print(line_x_hour_population)
    #line_df = line_df[line_df['deal_time'].dt.hour.isin(np.arange(6, 22))]
