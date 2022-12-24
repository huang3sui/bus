# -*- coding: utf-8 -*-
"""
# @Time    : 2022/12/20 09:27
# @Author  : Huang
# @File    : 处理刷卡数据.py
# @Software: PyCharm 
# @Comment : 
"""

import pandas as pd
import numpy as np


line_df = pd.read_csv('E:\毕设\公交\data\gd_train_data.txt', header = None,
                      names = ['use_city', 'line_name', 'terminal_id', 'card_id', 'creat_city', 'deal_time', 'card_type'])

# 处理时间字段
line_df['deal_time'] = pd.to_datetime(line_df['deal_time'].apply(lambda x: str(x) + '00'))
line_df['date'] = line_df['deal_time'].dt.date
line_df['time'] = line_df['deal_time'].dt.time
# 顺序调整
line_df = line_df[['deal_time', 'date', 'time', 'use_city', 'line_name', 'terminal_id', 'card_id', 'creat_city', 'card_type']]
# 交易时间排序
line_df = line_df.sort_values(by = 'deal_time')

# 广东省大部分公交运营时间为 6:00~23:00
line_df = line_df[line_df['deal_time'].dt.hour.isin(np.arange(6, 22))]

# 两条线路分开
for i in ['线路6', '线路11']:
    line_x_df = line_df[line_df['line_name'] == i]
    if i == '线路6':
        line_6_df = line_x_df.drop('line_name', axis = 1)
    else:
        line_11_df = line_x_df.drop('line_name', axis = 1)

# 分开存储
line_6_df.to_csv('E:\公交客流预测\data\\6路公交刷卡记录.csv', encoding = 'gbk', index = False)
line_11_df.to_csv('E:\公交客流预测\data\\11路公交刷卡记录.csv', encoding = 'gbk', index = False)











