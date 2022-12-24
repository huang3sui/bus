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

weather_df = pd.read_csv(r'E:\公交客流预测\data\天气情况.csv')

extend_feature = {}

# 星期
week_count = len(weather_df) // 7 + 1 # 周数
extend_feature['week'] = list(np.arange(1, 8)) * week_count
# 2014-08-01   周五
extend_feature['week'] = extend_feature['week'][: len(weather_df) - 3]
extend_feature['week'] = [5, 6, 7] + extend_feature['week']

# 是否节假日
# 中秋节：09—06 ~ 09-08
# 国庆节：10-01 ~ 10-07
# 元旦: 01-01 ~ 01-03
extend_feature['is_festival'] = [0] * len(weather_df)
extend_feature['is_festival'][36 : 36 + 3] = [1] * 3
extend_feature['is_festival'][61 : 61 + 7] = [1] * 7
extend_feature['is_festival'][153 : 153 + 3] = [1] * 3

# 节假日第几天
extend_feature['day_festival'] = [0] * len(weather_df)
extend_feature['day_festival'][36 : 36 + 3] = [1, 2, 3]
extend_feature['day_festival'][61 : 61 + 7] = [1, 1, 1, 2, 2, 3, 3] # 国庆放假方案：3,5,7
extend_feature['day_festival'][153 : 153 + 3] = [1, 2, 3]

# 是否工作日
extend_feature['is_work'] = [int(x < 6) for x in extend_feature['week']]
extend_feature['is_work'][36 : 36 + 3] = [0] * 3
extend_feature['is_work'][61 : 61 + 7] = [0] * 7
# 国庆调休：2014-09-28    2014-10-11
extend_feature['is_work'][61 - 3] = 1
extend_feature['is_work'][61 + 10] = 1
extend_feature['is_work'][153 : 153 + 3] = [0] * 3
# 元旦调休：2015-01-04
extend_feature['is_work'][153 + 3] = 1

# 时间类型
# 工作日 : 0   周末 : 1  节假日 : 2
extend_feature['date_type'] = [int(x >= 6) for x in extend_feature['week']]
# 中秋
extend_feature['date_type'][36 : 36 + 3] = [2] * 3
# 国庆
extend_feature['date_type'][61 : 61 + 7] = [2] * 7
# 国庆调休：2014-09-28    2014-10-11
extend_feature['date_type'][61 - 3] = 0
extend_feature['date_type'][61 + 10] = 0
# 元旦
extend_feature['date_type'][153 : 153 + 3] = [2] * 3
# 元旦调休：2015-01-04
extend_feature['date_type'][153 + 3] = 0

# 构建特征数据框
col = ['week', 'is_festival', 'day_festival','is_work', 'date_type']
feature_df = pd.DataFrame(extend_feature, columns = col)
print(feature_df)

weather_feature_df = pd.merge(weather_df, feature_df, left_index = True, right_index = True)

weather_feature_df.to_csv('E:\公交客流预测\data\天气&特征.csv', encoding = 'gbk', index = False)

weather_feature_df['date'] = pd.to_datetime(weather_feature_df['date'])
print(weather_feature_df)
for i in [6, 11]:
    line_x_df = pd.read_csv(r'E:\公交客流预测\data\{}路公交刷卡记录.csv'.format(i), encoding = 'gbk')
    line_x_df['date'] = pd.to_datetime(line_x_df['date'])

    line_x_day_population = line_x_df.groupby('date')['card_id'].count().reset_index()
    #print(line_x_df['date'])
    #print(line_x_day_population)
    #print(line_x_day_population['date'])
    # 各时段客流量
    line_x_hour_population = line_x_df.groupby('deal_time')['card_id'].count().reset_index()
    line_x_hour_population['deal_time'] = pd.to_datetime(line_x_hour_population['deal_time'])
    line_x_hour_population['date'] = line_x_hour_population['deal_time'].dt.date
    line_x_hour_population['date'] = pd.to_datetime(line_x_hour_population['date'])


    def split_time(A):
        A = pd.to_datetime(A)
        if A.hour in np.arange(6, 10):
            return 0
        elif A.hour in np.arange(16, 19):
            return 1
        else:
            return 2

    line_x_hour_population['time_type'] = line_x_hour_population['deal_time'].apply(lambda x: split_time(x))
    line_x_hour_feature_df =pd.merge(line_x_hour_population, weather_feature_df[weather_feature_df['date'].dt.year<2015], on = 'date', how = 'outer')
    line_x_hour_feature_df = line_x_hour_feature_df.rename(columns={"card_id" : "population"})
    line_x_hour_feature_df = line_x_hour_feature_df.sort_values(by = 'deal_time')
    line_x_hour_feature_df.to_csv('E:\公交客流预测\data\\{}路公交各时段客流量.csv'.format(i), encoding = 'gbk', index = False)

    line_x_feature_df = pd.merge(line_x_day_population, weather_feature_df[weather_feature_df['date'].dt.year<2015], on = 'date', how = 'outer')
    line_x_feature_df = line_x_feature_df.rename(columns={"card_id" : "population"})
    # 交易时间排序
    line_x_feature_df = line_x_feature_df.sort_values(by='date')
    print(line_x_feature_df)
    line_x_feature_df.to_csv('E:\公交客流预测\data\\{}路公交日客流量.csv'.format(i), encoding = 'gbk', index = False)