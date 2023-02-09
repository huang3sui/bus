# -*- coding: utf-8 -*-
"""
# @Time    : 2023/2/7 1:36
# @Author  : Huang
# @File    : 天气数据EDA.py
# @Software: PyCharm 
# @Comment : 
"""

import pandas as pd
from matplotlib import pyplot as plt
from datetime import timedelta
from matplotlib.patches import ConnectionPatch
import numpy as np
import warnings
warnings.filterwarnings("ignore")


plt.rcParams['font.sans-serif'] = ['simsun']  # 指定默认字体
plt.rcParams['font.size'] = 10.5
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
plt.rcParams['savefig.dpi'] = 400
# plt.rcParams['bbox_inches'] = 'tight'
#
#
#
#
# # 对应时间内的客流
# def date_flow(df, start, end):
#     flow_data = df[(df['deal_time']>= start) & (df['deal_time']<= end)]
#     flow_data['time'] = flow_data['deal_time'].dt.time
#     flow_data['time'] = flow_data['time'].astype(str)
#     return flow_data
#
#
# for i in ['6', '11']:
#     # print(flow)
#     df = pd.read_csv(r'E:\公交客流预测\data\{}路公交各时段客流量.csv'.format(i), parse_dates=["deal_time"])
#     # print(df)
#     flow = df[['deal_time', 'flow']]
#     data_9_1 = date_flow(flow, '2014-09-01 06:00:00', '2014-09-01 21:00:00')
#     data_12_1 = date_flow(flow, '2014-12-01 06:00:00', '2014-12-01 21:00:00')
#     plt.plot(data_9_1['time'].tolist(),data_9_1['flow'].tolist(), 'o--',label="9月1日")
#     plt.plot(data_12_1['time'].tolist(),data_12_1['flow'].tolist(), 'v:',label="12月1日")
#     plt.ylabel('客流量')
#     plt.xticks(rotation = 30)
#     plt.legend(loc="best")
#     plt.savefig('E:\公交客流预测\EDA图象\line_{}_工作日各时段客流量.png'.format(i),bbox_inches='tight')
#     plt.show()
#
#     data_10_01 = date_flow(flow, '2014-10-01 06:00:00', '2014-10-01 21:00:00')
#     data_10_18 = date_flow(flow, '2014-10-18 06:00:00', '2014-10-18 21:00:00')
#     data_10_19 = date_flow(flow, '2014-10-19 06:00:00', '2014-10-19 21:00:00')
#
#     plt.plot(data_10_01['time'].tolist(),data_10_01['flow'].tolist(), 'o--',label="10月1日")
#     plt.plot(data_10_18['time'].tolist(),data_10_18['flow'].tolist(), 'v:',label="10月18日 星期六")
#     plt.plot(data_10_19['time'].tolist(),data_10_19['flow'].tolist(), '*-.', label="10月19日 星期日")
#     plt.ylabel('客流量')
#     plt.xticks(rotation = 30)
#     plt.legend(loc="best")
#     plt.savefig('E:\公交客流预测\EDA图象\line_{}_非工作日各时段客流量.png'.format(i),bbox_inches='tight')
#     plt.show()

# df_day_6 = pd.read_csv(r'E:\公交客流预测\data\6路公交日客流量.csv', parse_dates=["date"], encoding = 'gbk')
df_day_6 = pd.read_csv(r'E:\公交客流预测\data\6路公交各时段客流量.csv', encoding = 'gbk')

print(df_day_6.columns)
df_day_6_weather = df_day_6[['flow', 'weather_d', 'weather_n',
       'temperature_h', 'temperature_l', 'temperature_avg', 'wind_d',
       'wind_n']]
print(df_day_6_weather.corr()['flow'])





# day_flow_6 = df_day_6[['date', 'flow']]
# df_day_11 = pd.read_csv(r'E:\公交客流预测\data\11路公交日客流量.csv', parse_dates=["date"], encoding = 'gbk')
# day_flow_11 = df_day_11[['date', 'flow']]
#
# def day_flow(df, satart, end):
#     day_flow_data = df[(df['date'] >= '2014-09-01') & (df['date'] <= '2014-09-07')]
#     return day_flow_data
#
# week_data_6 = day_flow(day_flow_6, '2014-09-01', '2014-09-07')
# week_data_11 = day_flow(day_flow_11, '2014-09-01', '2014-09-07')
#
# x = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
# plt.plot(x,week_data_11['flow'].tolist(), 'o--',label="线路11")
# plt.plot(x,week_data_6['flow'].tolist(), 'v:',label="线路6")
# plt.ylabel('客流量')
# plt.legend(loc="best")
# plt.savefig('E:\公交客流预测\EDA图象\\不同线路公交客流.png', bbox_inches='tight')
# plt.show()
#
# df_6 = pd.read_csv(r'E:\公交客流预测\data\6路公交各时段客流量.csv', parse_dates=["deal_time"])
#
#
