# -*- coding: utf-8 -*-
"""
# @Time    : 2022/12/20 10:37
# @Author  : Huang
# @File    : 处理天气数据.py
# @Software: PyCharm 
# @Comment : 
"""

import pandas as pd
import re


weather_report = pd.read_csv('E:\公交客流预测\data\gd_weather_report.txt', header = None,
                       names = ['date', 'weather', 'temperature', 'wind_direction_force'])
weather_report['date'] = pd.to_datetime(weather_report['date'], format='%Y-%m-%d')

def splitInfo(x):
    info = x.split('/')
    return info[0], info[-1]

# 各字段分开
# 天气状况
weather_report['weather_d'] = weather_report['weather'].apply(lambda x: splitInfo(x)[0])
weather_report['weather_n'] = weather_report['weather'].apply(lambda x: splitInfo(x)[1])

pd.concat([weather_report['weather_n'], weather_report['weather_d']]).drop_duplicates()
weather_map = {'晴' : 0,
               '多云' : 1, '阴' : 1, '霾' : 1,
               '阵雨' : 2, '雷阵雨' : 2,
               '小雨' : 3,
               '小到中雨' : 4,
               '中雨' : 5,
               '中到大雨' : 6,
               '大雨' : 7,
               '大到暴雨' : 8,
               }
weather_report['weather_d'] = weather_report['weather_d'].map(weather_map)
weather_report['weather_n'] = weather_report['weather_n'].map(weather_map)
weather_report['weather_avg'] = (weather_report['weather_d'] + weather_report['weather_n']) / 2


# 气温
weather_report['temperature_h'] = weather_report['temperature'].apply(lambda x: int(re.sub(r'\D', '',splitInfo(x)[0])))
weather_report['temperature_l'] = weather_report['temperature'].apply(lambda x: int(re.sub(r'\D', '',splitInfo(x)[1])))
# 日平均温度
weather_report['temperature_avg'] = (weather_report['temperature_h'] + weather_report['temperature_l']) / 2
# 日温差
weather_report['temperature_abs'] = abs(weather_report['temperature_h'] - weather_report['temperature_l'])

# 风向风力
weather_report['wind_direction_force_d'] = weather_report['wind_direction_force'].apply(lambda x: splitInfo(x)[0])
weather_report['wind_direction_force_n'] = weather_report['wind_direction_force'].apply(lambda x: splitInfo(x)[1])

pd.concat([weather_report['wind_direction_force_d'], weather_report['wind_direction_force_d']]).drop_duplicates()
wind_map = {'无持续风向≤3级' : 0,
            '东北风3-4级' : 1, '北风3-4级' : 1, '无持续风向微风转3-4级' : 1, '东南风3-4级' : 1, '北风微风转3-4级' : 1,
            '东风4-5级' : 2, '北风4-5级' : 2
            }
weather_report['wind_d'] = weather_report['wind_direction_force_d'].map(wind_map)
weather_report['wind_n'] = weather_report['wind_direction_force_n'].map(wind_map)
weather_report['wind_avg'] = (weather_report['wind_d'] + weather_report['wind_n']) / 2

weather_report = weather_report.drop(['weather', 'temperature', 'wind_direction_force',
                                      'wind_direction_force_d', 'wind_direction_force_n'], axis = 1)

weather_report.to_csv('E:\公交客流预测\data\天气情况.csv', encoding = 'utf-8', index = False)
