# -*- coding: utf-8 -*-
"""
# @Time    : 2022/12/29 22:40
# @Author  : Huang
# @File    : 特征标准化.py
# @Software: PyCharm
# @Comment :
"""

import pandas as pd
from sklearn import preprocessing

def sacle_line_day_population():
    for i in ['6', '11']:
        line_x_day_population = pd.read_csv(r'E:\公交客流预测\data\{}路公交日客流量.csv'.format(i))

        '''one-hot col
        'weekday', 'is_holiday', 'weather_d', 'weather_n','wind_d', 'wind_n',]'''
        weekday_dum = pd.get_dummies(line_x_day_population['weekday'], prefix='weekday')
        is_holiday_dum = pd.get_dummies(line_x_day_population['is_holiday'], prefix='isholiday')

        line_x_dum = pd.concat([line_x_day_population,weekday_dum, is_holiday_dum],axis=1)
        print(line_x_dum)
        line_x_dum.drop(['weekday', 'is_holiday'],axis=1, inplace=True)
        print(line_x_dum)

        maxmin_Scaler = preprocessing.MinMaxScaler()
        temperature_scale = maxmin_Scaler.fit(line_x_dum[['temperature_h', 'temperature_l', 'temperature_avg','temperature_abs', 'weather_d', 'weather_n', 'weather_avg']].values)
        temperature_scale_Df = pd.DataFrame(temperature_scale.transform(line_x_dum[['temperature_h', 'temperature_l', 'temperature_avg','temperature_abs', 'weather_d', 'weather_n', 'weather_avg']].values),
                                            columns=['temperature_h_scale', 'temperature_l_scale', 'temperature_avg_scale','temperature_abs_scale', 'weather_d_scale', 'weather_n_scale', 'weather_avg_scale'])
        line_x_dum_scale = pd.concat([line_x_dum, temperature_scale_Df], axis = 1)
        line_x_dum_scale.drop(['temperature_h', 'temperature_l', 'temperature_avg','temperature_abs', 'weather_d', 'weather_n', 'weather_avg'],
                              axis=1, inplace=True)
        # print(line_x_dum_scale[['temperature_h_scale', 'temperature_l_scale', 'temperature_avg_scale','temperature_abs_scale']])

        zScore_scale = preprocessing.StandardScaler()
        wea_wind_scale = zScore_scale.fit(line_x_dum[['wind_avg', 'wind_d', 'wind_n']].values)
        wea_wind_scale_Df = pd.DataFrame(wea_wind_scale.transform(line_x_dum[['wind_avg','wind_d', 'wind_n']].values),
                                         columns=['wind_avg_scale','wind_d_scale', 'wind_n_scale'])
        line_x_dum_scale = pd.concat([line_x_dum_scale, wea_wind_scale_Df], axis=1)
        line_x_dum_scale.drop(['wind_avg','wind_d', 'wind_n'],
                              axis = 1, inplace=True)

        line_x_dum_scale.to_csv('E:\公交客流预测\data\{}路公交日客流量_scale.csv'.format(i),encoding = 'gbk', index = False)
        print(line_x_dum_scale)

def sacle_line_hour_population():
    for i in ['6', '11']:
        line_x_hour_population = pd.read_csv(r'E:\公交客流预测\data\{}路公交各时段客流量.csv'.format(i))
        print(len(line_x_hour_population.columns.to_list()))

        # one-hot col    7
        # ['weekday', 'is_holiday', 'time_type', 'weather_d', 'weather_n',  'wind_d', 'wind_n']
        weekday_dum = pd.get_dummies(line_x_hour_population['weekday'], prefix='weekday')
        is_holiday_dum = pd.get_dummies(line_x_hour_population['is_holiday'], prefix='isholiday')
        time_type_dum = pd.get_dummies(line_x_hour_population['time_type'], prefix='time_type')

        line_x_hour_dum = pd.concat(
            [line_x_hour_population, weekday_dum, is_holiday_dum, time_type_dum], axis = 1)
        line_x_hour_dum.drop(['weekday', 'is_holiday', 'time_type'],
                             axis = 1, inplace = True)

        #  z_score
        # ['weather_avg', 'wind_avg']
        zScore_scale = preprocessing.StandardScaler()
        wea_wind_scale = zScore_scale.fit(line_x_hour_dum[['wind_avg',  'wind_d', 'wind_n']].values)
        wea_wind_scale_Df = pd.DataFrame(wea_wind_scale.transform(line_x_hour_dum[['wind_avg',  'wind_d', 'wind_n']].values),
                                         columns=['wind_avg_scale',  'wind_d_scale', 'wind_n_scale'])
        line_x_hour_dum_scale = pd.concat([line_x_hour_dum, wea_wind_scale_Df], axis=1)
        line_x_hour_dum_scale.drop(['wind_avg',  'wind_d', 'wind_n'],
                              axis=1, inplace=True)

        # MaxMin
        # ['temperature_h', 'temperature_l', 'temperature_avg', 'temperature_abs']
        maxmin_Scaler = preprocessing.MinMaxScaler()
        temperature_scale = maxmin_Scaler.fit(
            line_x_hour_dum[['temperature_h', 'temperature_l', 'temperature_avg', 'temperature_abs', 'weather_d', 'weather_n', 'weather_avg']].values)
        temperature_scale_Df = pd.DataFrame(temperature_scale.transform(
            line_x_hour_dum[['temperature_h', 'temperature_l', 'temperature_avg', 'temperature_abs', 'weather_d', 'weather_n', 'weather_avg']].values),
                                            columns=['temperature_h_scale', 'temperature_l_scale', 'temperature_avg_scale','temperature_abs_scale', 'weather_d_scale', 'weather_n_scale', 'weather_avg_scale'])
        line_x_hour_dum_scale = pd.concat([line_x_hour_dum_scale, temperature_scale_Df], axis=1)
        line_x_hour_dum_scale.drop(['temperature_h', 'temperature_l', 'temperature_avg', 'temperature_abs', 'weather_d', 'weather_n', 'weather_avg'],
                              axis=1, inplace=True)
        print(line_x_hour_dum_scale)
        line_x_hour_dum_scale.to_csv('E:\公交客流预测\data\{}路公交各时段客流量_scale.csv'.format(i),encoding = 'gbk', index = False)

if __name__ == '__main__':
    sacle_line_day_population()
    sacle_line_hour_population()




