# -*- coding: utf-8 -*-
"""
# @Time    : 2022/12/12 16:22
# @Author  : Huang
# @File    : LSTM.py
# @Software: PyCharm 
# @Comment : 
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM,Dense,Dropout
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
from math import sqrt

import tensorflow as tf
tf.random.set_seed(2)

df = pd.read_csv(r'E:\公交客流预测\data\6路公交刷卡记录.csv', encoding = 'gbk')
day_population = pd.DataFrame()
day_population = df.groupby('deal_time')['card_id'].count()




