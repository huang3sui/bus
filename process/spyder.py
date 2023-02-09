# -*- coding: utf-8 -*-
"""
# @Time    : 2023/2/8 19:49
# @Author  : Huang
# @File    : spyder.py
# @Software: PyCharm 
# @Comment : 
"""

import pandas as pd
import requests
from bs4 import BeautifulSoup

def set_link():
    #year参数为需要爬取数据的年份
    link = []
    for i in range(8,13):#这里因为今天是2022/10/3  故我们只获得 1-9月的url 其他的这里改为13即可
        #一年有12个月份
        if i < 10:
            url='http://www.tianqihoubao.com/aqi/guangzhou-20140{}.html'.format(i)
        else:
            url='http://www.tianqihoubao.com/aqi/guangzhou-2014{}.html'.format(i)
        link.append(url)
    return link

def get_data():
    link = set_link()
    for url in link:
        response = requests.get(url)
        html = response.text
        soup = BeautifulSoup(html)
        tr_list = soup.find_all("tr")
        
        for data in tr_list[1:]:
            item = {}
            sub_data = data.text.split()
            print(sub_data)
            item['date'] = sub_data[0]
            #item['air_level'] = sub_data[1]
            item['air'] = sub_data[2]
            datas.append(item)


datas = []
get_data()
df = pd.DataFrame(datas)
df.to_csv('E:/公交客流预测/data/air.csv',index =False,encoding = 'gbk')

'''
def get_data():
    link = set_link()
    for url in link:
        response = requests.get(url)
        html = response.text
        soup = BeautifulSoup(html)
        tr_list = soup.find_all("li")
        
        for data in tr_list[:]:
            item = {}
            sub_data = data.text.split()
            print(sub_data)

            item['date'] = sub_data[0]
            item['temperature_h'] = sub_data[2].split('°')[0]
            item['temperature_l'] = sub_data[3].split('°')[0]
            item['weather'] = sub_data[4]
            item['wind'] = sub_data[5]
            item['air'] = sub_data[6]
            # item['air_level'] = sub_data[7]
            datas.append(item)
            
        


datas = []
get_data()
df = pd.DataFrame(datas)
df.to_csv('E:/公交客流预测/data/weather.csv',index =False,encoding = 'gbk')

get_data()
'''