# -*- coding: utf-8 -*-
"""
# @Time    : 2022/12/4 23:06
# @Author  : Huang
# @File    : 刷卡数据EDA.py
# @Software: PyCharm 
# @Comment : 
"""

import pandas as pd
from pyecharts import options as opts
from pyecharts.charts import Calendar, Line
from matplotlib import pyplot as plt
from matplotlib.patches import ConnectionPatch
import numpy as np

plt.rcParams['font.sans-serif'] = ['simsun']  # 指定默认字体
plt.rcParams['font.size'] = 10.5
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
plt.rcParams['savefig.dpi'] = 400



# 数据读取
def data(path):
    line_x_df = pd.read_csv(path, encoding = 'gbk')
    #date_flow['客流量'] = line_6_df.groupby('date')['card_id'].count()
    day_flow = line_x_df.groupby(['date'])['card_id'].count()
    day_data = []
    for i in day_flow.items():
        day_data.append(i)

    hour_day_flow = line_x_df.groupby('deal_time')['card_id'].count()
    hour_day_data = []
    for i in hour_day_flow.items():
        hour_day_data.append(i)
    print(day_data)

    # 卡类型客流量
    x_type_count = line_x_df['card_type'].value_counts()
    print(x_type_count)
    labels = x_type_count.index.tolist()
    print(labels)
    sizes = x_type_count.tolist()
    print(sizes)

    return day_data, day_flow, hour_day_flow, labels, sizes

# 日历图
def draw_day_date(day_data, day_flow, line):
    (
        Calendar()
        .add(
            "",
            day_data,
            calendar_opts=opts.CalendarOpts(
                range_= ["2014-08-01",'2014-12-31'],
                daylabel_opts=opts.CalendarDayLabelOpts(name_map="cn"),
                monthlabel_opts=opts.CalendarMonthLabelOpts(name_map="cn"),
            ),
        )
        .set_global_opts(
            title_opts=opts.TitleOpts(title="公交客流日历图"),
            visualmap_opts=opts.VisualMapOpts(
                max_=day_flow.max(),
                min_=day_flow.min(),
                orient="horizontal",
                is_piecewise=True,
                pos_top="230px",
                pos_left="100px",
            ),
            toolbox_opts=opts.ToolboxOpts(
                is_show=True,
                pos_top="top",
                pos_left="right",
                feature={
                    "saveAsImage": {},
                    "restore": {},
                    "dataView": {}
                }
            )
        )
        .render("E:\公交客流预测\EDA图象\{}_公交客流日历图.html".format(line))
    )

# 日客流量折线图
def draw_day_flow(day_flow, line):
    (
        Line(init_opts=opts.InitOpts(width="1200px", height="600px"))
        .add_xaxis(xaxis_data=day_flow.index.tolist())
        .add_yaxis(
            series_name="客流量",
            y_axis=day_flow.tolist(),
            is_smooth=False
        )
        .set_global_opts(
            tooltip_opts=opts.TooltipOpts(trigger="axis", axis_pointer_type="cross"),
            legend_opts=opts.LegendOpts(pos_left="left"),
            datazoom_opts=[
                opts.DataZoomOpts(is_show = True)
            ],
            yaxis_opts=opts.AxisOpts(name="客流量", type_="value"),
            toolbox_opts=opts.ToolboxOpts(
                is_show=True,
                pos_top="top",
                pos_left="right",
                feature={
                    "saveAsImage": {},
                    "restore": {},
                    "dataView": {}
                }
            )
        )
        .set_series_opts(
            markarea_opts=opts.MarkAreaOpts(
                is_silent=False,
            ),
            axisline_opts=opts.AxisLineOpts(),
            label_opts=opts.LabelOpts(is_show=False)
        )
        .render("E:\公交客流预测\EDA图象\{}_日客流量.html".format(line))
    )

# 各时段客流量折线图
def draw_hour_day_flow(hour_day_flow, line):
    (
        Line(init_opts=opts.InitOpts(width="1200px", height="600px"))
        .add_xaxis(xaxis_data=hour_day_flow.index.tolist())
        .add_yaxis(
            series_name="客流量",
            y_axis=hour_day_flow.tolist(),
            is_smooth=False
        )
        .set_global_opts(
            tooltip_opts=opts.TooltipOpts(trigger="axis", axis_pointer_type="cross"),
            legend_opts=opts.LegendOpts(pos_left="left"),
            datazoom_opts=[
                opts.DataZoomOpts(is_show = True)
            ],
            yaxis_opts=opts.AxisOpts(name="客流量", type_="value"),
            toolbox_opts=opts.ToolboxOpts(
                is_show=True,
                pos_top="top",
                pos_left="right",
                feature={
                    "saveAsImage": {},
                    "restore": {},
                    "dataView": {}
                }
            )
        )
        .set_series_opts(
            markarea_opts=opts.MarkAreaOpts(
                is_silent=False,
            ),
            axisline_opts=opts.AxisLineOpts(),
            label_opts=opts.LabelOpts(is_show=False)
        )
        .render("E:\公交客流预测\EDA图象\{}_各时段客流量.html".format(line))
    )

# 列表求和
def ls_sum(ls):
    ls_count = 0
    for i in ls:
        ls_count += i
    return ls_count

# 列表分隔  转化为复合饼图数据所需格式
def split_ls(sizes, labels, ls_all):
    label1 = []
    label2 = []
    size1 = []
    size2 = []
    for i in range(len(sizes)):
        if float(sizes[i]) / ls_all < 0.05:
            label2.append(labels[i])
            size2.append(sizes[i])
        else:
            label1.append(labels[i])
            size1.append(sizes[i])
    label1.append('其他卡')
    print(size2)
    other_sum= ls_sum(size2)

    size1.append(other_sum)
    return label1, size1, label2, size2

def draw_complex_pie(label1, label2, size1, size2, line):
    #制画布
    fig = plt.figure(figsize=(9,5.0625))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    fig.subplots_adjust(wspace=0)

    #大饼图的制作
    explode=(0,0,0,0.1)
    ax1.pie(size1, autopct='%1.1f%%',startangle=30,labels=label1,explode=explode)
    #小饼图的制作
    width=0.2
    ax2.pie(size2, autopct='%1.1f%%',labels=label2,
            radius=0.5, explode = (0,0,0.2,0))

    #使用ConnectionPatch画出两个饼图的间连线
    #先得到饼图边缘的数据
    theta1, theta2 = ax1.patches[len(label1)-1].theta1, ax1.patches[len(label1)-1].theta2
    center, r = ax1.patches[len(label1)-1].center,ax1.patches[len(label1)-1].r
    #画出上边缘的连线
    x = r*np.cos(np.pi/180*theta2)+center[0]
    y = np.sin(np.pi/180*theta2)+center[1]
    con = ConnectionPatch(
        xyA=(-width/2,0.5),xyB=(x,y),
        coordsA='data',
        coordsB='data',
        axesA=ax2,
        axesB=ax1
    )
    con.set_linewidth(2)
    con.set_color=([0,0,0])
    ax2.add_artist(con)
    #画出下边缘的连线
    x = r*np.cos(np.pi/180*theta1)+center[0]
    y = np.sin(np.pi/180*theta1)+center[1]
    con = ConnectionPatch(
        xyA=(-width/2,-0.5),
        xyB=(x,y),
        coordsA='data',
        coordsB='data',
        axesA=ax2,
        axesB=ax1
    )
    con.set_linewidth(2)
    con.set_color=([0,0,0])
    ax2.add_artist(con)

    plt.savefig('E:\公交客流预测\EDA图象\{}路公交不同类型乘客消费占比.png'.format(line), dpi=400, bbox_inches = 'tight')
    plt.show()

if __name__ == "__main__":
    line_6_path = 'E:\公交客流预测\data\\6路公交刷卡记录.csv'
    line_11_path = 'E:\公交客流预测\data\\11路公交刷卡记录.csv'
    line_6_data = data(line_6_path)
    line_11_data = data(line_11_path)
    draw_day_date(line_6_data[0], line_6_data[1], 6)    # 6路公交客流日历图
    draw_day_date(line_11_data[0], line_11_data[1], 11)  # 11路公交客流日历图
    draw_day_flow(line_6_data[1], 6)   # 6路公交日客流折线图
    draw_day_flow(line_11_data[1], 11)  # 11路公交日客流折线图
    draw_hour_day_flow(line_6_data[2], 6)  # 6路公交各时段客流折线图
    draw_hour_day_flow(line_11_data[2], 11)  # 11路公交各时段客流折线图

    # 6路公交各种卡类型复合饼图
    line_6_size_sum = ls_sum(line_6_data[-1])
    line_6_split_data = split_ls(line_6_data[-1], line_6_data[-2], line_6_size_sum)
    draw_complex_pie(line_6_split_data[0], line_6_split_data[2], line_6_split_data[1], line_6_split_data[3], 6)

    # 11路公交各种卡类型复合饼图
    line_11_size_sum = ls_sum(line_11_data[-1])
    line_11_split_data = split_ls(line_11_data[-1], line_11_data[-2], line_11_size_sum)
    draw_complex_pie(line_11_split_data[0], line_11_split_data[2], line_11_split_data[1], line_11_split_data[3],11)






