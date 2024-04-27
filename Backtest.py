import pandas as pd
import datetime
import numpy as np


# 相关系数矩阵变为协方差矩阵
def corr2cov(corr, std):
    cov = corr * np.outer(std, std)
    return cov


# 协方差矩阵变为相关系数矩阵
def cov2corr(cov):
    std = np.sqrt(np.diag(cov))  # cov_df对角开平方根得出一个向量
    corr = cov / np.outer(std, std)  # R = C / sqrt(diag(C)) ，C满足多元正态分布
    # np.outer(a,b)计算矩阵的外积，把a当做列向量，b当做行向量，得到一个n*m的矩阵
    corr[corr < -1], corr[corr > 1] = -1, 1  # numerical error
    return corr


# 获得调仓日期数据
def position_date(all_data, vPeriod="month", Is_yc="ym"):
    if vPeriod == "week" and Is_yc == 'yc':
        Date_arr = all_data.index.astype(str).unique()
        Hold_date = [Date_arr[0]]
        s_date = Date_arr[0]
        Hold_date.append(s_date)
        for i in range(len(Date_arr) - 2):
            if (pd.to_datetime(Date_arr[i]) - pd.to_datetime(s_date)).days >= 7:
                Hold_date.append(Date_arr[i + 1])
                s_date = Date_arr[i]
    if vPeriod == "month" and Is_yc == 'yc':
        Date_arr = all_data.index.astype(str).unique()
        Hold_date = [Date_arr[0]]
        for i in range(len(Date_arr) - 2):
            if (Date_arr[i][0:7] != Date_arr[i + 1][0:7]):
                Hold_date.append(Date_arr[i + 1])
    if vPeriod == "quarter" and Is_yc == 'yc':
        Date_arr = all_data.index.astype(str).unique()
        tmp_Hold_date = [Date_arr[0]]
        for i in range(len(Date_arr) - 2):
            if (Date_arr[i][0:7] != Date_arr[i + 1][0:7]):
                tmp_Hold_date.append(Date_arr[i + 1])
        Hold_date = [Date_arr[0]]
        s_date = tmp_Hold_date[0]
        Hold_date.append(s_date)
        for i in range(len(tmp_Hold_date) - 2):
            if (pd.to_datetime(tmp_Hold_date[i]) - pd.to_datetime(s_date)).days >= 65:
                Hold_date.append(tmp_Hold_date[i])
                s_date = tmp_Hold_date[i]
    if vPeriod == "halfyear" and Is_yc == 'yc':
        Date_arr = all_data.index.astype(str).unique()
        tmp_Hold_date = [Date_arr[0]]
        for i in range(len(Date_arr) - 2):
            if (Date_arr[i][0:7] != Date_arr[i + 1][0:7]):
                tmp_Hold_date.append(Date_arr[i + 1])
        Hold_date = []
        s_date = tmp_Hold_date[0]
        Hold_date.append(s_date)
        for i in range(len(tmp_Hold_date) - 2):
            if (pd.to_datetime(tmp_Hold_date[i]) - pd.to_datetime(s_date)).days >= 175:
                Hold_date.append(tmp_Hold_date[i])
                s_date = tmp_Hold_date[i]
    if vPeriod == "year" and Is_yc == 'yc':
        Date_arr = all_data.index.astype(str).unique()
        tmp_Hold_date = [Date_arr[0]]
        for i in range(len(Date_arr) - 2):
            if (Date_arr[i][0:7] != Date_arr[i + 1][0:7]):
                tmp_Hold_date.append(Date_arr[i + 1])
        Hold_date = []
        s_date = tmp_Hold_date[0]
        Hold_date.append(s_date)
        for i in range(len(tmp_Hold_date) - 2):
            if (pd.to_datetime(tmp_Hold_date[i]) - pd.to_datetime(s_date)).days >= 350:
                Hold_date.append(tmp_Hold_date[i])
                s_date = tmp_Hold_date[i]
    # 获得调仓日期数据
    if vPeriod == "week" and Is_yc == 'ym':
        Date_arr = all_data.index.astype(str).unique()
        Hold_date = []
        s_date = Date_arr[0]
        Hold_date.append(s_date)
        for i in range(len(Date_arr) - 2):
            if (pd.to_datetime(Date_arr[i]) - pd.to_datetime(s_date)).days >= 7:
                Hold_date.append(Date_arr[i])
                s_date = Date_arr[i]
    if vPeriod == "month" and Is_yc == 'ym':
        Date_arr = all_data.index.astype(str).unique()
        Hold_date = []
        for i in range(len(Date_arr) - 2):
            if (Date_arr[i][0:7] != Date_arr[i + 1][0:7]):  # 截出月份，若月份不相同则已换月
                Hold_date.append(Date_arr[i])  # 筛出月底日期
    if vPeriod == "quarter" and Is_yc == 'ym':
        Date_arr = all_data.index.astype(str).unique()
        tmp_Hold_date = []
        for i in range(len(Date_arr) - 2):
            if (Date_arr[i][0:7] != Date_arr[i + 1][0:7]):
                tmp_Hold_date.append(Date_arr[i])
        Hold_date = []
        s_date = tmp_Hold_date[0]
        Hold_date.append(s_date)
        for i in range(len(tmp_Hold_date) - 2):
            if (pd.to_datetime(tmp_Hold_date[i]) - pd.to_datetime(s_date)).days >= 65:
                Hold_date.append(tmp_Hold_date[i])
                s_date = tmp_Hold_date[i]
    if vPeriod == "halfyear" and Is_yc == 'ym':
        Date_arr = all_data.index.astype(str).unique()
        tmp_Hold_date = []
        for i in range(len(Date_arr) - 2):
            if (Date_arr[i][0:7] != Date_arr[i + 1][0:7]):
                tmp_Hold_date.append(Date_arr[i])
        Hold_date = []
        s_date = tmp_Hold_date[0]
        Hold_date.append(s_date)
        for i in range(len(tmp_Hold_date) - 2):
            if (pd.to_datetime(tmp_Hold_date[i]) - pd.to_datetime(s_date)).days >= 175:
                Hold_date.append(tmp_Hold_date[i])
                s_date = tmp_Hold_date[i]
    if vPeriod == "year" and Is_yc == 'ym':
        Date_arr = all_data.index.astype(str).unique()
        tmp_Hold_date = []
        for i in range(len(Date_arr) - 2):
            if (Date_arr[i][0:7] != Date_arr[i + 1][0:7]):
                tmp_Hold_date.append(Date_arr[i])
        Hold_date = []
        s_date = tmp_Hold_date[0]
        Hold_date.append(s_date)
        for i in range(len(tmp_Hold_date) - 2):
            if (pd.to_datetime(tmp_Hold_date[i]) - pd.to_datetime(s_date)).days >= 350:
                Hold_date.append(tmp_Hold_date[i])
                s_date = tmp_Hold_date[i]
    return Hold_date


# 回测函数
def Back_test(Hold_df, price_table, vfee=0.0003):
    start_nav = 1
    vData = Hold_df.copy()
    Date_arr = vData['Date'].unique()
    nav = pd.DataFrame()
    for i_date in Date_arr:
        tmp_vData = vData[vData['Date'] == i_date]  # i_date的权重
        tmp_date = Date_arr[Date_arr > i_date]  # i_date之后的交易日
        if len(tmp_date) > 0:
            end_date = tmp_date[0]
        else:
            end_date = datetime.datetime.now()  # 如果tmp_date是权重的最后一个时间点，则该时间点后至最新日期的权重都为该点的权重
        tmp_price = price_table[(price_table.index > i_date) & (price_table.index <= end_date)][
            tmp_vData['S_INFO_WINDCODE']]  # i_date后一个月的各资产的净值序列
        for j_name in tmp_vData['S_INFO_WINDCODE']:
            tmp_price[j_name] = tmp_price[j_name] / (tmp_price[j_name][0])  # 改为从1开始的净值序列
        tmp_return = tmp_price.copy()
        for j_name in tmp_vData['S_INFO_WINDCODE']:
            tmp_return[j_name] = tmp_return[j_name].pct_change()  # i_date后一个月的各资产的收益率序列
        tmp_return = tmp_return.fillna(0)

        L = sum(tmp_vData['Weight_L'])

        for j_name in tmp_vData['S_INFO_WINDCODE']:
            tmp_return[j_name] = tmp_return[j_name] * L  # 各资产的收益率序列 * 杠杆
        for j_name in tmp_vData['S_INFO_WINDCODE']:
            tmp_price[j_name] = np.cumprod(1 + tmp_return[j_name])  # 计算各资产的净值序列

        tmp_nav = (np.dot(tmp_price.values, tmp_vData['Weight_L'] / L)) * start_nav * (1 - vfee)
        start_nav = tmp_nav[len(tmp_nav) - 1]  # 更新下一时段的初始净值
        tmp_nav_df = pd.DataFrame(tmp_nav, index=tmp_price.index)
        tmp_nav_df.columns = ['Nav']
        nav = pd.concat([nav, tmp_nav_df])
    return nav


# 计算累计收益函数
def accumReturn(Asset_nav):
    Asset_nav_values = Asset_nav.values
    accum_return = Asset_nav_values[len(Asset_nav_values) - 1] / Asset_nav_values[0] - 1
    return round(accum_return, 3)


# 计算年化收益函数
def annReturn(Asset_nav, annual_day=240):
    Asset_nav_values = Asset_nav.values
    during_day = len(Asset_nav_values)
    annual_rate = (Asset_nav_values[len(Asset_nav_values) - 1] / Asset_nav_values[0]) ** (annual_day / during_day) - 1
    return round(annual_rate, 3)


# 计算年化波动率
def annVolatility(Asset_nav, annual_day=240):
    ret = Asset_nav.pct_change(1).dropna()
    ann_vol = np.std(ret) * np.sqrt(annual_day)
    return round(ann_vol, 3)


# 计算滚动年化波动率（时间序列）
def RollannVolatility(Asset_nav, annual_day=240, windows=60):
    ret = Asset_nav.pct_change(1).dropna()
    roll_ann_vol = (ret.rolling(windows).std()) * np.sqrt(annual_day)
    roll_ann_vol = roll_ann_vol.dropna()
    return roll_ann_vol


# 计算年化夏普
def sharpRatio(Asset_nav, annual_day=240):
    sharp_ratio = annReturn(Asset_nav, annual_day) / annVolatility(Asset_nav, annual_day)
    return round(sharp_ratio, 3)


# 计算最大回撤
def max_drawdown(Asset_nav):
    Asset_nav_values = Asset_nav.values
    acc_max = np.maximum.accumulate(Asset_nav_values)
    max_drawdown = np.max((acc_max - Asset_nav_values) / acc_max)
    return round(max_drawdown, 3)
