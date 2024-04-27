import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# 假设我们有一些模拟的数据来演示这个策略
# 数据包括10个资产的实际月度回报率和模型预测的回报率

# 生成模拟数据
np.random.seed(0)
dates = pd.date_range('2020-01-01', periods=12, freq='M')
assets = [f'Asset_{i}' for i in range(1, 11)]
actual_returns = pd.DataFrame(np.random.randn(12, 10), index=dates, columns=assets)
predicted_returns = actual_returns * np.random.uniform(0.8, 1.2, size=(12, 10)) + np.random.uniform(-0.02, 0.02,
                                                                                                    size=(12, 10))

# 将模拟数据放入DataFrame
data = pd.concat([actual_returns.stack(), predicted_returns.stack()], axis=1)
data.reset_index(inplace=True)
data.columns = ['Date', 'Asset', 'Actual_Return', 'Predicted_Return']


# 进行时间序列和横截面分析
# 首先，定义时间序列和横截面回归的函数
def time_series_regression(data, asset):
    X = data[['Predicted_Return']]
    y = data['Actual_Return']
    model = LinearRegression().fit(X, y)
    return model.coef_[0], model.intercept_


def cross_sectional_regression(data):
    X = data[['Predicted_Return']]
    y = data['Actual_Return']
    model = LinearRegression().fit(X, y)
    return model.coef_[0], model.intercept_


# 按资产分组运行时间序列回归
ts_results = data.groupby('Asset').apply(time_series_regression)
ts_results.columns = ['TS_Slope', 'TS_Intercept']

# 按日期分组运行横截面回归
cs_results = data.groupby('Date').apply(cross_sectional_regression)
cs_results.columns = ['CS_Slope', 'CS_Intercept']

# 模拟回测
trading_signals = {}

for date, group in data.groupby('Date'):
    # 计算资产的残差，即实际回报与预测回报的差值
    group['Residual'] = group['Actual_Return'] - group['Predicted_Return']

    # 基于残差，找到反应过度的资产（预测回报高于实际回报的资产）
    over_traded_assets = group[group['Residual'] < 0]['Asset'].tolist()

    # 基于残差，找到反应不足的资产（预测回报低于实际回报的资产）
    under_traded_assets = group[group['Residual'] > 0]['Asset'].tolist()

    # 存储交易信号
    trading_signals[date] = {'Short': over_traded_assets, 'Long': under_traded_assets}

# 将交易信号转换为DataFrame
trading_signals_df = pd.DataFrame(trading_signals).T
trading_signals_df.head()

