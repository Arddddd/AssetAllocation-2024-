import numpy as np
import statsmodels.api as sm

# 设置随机种子
np.random.seed(0)

# 假设有250个交易日的数据
T = 250

# 生成模拟的市场因子数据，简化为正态分布的随机数
market_premium = np.random.normal(loc=0.05, scale=0.15, size=T)
SMB = np.random.normal(loc=0.02, scale=0.07, size=T)
HML = np.random.normal(loc=0.03, scale=0.08, size=T)

# 生成模拟的无风险利率，固定为3%
risk_free_rate = 0.03

# 生成一个证券的回报率数据，同样简化为正态分布的随机数
asset_returns = np.random.normal(loc=0.1, scale=0.2, size=T)

# 将无风险利率转换为与资产回报率相同长度的向量
risk_free_rate_series = np.full(T, risk_free_rate)

# 计算资产的超额回报率
excess_returns = asset_returns - risk_free_rate_series

# 准备回归分析的数据
X = np.column_stack((market_premium, SMB, HML))

# 添加常数项（截距）到因子数据中
X_sm = sm.add_constant(X)

# 创建回归模型
model = sm.OLS(excess_returns, X_sm)

# 拟合模型
results = model.fit()

# 获取回归残差（时间序列）
residuals = results.resid

# 检查回归结果和残差的前几个值
results.summary(), residuals[:10]