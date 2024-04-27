import numpy as np
import pandas as pd
import cvxpy as cp
import statsmodels.api as sm

# 基础设定
n_assets = 10
n_factors = 6
n_industries = 3
n_observations = 250  # 五年的交易日假设为250天

# 随机生成资产价格序列和因子时间序列
asset_prices = np.random.randn(n_observations, n_assets)
macro_factors = np.random.randn(n_observations, n_factors)

# 计算资产的日收益率
asset_returns = np.diff(asset_prices, axis=0) / asset_prices[:-1, :]
asset_returns = np.nan_to_num(asset_returns)

# 计算因子暴露矩阵 A (使用线性回归)
A = np.zeros((n_assets, n_factors))
for i in range(n_assets):
    model = sm.OLS(asset_returns[:, i], sm.add_constant(macro_factors[:-1, :]))
    results = model.fit()
    A[i, :] = results.params[1:]  # 排除截距，保留因子暴露

# 计算因子的协方差矩阵 Σ
factor_cov_matrix = np.cov(macro_factors[:-1, :], rowvar=False)

# 使用 Fama-French 三因子模型计算资产特质风险的协方差矩阵 Q
# 这里简化，直接使用资产收益率的残差方差
Q = np.zeros((n_assets, n_assets))
for i in range(n_assets):
    fama_model = sm.OLS(asset_returns[:, i], sm.add_constant(macro_factors[:-1, :]))
    fama_results = fama_model.fit()
    residuals = fama_results.resid
    Q[i, i] = np.var(residuals)

# 构造行业暴露矩阵 π
industry_exposure = np.random.randint(0, 2, size=(n_assets, n_industries))

# 已知的宏观因子目标暴露向量 e
e = np.random.rand(n_factors)

# 创建资产权重变量
weights = cp.Variable(n_assets)

# 目标函数
risk = cp.quad_form(weights, Q) + 0.8 * cp.quad_form(A.T @ weights - e, factor_cov_matrix)
objective = cp.Minimize(risk)

# 约束条件
constraints = [
    weights >= 0.05,  # 单个资产最小权重设为5%
    cp.sum(weights) == 1,  # 权重和为1
]

# 添加行业配置的约束
for i in range(n_industries):
    industry_weight = industry_exposure[:, i] @ weights
    constraints += [
        industry_weight >= 0.3,
        industry_weight <= 0.5
    ]

# 解决优化问题
problem = cp.Problem(objective, constraints)
problem.solve()

# # 输出结果
# weights.value, A, factor_cov_matrix, Q, industry_exposure