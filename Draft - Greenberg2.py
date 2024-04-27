import numpy as np
import cvxpy as cp
import statsmodels.api as sm

# 设置
n_assets = 10
n_factors = 6
n_observations = 250  # 五年假设250个交易日
n_industries = 3

# 生成资产收益率和宏观因子数据
asset_returns = np.random.randn(n_observations, n_assets)
macro_factors = np.random.randn(n_observations, n_factors)

# 计算因子暴露矩阵 A
A = np.zeros((n_assets, n_factors))
for i in range(n_assets):
    model = sm.OLS(asset_returns[:, i], sm.add_constant(macro_factors))
    results = model.fit()
    A[i, :] = results.params[1:]  # 排除截距

# 计算宏观因子的协方差矩阵
factor_cov_matrix = np.cov(macro_factors, rowvar=False)

# 简化资产特质风险协方差矩阵 Q 计算
Q = np.diag(np.var(asset_returns - macro_factors @ A.T, axis=0))

# 行业暴露矩阵 π
industry_exposure = np.random.randint(0, 2, size=(n_assets, n_industries))

# 宏观因子目标暴露向量 e
e = np.random.rand(n_factors)

# 资产权重变量
weights = cp.Variable(n_assets)

# 目标函数
risk = cp.quad_form(weights, Q) + 0.8 * cp.quad_form(A.T @ weights - e, factor_cov_matrix)
objective = cp.Minimize(risk)

# 约束
constraints = [weights >= 0.05, cp.sum(weights) == 1]
for i in range(n_industries):
    industry_weight = industry_exposure[:, i] @ weights
    constraints += [industry_weight >= 0.3, industry_weight <= 0.5]

# 解决优化问题
problem = cp.Problem(objective, constraints)
problem.solve(solver=cp.OSQP, verbose=True)

# # 输出结果
# weights.value