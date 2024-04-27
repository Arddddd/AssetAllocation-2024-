
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# 生成模拟数据
np.random.seed(42)  # 保证结果可复现
num_assets = 10
predicted_beta = np.random.rand(num_assets)  # 模拟每个资产对某个宏观经济因子的敞口β
actual_returns = np.random.randn(num_assets)  # 模拟每个资产的实际回报

# 用线性回归模型来拟合数据
model = LinearRegression()
model.fit(predicted_beta.reshape(-1, 1), actual_returns)

# 模型的斜率即Φ值
phi_value = model.coef_[0]

# 模型的截距
intercept = model.intercept_

# 绘制数据点和回归线
plt.figure(figsize=(10, 6))
plt.scatter(predicted_beta, actual_returns, color='blue', label='Actual Returns')
plt.plot(predicted_beta, model.predict(predicted_beta.reshape(-1, 1)), color='red', label='Cross-sectional Regression Line')
plt.axline((0, 0), slope=1, linestyle='--', color='grey', label='45-degree Line')

plt.title('Cross-sectional Regression to find Phi Value')
plt.xlabel('Predicted Beta')
plt.ylabel('Actual Returns')
plt.legend()
plt.grid(True)
plt.show()

# # 输出模型的斜率和截距
# phi_value, intercept