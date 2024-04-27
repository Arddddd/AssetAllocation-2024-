

'''
1.首先，生成一个50x50的正定协方差矩阵。
2.然后，对于每个T值（100到5000，步长100），我们将进行1000次模拟：
- 每次模拟中，基于该协方差矩阵生成一个50xT的资产收益率矩阵。
- 计算该收益率矩阵的样本协方差矩阵。
- 计算样本协方差矩阵与真实协方差矩阵的Frobeniusnorm，并记录下来。
3.对于每个T，计算这1000次模拟的Frobenius norm的均值。
4.最后，绘制x轴为T，y轴为每个T对应的Frobenius norm均值的折线图。
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import wishart

# 步骤1: 生成50x50的正定协方差矩阵
N = 5
df = N  # 自由度，保证矩阵正定
scale_matrix = np.eye(N)  # 缩放矩阵，这里使用单位矩阵
true_cov_matrix = wishart.rvs(df=df, scale=scale_matrix, size=1)

# 初始化
T_values = range(100, 3001, 100)  # T的取值
m = 1000  # 每个T的模拟次数
errors = []  # 存储每个T的误差

for T in T_values:
    error_sum = 0
    for _ in range(m):
        # 步骤2: 生成资产收益率矩阵
        asset_returns = np.random.multivariate_normal(np.zeros(N), true_cov_matrix, T)

        # 计算样本协方差矩阵
        sample_cov_matrix = np.cov(asset_returns, rowvar=False)

        # 计算Frobenius norm
        frob_norm = np.linalg.norm(sample_cov_matrix - true_cov_matrix, 'fro')
        error_sum += frob_norm

    # 步骤3: 计算平均误差
    avg_error = error_sum / m
    errors.append(avg_error)

# # 步骤4: 绘制折线图
plt.plot(T_values, errors, marker='o')
plt.xlabel('T')
plt.ylabel('Average Frobenius Norm Error')
plt.title('Error vs. T')
plt.grid(True)

plt.savefig(r'test.png')  # 保存图片
plt.close()

# plt.show()
