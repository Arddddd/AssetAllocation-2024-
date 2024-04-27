import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

'''
假设我们是一家在线零售商，想要深入了解影响产品销售额的因素，以优化我们的营销策略。我们选择了以下几个关键自变量：
广告费用（Advertising_Cost）： 我们在各种渠道上的广告投入，包括社交媒体和其他在线平台。
社交媒体宣传投入（Social_Media_Expense）： 我们在社交媒体上的宣传和广告支出。
产品价格（Product_Price）： 我们的产品定价，可能会影响销售量。
季节性因素（Seasonality_Factor）： 季节变化对销售的影响，考虑到一些产品在特定季节可能更受欢迎。
员工满意度（employee_satisfaction ）： 猜测员工满意度可能会影响到销售额。
天气情况（Monthly_Weather_Index）： 设想每个月的天气情况可能会影响到产品的销售。
'''

'''################################################# 构建测试数据集 ####################################################'''
np.random.seed(12)
num_products = 150
advertising_cost = np.random.uniform(500, 5000, num_products)
social_media_expense = np.random.uniform(100, 1000, num_products)
product_price = np.random.uniform(20, 200, num_products)
seasonality_factor = np.random.normal(1, 0.2, num_products)

# 增加两个与销售额无关的变量
employee_satisfaction = np.random.uniform(1, 5, num_products)
monthly_weather_index = np.random.uniform(-10, 10, num_products)

# 生成销售额，考虑以上因素和噪声
sales_revenue = 1000 * advertising_cost + 500 * social_media_expense - 10 * product_price + 200 * seasonality_factor + np.random.normal(
    0, 5000, num_products)

# 创建数据框
df_sales = pd.DataFrame({
    'Advertising_Cost': advertising_cost,
    'Social_Media_Expense': social_media_expense,
    'Product_Price': product_price,
    'Seasonality_Factor': seasonality_factor,
    'Employee_Satisfaction': employee_satisfaction,
    'Monthly_Weather_Index': monthly_weather_index,
    'Sales_Revenue': sales_revenue
})

'''################################################# 逐步回归分析实现 ##################################################'''


def stepwise_regression(X_train, y_train, X_test, y_test, threshold=0.05):
    models_info = []  # 用于存储每个模型的信息
    best_model = None
    best_aic = float('inf')  # 初始化最佳AIC为正无穷
    best_bic = float('inf')  # 初始化最佳BIC为正无穷
    best_features = None

    while True:
        # 添加截距项
        X_train = sm.add_constant(X_train)
        X_test = sm.add_constant(X_test)

        # 初始化模型
        model = sm.OLS(y_train, X_train).fit()
        models_info.append({
            'Features': X_train.columns[1:],
            'R-squared': model.rsquared,
            'AIC': model.aic,
            'BIC': model.bic,
            'MSE': mean_squared_error(y_test, model.predict(X_test))
        })

        # 获取当前模型的AIC和BIC
        current_aic = model.aic
        current_bic = model.bic

        # 如果当前模型的AIC或BIC更优，则更新最佳模型和特征
        if current_aic < best_aic and current_bic < best_bic:
            best_aic = current_aic
            best_bic = current_bic
            best_model = model
            best_features = X_train.columns[1:]

        # 获取当前模型的最大p值
        max_pvalue = model.pvalues[1:].idxmax()
        max_pvalue_value = model.pvalues[1:].max()

        # 如果最大p值大于阈值，去除该特征
        if max_pvalue_value > threshold:
            X_train = X_train.drop(max_pvalue, axis=1)
            X_test = X_test.drop(max_pvalue, axis=1)
        else:
            break

    return {
        'Best_Model': best_model,
        'Best_Features': best_features,
        'Models_Info': pd.DataFrame(models_info)
    }


# 准备数据
X = df_sales[
    ['Advertising_Cost', 'Social_Media_Expense', 'Product_Price', 'Seasonality_Factor', 'Employee_Satisfaction',
     'Monthly_Weather_Index']]
y = df_sales['Sales_Revenue']

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 进行逐步回归分析
result = stepwise_regression(X_train, y_train, X_test, y_test)

# 获取最终模型在训练集上的效果
final_train_model = sm.OLS(y_train, sm.add_constant(X_train)).fit()
final_train_r_squared = final_train_model.rsquared
final_train_aic = final_train_model.aic
final_train_bic = final_train_model.bic

# 打印最终模型效果
print("Best Final Model - AIC:", result['Best_Model'].aic, "BIC:", result['Best_Model'].bic)
print("Best Features:", result['Best_Features'])
print("\nFinal Model on Training Set - R-squared:", final_train_r_squared, "AIC:", final_train_aic, "BIC:",
      final_train_bic)

# 打印模型剔除过程中的关键参数数据
print("\nModels Information:")
print(result['Models_Info'])
