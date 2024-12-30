import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf

# 1. 数据准备
def generate_synthetic_data(n=500, num_assets=5, seed=42):
    """
    生成多个标的的合成时间序列数据，包括价格和收益率。
    """
    np.random.seed(seed)
    data = {}
    for i in range(num_assets):
        prices = np.cumsum(np.random.randn(n)) + 100  # 随机价格序列
        returns = np.log(prices[1:] / prices[:-1])  # 对数收益率
        dates = pd.date_range(start="2020-01-01", periods=n-1)
        asset_data = pd.DataFrame({"Price": prices[1:], "Return": returns}, index=dates)
        data[f"Asset_{i+1}"] = asset_data
    return data

# 2. 因子构造（包含标准化和未来收益率计算）
def construct_single_factor(data, lookback=5):
    """
    对所有标的构造单一因子（动量因子），并进行标准化。
    """
    for asset in data:
        asset_data = data[asset]
        
        # 动量因子
        asset_data['Factor'] = asset_data['Price'] - asset_data['Price'].shift(lookback)
        
        # Z-Score 标准化
        asset_data['Factor_standardized'] = (
            asset_data['Factor'] - asset_data['Factor'].rolling(lookback).mean()
        ) / asset_data['Factor'].rolling(lookback).std()
        
        # 构造未来 1 到 30 天的累积收益率
        for i in range(1, 31):  # 从未来 1 天到未来 30 天
            # 使用价格计算未来的累积对数收益率
            asset_data[f'FutureReturn_{i}'] = np.log(asset_data['Price'].shift(-i) / asset_data['Price'])
        
        # 更新数据，删除缺失值
        data[asset] = asset_data.dropna()
    return data
# 3. 因子相关性分析
def evaluate_correlation(data, factor_col='Factor_standardized'):
    """
    计算标准化因子与未来收益率在不同时间维度（1 天、2 天、... 30 天）的相关性。
    返回每个标的的相关性列表。
    """
    all_correlations = {}
    for asset in data:
        asset_data = data[asset]
        correlations = []
        for i in range(1, 31):  # 计算 1 到 30 天的相关性
            corr = asset_data[factor_col].corr(asset_data[f'FutureReturn_{i}'])
            correlations.append(corr)
        all_correlations[asset] = correlations
    return all_correlations

# 4. 时间序列稳定性检验
def check_stability(data, factor_col='Factor_standardized'):
    """
    检查因子的时间序列稳定性：分时间段计算相关性。
    """
    results = {}
    for asset in data:
        asset_data = data[asset]
        asset_data['Year'] = asset_data.index.year
        stability = asset_data.groupby('Year').apply(
            lambda x: x[factor_col].corr(x['Return'])
        )
        results[asset] = stability
        print(f"Stability for {asset}:\n{stability}\n")
    return results

# 5. 滚动窗口检验
def rolling_window_test(data, factor_col='Factor_standardized', window=60):
    """
    通过滚动窗口计算因子与目标变量的相关性。
    """
    rolling_results = {}
    for asset in data:
        asset_data = data[asset]
        rolling_corr = asset_data[factor_col].rolling(window=window).corr(asset_data['Return'])
        rolling_results[asset] = rolling_corr
        # 绘制滚动相关性
        plt.figure(figsize=(10, 6))
        plt.plot(rolling_corr, label=f"{asset} Rolling Correlation", color='blue', alpha=0.7)
        plt.axhline(0, color='red', linestyle='--', linewidth=1)
        plt.title(f"Rolling Correlation for {asset}")
        plt.xlabel("Date")
        plt.ylabel("Correlation")
        plt.legend()
        plt.grid()
        plt.show()
    return rolling_results

# 6. 自相关性检查
def check_autocorrelation(data, factor_col='Factor_standardized', lags=30):
    """
    检查因子的自相关性。
    """
    for asset in data:
        asset_data = data[asset]
        print(f"Autocorrelation for {asset}:")
        plot_acf(asset_data[factor_col].dropna(), lags=lags)
        plt.title(f"Autocorrelation for {asset}")
        plt.show()

# 7. 绘制相关性变化图
def plot_correlation(all_correlations):
    """
    绘制因子与未来收益率相关性的变化图，包括：
    - 每个标的的相关性曲线。
    - 平均相关性曲线。
    """
    plt.figure(figsize=(12, 8))

    # 绘制个体相关性曲线
    for asset, correlations in all_correlations.items():
        plt.plot(range(1, 31), correlations, label=asset, alpha=0.5)

    # 计算并绘制平均相关性曲线
    avg_correlations = np.mean(list(all_correlations.values()), axis=0)
    plt.plot(range(1, 31), avg_correlations, label='Average', color='black', linewidth=2)

    # 图表设置
    plt.axhline(0, color='red', linestyle='--', linewidth=1)
    plt.title('Factor Correlation with Future Returns (1-30 days)')
    plt.xlabel('Future Return Lag (Days)')
    plt.ylabel('Correlation')
    plt.xticks(range(1, 31))
    plt.legend()
    plt.grid()
    plt.show()


def perform_detailed_linear_regression(data, factor_col='Factor_standardized', future_return_col='FutureReturn_10'):
    """
    对因子和未来收益率进行线性回归，并返回详细统计指标。
    
    参数：
    - data: dict，每个标的的数据（DataFrame）。
    - factor_col: str，因子的列名。
    - future_return_col: str，未来收益率的列名。
    
    返回：
    - results_df: DataFrame，包含每个标的的回归结果，列包括斜率、截距、R²、t 值、p 值等。
    """
    results = []
    for asset in data:
        asset_data = data[asset]
        # 确保数据中无缺失值
        asset_data = asset_data[[factor_col, future_return_col]].dropna()
        
        # 因子值和未来收益率
        X = sm.add_constant(asset_data[factor_col])  # 添加常数项（截距）
        y = asset_data[future_return_col]

        # 使用 statsmodels 进行线性回归
        model = sm.OLS(y, X).fit()

        # 提取回归结果
        slope = model.params[factor_col]  # 斜率
        intercept = model.params['const']  # 截距
        r_squared = model.rsquared  # R²
        f_stat = model.fvalue  # F 值
        f_pval = model.f_pvalue  # F 值的 p 值
        t_value = model.tvalues[factor_col]  # 斜率的 t 值
        p_value = model.pvalues[factor_col]  # 斜率的 p 值
        std_err = model.bse[factor_col]  # 斜率的标准误差
        residual_std_err = model.mse_resid ** 0.5  # 残差标准误差

        # 将结果保存到列表
        results.append({
            'Asset': asset,
            'Slope': slope,
            'Intercept': intercept,
            'R²': r_squared,
            'F-stat': f_stat,
            'F_p-value': f_pval,
            't-value': t_value,
            'p-value': p_value,
            'Std Err': std_err,
            'Residual Std Err': residual_std_err
        })
    
    # 转换为 DataFrame
    results_df = pd.DataFrame(results)
    return results_df

# 主流程

if __name__ == "__main__":
    # 1. 生成多个标的的数据
    data = generate_synthetic_data(n=500, num_assets=5)

    # 2. 构造单一因子（动量因子）并标准化
    data = construct_single_factor(data, lookback=5)

    # 3. 因子相关性分析（使用标准化因子）
    all_correlations = evaluate_correlation(data, factor_col='Factor_standardized')

    # 4. 打印相关性结果
    print("Correlation with Future Returns (1-30 days):")
    for asset, correlations in all_correlations.items():
        print(f"{asset}: {['{:.4f}'.format(c) for c in correlations]}")

    # 5. 绘制相关性变化图
    plot_correlation(all_correlations)

    # 6. 增加时间序列稳定性检验
    print("\nChecking Time Series Stability:")
    stability_results = check_stability(data, factor_col='Factor_standardized')

    # 7. 增加滚动窗口检验
    print("\nPerforming Rolling Window Test:")
    rolling_results = rolling_window_test(data, factor_col='Factor_standardized', window=60)

    # 8. 增加自相关性检查
    print("\nChecking Autocorrelation:")
    check_autocorrelation(data, factor_col='Factor_standardized', lags=30)

    # 9. 线性回归分析（详细统计结果）
    print("\nPerforming Detailed Linear Regression Analysis:")
    future_return_col = 'FutureReturn_10'  # 使用未来 10 天的累积收益率
    detailed_regression_results = perform_detailed_linear_regression(data, factor_col='Factor_standardized', future_return_col=future_return_col)
    
    # 打印线性回归结果表格
    print("\nDetailed Linear Regression Results:")
    print(detailed_regression_results)
    
    # 保存结果为 CSV 文件（可选）
    # detailed_regression_results.to_csv("detailed_regression_results.csv", index=False)