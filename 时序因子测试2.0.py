import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf



# 1. 因子构造（包含标准化）
def construct_single_factor(target_assets, paths,window_1=34):
    """
    对所有标的构造单一因子（动量因子），并进行标准化。
    """
    for asset in data:
        asset_data = data[asset]
        daily_data = pd.read_csv(os.path.join(paths['daily'], f"{code}.csv"), index_col=[0])
        daily_data.index = pd.to_datetime(daily_data.index)
        df=daily_data.copy()
        df=df.loc["2021-01-04":,:]
        close = df["close"]
        open = df['open']    
        low = df["low"]
        high = df["high"]
        volume = df['volume']
        # 计算
        volup = (high-open)/open
        voldown = (open-low)/open
        ud= volup - voldown

        df["var_1"] = ud.rolling(window_1).mean()
        # 动量因子
        asset_data['Factor'] = asset_data['Price'] - asset_data['Price'].shift(lookback)
        
        # Z-Score 标准化
        asset_data['Factor_standardized'] = (
            asset_data['Factor'] - asset_data['Factor'].rolling(lookback).mean()
        ) / asset_data['Factor'].rolling(lookback).std()
        
        # 构造未来收益率
        for i in range(1, 31):  # 未来 1 到 30 天的收益率
            asset_data[f'FutureReturn_{i}'] = asset_data['Return'].shift(-i)
        
        # 更新数据，删除缺失值
        data[asset] = asset_data.dropna()
    return data

# 2. 因子相关性分析
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

# 3. 时间序列稳定性检验
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

# 4. 滚动窗口检验
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

# 5. 自相关性检查
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

# 6. 绘制相关性变化图
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

# 主流程
if __name__ == "__main__":
    
    #数据路径
    paths=paths = {
    'daily': r'D:\数据库\同花顺ETF跟踪指数量价数据\1d'}

    target_assets = [
    "000300.SH",
    "000852.SH",
    "000905.SH",
    "399006.SZ",
    "399303.SZ"]

    # 1. 构造单一因子（动量因子）并标准化
    data = construct_single_factor(data, lookback=5)

    # 2. 因子相关性分析（使用标准化因子）
    all_correlations = evaluate_correlation(data, factor_col='Factor_standardized')

    # 3. 打印相关性结果
    print("Correlation with Future Returns (1-30 days):")
    for asset, correlations in all_correlations.items():
        print(f"{asset}: {['{:.4f}'.format(c) for c in correlations]}")

    # 4. 绘制相关性变化图
    plot_correlation(all_correlations)

    # 5. 增加时间序列稳定性检验
    print("\nChecking Time Series Stability:")
    stability_results = check_stability(data, factor_col='Factor_standardized')

    # 6. 增加滚动窗口检验
    print("\nPerforming Rolling Window Test:")
    rolling_results = rolling_window_test(data, factor_col='Factor_standardized', window=60)

    # 7. 增加自相关性检查
    print("\nChecking Autocorrelation:")
    check_autocorrelation(data, factor_col='Factor_standardized', lags=30)