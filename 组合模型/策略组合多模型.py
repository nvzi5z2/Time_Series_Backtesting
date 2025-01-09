
import numpy as np
import pandas as pd
import riskfolio as rp
import os


class Portfolio_Model:

    def __init__(self):

        self.strategy_path=r'D:\量化交易构建\私募基金研究\股票策略研究\
                            Time_Series_Backtesting\组合模型\子策略净值\子策略表现.xlsx'
    
    def Risk_Parity_Model(self,data: pd.DataFrame, window: int = 252) -> pd.DataFrame:
        """
        计算风险平价模型下的资产权重。
        
        参数:
        data (pd.DataFrame): 包含股票收盘价或基金净值时间序列的数据的Dataframe。
        window (int): 计算窗口期，默认为252（大约一年的交易日）。
        
        返回:
        pd.DataFrame: 风险平价模型下的资产权重。
        """

        # 创建一个空的Dataframe来存储各资产的权重
        weights = pd.DataFrame(columns=data.columns)

        # 滚动计算风险平价权重
        for i in range(window, len(data)):
            # 提取滚动窗口期内的数据
            window_data = data.iloc[i - window:i]

            # 计算收益率
            returns = window_data.pct_change().dropna()

            # 使用Riskfolio-Lib构建投资组合对象
            port = rp.Portfolio(returns=returns)

            # 选择估计方法并计算输入参数
            method_mu = 'hist'
            method_cov = 'hist'
            port.assets_stats(method_mu=method_mu, method_cov=method_cov)

            # 估计风险平价投资组合
            model = 'Classic'
            rm = 'MV'
            rf = 0
            b = None
            hist = True
            w_rp = port.rp_optimization(model=model, rm=rm, rf=rf, b=b, hist=hist)

            # 将权重添加到结果Dataframe中
            weights = weights.append(w_rp.T, ignore_index=True)

        # 将结果Dataframe的索引与输入数据对齐
        weights.index = data.index[window:]

        return weights
   
    def clean_data(strategy_path):
        #读取数据
        strategy_path=self.strategy_path
        strategy_df=pd.read_excel(strategy_path,index_col=[0])
        strategy_df.index=pd.to_datetime(strategy_df.index)

        #净值归一
        strategy_nv=strategy_df/strategy_df.iloc[0,:]

        #剔除总组合净值
        strategy_nv=strategy_nv.drop(columns=['Combined','FS_A50'])

        return strategy_nv






strategy_path=r'D:\量化交易构建\Time_Serires_Backtesting\Time_Series_Backtesting\组合模型\子策略净值\子策略表现.xlsx'



def Risk_Parity_Model(data: pd.DataFrame, window: int = 252) -> pd.DataFrame:
    """
    计算风险平价模型下的资产权重。
    
    参数:
    data (pd.DataFrame): 包含股票收盘价或基金净值时间序列的数据的Dataframe。
    window (int): 计算窗口期，默认为252（大约一年的交易日）。
    
    返回:
    pd.DataFrame: 风险平价模型下的资产权重。
    """

    # 创建一个空的Dataframe来存储各资产的权重
    weights_list = []

    # 滚动计算风险平价权重
    for i in range(window, len(data)):
        # 提取滚动窗口期内的数据
        window_data = data.iloc[i - window:i]

        # 计算收益率
        returns = window_data.pct_change().dropna()

        # 使用Riskfolio-Lib构建投资组合对象
        port = rp.Portfolio(returns=returns)

        # 选择估计方法并计算输入参数
        method_mu = 'hist'
        method_cov = 'hist'
        port.assets_stats(method_mu=method_mu, method_cov=method_cov)

        # 估计风险平价投资组合
        model = 'Classic'
        rm = 'MV'
        rf = 0
        b = None
        hist = True
        w_rp = port.rp_optimization(model=model, rm=rm, rf=rf, b=b, hist=hist)

        # 将权重添加到结果列表中
        weights_list.append(w_rp.T)

    # 将结果列表转换为DataFrame
    weights = pd.concat(weights_list, ignore_index=True)

    # 将结果DataFrame的索引与输入数据对齐
    weights.index = data.index[window:]

    return weights



def clean_data(strategy_path):
    #读取数据
    strategy_df=pd.read_excel(strategy_path,index_col=[0])
    strategy_df.index=pd.to_datetime(strategy_df.index)

    #净值归一
    strategy_nv=strategy_df/strategy_df.iloc[0,:]

    #剔除总组合净值
    strategy_nv=strategy_nv.drop(columns=['Combined','FS_A50'])

    return strategy_nv


df=clean_data(strategy_path)

rf=Risk_Parity_Model(df,1260)


    