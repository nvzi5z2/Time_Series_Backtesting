import os
import pandas as pd
import backtrader as bt
import matplotlib.pyplot as plt
from analyzing_tools import Analyzing_Tools
from itertools import product
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from statsmodels.tsa.filters.hp_filter import hpfilter


def Inventory_Cycle(target_assets,paths,window_1=8):
    #PMI：原材料库存代码=M002043811
    #BCI:企业库存前瞻指数=M004488064 

    #信号结果字典
    results = {}
    #全数据字典，包含计算指标用于检查
    full_info={}
    
    #编写策略主体部分
    for code in target_assets:
        # 读取数据量价数据
        daily_data = pd.read_csv(os.path.join(paths['daily'], f"{code}.csv"), index_col=[0])
        daily_data.index = pd.to_datetime(daily_data.index)
        df=daily_data.copy()

        #读取EDB数据
        PMI_Inventory= pd.read_csv(paths['EDB']+'\\'+'M002043811.csv')
        PMI_Inventory=PMI_Inventory.set_index('time',drop=True)
        PMI_Inventory_value=PMI_Inventory[['value']]
        PMI_Inventory_value.columns=['PMI_Inventory_Index']
        BCI_Inventory=pd.read_csv(paths['EDB']+'\\'+'M004488064.csv')
        BCI_Inventory=BCI_Inventory.set_index('time',drop=True)
        BCI_Inventory_value=BCI_Inventory[['value']]
        BCI_Inventory_value.columns=['BCI_Inventory_Index']

        #合并所需的EDB数据
        Inventory_merged_df=pd.merge(PMI_Inventory_value,BCI_Inventory_value,right_index=True,left_index=True)
        Inventory_merged_df.index=pd.to_datetime(Inventory_merged_df.index)
        Inventory_merged_df=Inventory_merged_df.sort_index()

        #计算第一个信号（PMI原材料库存信号）

        #处理极值和zscore化
        def process_macro_data_rolling(data, column_name, window=36):
            """
            对输入的宏观数据进行滚动极端值处理和Zscore标准化处理（基于滚动窗口）。

            参数：
            data : pd.DataFrame
                包含待处理数据的DataFrame。
            column_name : str
                需要处理的列名。
            window : int
                滚动窗口的大小（默认为36列）。
            
            返回：
            pd.DataFrame
                包含处理后数据的DataFrame，新增列为 {column_name}_Zscore。
            """
            # 确保列存在
            if column_name not in data.columns:
                raise ValueError(f"列名 {column_name} 不存在于输入数据中")

            # 初始化新列
            zscore_column_name = f"{column_name}_Zscore"
            data[zscore_column_name] = np.nan

            # 滚动窗口计算
            for i in range(window - 1, len(data)):
                # 提取过去window列的数据
                rolling_window = data[column_name].iloc[i - window + 1:i + 1]

                # 计算滚动均值和标准差
                mean = rolling_window.mean()
                std = rolling_window.std()

                # 定义上下限
                upper_limit = mean + 3 * std
                lower_limit = mean - 3 * std

                # 当前值进行极端值处理
                current_value = data[column_name].iloc[i]
                clipped_value = np.clip(current_value, lower_limit, upper_limit)

                # 计算Zscore并赋值
                data.loc[data.index[i], zscore_column_name] = (clipped_value - mean) / std
                data.dropna()

            return data[['PMI_Inventory_Index_Zscore']]
        
        PMI_Inventory_Index_Zscore=process_macro_data_rolling(PMI_Inventory_value,'PMI_Inventory_Index')
        
        #输出PMI的信号
        multiplier=window_1/10
        def generate_signals_with_cleaning(data, column_name, window=36, multiplier=0.8):
            """
            根据Zscore生成信号，并删除前36个月的数据以避免回测时出错

            参数：
            data : pd.DataFrame
                包含处理后数据的DataFrame
            column_name : str
                需要处理的列名（Zscore列）
            window : int, 可选
                滚动窗口大小，默认为36个月
            multiplier : float, 可选
                标准差的倍数，默认为0.8

            返回：
            pd.DataFrame
                包含信号的DataFrame，删除了前36个月的数据
            """
            if column_name not in data.columns:
                raise ValueError(f"列名 {column_name} 不存在于输入数据中")

            # 计算滚动标准差
            rolling_std = data[column_name].rolling(window=window).std()

            # 计算正负0.8倍标准差
            upper_bound = multiplier * rolling_std
            lower_bound = -multiplier * rolling_std

            # 初始化信号
            signals = []

            # 生成信号
            for i in range(len(data)):
                if i < window:  # 在滚动窗口大小之前，无数据可用
                    signals.append(np.nan)
                else:
                    if data[column_name].iloc[i] > upper_bound.iloc[i]:
                        signals.append(-1)  # 看空信号
                    elif data[column_name].iloc[i] < lower_bound.iloc[i]:
                        signals.append(1)  # 看多信号
                    else:
                        signals.append(signals[-1])  # 维持上一个信号

            # 将信号加入数据框
            data['Signal'] = signals

            # 删除前36个月的数据（滚动窗口前的数据）
            data = data.iloc[window:].copy()

            return data[['Signal']]
                
        PMI_Signals = generate_signals_with_cleaning(PMI_Inventory_Index_Zscore, 'PMI_Inventory_Index_Zscore',multiplier=multiplier)
        PMI_Signals.columns=['PMI_Signal']
        PMI_Signals.index=pd.to_datetime(PMI_Signals.index)

        #计算第二个信号（BCI企业前瞻库存指数）反向指标

        BCI=Inventory_merged_df[['BCI_Inventory_Index']]

        BCI.loc[:,"BCI_pct_change"]=BCI.pct_change()
        # 生成信号列：环比变化为正时信号为1，为负时信号为-1
        
        BCI.loc[:, "BCI_Signal"] = BCI["BCI_pct_change"].apply(lambda x: 1 if x > 0 else -1 if x < 0 else 0)
        
        BCI_Signals=BCI[['BCI_Signal']]
        #计算综合信号

        Merged_Signal_DF=pd.merge(PMI_Signals,BCI_Signals,right_index=True,left_index=True)
        
        Merged_Signal_DF.loc[:,"sum_signal"]=Merged_Signal_DF.loc[:,"PMI_Signal"]+Merged_Signal_DF.loc[:,"BCI_Signal"]

        Merged_Signal_DF.loc[:, "signal"] = Merged_Signal_DF["sum_signal"].apply(lambda x: -1 if x in [-2, -1] else (1 if x in [1, 2] else 0))
        
        #将信号转换为日频

        Merged_Signal_DF = Merged_Signal_DF.resample('D').ffill()

        Merged_Signal_DF.index=pd.to_datetime(Merged_Signal_DF.index)
        
        #处理月频信号和日频信号不匹配的问题
        
        #合并最全的信息
        total_info=pd.concat([daily_data,Merged_Signal_DF],axis=1)

        #将有信号的日期df调出来
        first_signal_date=Merged_Signal_DF.index[0]
        first_signal_formatted_date=first_signal_date.strftime('%Y-%m-%d')
        total_info_selected=total_info.loc["2016-03-31":,:]

        #填充缺失值
        total_info_selected=total_info_selected.ffill()

        signal=total_info_selected[['signal']]

        #合并最终信号到daily_data

        daily_data.loc[:,"signal"]=signal

        results[code] = daily_data.dropna()

        full_info[code]=daily_data    
    
    return results,full_info

# 自定义数据类，包含 'signal'
class PandasDataPlusSignal(bt.feeds.PandasData):
    lines = ('signal',)
    params = (
        ('signal', -1),  # 默认情况下，'signal' 列在最后一列   
    )

# 策略类，包含调试信息和导出方法
class Inventory_Cycle_Strategy(bt.Strategy):
    params = (
        ('size_pct',0.198),  # 每个资产的仓位百分比
    )

    def __init__(self):
        self.orders = {}         # 用于跟踪每个资产的订单状态
        self.trade_counts = {}   # 记录每个资产的交易次数
        self.value = []          # 存储组合总净值
        self.dates = []          # 存储日期序列
        self.debug_info = []     # 存储调试信息

        for data in self.datas:
            name = data._name
            self.trade_counts[name] = 0
            self.orders[name] = None

    def next(self):
        total_value = self.broker.getvalue()
        self.value.append(total_value)
        current_date = self.datas[0].datetime.datetime(0)
        self.dates.append(current_date)

        for data in self.datas:
            name = data._name
            position_size = self.getposition(data).size
            signal = data.signal[0]

            # 根据信号执行交易
            if signal == 1 and position_size == 0:
                size = self.calculate_position_size(data)
                self.orders[name] = self.buy(data=data, size=size)
                self.trade_counts[name] += 1

            elif signal == -1 and position_size > 0:
                self.orders[name] = self.close(data=data)
                self.trade_counts[name] += 1


            # 存储调试信息
            self.debug_info.append({
                'Date': current_date,
                'Asset': name,
                'Position': position_size,
                'Signal': signal,
                'Size': self.calculate_position_size(data),
                'Open':data.open[0],
                'High':data.high[0],
                'Low':data.low[0],
                'Volume':data.volume[0],
                'Close': data.close[0],
                'Cash': self.broker.getcash(),
                'Value': total_value,
                'Trades': self.trade_counts[name],
            })


    def calculate_position_size(self, data):
        """
        计算仓位大小
        """
        available_cash = self.broker.getvalue()
        current_price = data.close[0]
        max_investment = available_cash * self.params.size_pct
        max_shares = int(max_investment / current_price)
        return max_shares

    def notify_order(self, order):
        """
        订单完成后重置状态
        """
        if order.status in [order.Completed, order.Canceled, order.Margin]:
            name = order.data._name
            self.orders[name] = None

    def get_net_value_series(self):
        """
        返回净值序列，用于后续分析
        """
        return pd.DataFrame(self.value, index=self.dates)

    def get_debug_df(self):
        """
        返回包含调试信息的DataFrame
        """
        df = pd.DataFrame(self.debug_info)
        df.set_index('Date', inplace=True)
        return df


def run_backtest(strategy, target_assets, strategy_results, cash=100000.0, commission=0.0002, slippage_perc=0.0005, slippage_fixed=None, **kwargs):
    
    cerebro = bt.Cerebro()  # 初始化Cerebro引擎
    cerebro.addstrategy(strategy, **kwargs)  # 添加策略
    
    for code in target_assets:
        data = PandasDataPlusSignal(dataname=strategy_results[code])
        data._name = code  # 为数据设置名称，便于识别
        cerebro.adddata(data)
    
    # 使用setcommission设置股票模式的佣金
    cerebro.broker.setcommission(
        commission=commission,  # 佣金百分比
        stocklike=True  # 将交易设置为股票模式
    )
    
    cerebro.broker.setcash(cash)  # 设置初始资金

    # 设置滑点
    if slippage_perc is not None:
        cerebro.broker.set_slippage_perc(slippage_perc)  # 设置百分比滑点
    elif slippage_fixed is not None:
        cerebro.broker.set_slippage_fixed(slippage_fixed)  # 设置固定点滑点
    
    strategies = cerebro.run()  # 运行回测
    return strategies[0]



#加载分析工具
AT=Analyzing_Tools()


# 定义数据路径
paths = {
    'daily': r'D:\数据库\同花顺ETF跟踪指数量价数据\1d',
    'EDB':r'D:\数据库\同花顺EDB数据',
    'pv_export':r"D:\量化交易构建\私募基金研究\股票策略研究\策略净值序列"
}


# 资产列表
target_assets = [
    "000300.SH",
    "000852.SH",
    "000905.SH",
    "399006.SZ",
    "399303.SZ"
]



# 生成信号
strategy_results,full_info = Inventory_Cycle(target_assets, paths)


# 获取策略实例
strat = run_backtest(Inventory_Cycle_Strategy,target_assets,strategy_results,10000000,0.0005,0.0005)

pv=strat.get_net_value_series()

strtegy_name='Inventory_Cycle_Strategy'


pv.to_excel(paths["pv_export"]+'\\'+strtegy_name+'.xlsx')

portfolio_value, returns, drawdown_ts, metrics = AT.performance_analysis(pv, freq='D')

# 获取净值序列
index_price_path=paths['daily']

AT.plot_results('000906.SH',index_price_path,portfolio_value, drawdown_ts, returns, metrics)

# 获取调试信息
debug_df = strat.get_debug_df()

#蒙特卡洛分析

# AT.monte_carlo_analysis(strat,num_simulations=10000,num_days=252,freq='D')


#定义参数优化函数
def parameter_optimization(parameter_grid, strategy_function, strategy_class, target_assets, paths, cash=100000.0, commission=0.0002, slippage_perc=0.0005, metric='sharpe_ratio'):   
    """
    执行参数优化，支持一个或两个参数。

    参数：
    - parameter_grid: 字典，包含参数名称和要测试的取值列表。例如：{'window_1': [30, 34, 38]}
    - strategy_function: 生成信号的策略函数，例如 UDVD
    - strategy_class: Backtrader 策略类，例如 UDVD_Strategy
    - target_assets: 资产列表
    - paths: 数据路径字典
    - cash: 初始资金
    - commission: 佣金
    - slippage_perc: 滑点百分比
    - metric: 选择用于评估的绩效指标，默认为 'sharpe_ratio'
    """

    #加载数据类

    # 获取参数名称和取值列表
    param_names = list(parameter_grid.keys())
    param_values = [parameter_grid[key] for key in param_names]

    # 生成所有参数组合
    param_combinations = [dict(zip(param_names, values)) for values in product(*param_values)]

    results = []

    for params in param_combinations:
        try:
            print(f"正在测试参数组合：{params}")
            # 生成当前参数下的信号
            strategy_results, full_info = strategy_function(target_assets, paths, **params)

            # 运行回测
            strat = run_backtest(strategy_class, target_assets, strategy_results, cash, commission, slippage_perc)

            # 获取净值序列
            pv = strat.get_net_value_series()

            # 计算绩效指标
            portfolio_value, returns, drawdown_ts, metrics =AT.performance_analysis(pv)

            # 收集指标和参数
            result_entry = {k: v for k, v in params.items()}
            result_entry.update(metrics)
            result_entry=pd.DataFrame(result_entry)
            results.append(result_entry)

        except:

            print(f"参数组合出现错误：{params}")

    # 将结果转换为 DataFrame
    results_df = pd.concat(results,axis=0)
    results_df=results_df.dropna()

    # 可视化结果
    if len(param_names) == 1:
        # 绘制参数与绩效指标的关系曲线
        param = param_names[0]
        plt.figure(figsize=(10, 6))
        plt.plot(results_df[param], results_df[metric], marker='o')
        plt.xlabel(param)
        plt.ylabel(metric)
        plt.title(f'{metric} vs {param}')
        plt.grid(True)
        plt.show()
    elif len(param_names) == 2:
        # 绘制热力图
        param1 = param_names[0]
        param2 = param_names[1]
        pivot_table = results_df.pivot(index=param1, columns=param2, values=metric)

        plt.figure(figsize=(15, 12))  # 调整图像大小
        sns.heatmap(pivot_table, annot=True, fmt=".4f", cmap='viridis',
                    annot_kws={"size": 8}, linewidths=0.5, linecolor='white')
        plt.title(f'{metric} Heatmap', fontsize=16)
        plt.ylabel(param1, fontsize=14)
        plt.xlabel(param2, fontsize=14)
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()  # 自动调整布局
        plt.show()
    else:
        print("无法可视化超过两个参数的结果，请减少参数数量。")

    # 返回结果 DataFrame
    return results_df


# 定义参数网格
parameter_grid = {
    'window_1': range(5, 21,1)
}

# # # 运行参数优化
# results_df = parameter_optimization(
#     parameter_grid=parameter_grid,
#     strategy_function=Inventory_Cycle,
#     strategy_class=Inventory_Cycle_Strategy,
#     target_assets=target_assets,
#     paths=paths,
#     cash=10000000,
#     commission=0.0005,
#     slippage_perc=0.0005,
#     metric='sharpe_ratio'
# )
