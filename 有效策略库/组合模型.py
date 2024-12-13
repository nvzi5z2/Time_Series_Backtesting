import backtrader as bt
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from analyzing_tools import Analyzing_Tools


# 定义自定义数据类
class PandasDataPlusSignal(bt.feeds.PandasData):
    lines = ('signal',)
    params = (
        ('datetime', None),
        ('open', 'open'),
        ('high', 'high'),
        ('low', 'low'),
        ('close', 'close'),
        ('volume', 'volume'),
        ('openinterest', -1),
        ('signal', 'signal'),
    )

class EqualWeightsStrategy(bt.Strategy):
    params = (
        ('size_pct',0.16),  # 每个资产的仓位百分比
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

        # 调试打印
        print(f"Date: {current_date}, Total Value: {total_value}")

        for data in self.datas:
            name = data._name
            position_size = self.getposition(data).size
            signal = data.signal[0]

            # 调试打印
            print(f"Asset: {name}, Position Size: {position_size}, Signal: {signal}")

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
        return pd.Series(self.value, index=self.dates, name='Net Value')

    def get_debug_df(self):
        """
        返回包含调试信息的DataFrame
        """
        df = pd.DataFrame(self.debug_info)
        df.set_index('Date', inplace=True)
        return df
# 定义策略类
class Strategies:

    def __init__(self):
        # 定义数据路径
        self.paths = {
            'daily': r'D:\数据库\同花顺ETF跟踪指数量价数据\1d',
            'hourly': r'D:\数据库\同花顺ETF跟踪指数量价数据\1h',
            'min15': r'D:\数据库\同花顺ETF跟踪指数量价数据\15min',
            'pv_export':r"D:\量化交易构建\私募基金研究\股票策略研究\策略净值序列"
        }
        # 定义选择的资产
        self.target_assets = ["000016.SH","000300.SH","000852.SH",
                              "000905.SH", "399006.SZ","399303.SZ"]

    # 突破类策略
    def UDVD(self,window_1=34):
        # 信号结果字典
        results = {}
        # 全数据字典，包含计算指标用于检查
        full_info={}

        target_assets=self.target_assets

        paths=self.paths

        # 编写策略主体部分
        for code in target_assets:
            # 读取数据
            daily_data = pd.read_csv(os.path.join(paths['daily'], f"{code}.csv"), index_col=[0])
            daily_data.index = pd.to_datetime(daily_data.index)
            df=daily_data.copy()
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
            df["var_2"] =0
            # 添加信号列
            df.loc[(df["var_1"].shift(1) <= df["var_2"].shift(1)) & (df["var_1"] >= df["var_2"]) , 'signal'] = 1
            df.loc[(df["var_1"].shift(1) > df["var_2"].shift(1)) & (df["var_1"] < df["var_2"]) , 'signal'] = -1
            df['signal'].fillna(method='ffill', inplace=True)
            result=df

            # 将信号合并回每日数据
            daily_data = daily_data.join(result[['signal']], how='left')
            daily_data[['signal']].fillna(0, inplace=True)
            daily_data=daily_data.dropna()

            # 存储结果
            results[code] = daily_data
            full_info[code]=result

        return results,full_info

    #均线类策略
    def Alligator_strategy_with_Ao_and_Fractal_Macd(self):
        # 信号结果字典
        results = {}
        # 全数据字典，包含计算指标用于检查
        full_info = {}

        paths=self.paths

        target_assets=self.target_assets

        for code in target_assets:
            # 读取数据
            daily_data = pd.read_csv(os.path.join(paths['daily'], f"{code}.csv"), index_col=[0])
            daily_data.index = pd.to_datetime(daily_data.index)
            hourly_data = pd.read_csv(os.path.join(paths['hourly'], f"{code}.csv"), index_col=[0])
            hourly_data.index = pd.to_datetime(hourly_data.index)
            mins_15_data = pd.read_csv(os.path.join(paths['min15'], f"{code}.csv"), index_col=[0])
            mins_15_data.index = pd.to_datetime(mins_15_data.index)
            mins_15_data = mins_15_data[~mins_15_data.index.duplicated(keep='first')]

            # 提取收盘价
            daily_data_close = daily_data[['close']]
            daily_data_high = daily_data[['high']]
            daily_data_low = daily_data[['low']]
            hourly_data_close = hourly_data[['close']]
            mins_15_data_close = mins_15_data[['close']]

            # 确保数据为数值类型
            daily_data_high = pd.to_numeric(daily_data_high.squeeze(), errors='coerce')
            daily_data_low = pd.to_numeric(daily_data_low.squeeze(), errors='coerce')

            # 计算鳄鱼线指标
            Jaw = daily_data_close.rolling(window=13).mean().shift(8)
            Jaw.columns=['Jaw']
            Teeth = hourly_data_close.rolling(window=8).mean().shift(5)
            Teeth.columns=['Teeth']
            Lips = mins_15_data_close.rolling(window=5).mean().shift(3)
            Lips.columns=['Lips']

            # 合并鳄鱼线数据并统一到日频
            combined = pd.concat([daily_data, Jaw, Teeth, Lips], axis=1)
            combined.fillna(method='ffill', inplace=True)
            result = combined.resample('D').last()
            result.dropna(inplace=True)

            # 生成交易信号
            def signal_generation(row):
                if row['Lips'] > row['Teeth'] > row['Jaw']:
                    return 1  # 多头信号
                elif row['Lips'] < row['Teeth'] < row['Jaw']:
                    return -1  # 空头信号
                else:
                    return 0  # 无信号

            result['Alligator_signal'] = result.apply(signal_generation, axis=1)

            #分形形态信号计算
            def identify_fractals_and_record_values(df):
                """
                识别分形并记录过去 5 日的最高价的最高值和最低价的最低值。
                :param data: 包含时间序列数据的 DataFrame，需包含 'high', 'low', 'close' 列。
                :return: 包含 up_fractal_value 和 down_fractal_value 的 DataFrame。
                """
                # 初始化两列用于记录分形值
                data=df.copy()
                data['up_fractal_value'] = None
                data['down_fractal_value'] = None

                # 遍历数据，识别分形并记录关键值
                for i in range(4, len(data) - 4):  # 确保有足够的数据进行完整分形判断
                    # 向上分形：中间高点最高，且左右两侧逐级降低
                    if (data['high'][i] > data['high'][i - 1] and data['high'][i] > data['high'][i + 1] and  # 中间点高于左右
                        data['high'][i - 1] > data['high'][i - 2] and data['high'][i + 1] > data['high'][i + 2] and  # 左右点高于更远点
                        data['high'][i - 2] > data['high'][i - 3] and data['high'][i + 2] > data['high'][i + 3]):  # 更远点高于最远点
                        # 记录过去 5 日的最高价的最高值
                        data.loc[data.index[i], 'up_fractal_value'] = data['high'][i - 4:i + 1].max()

                    # 向下分形：中间低点最低，且左右两侧逐级升高
                    if (data['low'][i] < data['low'][i - 1] and data['low'][i] < data['low'][i + 1] and  # 中间点低于左右
                        data['low'][i - 1] < data['low'][i - 2] and data['low'][i + 1] < data['low'][i + 2] and  # 左右点低于更远点
                        data['low'][i - 2] < data['low'][i - 3] and data['low'][i + 2] < data['low'][i + 3]):  # 更远点低于最远点
                        # 记录过去 5 日的最低价的最低值
                        data.loc[data.index[i], 'down_fractal_value'] = data['low'][i - 4:i + 1].min()

                # 向下填充分形值
                data['up_fractal_value'].fillna(method='ffill', inplace=True)
                data['down_fractal_value'].fillna(method='ffill', inplace=True)

                return data
            
            def calculate_fractal_signals(df):
                """
                基于分形值计算交易信号，并在特定情况下保持上一周期的信号。
                :param data: 包含 'close', 'up_fractal_value', 'down_fractal_value' 列的 DataFrame。
                :return: 包含 fractal_signal 的 DataFrame。
                """
                # 初始化信号列，默认值为 0
                data=df.copy()
                data['fractal_signal'] = 0

                # 遍历数据，每一行根据条件更新信号
                for i in range(1, len(data)):
                    if data['close'][i] > data['up_fractal_value'][i]:  # 收盘价高于最近的上分形的最高价，看多
                        data.loc[data.index[i], 'fractal_signal'] = 1
                    elif data['close'][i] < data['down_fractal_value'][i]:  # 收盘价低于最近的下分形的最低价，看空
                        data.loc[data.index[i], 'fractal_signal'] = -1
                    else:  # 其他情况，维持上一周期的信号
                        data.loc[data.index[i], 'fractal_signal'] = 0

                return data
    
            fractals_data= identify_fractals_and_record_values(daily_data)
            fractals_signals = calculate_fractal_signals(fractals_data)
            result['Fractal_signal'] = fractals_signals['fractal_signal']

            # 计算 AO 指标
            median_price = (daily_data_high + daily_data_low) / 2
            ao_short = median_price.rolling(window=5).mean()
            ao_long = median_price.rolling(window=34).mean()
            AO = ao_short - ao_long

            # 将 AO 转化为 DataFrame，并与鳄鱼线数据对齐
            ao_df = AO.to_frame(name='AO').dropna()  # 将 AO 转换为 DataFrame

            # 计算 AO 的变化方向
            ao_df['AO_Diff'] = ao_df['AO'].diff()

            # 判断连续上涨和下跌天数
            ao_df['Up_Count'] = (ao_df['AO_Diff'] > 0).astype(int).rolling(window=3).sum()
            ao_df['Down_Count'] = (ao_df['AO_Diff'] < 0).astype(int).rolling(window=3).sum()

            # 根据规则生成信号
            ao_df['AO_signal'] = np.where(ao_df['Up_Count'] == 3, 1, 
                                        np.where(ao_df['Down_Count'] == 3, -1, np.nan))

            # 延续上一个信号
            ao_df['AO_signal'] = ao_df['AO_signal'].fillna(0)

            # 删除辅助列，保留 AO 和 AO_signal
            ao_df = ao_df[['AO_signal']]

            result=pd.merge(result,ao_df,right_index=True,left_index=True)

            #计算MACD指标
            df = daily_data.copy()
            # 1. 计算快线（DIFF）和慢线（DEA）
            df['ema_short'] = df['close'].ewm(span=12, adjust=False).mean()
            df['ema_long'] = df['close'].ewm(span=26, adjust=False).mean()
            df['diff'] = df['ema_short'] - df['ema_long']  # DIFF 快线
            df['dea'] = df['diff'].ewm(span=9, adjust=False).mean()  # DEA 慢线
            
            # 2. 计算能量柱
            df['macd_bar'] = (df['diff'] - df['dea']) * 2

            # 添加macd信号列
            def macd_generate_signal(row):
                # 水上或零轴看多信号
                if row['diff'] > row['dea'] and row['macd_bar'] >= 0:
                    return 1
                # 水下或零轴看空信号
                elif row['diff'] < row['dea'] and row['macd_bar'] <= 0:
                    return -1
                # 无信号
                else:
                    return 0

            # 应用macd信号生成规则
            df['MACD_signal'] = df.apply(macd_generate_signal, axis=1)
            
            macd_df=df[['MACD_signal']]

            result=pd.merge(result,macd_df,right_index=True,left_index=True)
            #计算最终信号

            result['signal'] = 0
            for i in range(len(result)):
                long_condition = (result['Alligator_signal'][i] == 1 and 
                                (result['MACD_signal'][i] == 1 or result['Fractal_signal'][i] == 1
                                or result['AO_signal'][i] == 1)
                                )
                short_condition = (result['Alligator_signal'][i] == -1 or
                                result['AO_signal'][i] == -1 or result['Fractal_signal'][i] == -1 or
                                result['MACD_signal'][i]== -1)
                
                if long_condition and short_condition:  # 同时触发多空信号
                    result['signal'][i] = result['signal'][i - 1] if i > 0 else 0  # 延续上一周期信号
                elif long_condition:
                    result['signal'][i] = 1  # 看多
                elif short_condition:
                    result['signal'][i] = -1  # 看空
                else:
                    result['signal'][i] = result['signal'][i - 1] if i > 0 else 0 # 延续上一周期信号

            # 存储结果
            signal=result[['signal']]
            results[code]=pd.merge(daily_data,signal,right_index=True,left_index=True)
            full_info[code] = result  # 包含所有信号计算信息的数据

        return results, full_info

        # 自定义数据类，包含 'signal'
        class PandasDataPlusSignal(bt.feeds.PandasData):
            lines = ('signal',)
            params = (
                ('signal', -1),  # 默认情况下，'signal' 列在最后一列   
            )

    # 成交量类策略
    def V_MACD(self,window_1=39,window_2=0):
        # 信号结果字典
        results = {}
        # 全数据字典，包含计算指标用于检查
        full_info={}

        paths=self.paths

        target_assets=self.target_assets
        
        # 编写策略主体部分
        for code in target_assets:
            # 读取数据
            daily_data = pd.read_csv(os.path.join(paths['daily'], f"{code}.csv"), index_col=[0])
            daily_data.index = pd.to_datetime(daily_data.index)

            df=daily_data.copy()

            volume = df['volume']
            df['EMA_F'] =volume.ewm(12).mean()
            df['EMA_S'] = volume.ewm(26).mean()
            df['V_DIF'] = df['EMA_F'] - df['EMA_S']
            df['V_DEA'] = df['V_DIF'].ewm(9).mean()
            df['VMACD']=(df['V_DIF']-df['V_DEA'])*2

            # 标准化处理
            df['VMACD_mean'] = df['VMACD'].rolling(window_1).mean()
            df['VMACD_std'] = df['VMACD'].rolling(window_1).std()
            df['Z_VMACD'] = (df['VMACD'] - df['VMACD_mean']) / df['VMACD_std']
            df['VMACD_MTM'] = df['Z_VMACD'].rolling(window_1).sum()

            # 计算
            df["var_1"] = df['VMACD_MTM']
            df["var_2"] = window_2/10
            df["var_3"] = -window_2/10

            # 信号触发条件
            df.loc[(df["var_1"].shift(1) <= df["var_2"].shift(1)) & (df["var_1"] > df["var_2"]) , 'signal'] = 1
            df.loc[(df["var_1"].shift(1) > df["var_3"].shift(1)) & (df["var_1"] <= df["var_3"]), 'signal'] = -1

            # pos为空的，向上填充数字
            df['signal'].fillna(method='ffill', inplace=True)

            result=df
            # 将信号合并回每日数据
            daily_data = daily_data.join(result[['signal']], how='left')
            daily_data[['signal']].fillna(0, inplace=True)
            daily_data=daily_data.dropna()

            # 存储结果
            results[code] = daily_data
            full_info[code]=result

        return results,full_info

# 实例化策略类
strategies_instance = Strategies()

# 生成策略数据
UDVD_results,_ = strategies_instance.UDVD()
Alligator_results,_ = strategies_instance.Alligator_strategy_with_Ao_and_Fractal_Macd()
V_MACD_results,_ = strategies_instance.V_MACD()

# 定义添加信号的数据类
Adding_Signal = PandasDataPlusSignal

# 定义策略和资金分配比例
strategies_list = [
    {'strategy': EqualWeightsStrategy, 'allocation': 0.25, 'name': 'UDVD', 'datas': UDVD_results},
    {'strategy': EqualWeightsStrategy, 'allocation': 0.5, 'name': 'Alligator', 'datas': Alligator_results},
    {'strategy': EqualWeightsStrategy, 'allocation':0.25, 'name': 'V_MACD', 'datas': V_MACD_results}
]

def Portfolio(strategies,initial_cash=10000000):
    # 创建策略名称到资金分配比例的映射
    allocation_map = {strat['name']: strat['allocation'] for strat in strategies}

    # 初始化结果存储
    strategy_returns = {}
    strategy_values = {}
    strategy_debug_info = {}
    initial_cash = initial_cash  # 总初始资金

    # 运行每个策略的回测
    for strat in strategies:
        cerebro = bt.Cerebro()
        # 为该策略设置初始资金
        strat_cash = initial_cash * strat['allocation']
        cerebro.broker.setcash(strat_cash)
        # 添加数据到 Cerebro
        for code, dataframe in strat['datas'].items():
            # 调试打印数据的前几行
            print(f"Adding data for {code} in strategy {strat['name']}:")
            print(dataframe.head())
            # 检查必要的列是否存在
            required_columns = ['open', 'high', 'low', 'close', 'volume', 'signal']
            if not all(col in dataframe.columns for col in required_columns):
                print(f"Error: Data for {code} is missing required columns.")
                continue  # 跳过该数据

            dataframe = dataframe.sort_index()
            data_feed = Adding_Signal(dataname=dataframe, name=code)
            cerebro.adddata(data_feed)
        # 添加策略
        cerebro.addstrategy(strat['strategy'])
        # 运行回测
        result = cerebro.run()
        # 获取策略实例
        strat_instance = result[0]
        # 获取净值序列
        net_value = strat_instance.get_net_value_series()

        # 调试打印
        print(f"Strategy: {strat['name']}, Net Value Series Length: {len(net_value)}")
        print(net_value.head())

        # 检查 net_value 是否为空
        if net_value.empty:
            print(f"Warning: Strategy {strat['name']} generated an empty net value series.")

        # 存储净值序列
        strategy_values[strat['name']] = net_value
        # 收集调试信息
        debug_df = strat_instance.get_debug_df()
        strategy_debug_info[strat['name']] = debug_df

    # 合并净值序列，处理不同数据长度问题
    # 获取所有日期的并集
    all_dates = pd.to_datetime(sorted(set.union(*(set(v.index) for v in strategy_values.values()))))

    # 重新索引净值序列，并填充缺失值
    for name, net_value in strategy_values.items():
        # 去除重复的日期索引
        net_value = net_value[~net_value.index.duplicated(keep='first')]
        # 重新索引到所有日期
        net_value = net_value.reindex(all_dates)
        # 填充缺失值，前向填充
        net_value = net_value.fillna(method='ffill')
        # 填充初始缺失值为初始资金
        initial_net_value = initial_cash * allocation_map[name]
        net_value = net_value.fillna(initial_net_value)
        strategy_values[name] = net_value

    # 将净值序列合并为DataFrame
    df_values = pd.DataFrame(strategy_values)
    # 计算组合净值曲线
    df_values['Combined'] = df_values.sum(axis=1)

    # 收集所有子策略的调试信息
    combined_debug_df = pd.concat(strategy_debug_info.values(), keys=strategy_debug_info.keys())
    combined_debug_df = combined_debug_df.reset_index(level=0).rename(columns={'level_0': 'Strategy'})

    return df_values,combined_debug_df

# 运行组合回测
pf_nv, debug_df = Portfolio(strategies_list)

# 获取组合净值
Portfolio_nv = pf_nv[['Combined']].resample('D').last().dropna()


#组合分析

AT=Analyzing_Tools()

index_price_path=strategies_instance.paths['daily']

portfolio_value, returns, drawdown_ts, metrics = AT.performance_analysis(Portfolio_nv, freq='D')

AT.plot_results('000906.SH',index_price_path,Portfolio_nv, drawdown_ts, returns, metrics)

