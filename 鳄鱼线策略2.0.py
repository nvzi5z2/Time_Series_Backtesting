import os
import pandas as pd
import backtrader as bt
import matplotlib.pyplot as plt
from analyzing_tools import Analyzing_Tools
import numpy as np

def alligator_strategy_with_ao_and_fractal(target_assets, paths):
    # 信号结果字典
    results = {}
    # 全数据字典，包含计算指标用于检查
    full_info = {}

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
        Teeth = hourly_data_close.rolling(window=8).mean().shift(5)
        Lips = mins_15_data_close.rolling(window=5).mean().shift(3)

        # 合并鳄鱼线数据并统一到日频
        combined = pd.concat([daily_data_close, Jaw, Teeth, Lips], axis=1)
        combined.columns = ['Close', 'Jaw', 'Teeth', 'Lips']
        combined.fillna(method='ffill', inplace=True)
        result = combined.resample('D').last()
        result.dropna(inplace=True)

        # 鳄鱼线信号计算
        alligator_signals = []
        for i in range(len(result)):
            if result['Jaw'][i] > result['Teeth'][i] > result['Lips'][i]:
                alligator_signals.append(-1)  # 看空
            elif result['Jaw'][i] < result['Teeth'][i] < result['Lips'][i]:
                alligator_signals.append(1)  # 看多
            else:
                if i == 0:
                    alligator_signals.append(0)  # 初始信号为 0
                else:
                    alligator_signals.append(alligator_signals[-1])

        result['Alligator_Signal'] = alligator_signals

        #分形形态信号计算
        def identify_fractals_and_record_values(data):
            """
            识别分形并记录过去 5 日的最高价的最高值和最低价的最低值。
            :param data: 包含时间序列数据的 DataFrame，需包含 'high', 'low', 'close' 列。
            :return: 包含 up_fractal_value 和 down_fractal_value 的 DataFrame。
            """
            # 初始化两列用于记录分形值
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
        
        def calculate_fractal_signals(data):
            """
            基于分形值计算交易信号，并在特定情况下保持上一周期的信号。
            :param data: 包含 'close', 'up_fractal_value', 'down_fractal_value' 列的 DataFrame。
            :return: 包含 fractal_signal 的 DataFrame。
            """
            # 初始化信号列，默认值为 0
            data['fractal_signal'] = 0

            # 遍历数据，每一行根据条件更新信号
            for i in range(1, len(data)):
                if data['close'][i] > data['up_fractal_value'][i]:  # 收盘价高于最近的上分形的最高价，看多
                    data.loc[data.index[i], 'fractal_signal'] = 1
                elif data['close'][i] < data['down_fractal_value'][i]:  # 收盘价低于最近的下分形的最低价，看空
                    data.loc[data.index[i], 'fractal_signal'] = -1
                else:  # 其他情况，维持上一周期的信号
                    data.loc[data.index[i], 'fractal_signal'] = data['fractal_signal'][i - 1]

            return data
   
        daily_data = identify_fractals_and_record_values(daily_data)
        daily_data = calculate_fractal_signals(daily_data)
        result['Fractal_Signal'] = daily_data['fractal_signal']

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
        ao_df['AO_Signal'] = np.where(ao_df['Up_Count'] == 3, 1, 
                                    np.where(ao_df['Down_Count'] == 3, -1, np.nan))

        # 延续上一个信号
        ao_df['AO_Signal'] = ao_df['AO_Signal'].fillna(method='ffill').fillna(0)

        # 删除辅助列，保留 AO 和 AO_Signal
        ao_df = ao_df[['AO_Signal']]

        result=pd.merge(result,ao_df,right_index=True,left_index=True)


        # 计算最终信号
        result['Signal'] = 0
        for i in range(len(result)):
            if result['Alligator_Signal'][i] == 1 and (
                result['AO_Signal'][i] == 1 or result['Fractal_Signal'][i] == 1):
                result['Signal'][i] = 1  # 总体看多
            elif (result['Alligator_Signal'][i] == -1 or
                  result['AO_Signal'][i] == -1 or
                  result['Fractal_Signal'][i] == -1):
                result['Signal'][i] = -1  # 总体看空
            else:
                # 维持上一个信号
                if i > 0:
                    result['Signal'][i] = result['Signal'][i - 1]

        # 存储结果
        results[code] = result[['Close', 'Signal']]  # 包含最终信号的日线数据
        full_info[code] = result  # 包含所有信号计算信息的数据

    return results, full_info

# 自定义数据类，包含 'signal'
class PandasDataPlusSignal(bt.feeds.PandasData):
    lines = ('signal',)
    params = (
        ('signal', -1),  # 默认情况下，'signal' 列在最后一列   
    )

# 策略类，包含调试信息和导出方法
class Alligator_Strategy(bt.Strategy):
    params = (
        ('size_pct',0.166),  # 每个资产的仓位百分比
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
        计算每个资产的仓位大小
        """
        available_cash = self.broker.get_value()
        total_assets = len(self.datas)
        max_investment = available_cash * self.params.size_pct
        size = int(max_investment / data.close[0])
        return size

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


def run_backtest(strategy, target_assets, cash=100000.0, commission=0.0006, slippage_perc=0.001, slippage_fixed=None, **kwargs):
    
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
    'hourly': r'D:\数据库\同花顺ETF跟踪指数量价数据\1h',
    'min15': r'D:\数据库\同花顺ETF跟踪指数量价数据\15min',
}

# 资产列表
target_assets = [
    "000016.SH",
    "000300.SH",
    "000852.SH",
    "000905.SH",
    "399006.SZ",
    "399303.SZ"
]



# 生成信号
strategy_results,full_info = alligator_strategy_with_ao_and_fractal(target_assets, paths)


# 获取策略实例
strat = run_backtest(Alligator_Strategy,target_assets,10000000,0,0)

pv=strat.get_net_value_series()

portfolio_value, returns, drawdown_ts, metrics = AT.performance_analysis(pv, freq='D')

# 获取净值序列
AT.plot_results('000300.SH',portfolio_value, drawdown_ts, returns, metrics)

# 获取调试信息
debug_df = strat.get_debug_df()




