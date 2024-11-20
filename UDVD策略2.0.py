import os
import pandas as pd
import backtrader as bt
import matplotlib.pyplot as plt
from analyzing_tools import Analyzing_Tools

def UDVD(target_assets, paths,window_1=34):
    #信号结果字典
    results = {}
    #全数据字典，包含计算指标用于检查
    full_info={}
    
    #编写策略主体部分
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

# 自定义数据类，包含 'signal'
class PandasDataPlusSignal(bt.feeds.PandasData):
    lines = ('signal',)
    params = (
        ('signal', -1),  # 默认情况下，'signal' 列在最后一列   
    )

# 策略类，包含调试信息和导出方法
class UDVD_Strategy(bt.Strategy):
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
        available_cash = self.broker.getcash()
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


def run_backtest(strategy, target_assets, cash=100000.0, commission=0.0002, slippage_perc=0.0005, slippage_fixed=None, **kwargs):
    
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
    'daily': r'E:\数据库\同花顺ETF跟踪指数量价数据\1d',
    'hourly': r'E:\数据库\同花顺ETF跟踪指数量价数据\1h',
    'min15': r'E:\数据库\同花顺ETF跟踪指数量价数据\15min',
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
strategy_results,full_info = UDVD(target_assets, paths)


# 获取策略实例
strat = run_backtest(UDVD_Strategy,target_assets,10000000)

pv=strat.get_net_value_series()

portfolio_value, returns, drawdown_ts, metrics = AT.performance_analysis(pv, freq='D')

# 获取净值序列
AT.plot_results('000906.SH',portfolio_value, drawdown_ts, returns, metrics)

# 获取调试信息
debug_df = strat.get_debug_df()

#蒙特卡洛测试

AT.monte_carlo_analysis(strat,num_simulations=10000,num_days=252,freq='D')

