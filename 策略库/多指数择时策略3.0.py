import backtrader as bt
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import empyrical as ep
from bokeh.plotting import figure, show, output_file
from bokeh.layouts import  gridplot, column
from bokeh.models import ColumnDataSource, HoverTool,Div
from bokeh.models import Span

# 数据加载器和组合分析
class DataLoader:

    def __init__(self):

        self.index_price_path=r'D:\数据库\同花顺ETF跟踪指数量价数据\1d'
    # 标准数据加载
    # 自定义PandasData类，增加自由流通换手率字段
    class CustomPandasData(bt.feeds.PandasData):
        lines = ('new_data',)  # 增加新的数据列

        # 定义列索引，确保backtrader知道如何从pandas DataFrame中提取这些数据
        params = (
            ('new_data', -1),  # -1表示此字段未被使用，稍后会具体赋值
        )

    def performance_analysis(self,portfolio_value,freq='D'):
        
        # 计算收益率
        returns = portfolio_value.pct_change().dropna()

        # 确定年化系数
        if freq == 'D':
            annual_factor = 252  # 你可以调整为252，如果更适合你的数据
        elif freq == 'H' or freq == '1H':
            annual_factor = 252 * 4  # 每天4小时
        elif freq == '30m':
            annual_factor = 252 * 4 * 2  # 每小时2个30分钟
        elif freq == '15m':
            annual_factor = 252 * 4 * 4  # 每小时4个15分钟
        elif freq == '5m':
            annual_factor = 252 * 4 * 12  # 每小时12个5分钟
        elif freq == '1m':
            annual_factor = 252 * 4 * 60  # 每小时60个1分钟
        elif freq == '2H':
            annual_factor = 252 * 2  # 每天12个2小时
        elif freq == '4H':
            annual_factor = 252 * 1  # 每天6个4小时

        else:
            raise ValueError("Unsupported frequency")

        # 计算各项绩效指标
        total_return = ep.cum_returns_final(returns)  # 总收益率
        periods = len(returns)  # 总周期数

        # 手动计算年化波动率和年化收益率
        annual_volatility = returns.std() * np.sqrt(annual_factor)
        annual_return = (1 + total_return) ** (annual_factor / periods) - 1

        # 计算最大回撤
        max_drawdown = ep.max_drawdown(returns)

        # 手动计算卡尔马比率
        calmar_ratio = annual_return / abs(max_drawdown) 

        # 计算夏普比率
        sharpe_ratio = annual_return / annual_volatility

        # 计算胜率
        win_rate = (returns >= 0).sum() / len(returns)  # 胜率

        # 计算下行标准差（只考虑负收益）
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() * np.sqrt(annual_factor)

        # 计算索提诺比率
        sortino_ratio = annual_return / downside_std 

        # 计算回撤时间序列
        cumulative_returns = ep.cum_returns(returns, starting_value=1)
        running_max = cumulative_returns.cummax()
        drawdown_ts = (cumulative_returns - running_max) / running_max

        # 计算最大恢复时间
        max_time_to_recovery = 0
        recovery_start = None

        for date in drawdown_ts.index:
            drawdown_value = drawdown_ts.get(date, None)  # 使用 .get() 安全访问
            if drawdown_value is not None and drawdown_value < 0:
                if recovery_start is None:
                    recovery_start = date
            elif recovery_start is not None:
                recovery_time = (date - recovery_start).days
                max_time_to_recovery = max(max_time_to_recovery, recovery_time)
                recovery_start = None

        return portfolio_value, returns, drawdown_ts, {
            'total_return': total_return,
            'periods': periods,
            'annual_volatility': annual_volatility,
            'annual_return': annual_return,
            'sharpe_ratio': sharpe_ratio,
            'calmar_ratio': calmar_ratio,
            'sortino_ratio': sortino_ratio,      # 增加索提诺比率
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'max_time_to_recovery': max_time_to_recovery
        }
    
    def plot_results(self,benchmark_code, portfolio_value, drawdown_ts, returns, perf_metrics):
        # 假设 drawdown_ts 和 returns 是 DataFrame，转换为 Series
        drawdown_ts = drawdown_ts['Combined']
        returns = returns['Combined']

        # 读取基准指数价格数据
        index_price_path = r'D:\数据库\同花顺ETF跟踪指数量价数据\1d'
        benchmark_data = pd.read_csv(index_price_path + '\\' + benchmark_code + '.csv', index_col=[0])

        # 确保时间列是 datetime 格式
        benchmark_data.index = pd.to_datetime(benchmark_data.index)

        # 合并组合价值数据
        benchmark_data = pd.merge(benchmark_data, portfolio_value, right_index=True, left_index=True)

        # 计算标准化的价格
        benchmark_data_normalized = benchmark_data['close'] / benchmark_data['close'].iloc[0]

        # 创建一个图表，用于策略与底层资产价格对比
        p_comparison = figure(x_axis_type="datetime", title="策略与底层资产价格对比", plot_height=400, plot_width=1000)
        p_comparison.grid.grid_line_alpha = 0.3

        # 标准化的基准价格曲线
        p_comparison.line(x=benchmark_data.index, y=benchmark_data_normalized, color='blue', legend_label='标准化价格')

        # 计算组合价值的标准化值
        portfolio_value_series = portfolio_value['Combined']
        portfolio_value_series.index = pd.to_datetime(portfolio_value_series.index)
        portfolio_normalized = portfolio_value_series / portfolio_value_series.iloc[0]
        p_comparison.line(portfolio_normalized.index, portfolio_normalized, color='green', legend_label='策略组合价值')

        # 设置图例位置和点击策略
        p_comparison.legend.location = "top_left"
        p_comparison.legend.click_policy = "hide"

        # 创建一个图表，用于显示组合价值
        p_value = figure(x_axis_type="datetime", title="组合价值", plot_height=300, plot_width=1000)
        p_value.line(portfolio_value_series.index, portfolio_value_series, color='navy', legend_label='组合价值')
        p_value.grid.grid_line_alpha = 0.3

        # 增加悬停工具到 p_value 图表
        hover_p_value = HoverTool(
            tooltips=[
                ("日期", "@x{%F}"),
                ("组合价值", "@y{0.2f}")
            ],
            formatters={
                '@x': 'datetime',
            },
            mode='vline'
        )
        p_value.add_tools(hover_p_value)

        # 创建一个图表，用于显示回撤
        p_drawdown = figure(x_axis_type="datetime", title="回撤", plot_height=300, plot_width=1000)
        p_drawdown.line(drawdown_ts.index, drawdown_ts, color='red', legend_label='回撤')
        p_drawdown.grid.grid_line_alpha = 0.3

        # 创建一个图表，用于显示每日收益分布
        p_returns_hist = figure(title="每日收益分布", plot_height=300, plot_width=1000)
        hist, edges = np.histogram(returns, bins=50)
        p_returns_hist.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:], fill_color="navy", line_color="white", alpha=0.5)

        # 计算每周收益并创建相应的分布图表
        weekly_returns = returns.resample('W').sum()
        p_weekly_returns_hist = figure(title="每周收益分布", plot_height=300, plot_width=1000)
        hist_weekly, edges_weekly = np.histogram(weekly_returns, bins=50)
        p_weekly_returns_hist.quad(top=hist_weekly, bottom=0, left=edges_weekly[:-1], right=edges_weekly[1:], fill_color="orange", line_color="white", alpha=0.5)

        # 计算每月收益并创建相应的分布图表
        monthly_returns = returns.resample('M').sum()
        p_monthly_returns_hist = figure(title="每月收益分布", plot_height=300, plot_width=1000)
        hist_monthly, edges_monthly = np.histogram(monthly_returns, bins=50)
        p_monthly_returns_hist.quad(top=hist_monthly, bottom=0, left=edges_monthly[:-1], right=edges_monthly[1:], fill_color="green", line_color="white", alpha=0.5)

        # 计算累计收益并创建相应的图表
        cumulative_returns = (1 + returns).cumprod() - 1
        p_cum_returns = figure(x_axis_type="datetime", title="累计收益", plot_height=300, plot_width=1000)
        p_cum_returns.line(cumulative_returns.index, cumulative_returns, color='green', legend_label='累计收益')
        p_cum_returns.grid.grid_line_alpha = 0.3

        # 创建一个Div对象，用于显示绩效分析结果
        perf_text = f"""
        <div style="background-color: #f9f9f9; border: 1px solid #ddd; padding: 15px; border-radius: 8px; width: 100%;">
            <h3 style="color: #333; font-family: Arial, sans-serif; text-align: center; margin-bottom: 15px;">策略绩效分析</h3>
            <table style="width: 100%; border-collapse: collapse; font-family: Arial, sans-serif; font-size: 14px;">
                <tr>
                    <td style="padding: 5px; border-bottom: 1px solid #ddd;"><b>总收益率:</b></td>
                    <td style="padding: 5px; border-bottom: 1px solid #ddd;">{perf_metrics['total_return'].values[0]:.4f}</td>
                </tr>
                <tr>
                    <td style="padding: 5px; border-bottom: 1px solid #ddd;"><b>年化收益率:</b></td>
                    <td style="padding: 5px; border-bottom: 1px solid #ddd;">{perf_metrics['annual_return'].values[0]:.4f}</td>
                </tr>
                <tr>
                    <td style="padding: 5px; border-bottom: 1px solid #ddd;"><b>年化波动率:</b></td>
                    <td style="padding: 5px; border-bottom: 1px solid #ddd;">{perf_metrics['annual_volatility'].values[0]:.4f}</td>
                </tr>
                <tr>
                    <td style="padding: 5px; border-bottom: 1px solid #ddd;"><b>夏普比率:</b></td>
                    <td style="padding: 5px; border-bottom: 1px solid #ddd;">{perf_metrics['sharpe_ratio'].values[0]:.4f}</td>
                </tr>
                <tr>
                    <td style="padding: 5px; border-bottom: 1px solid #ddd;"><b>索提诺比率:</b></td>
                    <td style="padding: 5px; border-bottom: 1px solid #ddd;">{perf_metrics['sortino_ratio'].values[0]:.4f}</td>
                </tr>
                <tr>
                    <td style="padding: 5px; border-bottom: 1px solid #ddd;"><b>卡尔马比率:</b></td>
                    <td style="padding: 5px; border-bottom: 1px solid #ddd;">{perf_metrics['calmar_ratio'].values[0]:.4f}</td>
                </tr>
                <tr>
                    <td style="padding: 5px; border-bottom: 1px solid #ddd;"><b>最大回撤:</b></td>
                    <td style="padding: 5px; border-bottom: 1px solid #ddd;">{perf_metrics['max_drawdown'].values[0]:.4f}</td>
                </tr>
                <tr>
                    <td style="padding: 5px; border-bottom: 1px solid #ddd;"><b>胜率:</b></td>
                    <td style="padding: 5px; border-bottom: 1px solid #ddd;">{perf_metrics['win_rate'].values[0]:.4f}</td>
                </tr>
                <tr>
                    <td style="padding: 5px;"><b>最大恢复时间:</b></td>
                    <td style="padding: 5px;">{perf_metrics['max_time_to_recovery']}</td>
                </tr>
            </table>
        </div>
        """
        perf_div = Div(text=perf_text, width=1000, height=200)

        # 显示所有图表和绩效分析结果，移除 p 图表
        show(column(p_comparison, p_value, p_drawdown, p_returns_hist, p_weekly_returns_hist, p_monthly_returns_hist, p_cum_returns, perf_div))


# 定义您的两个策略
class UDVDStrategy(bt.Strategy):
    params = (
        ('window', 34),  # 计算rolling mean的窗口期
        ('size_pct', 0.166),  # 每个币种的仓位百分比
    )

    def __init__(self):
        # 初始化字典来存储每个资产的指标
        self.ud = {}             # 上升和下降波动的差异
        self.udvd = {}           # UDVD指标
        self.orders = {}
        self.trade_counts = {}   # 用于管理每个币种的订单状态
        self.value = []          # 存储组合总净值
        self.dates = []          # 存储日期序列
        self.debug_info = []    # 存储调试信息

        # 初始化每个数据集（即每个币种）的UDVD指标
        for d in self.datas:
            name = d._name
            self.trade_counts[name] = 0 
            self.orders[name] = None
            # 计算volup和voldown
            volup = (d.high - d.open) / d.open
            voldown = (d.open - d.low) / d.open
            self.ud[name] = volup - voldown
            # 通过rolling mean计算UDVD
            self.udvd[name] = bt.indicators.SimpleMovingAverage(self.ud[name], period=self.params.window)
            self.orders[name] = None  # 初始化每个币种的订单状态为None，表示没有未完成的订单

    def next(self):
        total_value = self.broker.getvalue()
        self.value.append(total_value)
        self.dates.append(self.datas[0].datetime.datetime(0))

        for d in self.datas:
            name = d._name
            position_size = self.getposition(d).size
            signal_type = 0  # 初始化为无信号

            # 生成买入或卖出信号
            if self.orders[name] is None:
                signal_type = self.generate_signals(d, position_size)

            # 存储调试信息
            self.debug_info.append({
                'Date': d.datetime.datetime(0),
                'Asset': name,
                'Position': position_size,
                'Size': self.calculate_position_size(d),
                'Cash': self.broker.getcash(),
                'Value': self.broker.getvalue(),
                'UDVD':self.udvd[name][0],
                'Close': d.close[0],
                'Volume': d.volume[0],
                'Trades': self.trade_counts[name],  # 添加交易次数
                'Signal': signal_type  # 添加买卖信号
            })

    def generate_signals(self, data, position_size):
        """
        生成交易信号，返回对应的信号值
        """
        name = data._name
        udvd_value = self.udvd[name][0]  # 获取当前UDVD值

        if udvd_value > 0 and position_size == 0:
            # 当UDVD大于0且当前没有持仓时，生成做多信号
            size = self.calculate_position_size(data)  # 计算仓位
            if size > 0:
                self.orders[name] = self.buy(data=data, size=size)
                self.trade_counts[name] += 1  # 执行买入操作
                return 1
        elif udvd_value < 0 and position_size > 0:
            # 当UDVD小于0且当前有持仓时，生成做空信号
            self.close(data=data)  # 平仓
            self.trade_counts[name] += 1 
            self.orders[name] = None  # 重置订单状态
            return 0

        return float('nan')   # 无信号

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
            self.orders[order.data._name] = None

    def get_net_value_series(self):
        """
        返回净值序列，用于后续分析
        """
        return pd.Series(self.value, index=self.dates)

    def get_debug_df(self):
        df=pd.DataFrame(self.debug_info)
        df=df.set_index('Date',drop=True)
        return df

class DonchianStrategy(bt.Strategy):
    params = (
        ('window', 46),  # 唐奇安通道的窗口大小，即过去多少天的最高和最低价
        ('size_pct', 0.166),  # 仓位大小百分比
    )
    def __init__(self):
        """
        初始化函数：定义需要的指标和变量
        """
        self.highest = {}
        self.lowest = {}
        self.orders = {}          # 用于管理每个资产的订单状态
        self.trade_counts = {}    # 用于统计每个资产的交易次数
        self.value = []           # 存储组合总净值
        self.dates = []           # 存储日期序列
        self.debug_info = []
        # 初始化每个资产的25日和350日移动平均线
        self.sma25 = {}
        self.sma350 = {}
      # 存储调试信息

        for d in self.datas:
            name = d._name
            self.trade_counts[name] = 0  # 初始化交易次数为0
            self.orders[name] = None     # 初始化订单状态

            # 计算过去window日的最高价和最低价
            self.highest[name] = bt.indicators.Highest(d.high(-1), period=self.params.window, plot=False)
            self.lowest[name] = bt.indicators.Lowest(d.low(-1), period=self.params.window, plot=False)
            # 计算25日和350日的移动平均线
            self.sma25[name] = bt.indicators.SMA(d.close, period=20, plot=False)
            self.sma350[name] = bt.indicators.SMA(d.close, period=120, plot=False)


    def next(self):
        total_value = self.broker.getvalue()
        self.value.append(total_value)
        self.dates.append(self.datas[0].datetime.datetime(0))

        for d in self.datas:
            name = d._name
            position_size = self.getposition(d).size
            signal_type = 0  # 初始化为无信号

            # 生成买入或卖出信号
            if self.orders[name] is None:
                signal_type = self.generate_signals(d, position_size)

            # 存储调试信息
            self.debug_info.append({
                'Date': d.datetime.datetime(0),
                'Asset': name,
                'Position': position_size,
                'Size': self.calculate_position_size(d),
                'Cash': self.broker.getcash(),
                'Value': self.broker.getvalue(),
                'Highest': self.highest[name][0],
                'Lowest': self.lowest[name][0],
                'High': d.high[0],
                'Low': d.low[0],
                'Close': d.close[0],
                'Volume': d.volume[0],                
                'SMA25': self.sma25[name][0],
                'SMA350': self.sma350[name][0],
                'Trades': self.trade_counts[name],  # 添加交易次数
                'Signal': signal_type  # 添加买卖信号
            })

    def generate_signals(self, data, position_size):
        """
        生成交易信号，返回对应的信号值
        """
        name = data._name
        highest = self.highest[name]
        lowest = self.lowest[name]
        close_price = data.close[0]
        sma25 = self.sma25[name][0]
        sma350 = self.sma350[name][0]

        # 过滤条件：仅当25日均线高于350日均线时允许做多
        if close_price > highest and position_size == 0 and sma25 > sma350:
            size = self.calculate_position_size(data)
            self.orders[name] = self.buy(data=data, size=size)
            self.trade_counts[name] += 1  # 增加交易次数
            return 1  # 买入信号

        elif close_price < lowest and position_size >= 0:
            self.close(data=data)  # 平仓
            self.trade_counts[name] += 1  # 增加交易次数
            return 0  # 卖出信号

        return float('nan')   # 无信号

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
            self.orders[order.data._name] = None

    def get_net_value_series(self):
        """
        返回净值序列，用于后续分析
        """
        return pd.Series(self.value, index=self.dates)

    def get_debug_df(self):

        df=pd.DataFrame(self.debug_info)

        df=df.set_index('Date',drop=True)

        return df

# 定义策略
class BBS_Strategy(bt.Strategy):
    params = (
        ('window', 144),  
        ('short_period', 20),
        ('long_period',60),
        ('size_pct', 0.2),  # 每个资产的仓位百分比
    )

    def __init__(self):
        # 初始化字典来存储每个资产的状态
        self.short_MA={}
        self.long_MA={}
        self.diff = {}  # 信号
        self.orders = {}
        self.trade_counts = {}   # 用于管理每个资产的订单状态
        self.value = []          # 存储组合总净值
        self.dates = []          # 存储日期序列
        self.debug_info = []     # 存储调试信息

        # 初始化每个数据集（即每个资产）的自由流通换手率
        for d in self.datas:
            name = d._name
            self.trade_counts[name] = 0 
            self.orders[name] = None
            # 计算自由流通换手率的滚动均值（free_turn_ma），窗口为 window
            free_turn_ma = bt.indicators.SimpleMovingAverage(d.lines.new_data, period=self.params.window, plot=False)

            # 计算收盘价的日收益率（pct）
            pct = bt.indicators.PctChange(d.close, period=1)

            # 计算收益率的滚动标准差（std），窗口为 window
            std = bt.indicators.StandardDeviation(pct, period=self.params.window, plot=False)

            # 计算 bull_bear = std / free_turn_ma
            bull_bear = bt.DivByZero(std, free_turn_ma, zero=0.0)

            # 计算短期和长期均线（short_MA 和 long_MA）
            self.short_MA[name]= bt.indicators.SimpleMovingAverage(bull_bear, period=self.params.short_period, plot=False)
            self.long_MA[name]= bt.indicators.SimpleMovingAverage(bull_bear, period=self.params.long_period, plot=False)

            # 计算差值 diff = short_MA - long_MA
            self.diff[name] = self.short_MA[name] - self.long_MA[name]

    def next(self):
        total_value = self.broker.getvalue()
        self.value.append(total_value)
        self.dates.append(self.datas[0].datetime.datetime(0))

        for d in self.datas:
            name = d._name
            position_size = self.getposition(d).size
            signal_type = 0  # 初始化为无信号

            # 生成买入或卖出信号
            if self.orders[name] is None:
                signal_type = self.generate_signals(d, position_size)

            # 存储调试信息
            self.debug_info.append({
                'Date': d.datetime.datetime(0),
                'Asset': name,
                'Position': position_size,
                'Size': self.calculate_position_size(d),
                'Cash': self.broker.getcash(),
                'Value': self.broker.getvalue(),
                'Short_MA': self.short_MA[name][0],  # 当前的自由流通换手率
                'Long_MA': self.long_MA[name][0],
                'Close': d.close[0],
                'Volume': d.volume[0],
                'Trades': self.trade_counts[name],  # 添加交易次数
                'Signal': signal_type  # 添加买卖信号
            })

    def generate_signals(self, data, position_size):
        """
        生成交易信号，返回对应的信号值
        """
        name = data._name
        short_MA = self.short_MA[name][0]
        long_MA=self.long_MA[name][0]  # 获取当前自由流通换手率值

        if short_MA < long_MA and position_size == 0:
            # 当换手率大于买入阈值且当前没有持仓时，生成买入信号
            size = self.calculate_position_size(data)  # 计算仓位
            self.orders[name] = self.buy(data=data, size=size)
            self.trade_counts[name] += 1  # 执行买入操作
            return 1
        elif short_MA >= long_MA and position_size > 0:
            # 当换手率小于卖出阈值且当前有持仓时，生成卖出信号
            self.close(data=data)  # 平仓
            self.trade_counts[name] += 1 
            self.orders[name] = None  # 不进行任何交易操作，确保空仓
            return 0

        return float('nan')  # 无信号

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
            self.orders[order.data._name] = None

    def get_net_value_series(self):
        """
        返回净值序列，用于后续分析
        """
        return pd.Series(self.value, index=self.dates)

    def get_debug_df(self):
        """
        返回调试信息的DataFrame
        """
        df = pd.DataFrame(self.debug_info)
        df = df.set_index('Date', drop=True)
        return df


# 主函数
if __name__ == '__main__':
    # 设置您的数据目录和目标资产
    data_loader = DataLoader()
    paths = {
        'daily': r'D:\数据库\同花顺ETF跟踪指数量价数据\1d',
        'hourly': r'D:\数据库\同花顺ETF跟踪指数量价数据\1h',
        'min15': r'D:\数据库\同花顺ETF跟踪指数量价数据\15min',
    }

    #目标资产代码
    # 资产列表
    target_assets = ["000016.SH","000300.SH","000852.SH","000905.SH",
    "399006.SZ","399303.SZ"]

    BBS_assets=['000016.SH', '000300.SH', '000905.SH','399303.SZ', '399006.SZ']            

    # 加载标准策略所需数据
    datas_standard = data_loader.load_selected_data(price_data_directory, target_assets, extension=".csv", price_factor=1)
    datas_standard_1H=data_loader.load_selected_data(price_data_directory_1H, target_assets, extension=".csv", price_factor=1)
    
    # 加载BBS_Strategy所需的额外数据
    selected_col = 'ths_free_turnover_ratio_index'
    datas_bbs = data_loader.load_selected_data_with_additional_data(price_data_directory, additional_data_directory,
                                                                    BBS_assets, selected_col)

    # 定义策略和资金分配比例
    strategies = [
        {'strategy': UDVDStrategy, 'allocation': 0.33, 'name': 'UDVDStrategy', 'datas': datas_standard},
        {'strategy': BBS_Strategy, 'allocation': 0.33, 'name': 'BBS_Strategy', 'datas': datas_bbs},
        {'strategy': DonchianStrategy, 'allocation':0.33, 'name': 'DonchianStrategy', 'datas': datas_standard_1H}]

    def Portfolio(strategies,initial_cash=1000000):
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
            # 添加数据到cerebro
            for data in strat['datas']:
                cerebro.adddata(data)
            # 添加策略
            cerebro.addstrategy(strat['strategy'])
            # 运行回测
            result = cerebro.run()
            # 获取策略实例
            strat_instance = result[0]
            # 获取净值序列
            net_value = strat_instance.get_net_value_series()
            # 存储净值序列
            strategy_values[strat['name']] = net_value
            # 计算收益率
            returns = net_value.pct_change().fillna(0)
            strategy_returns[strat['name']] = returns
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

    pf_nv,debug_df=Portfolio(strategies)

    Portfolio_nv=pf_nv[['Combined']].resample('D').last().dropna()


    #绩效分析
    portfolio_value, returns, drawdown_ts,meric=data_loader.performance_analysis(Portfolio_nv,'D')

    data_loader.plot_results('000906.SH',portfolio_value,drawdown_ts,returns,meric)
