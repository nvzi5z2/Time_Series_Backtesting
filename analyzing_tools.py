import pandas as pd
import numpy as np
from bokeh.plotting import figure, show, output_file
from bokeh.layouts import  gridplot, column
from bokeh.models import ColumnDataSource, HoverTool,Div
import empyrical as ep
import backtrader as bt
import matplotlib.pyplot as plt
import os
import backtrader as bt
from tqdm import tqdm
import seaborn as sns
from scipy.stats import norm
from bokeh.models import Span



class Analyzing_Tools():

    def plot_results(self,data, strat, portfolio_value, drawdown_ts, returns, perf_metrics):
        # 计算比特币价格的标准化值（归一化处理）
        btc_normalized = data['close'] / data['close'].iloc[0]

        # 创建一个图表，用于策略与底层资产价格对比
        p_comparison = figure(x_axis_type="datetime", title="策略与底层资产价格对比", plot_height=400, plot_width=1000)
        p_comparison.grid.grid_line_alpha = 0.3

        # 将数据转换为ColumnDataSource类型
        source = ColumnDataSource(data)
        p_comparison.line(data.index, btc_normalized, color='blue', legend_label='标准化价格')

        # 计算组合价值的标准化值
        portfolio_normalized = portfolio_value / portfolio_value.iloc[0]
        p_comparison.line(portfolio_normalized.index, portfolio_normalized, color='green', legend_label='策略组合价值')

        # 设置图例位置和点击策略
        p_comparison.legend.location = "top_left"
        p_comparison.legend.click_policy = "hide"

        # 创建一个图表，用于显示回测结果
        p = figure(x_axis_type="datetime", title="回测结果", plot_height=400, plot_width=1000)
        p.grid.grid_line_alpha = 0.3

        # 添加收盘价线
        p.line(data.index, data['close'], color='blue', legend_label='收盘价')

        # 获取买入和卖出信号的数据
        buy_signals = pd.DataFrame(strat.buy_signals, columns=['timestamp', 'price'])
        sell_signals = pd.DataFrame(strat.sell_signals, columns=['timestamp', 'price'])

        # 将买入和卖出信号转换为ColumnDataSource类型
        buy_source = ColumnDataSource(buy_signals)
        sell_source = ColumnDataSource(sell_signals)
        p.circle(x='timestamp', y='price', size=10, color='green', legend_label='买入信号', source=buy_source)
        p.triangle(x='timestamp', y='price', size=10, color='red', legend_label='卖出信号', source=sell_source)

        # 添加悬停工具，显示日期和价格
        hover = HoverTool(
            tooltips=[
                ("日期", "@timestamp{%F}"),
                ("价格", "@price{0.2f}")
            ],
            formatters={
                '@timestamp': 'datetime',
                '@price': 'printf',
            },
            mode='vline'
        )
        p.add_tools(hover)

        # 设置图例位置和点击策略
        p.legend.location = "top_left"
        p.legend.click_policy = "hide"

        # 创建一个图表，用于显示组合价值
        p_value = figure(x_axis_type="datetime", title="组合价值", plot_height=300, plot_width=1000)
        p_value.line(portfolio_value.index, portfolio_value, color='navy', legend_label='组合价值')
        p_value.grid.grid_line_alpha = 0.3

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
        # 使用更丰富的CSS样式美化Div
        perf_text = f"""
        <div style="background-color: #f9f9f9; border: 1px solid #ddd; padding: 15px; border-radius: 8px; width: 100%;">
            <h3 style="color: #333; font-family: Arial, sans-serif; text-align: center; margin-bottom: 15px;">策略绩效分析</h3>
            <table style="width: 100%; border-collapse: collapse; font-family: Arial, sans-serif; font-size: 14px;">
                <tr>
                    <td style="padding: 5px; border-bottom: 1px solid #ddd;"><b>总收益率:</b></td>
                    <td style="padding: 5px; border-bottom: 1px solid #ddd;">{perf_metrics['total_return']:.4f}</td>
                </tr>
                <tr>
                    <td style="padding: 5px; border-bottom: 1px solid #ddd;"><b>年化收益率:</b></td>
                    <td style="padding: 5px; border-bottom: 1px solid #ddd;">{perf_metrics['annual_return']:.4f}</td>
                </tr>
                <tr>
                    <td style="padding: 5px; border-bottom: 1px solid #ddd;"><b>年化波动率:</b></td>
                    <td style="padding: 5px; border-bottom: 1px solid #ddd;">{perf_metrics['annual_volatility']:.4f}</td>
                </tr>
                <tr>
                    <td style="padding: 5px; border-bottom: 1px solid #ddd;"><b>夏普比率:</b></td>
                    <td style="padding: 5px; border-bottom: 1px solid #ddd;">{perf_metrics['sharpe_ratio']:.4f}</td>
                </tr>
                <tr>
                    <td style="padding: 5px; border-bottom: 1px solid #ddd;"><b>索提诺比率:</b></td>
                    <td style="padding: 5px; border-bottom: 1px solid #ddd;">{perf_metrics['sortino_ratio']:.4f}</td>
                </tr>
                <tr>
                    <td style="padding: 5px; border-bottom: 1px solid #ddd;"><b>卡尔马比率:</b></td>
                    <td style="padding: 5px; border-bottom: 1px solid #ddd;">{perf_metrics['calmar_ratio']:.4f}</td>
                </tr>
                <tr>
                    <td style="padding: 5px; border-bottom: 1px solid #ddd;"><b>最大回撤:</b></td>
                    <td style="padding: 5px; border-bottom: 1px solid #ddd;">{perf_metrics['max_drawdown']:.4f}</td>
                </tr>
                <tr>
                    <td style="padding: 5px; border-bottom: 1px solid #ddd;"><b>胜率:</b></td>
                    <td style="padding: 5px; border-bottom: 1px solid #ddd;">{perf_metrics['win_rate']:.4f}</td>
                </tr>
                <tr>
                    <td style="padding: 5px;"><b>最大恢复时间:</b></td>
                    <td style="padding: 5px;">{perf_metrics['max_time_to_recovery']}</td>
                </tr>
            </table>
        </div>
        """
        perf_div = Div(text=perf_text, width=1000, height=200)

        # 显示所有图表和绩效分析结果
        show(column(p_comparison, p, p_value, p_drawdown, p_returns_hist, p_weekly_returns_hist, p_monthly_returns_hist, p_cum_returns, perf_div))

    def multi_asset_combined_performance_analysis(self,strat,freq='D'):
        """
        对多资产组合进行综合绩效分析，支持多种数据频段。
        
        参数:
        strat: Backtrader 策略实例
        freq: 数据频率，'D' 表示每日，'H' 表示每小时，支持 '30m', '15m', '5m', '1m', '2H', '4H' 等。
        
        返回:
        portfolio_value: 组合的净值序列
        returns: 组合的收益率序列
        drawdown_ts: 组合的回撤时间序列
        metrics: 各种绩效指标的字典
        """
        
        # 获取策略的整体组合净值序列
        portfolio_value = strat.get_net_value_series()
        
        # 计算收益率
        returns = portfolio_value.pct_change().dropna()
        
        # 确定年化系数
        if freq == 'D':
            annual_factor = 365  # 你可以调整为252，如果更适合你的数据
        elif freq == 'H' or freq == '1H':
            annual_factor = 365 * 4  # 每天4小时
        elif freq == '30m':
            annual_factor = 365 * 4 * 2  # 每小时2个30分钟
        elif freq == '15m':
            annual_factor = 365 * 4 * 4  # 每小时4个15分钟
        elif freq == '5m':
            annual_factor = 365 * 4 * 12  # 每小时12个5分钟
        elif freq == '1m':
            annual_factor = 365 * 4 * 60  # 每小时60个1分钟
        elif freq == '2H':
            annual_factor = 365 * 2  # 每天12个2小时
        elif freq == '4H':
            annual_factor = 365 * 1  # 每天6个4小时
        elif freq=='8H':
            annual_factor=365*3 # 每天3个8小时
        else:
            raise ValueError("Unsupported frequency")
        
        # 计算各项绩效指标
        total_return = ep.cum_returns_final(returns)  # 总收益率
        periods = len(returns)  # 总周期数
        annual_volatility = returns.std() * np.sqrt(annual_factor)
        annual_return = (1 + total_return) ** (annual_factor / periods) - 1
        max_drawdown = ep.max_drawdown(returns)
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else np.nan
        sharpe_ratio = annual_return / annual_volatility
        win_rate = (returns >= 0).sum() / len(returns)
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() * np.sqrt(annual_factor)
        sortino_ratio = annual_return / downside_std if downside_std != 0 else np.nan
        
        # 计算回撤时间序列
        cumulative_returns = ep.cum_returns(returns, starting_value=1)
        running_max = cumulative_returns.cummax()
        drawdown_ts = (cumulative_returns - running_max) / running_max
        
        # 计算最大恢复时间
        max_time_to_recovery = 0
        recovery_start = None
        for date in drawdown_ts.index:
            value = drawdown_ts.loc[date]  # 获取可能的标量值
            if isinstance(value, pd.Series):
                value = value.iloc[0]  # 如果是Series，取第一个元素
            if value < 0:  # 现在的 value 是一个单一标量值
                if recovery_start is None:
                    recovery_start = date
            elif recovery_start is not None:
                recovery_time = (date - recovery_start).days
                max_time_to_recovery = max(max_time_to_recovery, recovery_time)
                recovery_start = None
        metrics = {
            'total_return': total_return,
            'annual_volatility': annual_volatility,
            'annual_return': annual_return,
            'sharpe_ratio': sharpe_ratio,
            'calmar_ratio': calmar_ratio,
            'sortino_ratio': sortino_ratio,  # 索提诺比率
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'max_time_to_recovery': max_time_to_recovery
        }
        
        return portfolio_value, returns, drawdown_ts, metrics

    def plot_multi_asset_results(self,portfolio_value, drawdown_ts, returns, perf_metrics):
        """
        用于分析多资产组合净值的数据画图函数。

        参数:
        portfolio_value: 多资产组合的净值序列
        drawdown_ts: 回撤时间序列
        returns: 收益率序列
        perf_metrics: 绩效分析指标字典
        """

        # 创建一个图表，用于显示组合价值
        p_value = figure(x_axis_type="datetime", title="组合价值", plot_height=400, plot_width=1000)
        p_value.line(portfolio_value.index, portfolio_value, color='navy', legend_label='组合价值')
        p_value.grid.grid_line_alpha = 0.3

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
                    <td style="padding: 5px; border-bottom: 1px solid #ddd;">{perf_metrics['total_return']:.4f}</td>
                </tr>
                <tr>
                    <td style="padding: 5px; border-bottom: 1px solid #ddd;"><b>年化收益率:</b></td>
                    <td style="padding: 5px; border-bottom: 1px solid #ddd;">{perf_metrics['annual_return']:.4f}</td>
                </tr>
                <tr>
                    <td style="padding: 5px; border-bottom: 1px solid #ddd;"><b>年化波动率:</b></td>
                    <td style="padding: 5px; border-bottom: 1px solid #ddd;">{perf_metrics['annual_volatility']:.4f}</td>
                </tr>
                <tr>
                    <td style="padding: 5px; border-bottom: 1px solid #ddd;"><b>夏普比率:</b></td>
                    <td style="padding: 5px; border-bottom: 1px solid #ddd;">{perf_metrics['sharpe_ratio']:.4f}</td>
                </tr>
                <tr>
                    <td style="padding: 5px; border-bottom: 1px solid #ddd;"><b>索提诺比率:</b></td>
                    <td style="padding: 5px; border-bottom: 1px solid #ddd;">{perf_metrics['sortino_ratio']:.4f}</td>
                </tr>
                <tr>
                    <td style="padding: 5px; border-bottom: 1px solid #ddd;"><b>卡尔马比率:</b></td>
                    <td style="padding: 5px; border-bottom: 1px solid #ddd;">{perf_metrics['calmar_ratio']:.4f}</td>
                </tr>
                <tr>
                    <td style="padding: 5px; border-bottom: 1px solid #ddd;"><b>最大回撤:</b></td>
                    <td style="padding: 5px; border-bottom: 1px solid #ddd;">{perf_metrics['max_drawdown']:.4f}</td>
                </tr>
                <tr>
                    <td style="padding: 5px;"><b>最大恢复时间:</b></td>
                    <td style="padding: 5px;">{perf_metrics['max_time_to_recovery']}</td>
                </tr>
            </table>
        </div>
        """
        perf_div = Div(text=perf_text, width=1000, height=200)

        # 显示所有图表和绩效分析结果
        show(column(p_value, p_drawdown, p_returns_hist, p_weekly_returns_hist, p_monthly_returns_hist, p_cum_returns, perf_div))
    
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
        drawdown_ts = drawdown_ts[drawdown_ts.columns[0]]
        returns = returns[returns.columns[0]]
        # 读取基准指数价格数据
        index_price_path = r'E:\数据库\同花顺ETF跟踪指数量价数据\1d'
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
        portfolio_value_series = portfolio_value[portfolio_value.columns[0]]
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
   
    # 加载并调整数据以适应回测
    def load_and_adjust_data(self,file_path, asset_name, price_factor=1):
        # 读取数据并调整价格
        data = pd.read_csv(file_path, parse_dates=['time'], index_col='time')
        data['volume'] *= price_factor  # 对交易量进行调整
        return bt.feeds.PandasData(dataname=data, open='open', high='high', low='low', close='close', volume='volume', openinterest=1, name=asset_name)
    # 加载指定的多个品种的数据
    def load_selected_data(self,directory, target_assets, extension=".csv", price_factor=1):
        datafeeds = []
        for asset in target_assets:
            filename = f"{asset}{extension}"  # 构建文件名
            file_path = os.path.join(directory, filename)  # 构建文件路径
            if os.path.exists(file_path):
                datafeed = self.load_and_adjust_data(file_path, asset_name=asset, price_factor=price_factor)
                datafeeds.append(datafeed)  # 如果文件存在，则加载数据
            else:
                print(f"Warning: Data file for {asset} not found in directory.")  # 如果文件不存在，打印警告信息
        return datafeeds

        def safe_run_backtest(self, run_backtest_func, strategy, datafeeds, cash=100000.0, commission=0.003, **kwargs):
            """
            安全运行回测，捕获异常并返回策略实例。
            
            参数:
            - run_backtest_func: 回测函数，作为参数传入
            - strategy: 需要运行的策略
            - datafeeds: 数据源
            - cash: 初始现金（默认 100000）
            - commission: 交易佣金（默认 0.003）
            - kwargs: 其他传递给回测函数的参数
            
            返回:
            - 成功时返回策略实例，失败时返回 None
            """
        try:
            # 使用传入的 run_backtest_func 函数
            strat = run_backtest_func(strategy, datafeeds, cash=cash, commission=commission, slippage_perc=0.002, slippage_fixed=None, **kwargs)
            return strat
        except Exception as e:
            print(f"Error with params {kwargs}: {e}")
            return None
    
    
     # 加载并调整数据以适应回测
    
    #寻找最优参时报错程序
    def safe_run_backtest(self, run_backtest_func, strategy, datafeeds, cash=100000.0, commission=0.003, **kwargs):
        """
        安全运行回测，捕获异常并返回策略实例。
        
        参数:
        - run_backtest_func: 回测函数，作为参数传入
        - strategy: 需要运行的策略
        - datafeeds: 数据源
        - cash: 初始现金（默认 100000）
        - commission: 交易佣金（默认 0.003）
        - kwargs: 其他传递给回测函数的参数
        
        返回:
        - 成功时返回策略实例，失败时返回 None
        """
        try:
            # 使用传入的 run_backtest_func 函数
            strat = run_backtest_func(strategy, datafeeds, cash=cash, commission=commission, slippage_perc=0.002, slippage_fixed=None, **kwargs)
            return strat
        except Exception as e:
            print(f"Error with params {kwargs}: {e}")
            return None
    #寻找单一最优参
    def optimize_parameters(self, run_backtest_func, strategy, datafeeds, window_range, cash=100000.0, commission=0.003,freq='H'):
        """
        对策略参数进行优化，遍历指定的窗口范围，记录每个窗口的绩效指标。
        
        参数:
        - run_backtest_func: 回测函数，作为参数传入
        - strategy: 需要优化的策略
        - datafeeds: 数据源
        - window_range: 需要遍历的窗口范围
        - cash: 初始现金（默认 100000）
        - commission: 交易佣金（默认 0.003）
        
        返回:
        - 返回 DataFrame，包含每个窗口的 Sharpe 比率和对应的窗口大小
        """
        results = []
        for window in tqdm(window_range):
            # 使用 safe_run_backtest 方法运行回测
            strat = self.safe_run_backtest(run_backtest_func, strategy, datafeeds, cash=cash, commission=commission, window=window)
            if strat is None:
                continue
            
            # 分析回测结果，调用类中的 multi_asset_combined_performance_analysis 方法
            portfolio_value, returns, drawdown_ts, metrics = self.multi_asset_combined_performance_analysis(strat, freq=freq)
            
            # 将窗口参数加入到绩效指标中
            metrics['window'] = window
            results.append(metrics)

            # 输出当前窗口的 Sharpe 比率
            sharpe_ratio = metrics['sharpe_ratio']
            print(f'window: {window} -> sharpe_ratio: {sharpe_ratio:.2f}')

        # 将结果转换为 DataFrame
        results_df = pd.DataFrame(results)
        
        # 只保留需要的列，并按 Sharpe 比率排序
        results_df = results_df[['sharpe_ratio', 'window']]
        results_df = results_df.sort_values(by='window', ascending=True)
        
        # 绘制柱状图
        results_df.plot(x='window', y='sharpe_ratio', kind='bar', figsize=(10, 6), legend=False, color='skyblue')
        
        # 设置图表的标签和标题
        plt.xlabel('Window Size', fontsize=12)
        plt.ylabel('Sharpe Ratio', fontsize=12)
        plt.title('Sharpe Ratio vs Window Size', fontsize=14)
        
        # 显示图形
        plt.show()
        
        # 确保返回 DataFrame
        return results_df
    #寻找两个最优参
    def optimize_two_parameters(self, run_backtest_func, strategy, datafeeds, param_combinations, cash=100000.0, commission=0.003):
        """
        对策略的两个参数进行优化，遍历指定的窗口范围和第二参数范围，记录每个参数组合的绩效指标。
        
        参数:
        - run_backtest_func: 回测函数，作为参数传入
        - strategy: 需要优化的策略
        - datafeeds: 数据源
        - param_combinations: 参数组合列表 [(window_1, window_2), ...]
        - cash: 初始现金（默认 100000）
        - commission: 交易佣金（默认 0.003）
        
        返回:
        - 返回 DataFrame，包含每个参数组合的 Sharpe 比率和对应的参数组合
        """
        results = []
        
        # 遍历参数组合
        for window_1, window_2 in tqdm(param_combinations):
            # 使用 safe_run_backtest 方法运行回测
            strat = self.safe_run_backtest(run_backtest_func, strategy, datafeeds, cash=cash, commission=commission, window_1=window_1, window_2=window_2)
            if strat is None:
                continue
            
            # 分析回测结果，调用类中的 multi_asset_combined_performance_analysis 方法
            portfolio_value, returns, drawdown_ts, metrics = self.multi_asset_combined_performance_analysis(strat, freq='8H')
            
            # 将两个参数加入到绩效指标中
            metrics['window_1'] = window_1
            metrics['window_2'] = window_2
            results.append(metrics)

            # 输出当前参数组合的 Sharpe 比率
            sharpe_ratio = metrics['sharpe_ratio']
            print(f'window_1: {window_1}, window_2: {window_2} -> sharpe_ratio: {sharpe_ratio:.2f}')

        # 将结果转换为 DataFrame
        results_df = pd.DataFrame(results)
        
        # 只保留需要的列
        results_df = results_df[['sharpe_ratio', 'window_1', 'window_2']]
        
        # 将 DataFrame 转换为适合绘制热力图的格式
        pivot_df = results_df.pivot(index='window_1', columns='window_2', values='sharpe_ratio')
        
        # 使用 seaborn 绘制热力图
        plt.figure(figsize=(10, 6))
        sns.heatmap(pivot_df, annot=True, cmap='coolwarm', center=0, linewidths=0.5)
        
        # 设置图表的标签和标题
        plt.xlabel('window_2 Parameter', fontsize=12)
        plt.ylabel('window_1 Parameter', fontsize=12)
        plt.title('Sharpe Ratio Heatmap for Parameter Combinations', fontsize=14)
        
        # 显示图形
        plt.tight_layout()
        plt.show()
        
        # 确保返回 DataFrame
        return results_df
    # 批量测试单一资产
    def test_assets_performance(self, strategy, data_directory, run_backtest_func, target_assets=None, freq='D', cash=100000.0, commission=0.003, price_factor=1, **kwargs):
        """
        针对文件夹中的每个资产单独回测，并将其绩效表现存储到DataFrame中。
        
        :param strategy: 回测策略
        :param data_directory: 包含所有资产数据的文件夹路径
        :param run_backtest_func: 启动回测的函数（作为变量传入）
        :param target_assets: 要加载的资产列表，如果为None则加载文件夹中的所有文件
        :param freq: 数据的频率，用于绩效分析
        :param cash: 初始资金
        :param commission: 佣金率
        :param price_factor: 对价格进行调整的因子
        :param kwargs: 其他策略参数
        :return: 包含每个资产绩效表现的DataFrame
        """
        results = []  # 存储每个资产的绩效结果

        # 加载指定的多个币种的数据
        if target_assets:
            datafeeds = self.load_selected_data(data_directory, target_assets, price_factor=price_factor)
        else:
            # 如果没有提供指定资产列表，加载文件夹中所有可用资产
            datafeeds = []
            for file in os.listdir(data_directory):
                asset_name = os.path.splitext(file)[0]  # 资产名称
                file_path = os.path.join(data_directory, file)  # 数据文件路径
                if os.path.isfile(file_path):
                    datafeed = self.load_and_adjust_data(file_path, asset_name=asset_name, price_factor=price_factor)
                    datafeeds.append(datafeed)
        
        # 对每个数据源进行单独回测
        for datafeed in tqdm(datafeeds):
            asset_name = datafeed._name
            
            # 安全运行回测
            strat = self.safe_run_backtest(
                run_backtest_func=run_backtest_func, 
                strategy=strategy, 
                datafeeds=[datafeed],  # 将单个数据源放入列表
                cash=cash, 
                commission=commission, 
                **kwargs
            )

            if strat is None:
                print(f"Error processing asset {asset_name}. Skipping.")
                continue

            # 调用 multi_asset_combined_performance_analysis 进行绩效分析
            portfolio_value, returns, drawdown_ts, metrics = self.multi_asset_combined_performance_analysis(strat, freq=freq)
            
            # 将结果存储到字典中
            results.append({
                'Asset': asset_name,
                'Total Return': metrics['total_return'],
                'Annual Return': metrics['annual_return'],
                'Annual Volatility': metrics['annual_volatility'],
                'Sharpe Ratio': metrics['sharpe_ratio'],
                'Calmar Ratio': metrics['calmar_ratio'],
                'Sortino Ratio': metrics['sortino_ratio'],
                'Max Drawdown': metrics['max_drawdown'],
                'Win Rate': metrics['win_rate'],
                'Max Time to Recovery': metrics['max_time_to_recovery']
            })
        
        # 将结果转换为DataFrame
        results_df = pd.DataFrame(results)
        
        return results_df
    #绘画相关性热力图 
    def assets_correlation(data_directory, target_assets):
        path = data_directory

        # 获取文件名列表
        file_names = os.listdir(path)

        # 选择匹配的文件名
        selected_matched_files = [file for file in file_names if file.split('_')[0] in target_assets]

        corr_list = []
        asset_names = []  # 用于保存资产代码

        for file in selected_matched_files:
            # 读取每个文件的数据
            df = pd.read_csv(os.path.join(path, file), index_col=[0])

            # 将时间索引转换为datetime格式
            df.index = pd.to_datetime(df.index)

            # 只选择 'close' 列
            close = df[['close']]
            pct_change=close.pct_change()

            # 资产代码（从文件名中提取）
            asset_code = file.split('_')[0]
            asset_names.append(asset_code)

            # 将 'close' 列添加到相关性列表中
            corr_list.append(pct_change)

        # 将所有资产的 'close' 列合并到一个DataFrame
        corr_df = pd.concat(corr_list, axis=1)
        corr_df.columns = asset_names  # 设置列名为资产代码

        # 计算价格的相关性矩阵
        correlation_matrix = corr_df.corr()

        # 打印相关性矩阵
        print(correlation_matrix)

        # 绘制相关性热力图
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
        plt.title('Assets Correlation Heatmap')
        plt.show()

        return correlation_matrix

    #蒙塔卡罗分析策略业绩
    def monte_carlo_analysis(self, strat, num_simulations=10000, num_days=252, freq='D'):
        """
        蒙特卡洛模拟分析，支持多种数据频段的分析功(能。
        """
        # 获取策略的净值序列
        portfolio_value = strat.get_net_value_series()

        portfolio_value=portfolio_value.iloc[:,0].copy()

        # 计算收益率
        returns = portfolio_value.pct_change().dropna()

        # 确定年化系数
        annual_factor = self._get_annual_factor(freq)

        # 存储模拟结果
        annualized_returns = []
        sharpe_ratios = []
        max_drawdowns = []
        annual_volatilities = []
        sortino_ratios = []
        calmar_ratios = []

        # 进行蒙特卡洛模拟
        values = (returns + 1).cumprod().values
        for _ in range(num_simulations):
            simulated_values = self._monte_carlo_simulation(values, num_days)
            simulated_returns = np.diff(simulated_values) / simulated_values[:-1]

            # 计算各项指标
            total_return = np.cumprod(1 + simulated_returns)[-1] - 1
            periods = len(simulated_returns)
            annual_volatility = np.std(simulated_returns, ddof=1) * np.sqrt(annual_factor)
            annual_return = (1 + total_return) ** (annual_factor / periods) - 1
            sharpe_ratio = annual_return / annual_volatility if annual_volatility != 0 else np.nan
            max_drawdown = np.min(simulated_values) / np.max(simulated_values) - 1

            # Sortino比率计算
            downside_returns = simulated_returns[simulated_returns < 0]
            downside_deviation = np.std(downside_returns, ddof=1) * np.sqrt(annual_factor) if len(downside_returns) > 0 else np.nan
            sortino_ratio = annual_return / downside_deviation if downside_deviation != 0 else np.nan

            # Calmar比率计算
            calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else np.nan

            # 保存计算结果
            annualized_returns.append(annual_return)
            sharpe_ratios.append(sharpe_ratio)
            max_drawdowns.append(max_drawdown)
            annual_volatilities.append(annual_volatility)
            sortino_ratios.append(sortino_ratio)
            calmar_ratios.append(calmar_ratio)

        # 生成分析报告和可视化
        self._plot_results(annualized_returns, sharpe_ratios, max_drawdowns, annual_volatilities, sortino_ratios, calmar_ratios)

    def _get_annual_factor(self, freq):
        # 根据频率返回年化系数
        freq_map = {
            'D': 365, 'H': 365 * 4, '30m': 365 * 4 * 2, '15m': 365 * 4 * 4, 
            '5m': 365 * 4 * 12, '1m': 365 * 4 * 60, '2H': 365 * 2, '4H': 365 * 1, '8H': 365 * 3
        }
        return freq_map.get(freq, 365)

    def _monte_carlo_simulation(self, values, num_days):
        # 蒙特卡洛模拟函数
        start_index = np.random.randint(0, len(values) - num_days)
        return values[start_index:start_index + num_days]

    def _plot_results(self, annualized_returns, sharpe_ratios, max_drawdowns, annual_volatilities, sortino_ratios, calmar_ratios):
        # 计算分位数置信区间
        def quantile_confidence_interval(data, lower_quantile=2.5, upper_quantile=97.5):
            lower_bound = np.percentile(data, lower_quantile)
            upper_bound = np.percentile(data, upper_quantile)
            return lower_bound, upper_bound

        # 计算各项指标的分位数置信区间
        ci_ret = quantile_confidence_interval(annualized_returns, lower_quantile=0.5, upper_quantile=80)  # 99%置信区间
        ci_sharpe = quantile_confidence_interval(sharpe_ratios, lower_quantile=0.5, upper_quantile=80)
        ci_mdd = quantile_confidence_interval(max_drawdowns, lower_quantile=0.5, upper_quantile=80)
        ci_vol = quantile_confidence_interval(annual_volatilities, lower_quantile=0.5, upper_quantile=80)
        ci_sortino = quantile_confidence_interval(sortino_ratios, lower_quantile=0.5, upper_quantile=80)
        ci_calmar = quantile_confidence_interval(calmar_ratios, lower_quantile=0.5, upper_quantile=80)

        # 计算概率
        prob_ret_pos = np.mean(np.array(annualized_returns) > 0)
        prob_sharpe_gt_1 = np.mean(np.array(sharpe_ratios) > 1)
        prob_sortino_gt_1 = np.mean(np.array(sortino_ratios) > 1)
        prob_mdd_lt_neg_0_1 = np.mean(np.array(max_drawdowns) > -0.1)

        # 创建绘图函数
        def create_histogram_with_pdf_cdf(data, title, xlabel, ylabel, mu, std, ci):
            p = figure(title=title, x_axis_label=xlabel, y_axis_label=ylabel)
            # 绘制直方图
            hist, edges = np.histogram(data, bins=50, density=True)
            p.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:], fill_color="skyblue", line_color="black")

            # 绘制正态分布的PDF
            x = np.linspace(min(data), max(data), 100)
            pdf = norm.pdf(x, mu, std)
            p.line(x, pdf, line_color="black", line_width=2)

            # 绘制累积分布函数（CDF）
            cdf = norm.cdf(x, mu, std)
            p.extra_y_ranges = {"cdf": p.y_range}
            p.line(x, cdf, line_color="red", line_dash="dashed")

            # 绘制置信区间
            p.add_layout(Span(location=ci[0], dimension='height', line_color='green', line_dash='dashed'))
            p.add_layout(Span(location=ci[1], dimension='height', line_color='green', line_dash='dashed'))

            return p

        # 创建图表，包括卡玛比率
        p1 = create_histogram_with_pdf_cdf(annualized_returns, "年化收益率分布", "年化收益率", "密度", np.mean(annualized_returns), np.std(annualized_returns), ci_ret)
        p2 = create_histogram_with_pdf_cdf(sharpe_ratios, "夏普比率分布", "夏普比率", "密度", np.mean(sharpe_ratios), np.std(sharpe_ratios), ci_sharpe)
        p3 = create_histogram_with_pdf_cdf(max_drawdowns, "最大回撤分布", "最大回撤", "密度", np.mean(max_drawdowns), np.std(max_drawdowns), ci_mdd)
        p4 = create_histogram_with_pdf_cdf(annual_volatilities, "年化波动率分布", "年化波动率", "密度", np.mean(annual_volatilities), np.std(annual_volatilities), ci_vol)
        p5 = create_histogram_with_pdf_cdf(sortino_ratios, "索提诺比率分布", "索提诺比率", "密度", np.mean(sortino_ratios), np.std(sortino_ratios), ci_sortino)
        p6 = create_histogram_with_pdf_cdf(calmar_ratios, "卡玛比率分布", "卡玛比率", "密度", np.mean(calmar_ratios), np.std(calmar_ratios), ci_calmar)

        # 创建2x3的网格布局
        grid = gridplot([[p1, p2], [p3, p4], [p5, p6]])

        # 创建统计信息的Div，增加标准差信息
        stats = f"""
        <h2>蒙特卡洛模拟统计结果</h2>
        <ul>
            <li>年化收益率的均值: {np.mean(annualized_returns):.4f}, 标准差: {np.std(annualized_returns, ddof=1):.4f}, 置信区间: {ci_ret}</li>
            <li>夏普比率的均值: {np.mean(sharpe_ratios):.4f}, 标准差: {np.std(sharpe_ratios, ddof=1):.4f}, 置信区间: {ci_sharpe}</li>
            <li>最大回撤的均值: {np.mean(max_drawdowns):.4f}, 标准差: {np.std(max_drawdowns, ddof=1):.4f}, 置信区间: {ci_mdd}</li>
            <li>年化波动率的均值: {np.mean(annual_volatilities):.4f}, 标准差: {np.std(annual_volatilities, ddof=1):.4f}, 置信区间: {ci_vol}</li>
            <li>索提诺比率的均值: {np.mean(sortino_ratios):.4f}, 标准差: {np.std(sortino_ratios, ddof=1):.4f}, 置信区间: {ci_sortino}</li>
            <li>卡玛比率的均值: {np.mean(calmar_ratios):.4f}, 标准差: {np.std(calmar_ratios, ddof=1):.4f}, 置信区间: {ci_calmar}</li>
            <li>年化收益率大于0的概率: {prob_ret_pos:.4f}</li>
            <li>夏普比率大于1的概率: {prob_sharpe_gt_1:.4f}</li>
            <li>索提诺比率大于1的概率: {prob_sortino_gt_1:.4f}</li>
            <li>最大回撤大于-0.1的概率: {prob_mdd_lt_neg_0_1:.4f}</li>
        </ul>
        """
        div = Div(text=stats, width=800)

        layout = column(div, grid)

        # 使用Bokeh的show()函数直接在浏览器中展示图表
        show(layout)

    def optimize_two_parameters(self, run_backtest_func, strategy, datafeeds, param_combinations, cash=100000.0, commission=0.003):
        """
        对策略的两个参数进行优化，遍历指定的窗口范围和第二参数范围，记录每个参数组合的绩效指标。
        
        参数:
        - run_backtest_func: 回测函数，作为参数传入
        - strategy: 需要优化的策略
        - datafeeds: 数据源
        - param_combinations: 参数组合列表 [(window_1, window_2), ...]
        - cash: 初始现金（默认 100000）
        - commission: 交易佣金（默认 0.003）
        
        返回:
        - 返回 DataFrame，包含每个参数组合的 Sharpe 比率和对应的参数组合
        """
        results = []
        
        # 遍历参数组合
        for window_1, window_2 in tqdm(param_combinations):
            # 使用 safe_run_backtest 方法运行回测
            strat = self.safe_run_backtest(run_backtest_func, strategy, datafeeds, cash=cash, commission=commission, window_1=window_1, window_2=window_2)
            if strat is None:
                continue
            
            # 分析回测结果，调用类中的 multi_asset_combined_performance_analysis 方法
            portfolio_value, returns, drawdown_ts, metrics = self.multi_asset_combined_performance_analysis(strat, freq='8H')
            
            # 将两个参数加入到绩效指标中
            metrics['window_1'] = window_1
            metrics['window_2'] = window_2
            results.append(metrics)

            # 输出当前参数组合的 Sharpe 比率
            sharpe_ratio = metrics['sharpe_ratio']
            print(f'window_1: {window_1}, window_2: {window_2} -> sharpe_ratio: {sharpe_ratio:.2f}')

        # 将结果转换为 DataFrame
        results_df = pd.DataFrame(results)
        
        # 只保留需要的列
        results_df = results_df[['sharpe_ratio', 'window_1', 'window_2']]
        
        # 将 DataFrame 转换为适合绘制热力图的格式
        pivot_df = results_df.pivot(index='window_1', columns='window_2', values='sharpe_ratio')
        
        # 使用 seaborn 绘制热力图
        plt.figure(figsize=(10, 6))
        sns.heatmap(pivot_df, annot=True, cmap='coolwarm', center=0, linewidths=0.5)
        
        # 设置图表的标签和标题
        plt.xlabel('window_2 Parameter', fontsize=12)
        plt.ylabel('window_1 Parameter', fontsize=12)
        plt.title('Sharpe Ratio Heatmap for Parameter Combinations', fontsize=14)
        
        # 显示图形
        plt.tight_layout()
        plt.show()
        
        # 确保返回 DataFrame
        return results_df

    def parameter_optimization(self, parameter_grid, strategy_function, strategy_class, target_assets, paths, cash=100000.0, commission=0.0002, slippage_perc=0.0005, metric='sharpe_ratio'):   
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


        # 获取参数名称和取值列表
        param_names = list(parameter_grid.keys())
        param_values = [parameter_grid[key] for key in param_names]

        # 生成所有参数组合
        param_combinations = [dict(zip(param_names, values)) for values in product(*param_values)]

        results = []

        for params in param_combinations:
            print(f"正在测试参数组合：{params}")
            # 生成当前参数下的信号
            strategy_results, full_info = strategy_function(target_assets, paths, **params)

            # 运行回测
            strat = self.run_backtest(strategy_class, target_assets, strategy_results, cash, commission, slippage_perc)

            # 获取净值序列
            pv = strat.get_net_value_series()

            # 计算绩效指标
            portfolio_value, returns, drawdown_ts, metrics = self.performance_analysis(pv)

            # 收集指标和参数
            result_entry = {k: v for k, v in params.items()}
            result_entry.update(metrics)
            results.append(result_entry)

        # 将结果转换为 DataFrame
        results_df = pd.DataFrame(results)

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

            plt.figure(figsize=(10, 8))
            sns.heatmap(pivot_table, annot=True, fmt=".4f", cmap='viridis')
            plt.title(f'{metric} Heatmap')
            plt.ylabel(param1)
            plt.xlabel(param2)
            plt.show()
        else:
            print("无法可视化超过两个参数的结果，请减少参数数量。")

        # 返回结果 DataFrame
        return results_df

    def run_backtest(self,strategy, target_assets, strategy_results, cash=10000000.0, commission=0.0002, slippage_perc=0.0005, slippage_fixed=None, **kwargs):
        
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
    
    class PandasDataPlusSignal(bt.feeds.PandasData):
        lines = ('signal',)
        params = (
            ('signal', -1),  # 默认情况下，'signal' 列在最后一列   
        )
