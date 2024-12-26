import backtrader as bt
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from analyzing_tools import Analyzing_Tools
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from iFinDPy import *

def thslogindemo():
    # 输入用户的帐号和密码
    thsLogin = THS_iFinDLogin("hwqh100","155d50")
    print(thsLogin)
    if thsLogin != 0:
        print('登录失败')
    else:
        print('登录成功')

thslogindemo()


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
        ('size_pct',0.19),  # 每个资产的仓位百分比
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

#定义组合分析工具类，设置策略起始时间
class Tools:

    def __init__(self):

        self.Begin_Date='2019-01-04'

        self.current_position={'000300.SH':23844.90,
                "000852.SH":10282.80,
                '000905.SH':10978.60,
                "399006.SZ":23197.90,
                '399303.SZ':10210.00,
                'cash':920657.8}

        self.ETF_code={'000300.SH':'510310.SH',
                "000852.SH":"159845.SZ",
                '000905.SH':"512500.SH",
                "399006.SZ":"159915.SZ",
                '399303.SZ':"159628.SZ"}       
        
    def Portfolio(self,strategies,initial_cash=10000000):
        Begin_Date=self.Begin_Date
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
                select_dataframe=dataframe.loc[Begin_Date:,:]
                data_feed = Adding_Signal(dataname=select_dataframe, name=code)
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

    def Strategies_Corr_and_NV(self,pf_nv):
        """
        计算基金多个策略的相关性，并绘制热力图。

        参数:
            pf_nv (pd.DataFrame): 基金策略的净值数据，行是时间，列是策略名称。

        返回:
            corr_matrix (pd.DataFrame): 策略的相关性矩阵。
        """
        # 计算净值的每日收益率（百分比变化）
        pf_nv_pct_change = pf_nv.pct_change().dropna()

        # 计算相关性矩阵
        corr_matrix = pf_nv_pct_change.corr()

        # 设置绘图风格
        sns.set(style="whitegrid", font_scale=1.2)

        # 绘制热力图
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, square=True)
        plt.title("Strategies Correlation Heatmap", fontsize=16)
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()

        return corr_matrix

    def caculate_signals_and_trades(self,debug_df,T0_Date):

        current_position=self.current_position
        ETF_code=self.ETF_code
        T0_debug=debug_df.loc[T0_Date,:]

        #先把signal列的信号转换为0方便计算

        T0_debug.loc[T0_debug['Signal']==-1,'Signal']=0

        #计算应该下单的调整的size

        adjusted_signal_and_size=T0_debug[['Asset','Signal','Size','Close']]
        adjusted_signal_and_size.loc[:,"adjusted_size"]=adjusted_signal_and_size.loc[:,"Signal"]*adjusted_signal_and_size.loc[:,"Size"]
        adjusted_signal_and_size.loc[:,"trade_amount"]=adjusted_signal_and_size.loc[:,"adjusted_size"]*adjusted_signal_and_size.loc[:,"Close"]
        Amount_List= adjusted_signal_and_size.groupby("Asset")["trade_amount"].sum().reset_index()

        #提取当日总组合的价值

        strategy_value=T0_debug.groupby("Strategy")["Value"].sum().reset_index()
        number_of_assets= len(T0_debug['Asset'].unique())
        strategy_value.loc[:,"Value"]=strategy_value.loc[:,"Value"]/number_of_assets
        portfolio_value_sum=strategy_value.loc[:,"Value"].sum()
        
        #计算占比
        pd.set_option("display.float_format", lambda x: f"{x:,.4f}")
        Amount_List.loc[:,"portfolio_value"]=portfolio_value_sum
        Amount_List=Amount_List.set_index('Asset',drop=True)
        cash_trade_amount=portfolio_value_sum-Amount_List.loc[:,"trade_amount"].sum()
        cash_row = pd.DataFrame({
            'trade_amount': [cash_trade_amount],
            'portfolio_value': portfolio_value_sum
        }, index=['cash'])
        Amount_result=pd.concat([Amount_List,cash_row],axis=0)
        Amount_result.loc[:,"proportion"]=Amount_result.loc[:,"trade_amount"]/Amount_result.loc[:,"portfolio_value"]
        

        #和现有持仓进行对比

        current_position_df=pd.DataFrame.from_dict(current_position, orient='index', columns=['current_position_value'])
        # 计算 current_position_value 的总和
        total_value = current_position_df['current_position_value'].sum()

        # 添加占比列
        current_position_df['current_position_ratio'] = (
            current_position_df['current_position_value'] / total_value
        )

        #合并现有持仓和目标持仓表

        result=pd.merge(Amount_result,current_position_df,right_index=True,left_index=True)
        result.loc[:,"adjusted_position"]=result.loc[:,"proportion"]-result.loc[:,"current_position_ratio"]
        result.loc[:,"adjusted_value"]=total_value*result.loc[:,"adjusted_position"]


        #根据昨日收盘价计算调仓
        
        index_code=result.index.to_list()
        index_code_list=index_code[:-1]
        etf_close_price=[]
        for code in index_code_list:
            etf=ETF_code[code]
            close=THS_HQ(etf,'close','',T0_Date,T0_Date).data
            close_price=close["close"].values[0]
            etf_close_price.append(close_price)
        etf_close_price.append(0)
        result.loc[:,"etf_close_price"]=etf_close_price
        result.loc[:,"adjusted_shares"]=result.loc[:,"adjusted_value"]/result.loc[:,"etf_close_price"]
        
        #和昨日的信号精选对比

        T0_Date_datetime=pd.Timestamp(T0_Date)
        unique_dates =debug_df.index.unique()
        target_index =unique_dates.get_loc(T0_Date_datetime)
        if target_index > 0:
            previous_date = unique_dates[target_index - 1]

        previous_date_df=debug_df.loc[previous_date,:]

        #抽取信号信息和表弟信息
        previous_date_signal=previous_date_df[['Asset',"Strategy",'Signal']]
        previous_date_signal=previous_date_signal.reset_index(drop=True)
        yesterday_result = previous_date_signal.pivot(index='Asset', columns='Strategy', values='Signal')
        
        T0_debug_signal=debug_df.loc[T0_Date,:]

        T0_debug_signal=T0_debug_signal[['Asset',"Strategy",'Signal']]
        T0_debug_signal=T0_debug_signal.reset_index(drop=True)
        T0_result = T0_debug_signal.pivot(index='Asset', columns='Strategy', values='Signal')
        
        comparison = T0_result == yesterday_result

        if not comparison.values.all():  # 如果存在 False
            print("信号改变")
        else:
            print("信号没有改变")

        result=result[['trade_amount','proportion','current_position_ratio','adjusted_position','adjusted_value',
                        'etf_close_price','adjusted_shares']]
        
        return result,comparison
    
    def set_clusters(self,Corr_df,n_groups):
        # 假设 Corr_df 是已经计算好的相关性矩阵
        # 转换相关性为距离
        distance_matrix = 1 - Corr_df

        # 执行层次聚类
        linked = linkage(distance_matrix, 'average')

        # 绘制树状图来观察聚类情况
        dendrogram(linked, labels=Corr_df.index)
        plt.title('Dendrogram')
        plt.xlabel('Strategy Index')
        plt.ylabel('Distance')
        # plt.axhline(y=1.4, color='r', linestyle='--')  # 添加一条红线表示截断位置
        plt.show()

        # # 根据树状图选择的截断值进行聚类
        # clusters = fcluster(linked, 1.4, criterion='distance')

        # 直接指定生成三个聚类
        clusters = fcluster(linked, n_groups, criterion='maxclust')

        # 将聚类结果添加到原始 DataFrame 中
        Corr_df['Cluster'] = clusters

        # 输出聚类结果
        print(Corr_df['Cluster'])

        # 创建一个字典来存储每个聚类的成员列表
        cluster_dict = {}
        for cluster_id in np.unique(clusters):
            cluster_dict[cluster_id] = list(Corr_df.index[Corr_df['Cluster'] == cluster_id])

        # 输出每个聚类的成员
        for key, value in cluster_dict.items():
            print(f"Cluster {key}: {value}")

        return cluster_dict


# 定义策略类（设置数据路径和选择资产）
class Strategies:

    def __init__(self):
        # 定义数据路径
        self.paths = {
            'daily': r'D:\数据库\同花顺ETF跟踪指数量价数据\1d',
            'hourly': r'D:\数据库\同花顺ETF跟踪指数量价数据\1h',
            'min15': r'D:\数据库\同花顺ETF跟踪指数量价数据\15min',
            'option': r'D:\数据库\另类数据\ETF期权数据',
            'EDB':r'D:\数据库\同花顺EDB数据',
            'new_HL': r'D:\数据库\另类数据\新高新低\001005010.csv',
            'up_companies':r'D:\数据库\另类数据\涨跌家数\A股.csv',
            'up_down': r'D:\数据库\另类数据\涨停跌停\001005010.csv',
            'A50': r'D:\数据库\另类数据\A50数据\CN0Y.SG.csv',
            #'pv_export':r"D:\量化交易构建\私募基金研究\股票策略研究\策略净值序列"
        }
        # 定义选择的资产
        self.target_assets = ["000300.SH","000852.SH",
                              "000905.SH", "399006.SZ","399303.SZ"]

    # 突破类策略
    def UDVD(self,window_1=27):
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

    def SMA_H(self,window_1=61,window_2=101):
        #信号结果字典
        results = {}
        #全数据字典，包含计算指标用于检查
        full_info={}
        
        paths=self.paths

        target_assets=self.target_assets

        #编写策略主体部分
        for code in target_assets:
            # 读取数据
            daily_data = pd.read_csv(os.path.join(paths['daily'], f"{code}.csv"), index_col=[0])
            daily_data.index = pd.to_datetime(daily_data.index)
            df=daily_data.copy()
            hourly_data = pd.read_csv(os.path.join(paths['hourly'], f"{code}.csv"), index_col=[0])
            hourly_data.index = pd.to_datetime(hourly_data.index)

            hourly_data["var_1"] = hourly_data['close'].rolling(window_1).mean()
            hourly_data["var_2"] =hourly_data['close'].rolling(window_2).mean()
            # 添加信号列
            hourly_data.loc[(hourly_data["var_1"].shift(1) <= hourly_data["var_2"].shift(1)) & (hourly_data["var_1"] >= hourly_data["var_2"]) , 'signal'] = 1
            hourly_data.loc[(hourly_data["var_1"].shift(1) > hourly_data["var_2"].shift(1)) & (hourly_data["var_1"] < hourly_data["var_2"]) , 'signal'] = -1
            
            hourly_data['signal'].fillna(method='ffill', inplace=True)
            hourly_exchange = hourly_data.resample('D').last()

            df = pd.merge(df, hourly_exchange[['signal']], left_index=True, right_index=True, how='left')        
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

    # 成交量类策略
    def V_MACD(self,window_1=42,window_2=0):
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

    #期权类策略
    def PCR(self,window_1=62):
        #信号结果字典
        results = {}
        #全数据字典，包含计算指标用于检查
        full_info={}
        target_assets=self.target_assets
        paths=self.paths
        #编写策略主体部分
        for code in target_assets:
            # 读取数据
            daily_data = pd.read_csv(os.path.join(paths['daily'], f"{code}.csv"), index_col=[0])
            daily_data.index = pd.to_datetime(daily_data.index)

            df=daily_data.copy()
            option_code='510050'
            data = pd.read_csv(os.path.join(paths['option'], f"{option_code}.csv"), index_col=[0])
            data.index= pd.to_datetime(data.index)
            
            #和指数数据合并
            merged_df = pd.merge(df, data[['p02872_f005', 'p02872_f006']], left_index=True, right_index=True, how='left')        
            #计算PCR滚动五天的均值
            merged_df['PCR']= merged_df['p02872_f006'].rolling(5).mean()/merged_df['p02872_f005'].rolling(5).mean()
            # 定义一个函数来计算从小到大排列后排名第 70% 的值
            def calc_70th_percentile(x):
                return np.percentile(x, 70)

            # 使用 rolling 方法计算过去六十日从小到大排列后第 70 个百分位数
            merged_df['rolloing_70%'] = merged_df['PCR'].rolling(window_1).apply(calc_70th_percentile, raw=True)

            df['PCR'] = merged_df['PCR']
            df['rolloing_70%'] = merged_df['rolloing_70%']


            # 信号触发条件
            df.loc[(df["PCR"].shift(1) <= df["rolloing_70%"].shift(1)) & (df["PCR"] > df["rolloing_70%"]) , 'signal'] = 1
            df.loc[(df["PCR"].shift(1) > df["rolloing_70%"].shift(1)) & (df["PCR"] <= df["rolloing_70%"]), 'signal'] = -1

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

    #宏观类策略

    def Inventory_Cycle(self,window_1=7):
        #PMI：原材料库存代码=M002043811
        #BCI:企业库存前瞻指数=M004488064 

        #信号结果字典
        results = {}
        #全数据字典，包含计算指标用于检查
        full_info={}
        target_assets=self.target_assets
        paths=self.paths
        
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
            PMI_Inventory_value=PMI_Inventory_value.sort_index()
            #计算第一个信号（PMI原材料库存信号）

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

            PMI_Signals.columns=['signal']

            PMI_Signals.index=pd.to_datetime(PMI_Signals.index)

            # 使用 resample 将月频转换为日频
            PMI_Signals_daily = PMI_Signals.resample('D').ffill()
            PMI_Signals_daily.index=pd.to_datetime(PMI_Signals_daily.index)
            
            #存储结果
            total_info=pd.concat([daily_data,PMI_Signals_daily],axis=1)
            first_signal_date=PMI_Signals_daily.index[0]
            first_signal_formatted_date=first_signal_date.strftime('%Y-%m-%d')
            total_info_selected=total_info.loc[first_signal_formatted_date:,:]

            #填充缺失值
            total_info_selected=total_info_selected.ffill()

            signal=total_info_selected[['signal']]

            daily_data.loc[:,"signal"]=signal

            results[code] = daily_data.dropna()

            full_info[code]=daily_data    
        
        return results,full_info

    #情绪类

    def high_low(self,window_1=12):
        #信号结果字典
        results = {}
        #全数据字典，包含计算指标用于检查
        full_info={}

        target_assets=self.target_assets
        paths=self.paths
        
        #编写策略主体部分
        for code in target_assets:
            # 读取数据
            daily_data = pd.read_csv(os.path.join(paths['daily'], f"{code}.csv"), index_col=[0])
            daily_data.index = pd.to_datetime(daily_data.index)

            df=daily_data.copy()

            #将上涨、平、下跌数量和涨跌停数量合并
            high_low_path=paths['new_HL']
            high_low=pd.read_csv(high_low_path)
            up_company_path=paths['up_companies']
            data = pd.read_csv(up_company_path)
            data=data.rename(columns={'p00112_f001':'time'})
            high_low['time'] = pd.to_datetime(high_low['time'])
            data['time'] = pd.to_datetime(data['time'])
            num_df = pd.merge(high_low, data[['p00112_f002', 'p00112_f003', 'p00112_f004', 'time']], on='time', how='left')        
            num_df.set_index('time', inplace=True)

            #和指数数据合并
            merged_df = pd.merge(df, num_df[['ths_new_high_num_block','ths_new_low_num_block','p00112_f002', 'p00112_f003', 'p00112_f004']], left_index=True, right_index=True, how='left')        
            
            #计算涨跌停剪刀差
            merged_df['股票数量']=merged_df['p00112_f002'] +  merged_df['p00112_f003'] + merged_df['p00112_f004']
            merged_df['新高数量']=merged_df['ths_new_high_num_block']
            merged_df['新低数量']=merged_df['ths_new_low_num_block']
            merged_df['高低差']=(merged_df['新高数量']-merged_df['新低数量'])/merged_df['股票数量']
            merged_df['高低差'].fillna(method='ffill', inplace=True)
            # 确保merged_df的索引唯一
            merged_df = merged_df[~merged_df.index.duplicated(keep='first')]

            df['var_1'] = merged_df['高低差']
            df['var_2'] = -0.2
            df['var_3']=window_1/1000
            # 根据条件生成信号值列
            df.loc[(df["var_1"].shift(1) >= df["var_2"].shift(1)) & (df["var_1"] < df["var_2"]) , 'signal'] = 1
            df.loc[(df["var_1"].shift(1) < df["var_3"].shift(1)) & (df["var_1"] >= df["var_3"]) , 'signal'] = -1

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


    def UD(self,window_1=38,window_2=65):
            #信号结果字典
            results = {}
            #全数据字典，包含计算指标用于检查
            full_info={}
            target_assets=self.target_assets
            paths=self.paths
            
            #编写策略主体部分
            for code in target_assets:
                # 读取数据
                daily_data = pd.read_csv(os.path.join(paths['daily'], f"{code}.csv"), index_col=[0])
                daily_data.index = pd.to_datetime(daily_data.index)

                df=daily_data.copy()

                #将上涨、平、下跌数量和涨跌停数量合并
                up_down_path=paths['up_down']
                up_down=pd.read_csv(up_down_path)
                up_company_path=paths['up_companies']
                data = pd.read_csv(up_company_path)
                data=data.rename(columns={'p00112_f001':'time'})
                up_down['time'] = pd.to_datetime(up_down['time'])
                data['time'] = pd.to_datetime(data['time'])
                num_df = pd.merge(up_down, data[['p00112_f002', 'p00112_f003', 'p00112_f004', 'time']], on='time', how='left')        
                num_df.set_index('time', inplace=True)

                #和指数数据合并
                merged_df = pd.merge(df, num_df[['ths_limit_up_stock_num_sector','ths_limit_down_stock_num_sector','p00112_f002', 'p00112_f003', 'p00112_f004']], left_index=True, right_index=True, how='left')        
                
                #计算涨跌停剪刀差
                merged_df['股票数量']=merged_df['p00112_f002'] +  merged_df['p00112_f003'] + merged_df['p00112_f004']
                merged_df['涨停数量']=merged_df['ths_limit_up_stock_num_sector']
                merged_df['跌停数量']=merged_df['ths_limit_down_stock_num_sector']
                merged_df['涨跌停差']=(merged_df['涨停数量']-merged_df['跌停数量'])/merged_df['股票数量']

                # 计算AMA
                merged_df['AMA_30'] = merged_df['涨跌停差'].ewm(window_1).mean()
                merged_df['AMA_100'] = merged_df['涨跌停差'].ewm(window_2).mean()
                merged_df['AMA']=merged_df['AMA_30']/merged_df['AMA_100']


                merged_df['ration']=merged_df['AMA']
                # 确保merged_df的索引唯一
                merged_df = merged_df[~merged_df.index.duplicated(keep='first')]

                df['var_1'] = merged_df['ration']
                df['var_2'] = 1.15
                df['var_3']=merged_df['AMA_30']
                df['var_4']=merged_df['AMA_100']

                # 根据条件生成信号值列
                df['signal_1'] = -1  # 初始化信号列为 -1
                condition = (df['var_1'] > df['var_2']) & (df['var_3'] > 0) & (df['var_4'] > 0)
                df.loc[condition, 'signal_1'] = 1
                
                df['var_5'] = merged_df['涨跌停差']
                df['var_6'] = -0.2
                df['var_7']=0.019
                # 根据条件生成信号值列
                df.loc[(df["var_5"].shift(1) >= df["var_6"].shift(1)) & (df["var_5"] < df["var_6"]) , 'signal_2'] = 1
                df.loc[(df["var_5"].shift(1) < df["var_7"].shift(1)) & (df["var_5"] >= df["var_7"]) , 'signal_2'] = -1
                # pos为空的，向上填充数字
                df['signal_2'].fillna(method='ffill', inplace=True)

                df['signal_sum']=df['signal_1']+df['signal_2']
                # 添加signal列，使用apply函数
                df['signal'] = df['signal_sum'].apply(lambda x: 1 if x >= 0 else -1)
                result=df
                # 将信号合并回每日数据
                daily_data = daily_data.join(result[['signal']], how='left')
                daily_data[['signal']].fillna(0, inplace=True)
                daily_data=daily_data.dropna()

                # 存储结果
                results[code] = daily_data
                full_info[code]=result

            return results,full_info

    #外资类
    
    def FS_A50(self,window_1=20):
        #信号结果字典
        results = {}
        #全数据字典，包含计算指标用于检查
        full_info={}
        target_assets=self.target_assets
        paths=self.paths
        
        #编写策略主体部分
        for code in target_assets:
            # 读取数据
            daily_data = pd.read_csv(os.path.join(paths['daily'], f"{code}.csv"), index_col=[0])
            daily_data.index = pd.to_datetime(daily_data.index)

            df=daily_data.copy()
            df = df.round(0)
            # 使用需要的列，通常是高、低、收盘价
            close = df["close"]    
            low = df["low"]
            open= df['open']
            high = df["high"]
            volume = df['volume']

            #导入富时A50
            A50_Path=paths['A50']
            data = pd.read_csv(A50_Path)
            #导入中证全指
            zzqz=pd.read_csv(os.path.join(paths['daily'], f"{'000985.CSI'}.csv"))
            #日期
            data['time'] = pd.to_datetime(data['time'])
            zzqz['time'] = pd.to_datetime(zzqz['time'])
            #self.df['time'] = pd.to_datetime(self.df['time'])
            #按照时间升序排列
            data= data.sort_values(by='time')
            zzqz= zzqz.sort_values(by='time')
            #计算涨跌幅
            data['chg']=(data['ths_close_price_future']-data['ths_open_price_future'])/data['ths_open_price_future']
            merged_df = pd.merge(zzqz, data[['time', 'chg']], left_on='time', right_on='time', how='left')
            # 向下填充'value'列的NaN值
            merged_df['chg'].fillna(method='ffill', inplace=True)
            #计算富时A50及中证全指差值
            merged_df['diff']=merged_df['chg']-(merged_df['close']-merged_df['open'])/merged_df['open']
            #将中证全债及富时A50的差合并到df中
            merged_df.set_index('time', inplace=True)   

            df = pd.merge(df, merged_df[['diff']], left_index=True, right_index=True, how='left')


            df['var_1'] = df['diff']
            df['var_2'] = window_1/1000
            
            df.loc[(df["var_1"].shift(1) <= df["var_2"].shift(1)) & (df["var_1"] > df["var_2"]) , 'signal'] = 1
            df.loc[(df["var_1"].shift(1) > df["var_2"].shift(1)) & (df["var_1"] <= df["var_2"]) , 'signal'] = -1

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
tools=Tools()

# 生成策略数据

#趋势类
UDVD_results,_= strategies_instance.UDVD()
V_MACD_results,_ = strategies_instance.V_MACD()
SMA_H_results,_=strategies_instance.SMA_H()

#期权类
PCR_results,_=strategies_instance.PCR()

#宏观类
Inventory_Cycle_results,_=strategies_instance.Inventory_Cycle()

#情绪类
high_low_results,_=strategies_instance.high_low()
UD_reults,_=strategies_instance.UD()

#外资类
FS_A50_results,_=strategies_instance.FS_A50()

# 定义添加信号的数据类
Adding_Signal = PandasDataPlusSignal

# 定义策略和资金分配比例
strategies_list = [
    {'strategy': EqualWeightsStrategy, 'allocation': 0.0666, 'name': 'UDVD', 'datas': UDVD_results},
    {'strategy': EqualWeightsStrategy, 'allocation':0.0666, 'name': 'SMA_H', 'datas': SMA_H_results},
    {'strategy': EqualWeightsStrategy, 'allocation':0.0666, 'name': 'V_MACD', 'datas': V_MACD_results},
    {'strategy': EqualWeightsStrategy, 'allocation':0.10, 'name': 'UD', 'datas': UD_reults},
    {'strategy': EqualWeightsStrategy, 'allocation':0.20, 'name': 'PCR', 'datas': PCR_results},
    {'strategy': EqualWeightsStrategy, 'allocation':0.20, 'name': 'Inventory_Cycle', 'datas': Inventory_Cycle_results},
    {'strategy': EqualWeightsStrategy, 'allocation':0.20, 'name': 'high_low', 'datas': high_low_results},
    {'strategy': EqualWeightsStrategy, 'allocation':0.10, 'name': 'FS_A50', 'datas': FS_A50_results}
    ]

# 运行组合回测
pf_nv, debug_df = tools.Portfolio(strategies_list)

# 获取组合净值
Portfolio_nv = pf_nv[['Combined']]


#组合分析

AT=Analyzing_Tools()

index_price_path=strategies_instance.paths['daily']

portfolio_value, returns, drawdown_ts, metrics = AT.performance_analysis(Portfolio_nv, freq='D')

AT.plot_results('000906.SH',index_price_path,Portfolio_nv, drawdown_ts, returns, metrics)

Corr=tools.Strategies_Corr_and_NV(pf_nv)

tools.set_clusters(Corr,6)


#信号处理和目标仓位生成

T0_Date='2024-12-26'

target_assets_position,difference=tools.caculate_signals_and_trades(debug_df,T0_Date)

