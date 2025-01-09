# -*- coding: utf-8 -*-
from iFinDPy import *
from datetime import datetime
import pandas as pd
import time as _time
import json
from threading import Thread,Lock,Semaphore
import requests
import numpy as np
from datetime import datetime
from numpy import nan as NA
import matplotlib.pyplot as plt
import math
from pandas.tseries.offsets import Day,MonthEnd
import os
from tqdm import tqdm  # 用于显示进度条
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor


def thslogindemo():
    # 输入用户的帐号和密码
    thsLogin = THS_iFinDLogin("hwqh101","95df1a")
    print(thsLogin)
    if thsLogin != 0:
        print('登录失败')
    else:
        print('登录成功')

thslogindemo()


class Downloading_And_Updating_Data():

    def __init__(self):

        self.A_Share_Trading_Hours= ["10:30", "11:30", "14:00", "15:00"]

        self.A_Share_Trading_15mins=["9:45", "10:00", "10:15", "10:30", 
                                    "10:45", "11:00", "11:15", "11:30",
                                     "13:15", "13:30", "13:45","14:00",
                                      "14:15", "14:30", "14:45", "15:00"]
    
    def Downloading_Market_Vol_Price_Data(self,Code_List,Begin_Date,End_Date,Export_Path):

        Export_Path=Export_Path

        os.chdir(Export_Path)

        print("下载开始")

        for i in range(0,len(Code_List)):

            try:

                Data=THS_HQ(Code_List[i],'open,high,low,close,volume,amount,turnoverRatio,totalCapital,floatCapital','',Begin_Date,End_Date).data

                Data.index=Data.loc[:,"time"]

                Data=Data.iloc[:,1:]

                Data.to_csv(Code_List[i]+".csv")

                print(str(i/len(Code_List)))
            
            except:

                print(Code_List[i]+"出现错误")

        print("下载结束")

    def Updating_Market_Vol_Price_Data(self,End_Date,Data_Path):

        Data_Path=Data_Path

        os.chdir(Data_Path)

        List=os.listdir()

        Doc_Number=len(List)

        print("更新开始")

        for i in range(0,Doc_Number):

            try:

                df=pd.read_csv(Data_Path+"\\"+List[i],index_col=[0])

                df.index=pd.to_datetime(df.index)

                if df.index[-1].strftime('%Y-%m-%d')!=End_Date:

                    New_Begin_Date=df.index[-1]+Day()

                    New_Begin_Date=New_Begin_Date.strftime("%Y-%m-%d")

                    code = List[i].replace('.csv', "")

                    new_data= THS_HQ(code,'open,high,low,close,volume,amount,turnoverRatio,totalCapital,floatCapital','',New_Begin_Date,End_Date).data

                    new_data.index=pd.to_datetime(new_data.loc[:,"time"])

                    new_data=new_data.iloc[:,1:]

                    Updated_Data=pd.concat([df,new_data],axis=0)

                    Updated_Data.to_csv(List[i])

                    print(str(i/Doc_Number))

                else:

                    print("数据更新至最新")

            except:

                print(List[i]+"出现错误")

        print("更新结束")

        return

    def Updating_All_Market_Vol_Data(self,End_Date, Data_Path_List):

        def Updating_Market_Vol_Price_Data(End_Date, Data_Path):
            os.chdir(Data_Path)  # 切换到数据路径

            List = os.listdir()  # 获取路径下所有文件
            Doc_Number = len(List)

            print("更新开始")

            def update_file(file_name):
                try:
                    file_path = os.path.join(Data_Path, file_name)
                    df = pd.read_csv(file_path, index_col=[0])
                    df.index = pd.to_datetime(df.index)

                    if df.index[-1].strftime('%Y-%m-%d') != End_Date:
                        New_Begin_Date = (df.index[-1] + timedelta(days=1)).strftime("%Y-%m-%d")
                        code = file_name.replace('.csv', "")

                        # 获取新数据
                        new_data = THS_HQ(code, 'open,high,low,close,volume,amount,turnoverRatio,totalCapital,floatCapital', '', New_Begin_Date, End_Date).data
                        new_data.index = pd.to_datetime(new_data["time"])  # 将时间列设为索引
                        new_data = new_data.drop(columns=["time", "thscode"])  # 删除无关列

                        Updated_Data = pd.concat([df, new_data], axis=0)  # 合并新旧数据
                        Updated_Data.loc[:,"thscode"]=Updated_Data.iloc[1,0]
                        Updated_Data.to_csv(file_path)  # 保存更新后的数据
                        return f"{file_name} 更新成功"
                    else:
                        return f"{file_name} 数据更新至最新"
                except Exception as e:
                    return f"{file_name} 出现错误: {e}"

            with concurrent.futures.ThreadPoolExecutor() as executor:
                results = list(tqdm(executor.map(update_file, List), total=Doc_Number, desc="更新进度"))

            for result in results:
                print(result)

            print("更新结束")

        for data_path in Data_Path_List:
            try:
                Updating_Market_Vol_Price_Data(End_Date, data_path)
            except Exception as e:
                print(f"{data_path} 出现错误: {e}")

        return

    def Download_Index_Valuation_Data(self,Code_List,Begin_Date,End_Date,Export_Path):

        def Downloading_Index_Valuation_Data(Code,Begin_Date,End_Date):

        #日期循环
            Begin_Date=Begin_Date

            Begin_Date=pd.to_datetime(Begin_Date)

            End_Date=End_Date

            End_Date=pd.to_datetime(End_Date)

            Gap=(End_Date-Begin_Date)/250

            Period=Gap.days

            Timeseries=pd.date_range(Begin_Date,End_Date,periods=Period)

            Timeseries_List=Timeseries.strftime("%Y-%m-%d").tolist()

            Begin_Date_List=Timeseries_List[:-1]

            End_Date_List=Timeseries_List[1:]

            Data=[]

        #日期循环下载数据

            for i,j in zip(Begin_Date_List,End_Date_List):
                
                DF=THS_DS(Code,'ths_pe_ttm_index;ths_pb_index;ths_dividend_rate_index','100,100;100,100;','block:history',i,j).data

                Data.append(DF)
            
            Data=pd.concat(Data,axis=0)

            Data=Data.set_index('time',drop=True)

            return Data

        os.chdir(Export_Path)

        Error_Code=[]

        print("更新开始")
        
        for i in Code_List:

            try:
                
                Data=Downloading_Index_Valuation_Data(i,Begin_Date,End_Date)

                Data.to_csv(i+".csv")

                print(i+"输出完成")

            except:

                print(i+"出现错误")

                Error_Code.append(i)
        
        return

    def Updating_Index_Valuation_Data(self,End_Date, Data_Path):
        def process_file(filename):
            try:
                df = pd.read_csv(os.path.join(Data_Path, filename), index_col=[0])
                df.index = pd.to_datetime(df.index)

                if df.index[-1].strftime('%Y-%m-%d') != End_Date:
                    New_Begin_Date = df.index[-1] + Day()
                    New_Begin_Date = New_Begin_Date.strftime("%Y-%m-%d")

                    code = filename.replace('.csv', "")
                    new_data = THS_DS(code, 'ths_pe_ttm_index;ths_pb_index;ths_dividend_rate_index', '100,100;100,100;', 'block:history', New_Begin_Date, End_Date).data
                    new_data.index = pd.to_datetime(new_data.loc[:, "time"])
                    new_data = new_data.iloc[:, 2:]

                    Updated_Data = pd.concat([df, new_data], axis=0)
                    Updated_Data.to_csv(os.path.join(Data_Path, filename))

                    print(f"{filename} processed successfully.")
                else:
                    print(f"{filename} is already up to date.")
            except Exception as e:
                print(f"Error processing {filename}: {e}")

        os.chdir(Data_Path)
        List = os.listdir()
        Doc_Number = len(List)

        print("更新开始")

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(process_file, filename) for filename in List]
            for i, future in enumerate(futures):
                try:
                    future.result()  # Wait for thread to complete and handle exceptions
                except Exception as e:
                    print(f"Thread {i} generated an exception: {e}")

        print("更新结束")

        return


        Data_Path=Data_Path

        os.chdir(Data_Path)

        List=os.listdir()

        Doc_Number=len(List)

        print("更新开始")

        for i in range(0,Doc_Number):

            try:

                df=pd.read_csv(Data_Path+"\\"+List[i],index_col=[0])

                df.index=pd.to_datetime(df.index)

                if df.index[-1].strftime('%Y-%m-%d')!=End_Date:

                    New_Begin_Date=df.index[-1]+Day()

                    New_Begin_Date=New_Begin_Date.strftime("%Y-%m-%d")

                    code = List[i].replace('.csv', "")

                    new_data= THS_DS(code,'ths_pe_ttm_index;ths_pb_index;ths_dividend_rate_index','100,100;100,100;','block:history',New_Begin_Date,End_Date).data

                    new_data.index=pd.to_datetime(new_data.loc[:,"time"])

                    new_data=new_data.iloc[:,2:]

                    Updated_Data=pd.concat([df,new_data],axis=0)

                    Updated_Data.to_csv(List[i])

                    print(str(i/Doc_Number))

                else:

                    print("数据更新至最新")

            except:

                print(List[i]+"出现错误")

        print("更新结束")

        return

    def Download_Index_Free_Turn_Data(self,Code_List,Begin_Date,End_Date,Export_Path):

        def Downloading_Free_Trun_Data(Code,Begin_Date,End_Date):

        #日期循环
            Begin_Date=Begin_Date

            Begin_Date=pd.to_datetime(Begin_Date)

            End_Date=End_Date

            End_Date=pd.to_datetime(End_Date)

            Gap=(End_Date-Begin_Date)/250

            Period=Gap.days

            Timeseries=pd.date_range(Begin_Date,End_Date,periods=Period)

            Timeseries_List=Timeseries.strftime("%Y-%m-%d").tolist()

            Begin_Date_List=Timeseries_List[:-1]

            End_Date_List=Timeseries_List[1:]

            Data=[]

        #日期循环下载数据

            for i,j in zip(Begin_Date_List,End_Date_List):
                
                DF=THS_DS(Code,'ths_free_turnover_ratio_index','','block:history',i,j).data

                Data.append(DF)
            
            Data=pd.concat(Data,axis=0)

            return Data

        os.chdir(Export_Path)

        Error_Code=[]

        print("更新开始")
        
        for i in Code_List:

            try:
                
                Data=Downloading_Free_Trun_Data(i,Begin_Date,End_Date)

                Data.index=Data.loc[:,"time"]

                Data=Data.iloc[:,1:]

                Data.to_csv(i+".csv")

                print(i+"输出完成")

            except:

                print(i+"出现错误")

                Error_Code.append(i)
        
        return

    def Download_Index_Forcast_Neteprofit(self,Code_List,Begin_Date,End_Date,Export_Path):

        def Downloading_Forcast_Neteprofit(Code,Begin_Date,End_Date):

        #日期循环
            Begin_Date=Begin_Date

            Begin_Date=pd.to_datetime(Begin_Date)

            End_Date=End_Date

            End_Date=pd.to_datetime(End_Date)

            Gap=(End_Date-Begin_Date)/250

            Period=Gap.days

            Timeseries=pd.date_range(Begin_Date,End_Date,periods=Period)

            Timeseries_List=Timeseries.strftime("%Y-%m-%d").tolist()

            Begin_Date_List=Timeseries_List[:-1]

            End_Date_List=Timeseries_List[1:]

            Data=[]

        #日期循环下载数据

            for i,j in zip(Begin_Date_List,End_Date_List):
                
                Year=i[:4]

                FY1=int(Year)+1

                FY1=str(FY1)

                FY2=int(Year)+2

                FY2=str(FY2)

                #一致预期数据下载
                DF=THS_DS(Code,'ths_fore_np_index',Year,'block:history',i,j).data
                
                DF_FY1=THS_DS(Code,'ths_fore_np_index',FY1,'block:history',i,j).data

                DF_FY1=DF_FY1[["ths_fore_np_index"]]

                DF_FY2=THS_DS(Code,'ths_fore_np_index',FY2,'block:history',i,j).data

                DF_FY2=DF_FY2[["ths_fore_np_index"]]
                
                #财报数据下载

                DF_FY0=THS_DS(Code,'ths_np_index','','block:history',i,j).data

                DF_FY0=DF_FY0[["ths_np_index"]]

                Total=pd.concat([DF,DF_FY1,DF_FY2,DF_FY0],axis=1)

                Total.columns=["time","thscode","fore_np_FY0","fore_np_FY1","fore_np_FY2","np_index"]

                Data.append(Total)
            
            Data=pd.concat(Data,axis=0)

            return Data

        #code list循环下载
        os.chdir(Export_Path)

        Error_Code=[]

        print("更新开始")
        
        for i in Code_List:

            try:
                
                Data=Downloading_Forcast_Neteprofit(i,Begin_Date,End_Date)

                Data.index=Data.loc[:,"time"]

                Data=Data.iloc[:,1:]

                Data.to_csv(i+".csv")

                print(i+"输出完成")

        #将出现错误的Code编成list
            except:

                print(i+"出现错误")

                Error_Code.append(i)
        
        return Error_Code

    def Updating_Index_Free_Turn_Data(self,end_date, data_path):
        """
        更新指定路径下的所有CSV文件中的数据，如果数据不是最新的，则从最新日期开始更新至指定的结束日期。
        
        :param end_date: 数据更新的截止日期，格式为'YYYY-MM-DD'
        :param data_path: 存放CSV文件的路径
        """
        def update_file(file_name):
            try:
                # 读取CSV文件
                df = pd.read_csv(os.path.join(data_path, file_name), index_col=[0])
                df.index = pd.to_datetime(df.index)

                # 检查数据是否已经是最新的
                if df.index[-1].strftime('%Y-%m-%d') != end_date:
                    # 计算新的开始日期
                    new_begin_date = (df.index[-1] + Day(1)).strftime("%Y-%m-%d")
                    code = file_name.replace('.csv', "")

                    # 获取新的数据
                    new_data = THS_DS(code, 'ths_free_turnover_ratio_index', '', 'block:history', new_begin_date, end_date).data
                    new_data.index = pd.to_datetime(new_data.loc[:, "time"])
                    new_data = new_data.iloc[:, 2:]

                    # 合并旧数据和新数据
                    updated_data = pd.concat([df, new_data], axis=0)

                    # 保存更新后的数据
                    updated_data.to_csv(os.path.join(data_path, file_name))

                    print(f"{file_name} 更新完成")
                else:
                    print(f"{file_name} 数据已是最新")

            except Exception as e:
                print(f"{file_name} 出现错误: {e}")

        # 获取路径下所有文件的列表
        file_list = os.listdir(data_path)

        print("更新开始")

        # 使用线程池来并行处理文件
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(update_file, file_name) for file_name in file_list]

            for future in futures:
                future.result()  # 等待所有线程完成

        print("更新结束")

        return

    def Download_Index_Forcast_Neteprofit(self,Code_List,Begin_Date,End_Date,Export_Path):

        def Downloading_Forcast_Neteprofit(Code,Begin_Date,End_Date):

        #日期循环
            Begin_Date=Begin_Date

            Begin_Date=pd.to_datetime(Begin_Date)

            End_Date=End_Date

            End_Date=pd.to_datetime(End_Date)

            Gap=(End_Date-Begin_Date)/250

            Period=Gap.days

            Timeseries=pd.date_range(Begin_Date,End_Date,periods=Period)

            Timeseries_List=Timeseries.strftime("%Y-%m-%d").tolist()

            Begin_Date_List=Timeseries_List[:-1]

            End_Date_List=Timeseries_List[1:]

            Data=[]

        #日期循环下载数据

            for i,j in zip(Begin_Date_List,End_Date_List):
                
                Year=i[:4]

                FY1=int(Year)+1

                FY1=str(FY1)

                FY2=int(Year)+2

                FY2=str(FY2)

                #一致预期数据下载
                DF=THS_DS(Code,'ths_fore_np_index',Year,'block:history',i,j).data
                
                DF_FY1=THS_DS(Code,'ths_fore_np_index',FY1,'block:history',i,j).data

                DF_FY1=DF_FY1[["ths_fore_np_index"]]

                DF_FY2=THS_DS(Code,'ths_fore_np_index',FY2,'block:history',i,j).data

                DF_FY2=DF_FY2[["ths_fore_np_index"]]
                
                #财报数据下载

                DF_FY0=THS_DS(Code,'ths_np_index','','block:history',i,j).data

                DF_FY0=DF_FY0[["ths_np_index"]]

                Total=pd.concat([DF,DF_FY1,DF_FY2,DF_FY0],axis=1)

                Total.columns=["time","thscode","fore_np_FY0","fore_np_FY1","fore_np_FY2","np_index"]

                Data.append(Total)
            
            Data=pd.concat(Data,axis=0)

            return Data

        #code list循环下载
        os.chdir(Export_Path)

        Error_Code=[]

        print("更新开始")
        
        for i in Code_List:

            try:
                
                Data=Downloading_Forcast_Neteprofit(i,Begin_Date,End_Date)

                Data.index=Data.loc[:,"time"]

                Data=Data.iloc[:,1:]

                Data.to_csv(i+".csv")

                print(i+"输出完成")

        #将出现错误的Code编成list
            except:

                print(i+"出现错误")

                Error_Code.append(i)
        
        return Error_Code
    
    def Download_Index_Forcast_2yr_Data(self,Code_List,Begin_Date,End_Date,Export_Path):

            def Downloading_Forcast_2yr_Data(Code,Begin_Date,End_Date):

            #日期循环
                Begin_Date=Begin_Date

                Begin_Date=pd.to_datetime(Begin_Date)

                End_Date=End_Date

                End_Date=pd.to_datetime(End_Date)

                Gap=(End_Date-Begin_Date)/250

                Period=Gap.days

                Timeseries=pd.date_range(Begin_Date,End_Date,periods=Period)

                Timeseries_List=Timeseries.strftime("%Y-%m-%d").tolist()

                Begin_Date_List=Timeseries_List[:-1]

                End_Date_List=Timeseries_List[1:]

                Data=[]

            #日期循环下载数据

                for i,j in zip(Begin_Date_List,End_Date_List):
                    
                    DF=THS_DS(Code,'ths_fore_np_compound_growth_2y_index','','block:history',i,j).data

                    Data.append(DF)
                
                Data=pd.concat(Data,axis=0)

                return Data

            os.chdir(Export_Path)

            Error_Code=[]

            print("更新开始")
            
            for i in Code_List:

                try:
                    
                    Data=Downloading_Forcast_2yr_Data(i,Begin_Date,End_Date)

                    Data.index=Data.loc[:,"time"]

                    Data=Data.iloc[:,1:]

                    Data.to_csv(i+".csv")

                    print(i+"输出完成")

                except:

                    print(i+"出现错误")

                    Error_Code.append(i)
            
            return
    
    def Updating_Forcast_2yr_Data(self,End_Date,Data_Path):

        Data_Path=Data_Path

        os.chdir(Data_Path)

        List=os.listdir()

        Doc_Number=len(List)

        print("更新开始")

        for i in range(0,Doc_Number):

            try:

                df=pd.read_csv(Data_Path+"\\"+List[i],index_col=[0])

                df.index=pd.to_datetime(df.index)

                if df.index[-1].strftime('%Y-%m-%d')!=End_Date:

                    New_Begin_Date=df.index[-1]+Day()

                    New_Begin_Date=New_Begin_Date.strftime("%Y-%m-%d")

                    code = List[i].replace('.csv', "")

                    new_data=THS_DS(code,'ths_fore_np_compound_growth_2y_index','','block:history',New_Begin_Date,End_Date).data

                    new_data.index=pd.to_datetime(new_data.loc[:,"time"])

                    new_data=new_data.iloc[:,2:]

                    Updated_Data=pd.concat([df,new_data],axis=0)

                    Updated_Data.to_csv(List[i])

                    print(str(i/Doc_Number))

                else:

                    print("数据更新至最新")

            except:

                print(List[i]+"出现错误")

        print("更新结束")

        return

    def Download_EDB_Data(self,Code_List,Begin_Date,End_Date,Export_Path):

        for i in range(0,len(Code_List)):
        
            try:
                Data=THS_EDB(Code_List[i],'',Begin_Date,End_Date).data

                Data.index=Data.loc[:,"time"]

                Data.index=pd.to_datetime(Data.index)

                Data=Data.iloc[:,2:5]

                os.chdir(Export_Path)

                Data.to_csv(Code_List[i]+".csv")

                print('完成度: '+str(i/len(Code_List)))

            except:

                print(Code_List[i]+"出现错误")

        return

    def Update_EDB_Data(self,Data_Path,End_Date):

        def Download_EDB_Data(Code_List,Begin_Date,End_Date,Export_Path):

                for i in range(0,len(Code_List)):
                
                    try:
                        Data=THS_EDB(Code_List[i],'',Begin_Date,End_Date).data

                        Data.index=Data.loc[:,"time"]

                        Data.index=pd.to_datetime(Data.index)

                        Data=Data.iloc[:,2:5]

                        os.chdir(Export_Path)

                        Data.to_csv(Code_List[i]+".csv")

                        print('完成度: '+str(i/len(Code_List)))

                    except:

                        print(Code_List[i]+"出现错误")

                return

        List=os.listdir(Data_Path)

        Code_List=[]

        for i in List:

            Code=i.replace(".csv","")

            Code_List.append(Code)

        Download_EDB_Data(Code_List,"2003-01-01",End_Date,Data_Path)

        return

    def Downloading_ETF_Option_PCR_Data(self,Code_List,Begin_Date,End_Date,Export_Path):
    
        Export_Path=Export_Path

        os.chdir(Export_Path)

        print("下载开始")

        for i in range(0,len(Code_List)):

            try:

                Data=THS_DS(Code_List[i],'ths_option_total_volume_pcr_option;ths_option_total_oi_pcr_option;ths_option_total_amount_pcr_option',';;','',Begin_Date,End_Date).data

                Data.index=Data.loc[:,"time"]

                Data=Data.iloc[:,1:]

                Data.to_csv(Code_List[i]+".csv")

                print(str(i/len(Code_List)))
            
            except:

                print(Code_List[i]+"出现错误")

        print("下载结束")

    def Updating_ETF_Option_PCR(self,End_Date,Data_Path):

        Data_Path=Data_Path

        os.chdir(Data_Path)

        List=os.listdir()

        Doc_Number=len(List)

        print("更新开始")

        for i in range(0,Doc_Number):

            try:

                df=pd.read_csv(Data_Path+"\\"+List[i],index_col=[0])

                df.index=pd.to_datetime(df.index)

                if df.index[-1].strftime('%Y-%m-%d')!=End_Date:

                    New_Begin_Date=df.index[-1]+Day()

                    New_Begin_Date=New_Begin_Date.strftime("%Y-%m-%d")

                    code = List[i].replace('.csv', "")

                    new_data= THS_DS(code,'ths_option_total_volume_pcr_option;ths_option_total_oi_pcr_option;ths_option_total_amount_pcr_option',';;','',New_Begin_Date,End_Date).data

                    new_data.index=pd.to_datetime(new_data.loc[:,"time"])

                    new_data=new_data.iloc[:,1:]

                    Updated_Data=pd.concat([df,new_data],axis=0)

                    Updated_Data.to_csv(List[i])

                else:

                    print("数据更新至最新")

            except:

                print(List[i]+"出现错误")

        print("更新结束")

        return

    def Download_Index_Forcast_Neteprofit(self,Code_List,Begin_Date,End_Date,Export_Path):

        def Downloading_Forcast_Neteprofit(Code,Begin_Date,End_Date):

            def generate_date_intervals(start_date, end_date):
                # 将输入的日期字符串转换为datetime对象
                start_date = datetime.strptime(start_date, "%Y-%m-%d")
                end_date = datetime.strptime(end_date, "%Y-%m-%d")

                # 初始化开始日期和结束日期的列表
                begin_date_list = []
                end_date_list = []

                temp_date = start_date
                while temp_date <= end_date:
                    # 将当前日期添加到开始日期列表
                    begin_date_list.append(temp_date.strftime("%Y-%m-%d"))

                    if temp_date.year == end_date.year:
                        # 如果是结束日期的年份，那么结束日期就是输入的结束日期
                        temp_end_date = end_date
                    else:
                        # 否则，结束日期是当前年份的12月31日
                        temp_end_date = datetime(temp_date.year, 12, 31)

                    # 将结束日期添加到结束日期列表
                    end_date_list.append(temp_end_date.strftime("%Y-%m-%d"))

                    # 将临时日期设置为下一年的1月1日
                    temp_date = datetime(temp_date.year + 1, 1, 1)

                return begin_date_list, end_date_list

            Begin_Date_List,End_Date_List=generate_date_intervals(Begin_Date,End_Date)

            Data=[]

        #日期循环下载数据

            for i,j in zip(Begin_Date_List,End_Date_List):
                
                Year=i[:4]

                FY1=int(Year)+1

                FY1=str(FY1)

                date=FY1+';'+FY1

                #一致预期数据下载
                DF=THS_DS(Code,'ths_fore_eps_index;ths_fore_roe_mean_index',date,'block:history',i,j).data

                Data.append(DF)
            
            result=pd.concat(Data,axis=0)

            return result

        #code list循环下载
        os.chdir(Export_Path)

        Error_Code=[]

        print("更新开始")
        
        # 使用tqdm包装Code_List
        for i in Code_List: 

            try:
                
                Data=Downloading_Forcast_Neteprofit(i,Begin_Date,End_Date)

                Data.index=Data.loc[:,"time"]

                Data=Data.iloc[:,1:]

                Data.to_csv(i+".csv")

                print(i+"输出完成")

        #将出现错误的Code编成list
            except:

                print(i+"出现错误")

                Error_Code.append(i)
        
        return Error_Code

    def Update_Index_Forcast_Data(self,End_Date,Data_Path):

        os.chdir(Data_Path)

        List=os.listdir()

        Doc_Number=len(List)

        print("更新开始")

        for i in range(0,Doc_Number):

            try:

                df=pd.read_csv(Data_Path+"\\"+List[i],index_col=[0])

                df.index=pd.to_datetime(df.index)

                if df.index[-1].strftime('%Y-%m-%d')!=End_Date:

                    New_Begin_Date=df.index[-1]+Day()

                    New_Begin_Date=New_Begin_Date.strftime("%Y-%m-%d")

                    code = List[i].replace('.csv', "")

                    Year=New_Begin_Date[:4]

                    FY1=int(Year)+1

                    FY1=str(FY1)

                    date=FY1+';'+FY1

                    new_data=THS_DS(code,'ths_fore_eps_index;ths_fore_roe_mean_index',date,'block:history',New_Begin_Date,End_Date).data

                    new_data.index=pd.to_datetime(new_data.loc[:,"time"])

                    new_data=new_data.iloc[:,2:]

                    Updated_Data=pd.concat([df,new_data],axis=0)

                    Updated_Data.to_csv(List[i])

                else:

                    print("数据更新至最新")

            except:

                print(List[i]+"出现错误")

        print("更新结束")

        return

    def Downloading_Mutal_Fund_Netvalue(self,Code_List,Begin_Date,End_Date,Export_Path):

        Export_Path=Export_Path

        os.chdir(Export_Path)

        print("下载开始")

        for i in range(0,len(Code_List)):

            try:
                Data=THS_DS(Code_List[i],'ths_accum_unit_nv_fund','','',Begin_Date,End_Date).data

                Data.index=Data.loc[:,"time"]

                Data=Data[['ths_accum_unit_nv_fund']]

                Data.columns=[Code_List[i]]

                Data.to_csv(Code_List[i]+".csv")

                print(str(i/len(Code_List)))
            
            except:

                print(Code_List[i]+"出现错误")

        print("下载结束")

    def Downloading_F_Score_Data(self,Code_List,Begin_Date,End_Date,Export_Path):

        Export_Path = Export_Path

        os.chdir(Export_Path)

        print("下载开始")

        for i in range(0, len(Code_List)):

            try:
                Data=THS_DS(Code_List[i],'ths_np_atoopc_pit_stock;ths_total_assets_pit_stock;ths_ncf_from_oa_pit_stock;ths_total_liab_pit_stock;ths_total_current_assets_pit_stock;ths_total_current_liab_pit_stock;ths_total_shares_stock;ths_operating_total_revenue_stock;ths_gross_selling_rate_stock','0@103,1;0@103,1;0@103,1;0@103,1;0@103,1;0@103,1;;100;','',Begin_Date,End_Date).data   

                Data.index = Data.loc[:, "time"]

                Data = Data.iloc[:, 1:]

                Data.to_csv(Code_List[i] + ".csv")

                print(str(i / len(Code_List)))

            except:

                print(Code_List[i] + "出现错误")

        print("下载结束")

    def Update_F_Score_Data(self,End_Date,Data_Path):

        os.chdir(Data_Path)

        List=os.listdir()

        Doc_Number=len(List)

        print("更新开始")

        for i in range(0,Doc_Number):

            try:

                df=pd.read_csv(Data_Path+"\\"+List[i],index_col=[0])

                df.index=pd.to_datetime(df.index)

                if df.index[-1].strftime('%Y-%m-%d')!=End_Date:

                    New_Begin_Date=df.index[-1]+Day()

                    New_Begin_Date=New_Begin_Date.strftime("%Y-%m-%d")

                    code = List[i].replace('.csv', "")
                            
                    new_data=THS_DS(code,'ths_np_atoopc_pit_stock;ths_total_assets_pit_stock;ths_ncf_from_oa_pit_stock;ths_total_liab_pit_stock;ths_total_current_assets_pit_stock;ths_total_current_liab_pit_stock;ths_total_shares_stock;ths_operating_total_revenue_stock;ths_gross_selling_rate_stock','0@103,1;0@103,1;0@103,1;0@103,1;0@103,1;0@103,1;;100;','',New_Begin_Date,End_Date).data  
                    
                    new_data.index=pd.to_datetime(new_data.loc[:,"time"])

                    new_data=new_data.iloc[:,1:]

                    Updated_Data=pd.concat([df,new_data],axis=0)

                    Updated_Data.to_csv(List[i])

                    print(str(i/Doc_Number))

                else:

                    print("数据更新至最新")

            except:

                print(List[i]+"出现错误")

        print("个股财务数据更新结束")
        
    def Downloading_Cov_Bond_Vol_Price_Data(self,Code_List,Begin_Date,End_Date,Export_Path):

        Export_Path=Export_Path

        os.chdir(Export_Path)

        print("下载开始")

        for i in range(0,len(Code_List)):

            try:

                Data=THS_DS(Code_List[i],'ths_open_price_bond;ths_high_price_bond;ths_low_bond;ths_close_daily_settle_bond;ths_vol_yz_bond;ths_trans_amt_bond','103;103;103;103;100;','',Begin_Date,End_Date).data
                Data.index=Data.loc[:,"time"]

                Data=Data.iloc[:,1:]

                Data.to_csv(Code_List[i]+".csv")

                print(str(i/len(Code_List)))
            
            except:

                print(Code_List[i]+"出现错误")

        print("下载结束")

    def Update_Cov_Bond_Vol_Price_Data(self,End_Date,Data_Path):

        os.chdir(Data_Path)

        List=os.listdir()

        Doc_Number=len(List)

        print("更新开始")

        for i in range(0,Doc_Number):

            try:

                df=pd.read_csv(Data_Path+"\\"+List[i],index_col=[0])

                df.index=pd.to_datetime(df.index)

                if df.index[-1].strftime('%Y-%m-%d')!=End_Date:

                    New_Begin_Date=df.index[-1]+Day()

                    New_Begin_Date=New_Begin_Date.strftime("%Y-%m-%d")

                    code = List[i].replace('.csv', "")
                            
                    new_data=THS_DS(code,'ths_open_price_bond;ths_high_price_bond;ths_low_bond;ths_close_daily_settle_bond;ths_vol_yz_bond;ths_trans_amt_bond','103;103;103;103;100;','',New_Begin_Date,End_Date).data
                    new_data.index=pd.to_datetime(new_data.loc[:,"time"])

                    new_data=new_data.iloc[:,1:]

                    Updated_Data=pd.concat([df,new_data],axis=0)

                    Updated_Data.to_csv(List[i])

                    print(str(i/Doc_Number))

                else:

                    print("数据更新至最新")

            except:

                print(List[i]+"出现错误")

        print("可转债数据更新结束")

        return

    def Downloading_A50_Futures_Data(self,Code_List,Begin_Date,End_Date,Export_Path):

        Export_Path=Export_Path

        os.chdir(Export_Path)

        print("下载开始")

        for i in range(0,len(Code_List)):

            try:

                Data=THS_DS(Code_List[i],'ths_open_price_future;ths_high_price_future;ths_low_future;ths_close_price_future;ths_basis_trading_rate_future',';;;;','',Begin_Date,End_Date).data

                Data=Data.set_index('time',drop=True)

                Data.to_csv(Code_List[i]+".csv")

                print(str(i/len(Code_List)))
            
            except:

                print(Code_List[i]+"出现错误")

        print("下载结束")

    def Updating_A50_Futures_Data(self,End_Date,Data_Path):

        Data_Path=Data_Path

        os.chdir(Data_Path)

        List=os.listdir()

        Doc_Number=len(List)

        print("更新开始")

        for i in range(0,Doc_Number):

            try:

                df=pd.read_csv(Data_Path+"\\"+List[i],index_col=[0])

                df.index=pd.to_datetime(df.index)

                if df.index[-1].strftime('%Y-%m-%d')!=End_Date:

                    New_Begin_Date=df.index[-1]+Day()

                    New_Begin_Date=New_Begin_Date.strftime("%Y-%m-%d")

                    code = List[i].replace('.csv', "")

                    new_data=  THS_DS(code,'ths_open_price_future;ths_high_price_future;ths_low_future;ths_close_price_future;ths_basis_trading_rate_future',';;;;','',New_Begin_Date,End_Date).data
                        
                    new_data.index=pd.to_datetime(new_data.loc[:,"time"])

                    new_data=new_data.iloc[:,1:]

                    Updated_Data=pd.concat([df,new_data],axis=0)

                    Updated_Data.to_csv(List[i])

                else:

                    print("数据更新至最新")

            except:

                print(List[i]+"出现错误")

        print("更新结束")

        return

    def Downloading_newHL_Data(self,Begin_Date, End_Date, Export_Path, Code='001005010'):
        
        Begin_Date = pd.to_datetime(Begin_Date)
        End_Date = pd.to_datetime(End_Date)
        
        Gap = (End_Date - Begin_Date) / 250
        Period = Gap.days
        
        Timeseries = pd.date_range(Begin_Date, End_Date, periods=Period)
        Timeseries_List = Timeseries.strftime("%Y-%m-%d").tolist()
        
        Begin_Date_List = Timeseries_List[:-1]
        End_Date_List = Timeseries_List[1:]
        
        Data = []
        
        # 使用 tqdm 来显示进度条
        for i, j in tqdm(zip(Begin_Date_List, End_Date_List), total=len(Begin_Date_List), desc="Downloading Data"):
            
            DF = THS_DS(Code, 'ths_new_high_num_block;ths_new_low_num_block', '1,250,100;1,250,100', 'block:latest', i, j).data
            
            Data.append(DF)
        
        Data = pd.concat(Data, axis=0)
        Data = Data.set_index('time', drop=True)
        
        os.chdir(Export_Path)
        Data.to_csv(Code+'.csv')
        
        return Data
     
    def Updating_newHL_Data(self,End_Date, Data_Path):

        Data_Path=Data_Path

        os.chdir(Data_Path)

        List=os.listdir()

        Doc_Number=len(List)

        print("更新开始")

        for i in range(0,Doc_Number):

            try:

                df=pd.read_csv(Data_Path+"\\"+List[i],index_col=[0])

                df.index=pd.to_datetime(df.index)

                if df.index[-1].strftime('%Y-%m-%d')!=End_Date:

                    New_Begin_Date=df.index[-1]+Day()

                    New_Begin_Date=New_Begin_Date.strftime("%Y-%m-%d")

                    code = List[i].replace('.csv', "")

                    new_data=THS_DS(code, 'ths_new_high_num_block;ths_new_low_num_block', '1,250,100;1,250,100', 'block:latest', New_Begin_Date, End_Date).data

                    new_data.index=pd.to_datetime(new_data.loc[:,"time"])

                    new_data=new_data.iloc[:,2:]

                    Updated_Data=pd.concat([df,new_data],axis=0) 

                    Updated_Data.to_csv(List[i])

                    print(str(i/Doc_Number))

                else:

                    print("数据更新至最新")

            except:

                print(List[i]+"出现错误")

        print("更新结束")

        return

    def Downloading_High_Freq_Vol_Price_Data(self,Code_List,Begin_Date,End_Date,Export_Path,Interval=60):

        Export_Path=Export_Path

        os.chdir(Export_Path)

        print("下载开始")

        for i in range(0,len(Code_List)):

            try:

                Data=THS_HF(Code_List[i],'open;high;low;close;volume;amount','Fill:Original,Interval:'+str(Interval),Begin_Date,End_Date).data
                
                Data.index=Data.loc[:,"time"]

                Data=Data.iloc[:,1:]

                Data.to_csv(Code_List[i]+".csv")

                print(str(i/len(Code_List)))
            
            except:

                print(Code_List[i]+"出现错误")

        print("下载结束")

    def get_next_trading_time_1h(self,last_time):
        """
        根据最后一条数据的时间，计算下一个交易时间。
        :param last_time: 最后一条数据的时间 (pd.Timestamp)
        :return: 下一个交易时间 (str)
        """
        last_date = last_time.date()
        current_time_str = last_time.strftime("%H:%M")
        TRADING_HOURS=self.A_Share_Trading_Hours
        # 查找下一个交易时间
        for trading_time in TRADING_HOURS:
            if current_time_str < trading_time:
                next_trading_time = datetime.combine(last_date, datetime.strptime(trading_time, "%H:%M").time())
                return next_trading_time.strftime("%Y-%m-%d %H:%M")
        
        # 如果当天所有交易时间都过了，返回下一个交易日的第一个交易时间 "10:30"
        next_trading_day = last_date + timedelta(days=1)
        return datetime.combine(next_trading_day, datetime.strptime("10:30", "%H:%M").time()).strftime("%Y-%m-%d %H:%M")

    def get_next_trading_time_15min(self,last_time):
        """
        根据最后一条数据的时间，计算下一个交易时间。
        :param last_time: 最后一条数据的时间 (pd.Timestamp)
        :return: 下一个交易时间 (str)
        """
        last_date = last_time.date()
        current_time_str = last_time.strftime("%H:%M")
        TRADING_HOURS=self.A_Share_Trading_15mins
        # 查找下一个交易时间
        for trading_time in TRADING_HOURS:
            if current_time_str < trading_time:
                next_trading_time = datetime.combine(last_date, datetime.strptime(trading_time, "%H:%M").time())
                return next_trading_time.strftime("%Y-%m-%d %H:%M")
        
        # 如果当天所有交易时间都过了，返回下一个交易日的第一个交易时间 "09:45"
        next_trading_day = last_date + timedelta(days=1)
        return datetime.combine(next_trading_day, datetime.strptime("09:45", "%H:%M").time()).strftime("%Y-%m-%d %H:%M")
        
    def Update_High_Freq_Vol_Price_Data(self, End_Date, Export_Path, Interval=60):
        """
        更新高频数据 (小时级别)，确保数据不重复下载，并且增量更新。
        
        :param End_Date: 数据更新的结束时间（字符串格式 'YYYY-MM-DD HH:MM'）
        :param Export_Path: 导出文件夹的路径
        :param Interval: 数据的时间间隔，默认是 60 分钟（1 小时）
        """
        
        os.chdir(Export_Path)  # 切换到数据存放的目录
        
        # 获取所有以 .csv 结尾的文件名，并去掉 .csv 后缀，生成股票代码列表
        Code_List = [f.replace('.csv', '') for f in os.listdir(Export_Path) if f.endswith('.csv')]
        
        print("更新开始")
        
        if Interval == 60:
            # 使用 tqdm 显示进度条
            for code in tqdm(Code_List):
                try:
                    file_path = os.path.join(Export_Path, f"{code}.csv")
                    
                    # 读取现有的 CSV 文件
                    if os.path.exists(file_path):
                        df_existing = pd.read_csv(file_path, parse_dates=['time'])
                        df_existing.set_index('time', inplace=True)  # 确保 'time' 列作为索引
                        
                        # 获取最后一条数据的时间，并确保是 pd.Timestamp 类型
                        last_time = pd.to_datetime(df_existing.index.max())
                        
                        # 获取下一个交易时间
                        start_time = self.get_next_trading_time_1h(last_time)
                        
                        # 如果最后一条时间已经超过了指定的结束时间，则跳过更新
                        if pd.to_datetime(start_time) > pd.to_datetime(End_Date):
                            print(f"{code} 数据已经是最新的，跳过更新。")
                            continue
                    else:
                        # 如果文件不存在，从头开始下载
                        start_time = "2010-01-01 00:00"
                        df_existing = pd.DataFrame()  # 空的 DataFrame 用于拼接

                    # 调用 API 下载新数据，从 start_time 到 End_Date
                    new_data = THS_HF(code, 'open;high;low;close;volume;amount', 
                                    f'Fill:Original,Interval:{Interval}', start_time, End_Date).data

                    # 确保 'time' 列存在，将其设置为索引
                    if 'time' in new_data.columns:
                        new_data.set_index('time', inplace=True)

                    # 如果没有新数据，跳过处理
                    if new_data.empty:
                        print(f"{code} 无新数据需要更新。")
                        continue

                    # 将新数据与旧数据拼接，并去重
                    if not df_existing.empty:
                        df_combined = pd.concat([df_existing, new_data])
                    else:
                        df_combined = new_data
                    
                    # 将索引转换为日期时间格式
                    df_combined.index = pd.to_datetime(df_combined.index, errors='coerce')
                    
                    # 检查是否有 NaT 索引，删除无效的时间索引
                    df_combined = df_combined[df_combined.index.notna()]

                    # 保存更新后的数据，确保 'time' 列作为索引
                    df_combined.to_csv(file_path, index=True)  # index=True 保留时间索引
                    print(f"{code} 数据更新成功。")

                except Exception as e:
                    print(f"{code} 更新时出现错误: {e}")
            
            print("数据更新结束")
        
        if Interval == 15:
            # 使用 tqdm 显示进度条
            for code in tqdm(Code_List):
                try:
                    file_path = os.path.join(Export_Path, f"{code}.csv")
                    
                    # 读取现有的 CSV 文件
                    if os.path.exists(file_path):
                        df_existing = pd.read_csv(file_path, parse_dates=['time'])
                        df_existing.set_index('time', inplace=True)  # 确保 'time' 列作为索引
                        
                        # 获取最后一条数据的时间，并确保是 pd.Timestamp 类型
                        last_time = pd.to_datetime(df_existing.index.max())
                        
                        # 获取下一个交易时间
                        start_time = self.get_next_trading_time_15min(last_time)
                        
                        # 如果最后一条时间已经超过了指定的结束时间，则跳过更新
                        if pd.to_datetime(start_time) > pd.to_datetime(End_Date):
                            print(f"{code} 数据已经是最新的，跳过更新。")
                            continue
                    else:
                        # 如果文件不存在，从头开始下载
                        start_time = "2010-01-01 00:00"
                        df_existing = pd.DataFrame()  # 空的 DataFrame 用于拼接

                    # 调用 API 下载新数据，从 start_time 到 End_Date
                    new_data = THS_HF(code, 'open;high;low;close;volume;amount', 
                                    f'Fill:Original,Interval:{Interval}', start_time, End_Date).data

                    # 确保 'time' 列存在，将其设置为索引
                    if 'time' in new_data.columns:
                        new_data.set_index('time', inplace=True)

                    # 如果没有新数据，跳过处理
                    if new_data.empty:
                        print(f"{code} 无新数据需要更新。")
                        continue

                    # 将新数据与旧数据拼接，并去重
                    if not df_existing.empty:
                        df_combined = pd.concat([df_existing, new_data])
                        df_combined = df_combined[~df_combined.index.duplicated(keep='last')]  # 去重并保留最新的数据
                    else:
                        df_combined = new_data

                    # 将索引转换为日期时间格式
                    df_combined.index = pd.to_datetime(df_combined.index, errors='coerce')

                    # 检查是否有 NaT 索引，删除无效的时间索引
                    df_combined = df_combined[df_combined.index.notna()]

                    # 保存更新后的数据，确保 'time' 列作为索引
                    df_combined.to_csv(file_path, index=True)  # index=True 保留时间索引
                    print(f"{code} 数据更新成功。")

                except Exception as e:
                    print(f"{code} 更新时出现错误: {e}")
            
            print("数据更新结束")

    def Downloading_ETF_Option_Data(self,Code_List,Begin_Date,End_Date,Export_Path):
            # 期权code 上证50=510050 沪深300=510300 500ETF=510500 科创50ETF=588000 科创板50=588080

            Export_Path=Export_Path

            print("下载开始")

            for i in range(0,len(Code_List)):

                try:
                    code=Code_List[i]+'O'
                    sdate=Begin_Date.replace('-','')
                    edate=End_Date.replace('-','')
                    Data=THS_DR('p02872','qqbd='+code+';'+'sdate='+sdate+';'+'edate='+edate+';jys=212001;bdlx=基金;date=全部',
                        'p02872_f001:Y,p02872_f002:Y,p02872_f003:Y,p02872_f004:Y,p02872_f005:Y,p02872_f006:Y,p02872_f007:Y,p02872_f008:Y,p02872_f009:Y,p02872_f010:Y,p02872_f014:Y,p02872_f011:Y,p02872_f015:Y,p02872_f016:Y,p02872_f017:Y',
                        'format:dataframe').data
                    
                    Data=Data.set_index('p02872_f001',drop=True)

                    Data=Data.sort_index()

                    Data.to_csv(Export_Path+'\\'+Code_List[i]+".csv")

                    print(str(i/len(Code_List)))
                
                except:

                    print(Code_List[i]+"出现错误")

            print("下载结束")

    def Updating_ETF_Option(self,End_Date,Data_Path):

        Data_Path=Data_Path

        os.chdir(Data_Path)

        List=os.listdir()

        Doc_Number=len(List)

        print("更新开始")

        for i in range(0,Doc_Number):

            try:

                df=pd.read_csv(Data_Path+"\\"+List[i],index_col=[0])

                df.index=pd.to_datetime(df.index)

                if df.index[-1].strftime('%Y-%m-%d')!=End_Date:

                    New_Begin_Date=df.index[-1]+Day()

                    New_Begin_Date=New_Begin_Date.strftime("%Y-%m-%d")

                    code_original = List[i].replace('.csv', "")
                    code=code_original+'O'
                    code=Code_List[i]+'O'

                    new_sdate=New_Begin_Date.replace('-','')
                    new_edate=End_Date.replace('-','')

                    new_data=THS_DR('p02872','qqbd='+code+';'+'sdate='+new_sdate+';'+'edate='+new_edate+';jys=212001;bdlx=基金;date=全部',
                        'p02872_f001:Y,p02872_f002:Y,p02872_f003:Y,p02872_f004:Y,p02872_f005:Y,p02872_f006:Y,p02872_f007:Y,p02872_f008:Y,p02872_f009:Y,p02872_f010:Y,p02872_f014:Y,p02872_f011:Y,p02872_f015:Y,p02872_f016:Y,p02872_f017:Y',
                        'format:dataframe').data
                    new_data=new_data.set_index('p02872_f001',drop=True)
                    new_data.index=pd.to_datetime(new_data.index)
                    new_data=new_data.sort_index()
                    Updated_Data=pd.concat([df,new_data],axis=0)

                    Updated_Data.to_csv(List[i])

                else:

                    print("数据更新至最新")

            except:

                print(List[i]+"出现错误")

        print("更新结束")

        return

    def Download_Index_DDE_Data(self,Code_List,Begin_Date,End_Date,Export_Path):

        def Downloading_Index_DDE_Data(Code,Begin_Date,End_Date):

            #日期循环
                Begin_Date=Begin_Date

                Begin_Date=pd.to_datetime(Begin_Date)

                End_Date=End_Date

                End_Date=pd.to_datetime(End_Date)

                Gap=(End_Date-Begin_Date)/250

                Period=Gap.days

                Timeseries=pd.date_range(Begin_Date,End_Date,periods=Period)

                Timeseries_List=Timeseries.strftime("%Y-%m-%d").tolist()

                Begin_Date_List=Timeseries_List[:-1]

                End_Date_List=Timeseries_List[1:]

                Data=[]

            #日期循环下载数据

                for i,j in zip(Begin_Date_List,End_Date_List):
                    
                    DF=THS_DS(Code,'ths_dde_5d_hb_index;ths_dde_10d_hb_index;ths_dde_20d_hb_index',';;','block:history',i,j).data
                    Data.append(DF)
                
                Data=pd.concat(Data,axis=0)

                return Data

        os.chdir(Export_Path)

        Error_Code=[]

        print("更新开始")
        
        for i in Code_List:

            try:
                
                Data=Downloading_Index_DDE_Data(i,Begin_Date,End_Date)

                Data=Data.set_index('time',drop=True)

                Data.to_csv(i+".csv")

                print(i+"输出完成")

            except:

                print(i+"出现错误")

                Error_Code.append(i)
        
        return

    def Updating_Index_DDE_Data(self,End_Date,Data_Path):

        Data_Path=Data_Path

        os.chdir(Data_Path)

        List=os.listdir()

        Doc_Number=len(List)

        print("更新开始")

        for i in range(0,Doc_Number):

            try:

                df=pd.read_csv(Data_Path+"\\"+List[i],index_col=[0])

                df.index=pd.to_datetime(df.index)

                if df.index[-1].strftime('%Y-%m-%d')!=End_Date:

                    New_Begin_Date=df.index[-1]+Day()

                    New_Begin_Date=New_Begin_Date.strftime("%Y-%m-%d")

                    code = List[i].replace('.csv', "")
                    THS_DS('801017.SL','ths_dde_5d_hb_index;ths_dde_10d_hb_index;ths_dde_20d_hb_index',';;','block:history',New_Begin_Date,End_Date).data
                    new_data=THS_DS(code,'ths_dde_5d_hb_index;ths_dde_10d_hb_index;ths_dde_20d_hb_index',';;','block:history',New_Begin_Date,End_Date).data

                    new_data.index=pd.to_datetime(new_data.loc[:,"time"])

                    new_data=new_data.iloc[:,2:]

                    Updated_Data=pd.concat([df,new_data],axis=0)
                    
                    Updated_Data.loc[:,"thscode"]=Updated_Data.iloc[0,0]

                    Updated_Data.to_csv(List[i])

                    print(str(i/Doc_Number))

                else:

                    print("数据更新至最新")

            except:

                print(List[i]+"出现错误")

        print("更新结束")

        return
    
        
        def Updating_updown_Data(End_Date, Data_Path):

            Data_Path=Data_Path

            os.chdir(Data_Path)

            List=os.listdir()

            Doc_Number=len(List)

            print("更新开始")

            for i in range(0,Doc_Number):

                try:

                    df=pd.read_csv(Data_Path+"\\"+List[i],index_col=[0])

                    df.index=pd.to_datetime(df.index)

                    if df.index[-1].strftime('%Y-%m-%d')!=End_Date:

                        New_Begin_Date=df.index[-1]+Day()

                        New_Begin_Date=New_Begin_Date.strftime("%Y-%m-%d")

                        code = List[i].replace('.csv', "")

                        new_data=THS_DS(code,'ths_limit_up_stock_num_sector;ths_limit_down_stock_num_sector',';', 'block:latest', New_Begin_Date, End_Date).data

                        new_data.index=pd.to_datetime(new_data.loc[:,"time"])

                        new_data=new_data.iloc[:,2:]

                        Updated_Data=pd.concat([df,new_data],axis=0)

                        Updated_Data.to_csv(List[i])

                        print(str(i/Doc_Number))

                    else:

                        print("数据更新至最新")

                except:

                    print(List[i]+"出现错误")

            print("更新结束")

            return

    def Updating_updown_Data(self,End_Date, Data_Path):

        Data_Path=Data_Path

        os.chdir(Data_Path)

        List=os.listdir()

        Doc_Number=len(List)

        print("更新开始")

        for i in range(0,Doc_Number):

            try:

                df=pd.read_csv(Data_Path+"\\"+List[i],index_col=[0])

                df.index=pd.to_datetime(df.index)

                if df.index[-1].strftime('%Y-%m-%d')!=End_Date:

                    New_Begin_Date=df.index[-1]+Day()

                    New_Begin_Date=New_Begin_Date.strftime("%Y-%m-%d")

                    code = List[i].replace('.csv', "")

                    new_data=THS_DS(code,'ths_limit_up_stock_num_sector;ths_limit_down_stock_num_sector',';', 'block:latest', New_Begin_Date, End_Date).data

                    new_data.index=pd.to_datetime(new_data.loc[:,"time"])

                    new_data=new_data.iloc[:,2:]

                    Updated_Data=pd.concat([df,new_data],axis=0)

                    Updated_Data.to_csv(List[i])

                    print(str(i/Doc_Number))

                else:

                    print("数据更新至最新")

            except:

                print(List[i]+"出现错误")

        print("更新结束")

        return

    def Updating_Up_Down_Companies(self,End_Date,Data_Path):

        Data_Path=Data_Path

        os.chdir(Data_Path)

        List=os.listdir()

        Doc_Number=len(List)

        print("更新开始")

        for i in range(0,Doc_Number):

            try:

                df=pd.read_csv(Data_Path+"\\"+List[i],index_col=[0])

                df.index=pd.to_datetime(df.index)

                if df.index[-1].strftime('%Y-%m-%d')!=End_Date:

                    New_Begin_Date=df.index[-1]+Day()

                    New_Begin_Date=New_Begin_Date.strftime("%Y-%m-%d")

                    code= List[i].replace('.csv', "")

                    new_sdate=New_Begin_Date.replace('-','')

                    new_edate=End_Date.replace('-','')

                    new_data=THS_DR('p00112','sdate='+new_sdate+';'+'edate='+new_edate+';p0='+code,'p00112_f001:Y,p00112_f002:Y,p00112_f003:Y,p00112_f004:Y,p00112_f005:Y,p00112_f006:Y,p00112_f007:Y,p00112_f008:Y,p00112_f009:Y,p00112_f010:Y,p00112_f011:Y,p00112_f012:Y,p00112_f013:Y,p00112_f014:Y,p00112_f021:Y,p00112_f022:Y,p00112_f023:Y,p00112_f024:Y,p00112_f025:Y',
                                    'format:dataframe').data
                    new_data=new_data.set_index('p00112_f001',drop=True)
                    new_data.index=pd.to_datetime(new_data.index)
                    new_data=new_data.sort_index()
                    Updated_Data=pd.concat([df,new_data],axis=0)

                    Updated_Data.to_csv(List[i])

                else:

                    print("数据更新至最新")

            except:

                print(List[i]+"出现错误")

        print("更新结束")

        return


DUD=Downloading_And_Updating_Data()



def Update_All(End_Date):

    Price_Data_Path_1=r"D:\数据库\同花顺指数量价数据"

    Price_Data_Path_2=r"D:\数据库\同花顺ETF跟踪指数量价数据\1d"

    Price_Data_Path_3=r'D:\数据库\同花顺ETF量价数据'

    Price_Data_Path_4=r'D:\数据库\同花顺商品指数量价数据'

    Path_List=[Price_Data_Path_1,Price_Data_Path_2,Price_Data_Path_3,Price_Data_Path_4]

    DUD.Updating_All_Market_Vol_Data(End_Date,Path_List)

    Forcast_Path=r"D:\数据库\同花顺ETF跟踪指数一致预期数据\预测净利润两年复合增长"

    #DUD.Updating_Forcast_2yr_Data(End_Date,Forcast_Path)

    Free_Turn_Path=r"D:\数据库\同花顺指数自由流通换手率"

    DUD.Updating_Index_Free_Turn_Data(End_Date,Free_Turn_Path)

    Valuation_Path=r"D:\数据库\同花顺ETF跟踪指数估值数据"

    DUD.Updating_Index_Valuation_Data(End_Date,Valuation_Path)

    EDB_Data_Path=r"D:\数据库\同花顺EDB数据"

    DUD.Update_EDB_Data(EDB_Data_Path,End_Date)

    ETF_Option_Path=r"D:\数据库\另类数据\ETF期权数据"

    DUD.Updating_ETF_Option(End_Date,ETF_Option_Path)

    Forcast_Data_Path=r'D:\数据库\同花顺ETF跟踪指数一致预期数据\盈利预测综合值'

    #DUD.Update_Index_Forcast_Data(End_Date,Forcast_Data_Path)

    Cov_Bond_Path=r'D:\数据库\同花顺可转债数据'

    #DUD.Update_Cov_Bond_Vol_Price_Data(End_Date,Cov_Bond_Path)

    #新高新低
    H_L_Path=r'D:\数据库\另类数据\新高新低'

    DUD.Updating_newHL_Data(End_Date,H_L_Path)

    #高频数据更新

    Export_Path_60 = r'D:\数据库\同花顺ETF跟踪指数量价数据\1h' 

    Export_Path_15=r'D:\数据库\同花顺ETF跟踪指数量价数据\15min'

    HF_End_Date=End_Date+' 15:00'

    DUD.Update_High_Freq_Vol_Price_Data(HF_End_Date, Export_Path_60,60)

    DUD.Update_High_Freq_Vol_Price_Data(HF_End_Date,Export_Path_15,15)

    #公司上涨数量更新
    
    Up_Down_Company_Path=r'D:\数据库\另类数据\涨跌家数'

    DUD.Updating_Up_Down_Companies(End_Date,Up_Down_Company_Path)

    #涨停跌停数量

    up_path=r'D:\数据库\另类数据\涨停跌停'

    DUD.Updating_updown_Data(End_Date,up_path)


    return print('updating finished')

#海外数据第二天更新
def Update_Oversea_Data(End_Date):

    A50_Path=r'D:\数据库\另类数据\A50期货数据'

    DUD.Updating_A50_Futures_Data(End_Date,A50_Path)

    return print('海外数据更新完毕')

Update_Oversea_Data('2025-01-08')

Update_All('2025-01-08')

