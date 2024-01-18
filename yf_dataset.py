import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data
import torch

def min_max_scaling(data, feature_range=(0, 1)):
    """
    Min-Max Scaling for data.

    Parameters:
    - data: A NumPy array containing the data to be scaled.
    - feature_range: A tuple specifying the desired feature range (default is [0, 1]).

    Returns:
    - scaled_data: The scaled data within the specified feature range.
    """
    min_val, max_val = feature_range
    min_data = np.min(data, axis=0)
    max_data = np.max(data, axis=0)
    scaled_data = (data - min_data) / (max_data - min_data) * (max_val - min_val) + min_val
    return scaled_data

def getInput(withGold, withOil):
    start_date = "2011-01-30"
    end_date = "2019-09-01"
    yfdata = yf.download('^GSPC', start=start_date, end=end_date)
    golddata = yf.download('GC=F', start=start_date, end=end_date)
    oildata = yf.download('CL=F', start=start_date, end=end_date)

    yfdata_dates = yfdata.index
    golddata_dates = golddata.index

    # 找出 yfdata 有而 golddata 没有的日期
    missing_dates = yfdata_dates[~yfdata_dates.isin(golddata_dates)]
    missing_data = pd.DataFrame(index=missing_dates,columns=golddata.columns)
    for date in missing_dates:
        previous_date = golddata_dates[golddata_dates < date]
        if not previous_date.empty:
            last_known_data = golddata.loc[previous_date.max()]
            missing_data.loc[date] = last_known_data

    golddata = pd.concat([golddata, missing_data])
    golddata = golddata.add_prefix('gold ')
    golddata = golddata.sort_index()

    # 找出 yfdata 有而 oildata 没有的日期
    oildata_dates = oildata.index
    missing_dates = yfdata_dates[~yfdata_dates.isin(oildata_dates)]
    missing_data = pd.DataFrame(index=missing_dates,columns=oildata.columns)
    for date in missing_dates:
        previous_date = oildata_dates[oildata_dates < date]
        if not previous_date.empty:
            last_known_data = oildata.loc[previous_date.max()]
            missing_data.loc[date] = last_known_data

    oildata = pd.concat([oildata, missing_data])
    oildata = oildata.add_prefix('oil ')
    oildata = oildata.sort_index()
#--------------------------------------------------------------------------------
    days = 8 #多取一天後續方便計算label
    date_cols = []
    price_cols = yfdata.columns.tolist()
    price_cols.append('gold Adj Close')
    price_cols.append('oil Adj Close')
    all_data = {}
    sliding_window = 1

    def group_by_week(data,price_col):
        local_df = []
        for i in range(len(data)):
            if np.isnan(data[price_col][i]): 
                print(str(data.index[i]))
                break
            local_df.append(data[price_col][i])
            
            if len(local_df) == days:
                col_name = str(data.index[i - days + 1]) + ' ~ ' + str(data.index[i])
                if price_col == price_cols[0]:
                    date_cols.append(col_name)
                all_data[(price_col, col_name)] = local_df.copy()
                for _ in range(sliding_window):
                    local_df.pop(0) 
                    
    for price_col in price_cols:
        if price_col.startswith('gold'):
            group_by_week(golddata, price_col)
        elif price_col.startswith('oil'):
            group_by_week(oildata, price_col)
        else: 
            group_by_week(yfdata, price_col)


    multi_columns = pd.MultiIndex.from_product([price_cols,date_cols],names=['price','date'])
    group_by_days_yfdata = pd.DataFrame(data=all_data,columns=multi_columns)
#---------------------------------------------------------------------------------------------------
    data = group_by_days_yfdata
    print("If there is NAN in data? ",data.isnull().values.any())
    returns = data['Adj Close'].pct_change().fillna(0) 
    print(returns.keys())
    scaled_returns = StandardScaler().fit_transform(returns) #Z-score標準化:mean = 0,std = 1，後續用來計算相關性
    # 計算各週之間的相關性
    correlation_matrix = np.corrcoef(scaled_returns, rowvar=False)

    # 將相關性矩陣轉換為鄰接矩陣
    adjacency_matrix = torch.tensor(correlation_matrix)

    src_nodes, dst_nodes = np.where((adjacency_matrix > 0.7))
    directed_index = np.where((src_nodes < dst_nodes) & ((dst_nodes - src_nodes) <= 495))
    src_nodes = src_nodes[directed_index]
    dst_nodes = dst_nodes[directed_index]

    edge_index = torch.tensor([
        src_nodes,  
        dst_nodes  
    ],dtype=torch.long) #邊

    features = ['Adj Close', 'Open', 'High', 'Low', 'Volume']
    if (withGold):
        features.append('gold Adj Close')
    if (withOil):
        features.append('oil Adj Close')
    num_nodes = len(returns.keys())
    num_features = len(features)
    feature_matrix = np.zeros((num_nodes, num_features,len(data)-1)) 

    # 將數據填充到特徵矩陣中
    for i, symbol in enumerate(returns.keys()):
        for j,feature in enumerate(features):
            company_data = data[(feature,symbol)]
            feature_matrix[i, j] = torch.tensor(min_max_scaling(company_data.values[:-1]),dtype=torch.float32) #取得第一天到第七天

    feature_matrix = torch.tensor(feature_matrix,dtype=torch.float32)
    feature_matrix = feature_matrix.reshape(feature_matrix.shape[0], -1)

    y_list = []
    for i, symbol in enumerate(returns.keys()):
        lastDay = data.iloc[-2][('Adj Close',symbol)] #input中最後一天的資料
        predictDay = data.iloc[-1][('Adj Close',symbol)] #隔天的資料
        if (lastDay/predictDay > 1.005):
            y_list.append([2]) #下跌
        elif ( 0.995 < lastDay/predictDay < 1.005): #應該在某個範圍內
            y_list.append([1]) #持平
        else:
            y_list.append([0]) #上漲

    y = torch.tensor(y_list, dtype=torch.long)

    print("feature_matrix_shape = ",feature_matrix.shape)
    print("edge size= ",edge_index.shape)
    print('y = ',y.shape)
    gnnInputData = Data(x=feature_matrix,edge_index=edge_index,y=y)

    train_eval_test_index = []
    for target_value in ['~ 2013-01-30','~ 2017-05-23','~ 2018-03-27','~ 2019-08-29']:
        for index, item in enumerate(returns.keys()):
            if target_value in item:
                train_eval_test_index.append(index)
                print(f"找到日期 '{item}'，index= {index}")
                break

    train_idx = torch.tensor(np.arange(train_eval_test_index[0],train_eval_test_index[1]+1),dtype=torch.long)
    #evaluate input節點
    eval_idx = torch.tensor(np.arange(train_eval_test_index[1]+1,train_eval_test_index[2]+1),dtype=torch.long)
    #test input
    test_idx = torch.tensor(np.arange(train_eval_test_index[2]+1,train_eval_test_index[3]+1),dtype=torch.long)

    count_0 = np.count_nonzero(y == 0)
    count_1 = np.count_nonzero(y == 1)
    count_2 = np.count_nonzero(y == 2)

    print("上漲(0)的数量：", count_0)
    print("持有(1)的数量：", count_1)
    print("下跌(2)的数量：", count_2)

    weight = [len(y) / (count_0 + 1e-8), len(y) / (count_1 + 1e-8), len(y) / (count_2 + 1e-8)]
    print("weight: ",weight)

    return train_idx, eval_idx, test_idx, weight, yfdata, gnnInputData