import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data
import torch
import datetime
from dateutil.relativedelta import relativedelta

def get_macd_param (yfdata):
    def condition(macd_yearMonth, date):
        year = int(macd_yearMonth.split("-")[0])
        month_start, month_end = map(int, macd_yearMonth.split("-")[1].split("~"))

        date_three_months_prior = date - relativedelta(months=3)
        
        if (year == date_three_months_prior.year and
            month_start <= date_three_months_prior.month <= month_end):
            return True
        else:
            return False
        
    macd_params = pd.read_csv('macdBestParam_per_3_month.csv')
    
    # Initialize a column for MACD parameters
    yfdata['macd Param'] = None
    
    # Iterate through yfdata to find and assign the correct MACD parameters
    for date in yfdata.index:
        # Find the corresponding MACD parameter
        for _, row in macd_params.iterrows():
            if condition(row['yearMonth'], date):
                yfdata.at[date, 'macd Param'] = row['macdParam']
                break 

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

def parse_macd_params(param_string):
    return map(int, param_string.split(','))

def getMacdSignal(yfdata, macdParamOptimize):
    if macdParamOptimize:
        print('macdParamOptimize = True!!!!!!')
        for index, row in yfdata.iterrows():
            fast_span, slow_span, signal_span = parse_macd_params(row['macd Param'])

            current_data = yfdata.loc[:index]
            
            current_data['Fast EMA'] = current_data['Adj Close'].ewm(span=fast_span, adjust=False).mean()
            current_data['Slow EMA'] = current_data['Adj Close'].ewm(span=slow_span, adjust=False).mean()
            
            current_data['MACD'] = current_data['Fast EMA'] - current_data['Slow EMA']
            current_data['Signal Line'] = current_data['MACD'].ewm(span=signal_span, adjust=False).mean()

            last_macd = current_data['MACD'].iloc[-1]
            last_signal_line = current_data['Signal Line'].iloc[-1]
            prev_macd = current_data['MACD'].iloc[-2] if len(current_data) > 1 else last_macd
            prev_signal_line = current_data['Signal Line'].iloc[-2] if len(current_data) > 1 else last_signal_line

            if last_macd > last_signal_line and prev_macd <= prev_signal_line:
                yfdata.at[index, 'Signal'] = 0  
            elif last_macd < last_signal_line and prev_macd >= prev_signal_line:
                yfdata.at[index, 'Signal'] = 2  
            else:
                yfdata.at[index, 'Signal'] = 1  
    else:
        fast_span = 5  
        slow_span = 20  
        signal_span = 9  
        yfdata['Fast EMA'] = yfdata['Adj Close'].ewm(span=fast_span, adjust=False).mean()
        yfdata['Slow EMA'] = yfdata['Adj Close'].ewm(span=slow_span, adjust=False).mean()

        yfdata['MACD'] = yfdata['Fast EMA'] - yfdata['Slow EMA']

        yfdata['Signal Line'] = yfdata['MACD'].ewm(span=signal_span, adjust=False).mean()
        yfdata['Signal'] = 1

        for i in range(1, len(yfdata)):
            if yfdata['MACD'].iloc[i] > yfdata['Signal Line'].iloc[i] and yfdata['MACD'].iloc[i - 1] <= yfdata['Signal Line'].iloc[i - 1]:
                yfdata['Signal'].iloc[i] = 0  
            elif yfdata['MACD'].iloc[i] < yfdata['Signal Line'].iloc[i] and yfdata['MACD'].iloc[i - 1] >= yfdata['Signal Line'].iloc[i - 1]:
                yfdata['Signal'].iloc[i] = 2  
    return

def getInput(withGold, withOil, withMacdSignal=False, macdParamOptimize=False, corr=0.7, begin_days=1096, edge_weight_based_on='corr' , edge_weight_lambda_decay=0.5, window_size=7):
    base_date = datetime.date(2008, 1, 30)
    start_date = (base_date + datetime.timedelta(days=begin_days)).strftime('%Y-%m-%d')
    print("start_date =",start_date)
    # start_date = "2011-01-30"
    end_date = "2019-09-01"
    yfdata = yf.download('^GSPC', start=start_date, end=end_date)
    golddata = yf.download('GC=F', start=start_date, end=end_date)
    oildata = yf.download('CL=F', start=start_date, end=end_date)

    train_date_begin = pd.Timestamp('2013-01-30')
    if train_date_begin in yfdata.index:
        train_date_index = yfdata.index.get_loc(train_date_begin)
        print(f"The index for {train_date_begin.date()} is {train_date_index}")
    else:
        print(f"{train_date_begin.date()} is not in the index.")

    get_macd_param(yfdata)
    getMacdSignal(yfdata, macdParamOptimize)

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
    days = window_size + 1 #多取一天後續方便計算label
    date_cols = []
    price_cols = yfdata.columns.tolist()
    price_cols.remove('macd Param')
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
    neighbor_range = train_date_index - window_size
    print("neighbor_range =",neighbor_range)
    src_nodes, dst_nodes = np.where((adjacency_matrix > corr))
    directed_index = np.where((src_nodes < dst_nodes) & ((dst_nodes - src_nodes) <= neighbor_range))
    src_nodes = src_nodes[directed_index]
    dst_nodes = dst_nodes[directed_index]

    edge_index = torch.tensor([
        src_nodes,  
        dst_nodes  
    ],dtype=torch.long) #邊

    if edge_weight_based_on == 'corr':
        edge_weights = adjacency_matrix[src_nodes, dst_nodes]
        # 將邊權重轉為Tensor
        edge_weights = torch.tensor(edge_weights, dtype=torch.float)
    else:
        time_diff = dst_nodes - src_nodes
        edge_weights = np.exp(-edge_weight_lambda_decay * time_diff)
        edge_weights = torch.tensor(edge_weights, dtype=torch.float)
        print('edge_weight_based_on ',edge_weight_based_on,' lambda_decay = ',edge_weight_lambda_decay,' edge_weights = ',edge_weights)


    features = ['Adj Close', 'Open', 'High', 'Low', 'Volume']
    if (withGold):
        features.append('gold Adj Close')
    if (withOil):
        features.append('oil Adj Close')
    if (withMacdSignal):
        features.append('Signal')
    num_nodes = len(returns.keys())
    num_features = len(features)
    feature_matrix = np.zeros((num_nodes, num_features,len(data)-1)) 

    # 將數據填充到特徵矩陣中
    for i, symbol in enumerate(returns.keys()):
        for j,feature in enumerate(features):
            company_data = data[(feature,symbol)]
            if feature == 'Signal':
                feature_matrix[i, j] = torch.tensor(company_data.values[:-1],dtype=torch.float32) #取得第一天到第七天
            else:
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
    print("edge_weights size= ",edge_weights.shape)
    print('y = ',y.shape)
    gnnInputData = Data(x=feature_matrix,edge_index=edge_index,y=y,edge_weights=edge_weights)

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