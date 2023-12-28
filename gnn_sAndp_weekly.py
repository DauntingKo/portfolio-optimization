import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch_geometric.loader import NeighborLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.nn import MessagePassing, SAGEConv
from ogb.nodeproppred import Evaluator, PygNodePropPredDataset
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
from early_stopping import EarlyStopping
import random

seed = 42
os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class FocalLoss(torch.nn.Module):
    def __init__(self, gamma=2, alpha=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, input, target):
        ce_loss = F.nll_loss(input, target)
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        self.alpha = self.alpha.to(input.device)
        if self.alpha is not None:
            focal_loss = self.alpha[target] * focal_loss

        if self.reduction == 'mean':
            return torch.mean(focal_loss)
        elif self.reduction == 'sum':
            return torch.sum(focal_loss)
        else:
            return focal_loss

#Graph Sample and Aggregation
class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, n_layers=2):
        super(SAGE, self).__init__()
        self.n_layers = n_layers
        self.layers = torch.nn.ModuleList()
        self.layers_bn = torch.nn.ModuleList() #batch normalization

        if n_layers == 1:
            self.layers.append(SAGEConv(in_channels, out_channels, normalize=False))
        elif n_layers == 2:
            self.layers.append(SAGEConv(in_channels, hidden_channels, normalize=False))
            self.layers_bn.append(torch.nn.BatchNorm1d(hidden_channels))
            self.layers.append(SAGEConv(hidden_channels, out_channels, normalize=False))
        else:
            self.layers.append(SAGEConv(in_channels, hidden_channels, normalize=False))
            self.layers_bn.append(torch.nn.BatchNorm1d(hidden_channels))

        for _ in range(n_layers - 2):
            self.layers.append(SAGEConv(hidden_channels, hidden_channels, normalize=False))
            self.layers_bn.append(torch.nn.BatchNorm1d(hidden_channels))

            self.layers.append(SAGEConv(hidden_channels, out_channels, normalize=False))

        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, x, edge_index):
        if len(self.layers) > 1:
            looper = self.layers[:-1]
        else:
            looper = self.layers
        
        for i, layer in enumerate(looper):
            x = layer(x, edge_index)
            try:
                x = self.layers_bn[i](x)
            except Exception as e:
                abs(1)
            finally:
                x = F.relu(x)
                x = F.dropout(x, p=0.5, training=self.training)

        if len(self.layers) > 1:
            x = self.layers[-1](x, edge_index)
        return F.log_softmax(x, dim=-1), torch.var(x)

    def inference(self, total_loader, device):
        xs = []
        var_ = []
        for batch in total_loader:
            out, var = self.forward(batch.x.to(device), batch.edge_index.to(device))
            out = out[:batch.batch_size]
            xs.append(out.cpu())
            var_.append(var.item())

        out_all = torch.cat(xs, dim=0)

        return out_all, var_


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

def startGNN(startLr, withGold, withOil, numNeighbors, lossFunction):
    start_date = "2011-01-30"
    end_date = "2019-08-30"
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
            
    # print("Any NAN in feature_matrix= ",np.isnan(feature_matrix).any())

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
    test_idx = torch.tensor(np.arange(train_eval_test_index[2]+1,len(feature_matrix)),dtype=torch.long)

    count_0 = np.count_nonzero(y == 0)
    count_1 = np.count_nonzero(y == 1)
    count_2 = np.count_nonzero(y == 2)

    print("上漲(0)的数量：", count_0)
    print("持有(1)的数量：", count_1)
    print("下跌(2)的数量：", count_2)

    weight = [len(y) / (count_0 + 1e-8), len(y) / (count_1 + 1e-8), len(y) / (count_2 + 1e-8)]
    print("weight: ",weight)

    train_loader =  NeighborLoader(gnnInputData, input_nodes=train_idx,
                              shuffle=False, num_workers=os.cpu_count() - 2,
                              batch_size=32, num_neighbors=[numNeighbors]*2)
    total_loader = NeighborLoader(gnnInputData, input_nodes=None, num_neighbors=[-1],
                                shuffle=False,
                                num_workers=os.cpu_count() - 2)
    
    target_dataset = 'ogbn-arxiv'
    def test(model, device):
        evaluator = Evaluator(name=target_dataset)
        model.eval()
        out, var = model.inference(total_loader, device)
        y_true = gnnInputData.y.cpu()
        y_pred = out.argmax(dim=-1, keepdim=True)
        
        train_acc = evaluator.eval({
            'y_true': y_true[train_idx],
            'y_pred': y_pred[train_idx],
        })['acc']
        val_acc = evaluator.eval({
            'y_true': y_true[eval_idx],
            'y_pred': y_pred[eval_idx],
        })['acc']
        test_acc = evaluator.eval({
            'y_true': y_true[test_idx],
            'y_pred': y_pred[test_idx],
        })['acc']
        
        return train_acc, val_acc, test_acc, torch.mean(torch.Tensor(var)), y_true, out
    
    def creterion(lossFunction , weight, out, y_true):
        loss = 1
        match lossFunction:
            case 'CrossEntrophyLoss':
                loss = F.nll_loss(out, torch.reshape(y_true, (-1,)))
            case 'FocalLoss':
                focalLoss = FocalLoss(gamma=2,alpha=torch.tensor(weight),reduction='mean')
                loss = focalLoss(out, torch.reshape(y_true, (-1,)))

        return loss
    
    title = lossFunction
    if withGold: title += ' with gold'
    if withOil: title += ' with oil'
    title += f' startLr={startLr:e} numNeighbors={numNeighbors}'
    if not os.path.exists(f'result/{lossFunction}/'):
        os.makedirs(f'result/{lossFunction}/')
    if not os.path.exists(f'result/{lossFunction}/{title}'):
        os.makedirs(f'result/{lossFunction}/{title}')
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SAGE(gnnInputData.x.shape[1], 256, 3, n_layers=2)
    model.to(device)
    epochs = 100
    optimizer = torch.optim.Adam(model.parameters(), lr=startLr, weight_decay=1e-3)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=3)

    save_path = f'result/{lossFunction}/{title}/'
    early_stopping = EarlyStopping(save_path,patience=15,verbose=True)

    train_accs = []
    val_accs = []
    test_accs = []
    total_train_loss = []
    total_eval_loss = []
    total_test_loss = []
    for epoch in range(1, epochs):
        model.train()
        pbar = tqdm(total=len(train_idx))
        pbar.set_description(f'Epoch {epoch:02d}')
        batch_loss = total_correct = 0
        for batch in train_loader:
            batch_size = batch.batch_size
            optimizer.zero_grad()
            out, _ = model(batch.x.to(device), batch.edge_index.to(device))
            out = out[:batch_size]
            batch_y = batch.y[:batch_size].to(device)
            batch_y = torch.reshape(batch_y, (-1,))
            loss = creterion(lossFunction, weight, out, batch_y)
            loss.backward()
            optimizer.step()
            batch_loss += float(loss)
            total_correct += int(out.argmax(dim=-1).eq(batch_y).sum())
            pbar.update(batch.batch_size)
        pbar.close()
        # loss = batch_loss / len(train_loader)
        approx_acc = total_correct / train_idx.size(0)
        train_acc, val_acc, test_acc, var, y_true, out = test(model, device)
        train_loss = creterion(lossFunction, weight, out[train_idx], torch.reshape(y_true[train_idx], (-1,)))  
        eval_loss = creterion(lossFunction, weight, out[eval_idx], torch.reshape(y_true[eval_idx], (-1,)))
        test_loss = creterion(lossFunction, weight, out[test_idx], torch.reshape(y_true[test_idx], (-1,)))
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        test_accs.append(test_acc)
        total_train_loss.append(train_loss)
        total_eval_loss.append(eval_loss)
        total_test_loss.append(test_loss)
        scheduler.step(eval_loss)

        cm = confusion_matrix(y_true[test_idx], out[test_idx].argmax(dim=-1, keepdim=True))

        print(f'TrainAcc: {train_acc:.4f}, ValAcc: {val_acc:.4f}, TestAcc: {test_acc:.4f}, trainLoss: {train_loss:.4f}, evalLoss: {eval_loss:.4f}, testLoss: {test_loss:.4f}')
        #early_stopping
        early_stopping(test_loss, model, confusion_matrix=cm, epoch=epoch, test_acc=test_acc, test_loss=test_loss)
        if early_stopping.early_stop:
            print("Early stopping")
            break 

    plt.plot(np.arange(1, len(train_accs)+1), train_accs, label='Train Accuracy', marker='o')
    # plt.plot(np.arange(0, len(val_accs)), val_accs, label='Validation Accuracy', marker='o')
    plt.plot(np.arange(1, len(test_accs)+1), test_accs, label='Test Accuracy', marker='o')

    plt.annotate(f'epoch: {early_stopping.best_epoch}, test_accs: {test_accs[early_stopping.best_epoch-1]:.2f}', 
                xy=(early_stopping.best_epoch, test_accs[early_stopping.best_epoch-1]), 
                xytext=(early_stopping.best_epoch + 3, test_accs[early_stopping.best_epoch-1] - 0.02),  # 文本的位置
                arrowprops=dict(facecolor='black', shrink=0.05),  # 箭頭的屬性
                )

    plt.legend()
    plt.title(f'Accuracy vs. Epochs ({title})')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.savefig(f'result/{lossFunction}/{title}/acc.png',bbox_inches = 'tight')
    plt.show()

    plt.plot(np.arange(1, len(total_train_loss)+1), [tensor.item() for tensor in total_train_loss], label='training loss', marker='o')
    plt.plot(np.arange(1, len(total_test_loss)+1), [tensor.item() for tensor in total_test_loss], label='test_loss', marker='o')

    plt.legend()
    plt.annotate(f'epoch: {early_stopping.best_epoch}, test_loss: {total_test_loss[early_stopping.best_epoch-1]:.2f}', 
                xy=(early_stopping.best_epoch, total_test_loss[early_stopping.best_epoch-1]), 
                xytext=(early_stopping.best_epoch + 3, total_test_loss[early_stopping.best_epoch-1] - 0.02),  # 文本的位置
                arrowprops=dict(facecolor='black', shrink=0.05),  # 箭頭的屬性
                )

    plt.title(f'Loss vs. Epochs ({title})')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.savefig(f'result/{lossFunction}/{title}/loss.png',bbox_inches = 'tight')
    plt.show()

    labels=['rising','hold','drop']     
    fig, ax= plt.subplots()
    sns.heatmap(early_stopping.best_test_confusion_matrix, annot=True, fmt='g', ax=ax);  #annot=True to annotate cells, ftm='g' to disable scientific notation

    # labels, title and ticks
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('CM (' + title +')')
    ax.xaxis.set_ticklabels(labels); ax.yaxis.set_ticklabels(labels)
    fig.savefig(f'result/{lossFunction}/{title}/cm.png',bbox_inches = 'tight')
    fig.show()
    return early_stopping.best_test_acc, early_stopping.best_test_loss


    
