# 加權 GraphSAGE 模型捕捉股票數據中時間關係的有效性 實驗</br>
**Effectiveness of Weighted GraphSAGE Model in Capturing Temporal Relationships in Stock Data**
## 簡介
**Abstract**
> 運用WeightedSAGE模型，用來預測單一股票(s&p500)未來一天為"上漲"、"持平"還是"下跌"</br>
> Using the WeightedSAGE model to predict whether a single stock (S&P 500) will be "rise," "remain," or "drop" the next day.
## 環境設定
> 先安裝 miniconda，並透過 gnnreq.txt 建立一個新的環境，以下為指令:</br>
> First,Install miniconda and create a environment using gnnreq.txt with the following command:
```
conda create -n <environment-name> --gnnreq.txt
```
## 各程式功能簡介</br>
**Overview of Each Program Function**
### model.py
> #### WeightedSAGEConv :
> 繼承 torch_geometric.nn.conv 的 MessagePassing，與原生SAGEConv不同之處在於message在聚合鄰居節點訊息時，加入了權重。</br>
> Inheriting from torch_geometric.nn.conv.MessagePassing, the key difference from the native SAGEConv is that the message function incorporates weights when aggregating information from neighboring nodes.
> #### SAGEWEIGHT :
> 實作過程如圖</br>
> The implementation process is illustrated below
> ![螢幕擷取畫面 2024-06-29 135558](https://github.com/DauntingKo/portfolio-optimization/assets/145321309/e159c107-02b2-43ad-a808-acd8ece15f05)
> #### SAGE:
> 實作原生SAGE模型</br>
> The implementation of native SAGE model
> #### FocalLoss:
> $FL(p_t )= -α(1 -p_t )^γ  log⁡(p_t )$

### yf_dataset.py
>#### get_macd_param:
>>取得目標日期前三個月最佳之macdParam</br>
>>Retrieve the Best MACD Parameters for the Target Date
>#### min_max_scaling:
>>將陣列資料範圍壓縮在0到1之間</br>
>>Compress the range of array data between 0 and 1.
>#### getMacdSignal
>>若macdParamOptimize = true，參數設定為get_macd_param找到的參數，否則將參數固定為 5,20,9，最後透過參數產出macd交易訊號</br>
>>If macdParamOptimize = true, set the parameters to those found by get_macd_param. Otherwise, fix the parameters to 5, 20, 9. Finally, generate MACD trading signals using these parameters.
>#### getInput 訓練前資料的前處理</br>for preprocessing data before training.
>>Input:
>>>1. withGold:決定特徵是否放入黃金價格</br>Determine whether to include the gold price as a feature.
>>>2. withOil:決定特徵是否放入石油價格</br>Determine whether to include the oil price as a feature.
>>>3. withMacdSignal:決定特徵是否放入MACD訊號</br>Determine whether to include the MACD signal as a feature.
>>>4. corr:相關性為多少以上，才作為鄰居</br>Only consider as neighbors if the correlation is above a certain threshold.
>>>5. begin_days:決定要取2008/1/30後幾天作為第一個可使用的鄰居節點</br>Determine the number of days after 2008/1/30 to use as the first available neighboring node.
>>>6. edge_weight_based_on: 'corr'代表用相關性作為邊權重，否則採用時間衰退性作為邊權重</br>'corr' represents using correlation as the edge weight; otherwise, use time decay as the edge weight.
>>>7. edge_weight_lambda_decay:計算時間衰退性的超參數</br>The hyperparameters for calculating time decay.
>>>8. window_size:一個節點包含幾天的資料</br>data size(days) which is contained in a node.

>>output:
>>>1. train_idx:訓練集的index</br>training index
>>>2. eval_idx:驗證集的index</br>evaluating index
>>>3. test_idx:測試集的index</br>testing index
>>>4. weight:focal loss的alpha權重</br>Alpha weight for focal loss
>>>5. yfdata:所有原始資料</br>All raw data
>>>6. gnnInputData:包含特徵矩陣、邊、label、邊權重</br>Include the feature matrix, edges, labels, and edge weights.

>>流程如下:
>>>1. 下載s&p500、原油、黃金之股市資料</br>Download stock market data for S&P 500, oil, and gold.
>>>2. 取得macd信號</br>get MACD signal
>>>3. 處理缺失值</br>Handle missing values.
>>>4. 依據window_size將資料切割成若干節點</br>Segment the data into multiple nodes based on window_size.
>>>5. 計算節點間Adj Close之相關性</br>Calculate the correlation of Adj Close between nodes.
>>>6. 透過相關性定義目標節點、鄰居節點</br>Define the target nodes and neighboring nodes based on correlation.
>>>7. 透過相關性或時間衰退來定義節點間邊的權重(edge_weights)</br>Define the edge weights between nodes based on correlation or time decay.
>>>8. 選擇特徵</br>Feature selection
>>>9. 制定label:</br>Define labels
>>>> 隔日收盤價 / 今日收盤價 > 1.005，則為 0 (上漲)        
>>>> 0.995 < 隔日收盤價 / 今日收盤價 < 1.005，則為 1 (持平)          
>>>> 隔日收盤價 / 今日收盤價 < 0.995，則為 2 (下跌)</br>
>>>> The next day's closing price / today's closing price > 1.005, then label it as 0 (rise).</br>
0.995 < next day's closing price / today's closing price < 1.005, then label it as 1 (remain).</br>
The next day's closing price / today's closing price < 0.995, then label it as 2 (drop).
>>>9. 切割為訓練集、驗證集、測試集</br>Split the data into training, validation, and test sets.
>>>10. 制定focal loss的權重</br>Set the weights for focal loss.

### gnn_sAndp_weekly.py
>#### startGNN:
>>Input:
>>>1. startLr:起始學習率</br>Start learning rate.
>>>1. withGold:決定特徵是否放入黃金價格</br>Determine whether to include the gold price as a feature.
>>>2. withOil:決定特徵是否放入石油價格</br>Determine whether to include the oil price as a feature.
>>>3. numNeighbors:NeighborLoader是一種用於圖數據批處理的加載器，它按照指定的num_neighbors列表來決定從每一層中抽取多少鄰居節點。</br>NeighborLoader is a loader used for batching graph data, which determines how many neighboring nodes to sample from each layer based on the specified num_neighbors list.
>>>4. lossFunction:'CrossEntrophyLoss'或'FocalLoss'</br>'CrossEntrophyLoss' or 'FocalLoss'
>>>5. withMacdSignal:決定特徵是否放入MACD訊號</br>Determine whether to include the MACD signal as a feature.
>>>6. macdParamOptimize:是否採用macdParam最佳化</br>Whether to use MACD parameter optimization(used only if withMacdSignal is set to True).
>>>7. gamma:focal loss之超參數(若lossFunction選擇FocalLoss才會用到)</br>Hyperparameters for focal loss (used only if lossFunction is set to FocalLoss).
>>>8. withAlpha:focal loss之超參數(若lossFunction選擇FocalLoss才會用到)</br>Hyperparameters for focal loss (used only if lossFunction is set to FocalLoss).
>>>9. aggr:聚合方式，'mean'或'weight'</br>Aggregation method, either 'mean' or 'weight'.
>>>10. corr:相關性為多少以上，才作為鄰居</br>Only consider as neighbors if the correlation is above a certain threshold.
>>>11. begin_days:決定要取2008/1/30後幾天作為第一個可使用的鄰居節點</br>Determine the number of days after 2008/1/30 to use as the first available neighboring node.
>>>12. edge_weight_based_on: 'corr'代表用相關性作為邊權重，否則採用時間衰退性作為邊權重</br>'corr' represents using correlation as the edge weight; otherwise, use time decay as the edge weight.
>>>13. edge_weight_lambda_decay:計算時間衰退性的超參數(edge_weight_based_on為時間衰退性時才會用到)</br>Calculate the hyperparameters for time decay (used only if edge_weight_based_on is set to time decay).
>>>14. window_size:一個節點包含幾天的資料</br>data size(days) which is contained in a node.

>>Output:
>>>1. best_test_acc:測試集最佳準確度</br>Best accuracy on the test set.
>>>2. best_test_loss:測試集最小loss</br>Minimum loss on the test set.
   
### early_stopping.py
>若模型loss不再降低，則提早結束訓練並儲存最佳結果。</br>If the model loss no longer decreases, terminate the training early and save the best results.

### bayes_search.ipynb
>透過optuna預設之超參數優化方法，即Tree Parzen Estimator找出超參數的最優解。</br>Find the optimal hyperparameters using Optuna's default optimization method, Tree Parzen Estimator (TPE).

### backtest.ipynb
>針對不同的情況進行回測，算出其測試集期間的回報率，包含每次只買賣一股或依據信心度決定交易量</br>Backtest different scenarios to calculate the return during the test period, including scenarios where each trade involves buying or selling only one share or where the trading volume is determined based on confidence level.
