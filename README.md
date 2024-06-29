# 加權 GraphSAGE 模型捕捉股票數據中時間關係的有效性 實驗
## 簡介
> 運用WeightedSAGE模型，用來預測單一股票(s&p500)未來一天為"上漲"、"持平"還是"下跌"
## 環境設定
> 先安裝 miniconda，並透過 gnnreq.txt 建立一個新的環境，以下為指令:
```
conda create -n <environment-name> --gnnreq.txt
```
## 各程式功能簡介
### model.py
> #### WeightedSAGEConv :
> 繼承 torch_geometric.nn.conv 的 MessagePassing，與原生SAGEConv不同之處在於message在聚合鄰居節點訊息時，加入了權重。
> #### SAGEWEIGHT :
> 實作過程如圖
> ![螢幕擷取畫面 2024-06-29 135558](https://github.com/DauntingKo/portfolio-optimization/assets/145321309/e159c107-02b2-43ad-a808-acd8ece15f05)
> #### SAGE:
> 實作原生SAGE模型
> #### FocalLoss:
> $FL(p_t )= -α(1 -p_t )^γ  log⁡(p_t )$

### yf_dataset.py
>#### get_macd_param:
>>取得目標日期前三個月最佳之macdParam
>#### min_max_scaling:
>>將陣列資料範圍壓縮在0到1之間
>#### getMacdSignal
>>若macdParamOptimize = true，參數設定為get_macd_param找到的參數，否則將參數固定為 5,20,9，最後透過參數產出macd交易訊號
>#### getInput 訓練前資料的前處理
>>Input:
>>>1. withGold:決定特徵是否放入黃金價格
>>>2. withOil:決定特徵是否放入石油價格
>>>3. withMacdSignal:決定特徵是否放入MACD訊號
>>>4. corr:相關性為多少以上，才作為鄰居
>>>5. begin_days:決定要取2008/1/30後幾天作為第一個可使用的鄰居節點
>>>6. edge_weight_based_on: 'corr'代表用相關性作為邊權重，否則採用時間衰退性作為邊權重
>>>7. edge_weight_lambda_decay:計算時間衰退性的超參數
>>>8. window_size:一個節點包含幾天的資料

>>output:
>>>1. train_idx:訓練集的index
>>>2. eval_idx:驗證集的index
>>>3. test_idx:測試集的index
>>>4. weight:focal loss的alpha權重
>>>5. yfdata:所有原始資料
>>>6. gnnInputData:包含特徵矩陣、邊、label、邊權重

>>流程如下:
>>>1. 下載s&p500、原油、黃金之股市資料
>>>2. 取得macd信號->處理缺失值
>>>3. 依據window_size將資料切割成若干節點
>>>4. 計算節點間Adj Close之相關性
>>>5. 透過相關性定義目標節點、鄰居節點
>>>6. 透過相關性或時間衰退來定義節點間邊的權重(edge_weights)
>>>7. 選擇特徵
>>>8. 制定label:
>>>> 隔日收盤價 / 今日收盤價 > 1.005，則為 0 (上漲)          
>>>> 0.995 < 隔日收盤價 / 今日收盤價 < 1.005，則為 1 (持平)          
>>>> 隔日收盤價 / 今日收盤價 < 0.995，則為 2 (下跌)
>>>9. 切割為訓練集、驗證集、測試集
>>>10. 制定focal loss的權重

### gnn_sAndp_weekly.py
>#### startGNN:
>>Input:
>>>1. startLr:起始學習率
>>>1. withGold:決定特徵是否放入黃金價格
>>>2. withOil:決定特徵是否放入石油價格
>>>3. numNeighbors:NeighborLoader是一種用於圖數據批處理的加載器，它按照指定的num_neighbors列表來決定從每一層中抽取多少鄰居節點。
>>>4. lossFunction:'CrossEntrophyLoss'或'FocalLoss'
>>>5. withMacdSignal:決定特徵是否放入MACD訊號
>>>6. macdParamOptimize:是否採用macdParam最佳化
>>>7. gamma:focal loss之超參數(若lossFunction選擇FocalLoss才會用到)
>>>8. withAlpha:focal loss之超參數(若lossFunction選擇FocalLoss才會用到)
>>>9. aggr:聚合方式，'mean'或'weight'
>>>10. corr:相關性為多少以上，才作為鄰居
>>>11. begin_days:決定要取2008/1/30後幾天作為第一個可使用的鄰居節點
>>>12. edge_weight_based_on: 'corr'代表用相關性作為邊權重，否則採用時間衰退性作為邊權重
>>>13. edge_weight_lambda_decay:計算時間衰退性的超參數(edge_weight_based_on為時間衰退性時才會用到)
>>>14. window_size:一個節點包含幾天的資料

>>Output:
>>>1. best_test_acc:測試集最佳準確度
>>>2. best_test_loss:測試集最小loss
   
### early_stopping.py
>若模型loss不再降低，則提早結束訓練並儲存最佳結果。

### bayes_search.ipynb
>透過optuna預測之超參數優化方法，即Tree Parzen Estimator找出超參數的最優解。

### backtest.ipynb
>針對不同的情況進行回測，算出其測試集期間的回報率，包含每次只買賣一股或依據信心度決定交易量

