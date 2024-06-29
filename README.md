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
> WeightedSAGEConv : 繼承 torch_geometric.nn.conv 的 MessagePassing，與原生SAGEConv不同之處在於message在聚合鄰居節點訊息時，加入了權重
> SAGEWEIGHT : 
