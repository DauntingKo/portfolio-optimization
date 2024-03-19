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
from model import SAGE,FocalLoss
from yf_dataset import getInput



def startGNN(startLr, withGold, withOil, numNeighbors, lossFunction, withMacdSignal=False, macdParamOptimize=False, gamma=2, withAlpha=True):
    seed = 42
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    train_idx, eval_idx, test_idx, weight, yfdata, gnnInputData = getInput(False, False, withMacdSignal=withMacdSignal, macdParamOptimize=macdParamOptimize)

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
    
    def creterion(lossFunction , weight, out, y_true, gamma, withAlpha):
        loss = 1
        match lossFunction:
            case 'CrossEntrophyLoss':
                loss = F.nll_loss(out, torch.reshape(y_true, (-1,)))
            case 'FocalLoss':
                alpha = None
                if withAlpha:
                    alpha=torch.tensor(weight)
                focalLoss = FocalLoss(gamma=gamma,alpha=alpha,reduction='mean')
                loss = focalLoss(out, torch.reshape(y_true, (-1,)))

        return loss
    
    title = lossFunction
    if lossFunction == 'FocalLoss':
        title += f' (gamma={gamma} withAlpha={withAlpha})'
    if withGold: title += ' with gold'
    if withOil: title += ' with oil'
    if withMacdSignal: title += ' with MACD Signal'
    if macdParamOptimize: title += '(macdParamOptimize)'
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
    test_precision = []
    test_recall = []
    test_f1_score = []
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
            loss = creterion(lossFunction, weight, out, batch_y, gamma, withAlpha)
            loss.backward()
            optimizer.step()
            batch_loss += float(loss)
            total_correct += int(out.argmax(dim=-1).eq(batch_y).sum())
            pbar.update(batch.batch_size)
        pbar.close()
        # loss = batch_loss / len(train_loader)
        approx_acc = total_correct / train_idx.size(0)
        train_acc, val_acc, test_acc, var, y_true, out = test(model, device)
        train_loss = creterion(lossFunction, weight, out[train_idx], torch.reshape(y_true[train_idx], (-1,)), gamma, withAlpha)  
        eval_loss = creterion(lossFunction, weight, out[eval_idx], torch.reshape(y_true[eval_idx], (-1,)), gamma, withAlpha)
        test_loss = creterion(lossFunction, weight, out[test_idx], torch.reshape(y_true[test_idx], (-1,)), gamma, withAlpha)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        test_accs.append(test_acc)
        total_train_loss.append(train_loss)
        total_eval_loss.append(eval_loss)
        total_test_loss.append(test_loss)
        scheduler.step(eval_loss)

        precisionScore = precision_score(y_true[test_idx], out[test_idx].argmax(dim=-1, keepdim=True), average='micro')
        recallScore = recall_score(y_true[test_idx], out[test_idx].argmax(dim=-1, keepdim=True), average='micro')
        f1Score = f1_score(y_true[test_idx], out[test_idx].argmax(dim=-1, keepdim=True), average='micro')
        test_precision.append(precisionScore)
        test_recall.append(recallScore)
        test_f1_score.append(f1Score)
        cm = confusion_matrix(y_true[test_idx], out[test_idx].argmax(dim=-1, keepdim=True))

        print(f'TrainAcc: {train_acc:.4f}, ValAcc: {val_acc:.4f}, TestAcc: {test_acc:.4f}, trainLoss: {train_loss:.4f}, evalLoss: {eval_loss:.4f}, testLoss: {test_loss:.4f}')
        #early_stopping
        early_stopping(test_loss, model, confusion_matrix=cm, epoch=epoch, test_acc=test_acc, test_loss=test_loss)
        if early_stopping.early_stop:
            print("Early stopping")
            break 

    plt.figure()
    plt.plot(np.arange(1, len(train_accs)+1), train_accs, label='Train Accuracy', marker='o')
    # plt.plot(np.arange(0, len(val_accs)), val_accs, label='Validation Accuracy', marker='o')
    plt.plot(np.arange(1, len(test_accs)+1), test_accs, label='Test Accuracy', marker='o')

    plt.annotate(f'epoch: {early_stopping.best_epoch}, test_accs: {test_accs[early_stopping.best_epoch-1]:.2f}\ntest_precision: {test_precision[early_stopping.best_epoch-1]:.2f}, test_recall: {test_recall[early_stopping.best_epoch-1]:.2f}, test_f1_score: {test_f1_score[early_stopping.best_epoch-1]:.2f}', 
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

    plt.figure()
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
    
    plt.figure()
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
    plt.close(fig)
    return early_stopping.best_test_acc, early_stopping.best_test_loss


    
