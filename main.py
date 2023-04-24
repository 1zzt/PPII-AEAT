import numpy as np
import os 
import pandas as pd 
from molfeaturizer import *
from metrics import compute_cls_metrics, compute_reg_metrics
from prettytable import PrettyTable
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torch 
import torch.nn as nn
from sklearn import preprocessing
from adan import * 
from model import *
import random
import argparse

parser = argparse.ArgumentParser(description='PPII-AEAT')

parser.add_argument('--dataset', type=str, default='bcl2_bak', 
                    help='dataset: bcl2_bak; bromodomain_histone; cyclophilins; hif1a_p300; integrins; ledgf_in; lfa_icam; mdm2_p53; xiap_smac')
parser.add_argument('--task', type=str, default='regression',
                    help="prediction task: classification or regression")
parser.add_argument('--num_epochs', type=int, default=100,
                    help='num_epochs')
parser.add_argument('--batch_size', type=int, default=32,
                    help="batch size")
parser.add_argument('--lr', type=int, default=1e-3,
                    help="learning rate")
parser.add_argument('--gpu', type=int, default=0,
                    help="train on which cuda device")


args = parser.parse_args()
dataset_name = args.dataset
task_name = args.task_name
num_epochs = args.num_epochs
batch_size = args.batch_size
lr = args.lr

data_path = './Datasets/'

def set_seed(seed):
    """
    Set of random seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic =True
    
set_seed(24)

def run_an_eval_epoch(model, data_loader, task_name, criterion, device):
    model.eval()
    running_loss = AverageMeter()

    with torch.no_grad():
        preds =  torch.Tensor()
        trues = torch.Tensor()

        for batch_id, (*x, y) in tqdm(enumerate(data_loader)):
            for i in range(len(x)):
                x[i] = x[i].float().to(device)

            _, _, _ ,_, _, _ , logits =  model(*x)
               
            if task_name == 'classification':
                
                y = y.float().to(device)
                loss = criterion(logits.view(-1), y.view(-1))
                logits = torch.sigmoid(logits)
            else:
                y = y.float().to(device)
                loss = criterion(logits.view(-1), y.view(-1))
            preds = torch.cat((preds, logits.cpu()), 0)
            trues = torch.cat((trues, y.view(-1, 1).cpu()), 0)

            running_loss.update(loss.item(), y.size(0))

        preds, trues = preds.numpy().flatten(), trues.numpy().flatten()

    val_loss =  running_loss.get_average()
 
    return preds, trues, val_loss 


def run_a_train_epoch(epoch, model, data_loader, optimizer, scheduler, loss_criterion, device ):
        

    n = epoch + 1
     
    tbar = tqdm(enumerate(data_loader), total=len(data_loader))
    for id,  (*x, y) in tbar:
        for i in range(len(x)):
            x[i] = x[i].float().to(device)
        
        
        fp = x[0]
        dp = x[1] 
        y = y.float().to(device)

        ae1_fp, ae2_fp, ae2ae1_fp, ae1_dp, ae2_dp, ae2ae1_dp, output =  model(*x)
        loss = loss_criterion(output.view(-1), y.view(-1))  

        fp_loss = ae_loss_function(fp, ae1_fp, ae2_fp, ae2ae1_fp, n)
        dp_loss = ae_loss_function(dp, ae1_dp, ae2_dp, ae2ae1_dp, n)
        loss +=  10 * fp_loss
        loss +=  dp_loss


        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)

        optimizer.step()

    tqdm.write(f'Epoch {epoch}, \tLoss = { loss.item() } \tFPLoss = { fp_loss.item() } \tDPLoss = { dp_loss.item() }  ')
    
        

def ptable_to_csv(table, filename, headers=True):
    # Save PrettyTable results to a CSV file.

    raw = table.get_string()
    data = [tuple(filter(None, map(str.strip, splitline)))
            for line in raw.splitlines()
            for splitline in [line.split('|')] if len(splitline) > 1]
    if table.title is not None:
        data = data[1:]
    if not headers:
        data = data[1:]
    with open(filename, 'w') as f:
        for d in data:
            f.write('{}\n'.format(','.join(d)))

class FPDPDataset(Dataset):
    def __init__(self, f_features, d_features, labels):
        self.labels =  labels 
        self.length = len(self.labels)
        self.f_features =  f_features
        self.d_features =  d_features
         
             
    def __getitem__(self, idx):
        label = self.labels[idx]
        fp = self.f_features[idx]
        dp = self.d_features[idx]
        return torch.FloatTensor(fp),  torch.FloatTensor(dp),   torch.FloatTensor([label]) 

    def __len__(self):
        return self.length

def main():
    print('**********start**********')
    print('\n')
    print('Arguments:')
    print('Datasets: '+ dataset_name)
    print('Datasets: '+ dataset_name)
    print('Task: '+ task_name)
    print('Epochs: '+ str(num_epochs))
    print('Batch size: '+ str(batch_size))
    print('Learning rate: '+ str(lr))
    fp_size = 2048
    ecfp = MorganFPFeaturizer(fp_size=fp_size, radius=2, use_counts=True, use_features=False)
    rdkit_norm_fp = lambda smiles: ecfp.transform(smiles)

    mordred_norm_dp = lambda smiles: mordred_fp(smiles)

    
    file_path = os.path.join(data_path, task_name)
    
    train_df = pd.read_csv(os.path.join(file_path, dataset_name+'_train.csv'))
    test_df = pd.read_csv(os.path.join(file_path, dataset_name+'_test.csv'))

    train_smiles = train_df[train_df.columns[0]].values
    train_labels = train_df[train_df.columns[-1]].values
    
    test_smiles = test_df[test_df.columns[0]].values
    test_labels = test_df[test_df.columns[-1]].values

    test_features_fp = np.array(rdkit_norm_fp(test_smiles))
    test_features_dp_all = np.array(mordred_norm_dp(test_smiles))


    if task_name == 'classification':
        t_tables = PrettyTable(['epoch', 'MCC', 'F1', 'AUC'])
    else:
        t_tables = PrettyTable(['epoch','R', 'Kendall', 'Spearman', 'RMSE', 'MAE' ])

    t_tables.float_format = '.3'  

    train_features_fp = np.array(rdkit_norm_fp(train_smiles))
    train_features_dp_all = np.array(mordred_norm_dp(train_smiles))


    scaler = preprocessing.StandardScaler().fit(train_features_dp_all)
    train_features_dp = scaler.transform(train_features_dp_all)
    test_features_dp = scaler.transform(test_features_dp_all)

    
    
    train_ds = FPDPDataset(train_features_fp,train_features_dp, train_labels)
    test_ds = FPDPDataset(test_features_fp, test_features_dp, test_labels)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)
        
    device = torch.device('cuda:'+str(args.gpu) if torch.cuda.is_available() else "cpu") 

    model = AES_FP_DP(n_inputs_fp=train_features_fp.shape[1], n_inputs_dp = train_features_dp.shape[1]).to(device)

    optimizer = Adan(
            model.parameters(),
            # optimizer_grouped_parameters,
            lr = lr,                  # learning rate (can be much higher than Adam, up to 5-10x)
            betas = (0.02, 0.08, 0.001), # beta 1-2-3 as described in paper - author says most sensitive to beta3 tuning
            weight_decay = 0.02        # weight decay 0.02 is optimal per author
        )

    if task_name == 'classification':
        
        loss_criterion = nn.BCEWithLogitsLoss()
    else:
        loss_criterion = nn.MSELoss()
    
    print('**********Training**********') 
    print('\n')
    
    # trained with all training set
    # 10-fold cross-validation code is omitted (used for parameter optimization)
    # kf = KFold(n_splits=10, shuffle=True) 
    # for split, (train_index, valid_index) in enumerate(kf.split(train_labels)): 
    
    for epoch in tqdm(list(range(1, num_epochs+1))):
      
        run_a_train_epoch(epoch, model, train_loader, optimizer, None, loss_criterion, device)
    print('**********Test**********')
    test_pred, test_y, test_loss = run_an_eval_epoch(model, test_loader,task_name, loss_criterion, device)

    if task_name == 'classification':
        F1, roc_auc, mcc = compute_cls_metrics(test_y,test_pred)
        row = [ 'test', mcc, F1, roc_auc]
    else:
        tau, rho, r, rmse, mae =  compute_reg_metrics(test_y,test_pred)
        row = [ 'test', r, tau, rho, rmse, mae]
    
    t_tables.add_row(row)   
    print(t_tables)
    print(task_name, dataset_name)
        
if __name__ == '__main__':
    main()
    

