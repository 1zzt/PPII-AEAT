import sklearn.metrics as metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score, confusion_matrix,roc_auc_score,matthews_corrcoef
from sklearn.metrics import confusion_matrix,mean_squared_error,mean_absolute_error
from scipy import stats
import numpy as np 

def compute_cls_metrics(y_true, y_prob):
    
    y_pred = np.array(y_prob) > 0.5

    roc_auc = roc_auc_score(y_true, y_prob)

    F1 = f1_score(y_true, y_pred, average = 'binary')
   
    mcc = matthews_corrcoef(y_true, y_pred)
    
    acc = accuracy_score(y_true, y_pred)
 
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    SE = tp / (tp + fn)
    SP = tn / (tn + fp)
    return   F1, roc_auc, mcc, acc, SE, SP,  tn, fp, fn, tp


def compute_reg_metrics(y_true, y_pred):

    y_true = y_true.flatten().astype(float)
    y_pred = y_pred.flatten().astype(float)

    tau, _ = stats.kendalltau(y_true, y_pred)
    rho, _ =stats.spearmanr(y_true, y_pred)
    r, _ =stats.pearsonr(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    mae = mean_absolute_error(y_true, y_pred)
    return tau, rho, r, rmse, mae