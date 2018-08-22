import torch
import numpy as np
from sklearn.metrics import accuracy_score,precision_score, recall_score, f1_score

def batch_metrics(batch_gt_vector,
                  batch_pred_vector,
                  threshold=0.5,
                  f1_only=False):
    
    if type(batch_gt_vector) == torch.Tensor:
        batch_gt_vector = batch_gt_vector.detach().cpu().numpy()
    if type(batch_pred_vector) == torch.Tensor:
        batch_pred_vector = batch_pred_vector.detach().cpu().numpy()
    
    metric_list = []
    
    for (_,__) in zip(batch_gt_vector,batch_pred_vector):
        metrics = calculate_precision_metrics(_,__,threshold=threshold,f1_only=f1_only)
        metric_list.append(metrics)

    return metric_list

def calculate_precision_metrics(gt_vector,
                                pred_vector,
                                threshold=0.5,
                                f1_only=False):
    
    assert type(gt_vector) == np.ndarray 
    assert type(pred_vector) == np.ndarray 
    
    pred_vector[pred_vector<0.5] = 0
    pred_vector[pred_vector>0] = 1
    
    if f1_only==False:
        f1 = f1_score(gt_vector,pred_vector, average='macro')
        acc =  accuracy_score(gt_vector,pred_vector)
        pr = precision_score(gt_vector,pred_vector, average='macro')
        re = recall_score(gt_vector,pred_vector, average='macro') 
        return  acc,f1,pr,re
    else:
        f1 = f1_score(gt_vector,pred_vector, average='macro')
        return  [f1] 