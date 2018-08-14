import torch
import numpy as np
from sklearn.metrics import accuracy_score,precision_score, recall_score, f1_score

def batch_metrics(batch_gt_vector,
                  batch_pred_vector,
                  threshold=0.5,
                  f1_only=False):
    
    if type(batch_gt_vector) == torch.Tensor:
        batch_gt_vector = batch_gt_vector.cpu().numpy()
    if type(batch_pred_vector) == torch.Tensor:
        batch_pred_vector = batch_pred_vector.cpu().numpy()
    
    metric_list = []
    
    for (_,__) in zip(batch_gt_vector,batch_pred_vector):
        metrics = calculate_precision_metrics(_,__,threshold=threshold,f1_only=f1_only)
        metric_list.append(metrics)

    return metric_list