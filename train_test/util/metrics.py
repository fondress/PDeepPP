import torch
import numpy as np
from sklearn.metrics import roc_auc_score, auc, precision_recall_curve

class Config:
    def __init__(self):
        self.alpha = 1.0
        self.num_class = 2

def calculate_metrics(logits, labels):
    labels = labels.int()
    probs = torch.sigmoid(logits)
    preds = torch.round(probs)
    
    tp = (preds * labels).sum().float()
    tn = ((1-preds)*(1-labels)).sum().float()
    fp = (preds*(1-labels)).sum().float()
    fn = ((1-preds)*labels).sum().float()
    
    acc = (tp + tn) / (tp + tn + fp + fn)
    auc = roc_auc_score(labels.cpu(), probs.cpu())
    precision, recall, _ = precision_recall_curve(labels.cpu(), probs.cpu())
    pr_auc = auc(recall, precision)
    bacc = (tp/(tp+fn) + tn/(tn+fp))/2
    sn = tp/(tp+fn)
    sp = tn/(tn+fp)
    mcc = (tp*tn - fp*fn)/torch.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    
    return (acc.item(), auc, bacc.item(), sn.item(), 
            sp.item(), mcc.item(), pr_auc)

def get_val_loss(logits, labels, criterion, lambda_, alpha=1.0):
    bce_loss = criterion(logits, labels.view(-1))
    probs = torch.sigmoid(logits)
    
    ent = - (probs*torch.log(probs+1e-12) + 
            (1-probs)*torch.log(1-probs+1e-12)).mean()
    cond_ent = - (probs*torch.log(probs+1e-12)).mean()
    
    reg_loss = alpha * (ent - cond_ent)
    return lambda_*bce_loss + (1-lambda_)*reg_loss