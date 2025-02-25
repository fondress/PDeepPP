import torch
import torch.nn as nn
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class Config:
    def __init__(self):
        self.alpha = 1.0 
        self.num_class = 2
        
parameters = Config()

def get_cond_entropy(probs):
    cond_ent = -(probs * torch.log(probs + 1e-12)).sum(1).mean(0, keepdim=True)
    return cond_ent

def get_val_loss(logits, label, criterion, lambda_):
    bce_loss = criterion(logits, label.view(-1))
    logits = torch.sigmoid(logits)  # Apply sigmoid to the output
    ent = -(logits * torch.log(logits + 1e-12) + (1 - logits) * torch.log(1 - logits + 1e-12)).mean()
    cond_ent = -(logits * torch.log(logits + 1e-12)).mean()
    regularization_loss = parameters.alpha * ent - parameters.alpha * cond_ent

    weighted_loss = lambda_ * bce_loss + (1 - lambda_) * regularization_loss
    return weighted_loss


criterion = nn.BCEWithLogitsLoss().to(device)