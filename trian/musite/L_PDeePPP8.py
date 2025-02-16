import pandas as pd
import torch
import torch.nn as nn
import os
import warnings
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, auc, precision_recall_curve
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from glob import glob

os.environ["CUDA_VISIBLE_DEVICES"] = "7"
warnings.filterwarnings('ignore', category=UserWarning)
esm_array = [0.9,0.95,1] 
lambda_array = [0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 1]
ptm_type = "SUMOylation_K"

learning_rate = 0.001
batch_size = 128
num_epochs = 100
early_stopping_patience = 20
# 自定义数据集类
for esm_ratio in esm_array:
    class CustomDataset(Dataset):
        def __init__(self, data_prefix, label_prefix):
            data_files = sorted(glob(f"{data_prefix}_*.npy"))
            label_files = sorted(glob(f"{label_prefix}_*.npy"))

            assert len(data_files) == len(label_files), "数据文件和标签文件的数量不匹配。"

            self.data = np.concatenate([np.load(file) for file in data_files], axis=0)
            self.labels = np.concatenate([np.load(file) for file in label_files], axis=0)

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            X = torch.tensor(self.data[idx], dtype=torch.float32)
            y = torch.tensor(self.labels[idx], dtype=torch.float32)
            return X, y

    train_dataset = CustomDataset(
        f'/data/liuyuhuan/zhaijx/weight/musite/{ptm_type}/{esm_ratio}/train_combined_representations',
        f'/data/liuyuhuan/zhaijx/weight/musite/{ptm_type}/{esm_ratio}/train_labels'
    )

    test_dataset = CustomDataset(
        f'/data/liuyuhuan/zhaijx/weight/musite/{ptm_type}/{esm_ratio}/test_combined_representations',
        f'/data/liuyuhuan/zhaijx/weight/musite/{ptm_type}/{esm_ratio}/test_labels'
    )

    train_size = int(0.9 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    class SelfAttentionGlobalFeatures(nn.Module):
        def __init__(self, input_size, output_size, num_heads=8):
            super(SelfAttentionGlobalFeatures, self).__init__()
            self.self_attention = nn.MultiheadAttention(embed_dim=input_size, num_heads=num_heads, batch_first=True)
            self.fc1 = nn.Linear(input_size, 256)
            self.fc2 = nn.Linear(256, output_size)
            self.layer_norm = nn.LayerNorm(input_size)
            self.dropout = nn.Dropout(0.3)

        def forward(self, x):
            attn_output, _ = self.self_attention(x, x, x)
            x = self.layer_norm(x + attn_output)
            x = self.fc1(x)
            x = self.dropout(x)
            x = self.fc2(x)
            return x

    class TransConv1d(nn.Module):
        def __init__(self, input_size, output_size):
            super(TransConv1d, self).__init__()
            self.self_attention_global_features = SelfAttentionGlobalFeatures(input_size, output_size)
            self.transformer_encoder = nn.TransformerEncoderLayer(d_model=output_size, nhead=8, dim_feedforward=512, dropout=0.3, batch_first=True)
            self.transformer = nn.TransformerEncoder(self.transformer_encoder, num_layers=4)
            self.fc1 = nn.Linear(output_size, output_size)
            self.fc2 = nn.Linear(output_size, output_size)
            self.layer_norm = nn.LayerNorm(output_size)

        def forward(self, x):
            x = self.self_attention_global_features(x)
            residual = x
            x = self.transformer(x)
            x = self.fc1(x)
            residual = x
            x = self.fc2(x)
            x = self.layer_norm(x + residual)
            return x

    class PosCNN(nn.Module):
        def __init__(self, input_size, output_size, use_position_encoding=True):
            super(PosCNN, self).__init__()
            self.use_position_encoding = use_position_encoding
            self.conv1d = nn.Conv1d(in_channels=input_size, out_channels=64, kernel_size=3, padding=1)
            self.relu = nn.ReLU()
            self.global_pooling = nn.AdaptiveAvgPool1d(1)
            self.fc = nn.Linear(64, output_size)
            
            if self.use_position_encoding:
                self.position_encoding = nn.Parameter(torch.zeros(64, input_size))

        def forward(self, x):
            x = x.permute(0, 2, 1)
            x = self.conv1d(x)
            x = self.relu(x)
            
            if self.use_position_encoding:
                seq_len = x.size(2)
                pos_encoding = self.position_encoding[:, :seq_len].unsqueeze(0)
                x = x + pos_encoding
            
            x = self.global_pooling(x)
            x = x.squeeze(-1)
            x = self.fc(x)
            return x

    class PredictModule(nn.Module):
        def __init__(self, input_size, output_size):
            super(PredictModule, self).__init__()
            self.transformer = TransConv1d(input_size, output_size)
            self.cnn = PosCNN(input_size, output_size)
            self.cnn_layers = nn.Sequential(
                nn.Conv1d(in_channels=output_size*2, out_channels=32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.AdaptiveMaxPool1d(1),
                nn.Dropout(0.15),
                nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.AdaptiveMaxPool1d(1),
                nn.Dropout(0.15),
                nn.Flatten(),
                nn.Linear(64, 1)
            )

        def forward(self, x):
            transformer_output = self.transformer(x)
            cnn_output = self.cnn(x)
            cnn_output = cnn_output.unsqueeze(1).expand(-1, transformer_output.size(1), -1)
            combined_output = torch.cat([transformer_output, cnn_output], dim=2)
            combined_output = combined_output.permute(0, 2, 1)
            output = self.cnn_layers(combined_output)
            output = output.squeeze(1)
            return output

    device = torch.device("cuda")

    input_size = 1280
    output_size = 128
    predict_module = PredictModule(input_size, output_size).to(device)

    if torch.cuda.device_count() > 1:
        predict_module = torch.nn.DataParallel(predict_module)

    predict_module.to(device).float()

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
    optimizer = torch.optim.Adam(predict_module.parameters(), lr=learning_rate)
    torch.backends.cudnn.benchmark = True

    # 保存结果的路径
    save_dir = f'/data/liuyuhuan/zhaijx/model/musite/{ptm_type}'
    os.makedirs(save_dir, exist_ok=True)

    def calculate_metrics(logits, labels):
        if isinstance(logits, np.ndarray):
            logits = torch.tensor(logits)

    # 同样检查 labels 是否为 numpy.ndarray
        if isinstance(labels, np.ndarray):
            labels = torch.tensor(labels)
            
        preds = torch.round(torch.sigmoid(logits))
        labels = labels.int()
        
        tp = (preds * labels).sum().to(torch.float32)
        tn = ((1 - preds) * (1 - labels)).sum().to(torch.float32)
        fp = (preds * (1 - labels)).sum().to(torch.float32)
        fn = ((1 - preds) * labels).sum().to(torch.float32)
        
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        auc_score = roc_auc_score(labels.cpu().numpy(), logits.cpu().numpy())
        precision, recall, _ = precision_recall_curve(labels.cpu().numpy(), logits.cpu().numpy())
        pr_auc = auc(recall, precision)
        bacc = (tp / (tp + fn) + tn / (tn + fp)) / 2
        sn = tp / (tp + fn)
        sp = tn / (tn + fp)
        mcc = (tp * tn - fp * fn) / torch.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        
        return accuracy.item(), auc_score, bacc.item(), sn.item(), sp.item(), mcc.item(), pr_auc

    for lambda_ in lambda_array:
        predict_module = PredictModule(input_size, output_size).to(device)
        optimizer = torch.optim.Adam(predict_module.parameters(), lr=learning_rate)
        best_metrics = {'accuracy': 0, 'auc': 0, 'bacc': 0, 'sn': 0, 'sp': 0, 'mcc': 0, 'pr_auc': 0}
        best_epoch = 0
        early_stopping_counter = 0

        # 保存模型权重的路径
        save_path = os.path.join(save_dir, f'{ptm_type}_{lambda_}_esm_{esm_ratio}.pth')

        for epoch in range(num_epochs):
            predict_module.train()
            train_loss = 0.0

            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = predict_module(inputs)
                loss = get_val_loss(outputs, labels, criterion, lambda_)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * inputs.size(0)

            # 验证阶段
            predict_module.eval()
            val_loss = 0.0
            all_val_logits = []
            all_val_labels = []

            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = predict_module(inputs)
                    loss = get_val_loss(outputs, labels, criterion, lambda_)
                    val_loss += loss.item() * inputs.size(0)
                    all_val_logits.append(outputs)
                    all_val_labels.append(labels)

            all_val_logits = torch.cat(all_val_logits)
            all_val_labels = torch.cat(all_val_labels)

            val_acc, val_auc_score, val_bacc, val_sn, val_sp, val_mcc, val_pr_auc = calculate_metrics(all_val_logits, all_val_labels)

            # 保存最佳模型权重
            if val_acc > best_metrics['accuracy']:
                best_metrics['accuracy'] = val_acc
                best_epoch = epoch
                early_stopping_counter = 0

                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': predict_module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_metrics': best_metrics,
                    'lambda': lambda_,
                    'lr': learning_rate,
                    'early_stopping_patience': early_stopping_patience,
                    'esm_ratio': esm_ratio,
                    'pos_embedding':True,
                }, save_path)
            else:
                early_stopping_counter += 1

            if early_stopping_counter >= early_stopping_patience:
                break

    