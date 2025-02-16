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
import torch.optim as optim
from glob import glob

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
warnings.filterwarnings('ignore', category=UserWarning)

esm_array = [1] 
lambda_array =[0.97, 0.98, 0.99, 1]
ptm_type = "Deep-Kcr"

learning_rate = 0.001
batch_size = 32
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
        f'/data/liuyuhuan/zhaijx/data/{ptm_type}/{esm_ratio}/train_combined_representations',
        f'/data/liuyuhuan/zhaijx/data/{ptm_type}/{esm_ratio}/train_labels'
    )

    test_dataset = CustomDataset(
        f'/data/liuyuhuan/zhaijx/data/{ptm_type}/{esm_ratio}/test_combined_representations',
        f'/data/liuyuhuan/zhaijx/data/{ptm_type}/{esm_ratio}/test_labels'
    )

    train_size = int(0.9 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    class ConvNet1D(nn.Module):
        def __init__(self):
            super(ConvNet1D, self).__init__()
            # 输入层（假设输入形状为 [batch_size, 35, 1280]，需要在 forward 中进行 permute）
            self.conv1 = nn.Conv1d(in_channels=1280, out_channels=32, kernel_size=3, stride=1, padding=0)
            self.pool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
            self.dropout = nn.Dropout(0.75)
            self.fc1 = nn.Linear(32 * 14, 32)  # 修改后的 Conv1D 输出尺寸为 (batch_size, 32, 14)
            self.fc2 = nn.Linear(32, 16)
            self.fc3 = nn.Linear(16, 8)
            self.fc4 = nn.Linear(8, 1)
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            # 调整输入维度：从 [batch_size, 35, 1280] -> [batch_size, 1280, 35]
            x = x.permute(0, 2, 1)
            
            # 通过卷积层
            x = self.conv1(x)
            x = torch.relu(x)
            
            # 池化
            x = self.pool(x)
            
            # 展平
            x = x.view(x.size(0), -1)  # Flatten
            
            # Dropout
            x = self.dropout(x)
            
            # 全连接层
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            x = torch.relu(self.fc3(x))
            
            # 输出层
            x = self.sigmoid(self.fc4(x))
            x = x.squeeze(1)
            
            return x

    device = torch.device("cuda")

    sample_inputs, _ = next(iter(train_loader))
    input_row = sample_inputs.size(1)  # 通常是序列长度
    input_col = sample_inputs.size(2)
    predict_module = ConvNet1D().to(device).float()

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
        logits = torch.sigmoid(logits)
        ent = -(logits * torch.log(logits + 1e-12) + (1 - logits) * torch.log(1 - logits + 1e-12)).mean()
        cond_ent = -(logits * torch.log(logits + 1e-12)).mean()
        regularization_loss = parameters.alpha * ent - parameters.alpha * cond_ent

        weighted_loss = lambda_ * bce_loss + (1 - lambda_) * regularization_loss
        return weighted_loss

    criterion = nn.BCEWithLogitsLoss().to(device)
    optimizer = optim.Adam(predict_module.parameters(), lr=learning_rate)
    torch.backends.cudnn.benchmark = True
    # 保存结果的路径
    save_dir = f'/data/liuyuhuan/zhaijx/model/musite/{ptm_type}'
    os.makedirs(save_dir, exist_ok=True)

    # output_save_dir = f'/data/liuyuhuan/zhaijx/output/musite/{ptm_type}/PDeePPP_{esm_ratio}.txt'

    # # 提取目录路径
    # output_directory = os.path.dirname(output_save_dir)

    # # 确保父目录存在
    # if not os.path.exists(output_directory):
    #     os.makedirs(output_directory)
        
    # with open(output_save_dir, 'a') as file:
    #     file.write(f"esm_ratio: {esm_ratio}\n")
        
    # with open(output_save_dir, 'a') as file:
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
        save_path = os.path.join(save_dir, f'{ptm_type}_RE_{lambda_}_esm_{esm_ratio}.pth')
        
        # 重置模型和优化器
        predict_module = ConvNet1D().to(device).float()
        optimizer = optim.Adam(predict_module.parameters(), lr=learning_rate)
        best_metrics = {'accuracy': 0, 'auc': 0, 'bacc': 0, 'sn': 0, 'sp': 0, 'mcc': 0, 'pr_auc': 0}
        best_epoch = 0
        early_stopping_counter = 0


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
            
            