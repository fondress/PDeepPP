import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
from sklearn.metrics import roc_curve, roc_auc_score, auc, precision_recall_curve
import numpy as np
import os
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = "6"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

esm_array = [0.8] 
lambda_array =[0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 1]

learning_rate = 0.001
batch_size = 32
num_epochs = 100
early_stopping_patience = 20

for esm_ratio in esm_array:
    class ConvNet1D(nn.Module):
        def __init__(self):
            super(ConvNet1D, self).__init__()
            
            # 第一部分：三个1D卷积层 + Dropout + 平均池化
            self.conv1 = nn.Conv1d(in_channels=1280, out_channels=128, kernel_size=3, padding=1)
            self.conv2 = nn.Conv1d(128, 128, kernel_size=3, padding=1)
            self.conv3 = nn.Conv1d(128, 128, kernel_size=3, padding=1)
            self.dropout1 = nn.Dropout(0.4)
            self.pool1 = nn.AvgPool1d(kernel_size=2, stride=2)

            # 第二部分：三个1D卷积层 + Dropout + 平均池化
            self.conv4 = nn.Conv1d(128, 64, kernel_size=3, padding=1)
            self.conv5 = nn.Conv1d(64, 64, kernel_size=3, padding=1)
            self.conv6 = nn.Conv1d(64, 64, kernel_size=3, padding=1)
            self.dropout2 = nn.Dropout(0.4)
            self.pool2 = nn.AvgPool1d(kernel_size=2, stride=2)

            # 全连接层
            self.fc1 = nn.Linear(512, 256)  # 计算展平后的输出大小
            self.fc2 = nn.Linear(256, 256)
            self.fc3 = nn.Linear(256, 1)  # 二分类问题，输出一个值

        def forward(self, x):
            x = x.permute(0, 2, 1)
            # 第一部分：卷积 -> 激活 -> Dropout -> 池化
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = F.relu(self.conv3(x))
            x = self.dropout1(x)
            x = self.pool1(x)

            # 第二部分：卷积 -> 激活 -> Dropout -> 池化
            x = F.relu(self.conv4(x))
            x = F.relu(self.conv5(x))
            x = F.relu(self.conv6(x))
            x = self.dropout2(x)
            x = self.pool2(x)

            # 展平
            x = x.view(x.size(0), -1)  # Flatten the tensor

            # 全连接层
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x).squeeze(1)  # 因为是二分类问题，输出1个值，使用 squeeze(1) 去掉多余维度

            return x
    # 数据加载
    train_representations = np.load(f'/data/liuyuhuan/zhaijx/data/deepglut/{esm_ratio}/train_combined_representations.npy')
    train_labels = np.load(f'/data/liuyuhuan/zhaijx/data/deepglut/{esm_ratio}/train_labels.npy')
    test_representations = np.load(f'/data/liuyuhuan/zhaijx/data/deepglut/{esm_ratio}/test_combined_representations.npy')
    test_labels = np.load(f'/data/liuyuhuan/zhaijx/data/deepglut/{esm_ratio}/test_labels.npy')

    # 转换为张量
    X_train = torch.tensor(train_representations, dtype=torch.float32)
    y_train = torch.tensor(train_labels, dtype=torch.float32)
    X_test = torch.tensor(test_representations, dtype=torch.float32)
    y_test = torch.tensor(test_labels, dtype=torch.float32)
    
    train_dataset = TensorDataset(X_train, y_train)
    val_size = int(0.1 * len(train_dataset))
    train_size = len(train_dataset) - val_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
    
    # 初始化模型
    predict_module = ConvNet1D().to(device).float()  # 替换 MultiCNN 为 ConvNet

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

    def calculate_metrics(logits, labels):
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

    # 模型保存目录
    save_dir = f'/data/liuyuhuan/zhaijx/model/deepglut'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    
    for lambda_ in lambda_array:

            predict_module = ConvNet1D().to(device).float()
            optimizer = optim.Adam(predict_module.parameters(), lr=learning_rate)
            best_metrics = {
                'accuracy': {'value': 0}, 'auc': {'value': 0}, 'bacc': {'value': 0}, 
                'sn': {'value': 0}, 'sp': {'value': 0}, 'mcc': {'value': 0}, 'pr_auc': {'value': 0}
            }
            best_epoch = 0
            early_stopping_counter = 0

            # 训练和验证循环
            for epoch in range(num_epochs):
                # 训练阶段
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

                train_loss /= len(train_loader.dataset)

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

                val_loss /= len(val_loader.dataset)
                all_val_logits = torch.cat(all_val_logits)
                all_val_labels = torch.cat(all_val_labels)

                val_acc, val_auc_score, val_bacc, val_sn, val_sp, val_mcc, val_pr_auc = calculate_metrics(all_val_logits, all_val_labels)

                # 更新最佳验证指标
                if val_acc > best_metrics['accuracy']['value']:
                    best_metrics['accuracy']['value'] = val_acc
                    best_epoch = epoch
                    early_stopping_counter = 0  # 重置早停计数器

                    # 保存模型权重及超参数
                    torch.save({
                        'epoch': epoch + 1,
                        'model_state_dict': predict_module.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'best_metrics': best_metrics,
                        'lambda': lambda_,
                        'lr' : learning_rate,
                        'early_stopping_patience' : early_stopping_patience,
                        'esm_ratio': esm_ratio,
                    }, os.path.join(save_dir, f'deepglut_RE_{lambda_}_esm_{esm_ratio}.pth'))

                else:
                    early_stopping_counter += 1

                if early_stopping_counter >= early_stopping_patience:
                    break
