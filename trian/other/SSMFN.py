import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
import numpy as np
import os

# 设置设备
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

esm_array = [0.9,1] 
lambda_array = [0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 1]
type = "SSMFN"
learning_rate = 0.001
batch_size = 32
num_epochs = 100
early_stopping_patience = 20
# 调整后的模型 (来自 ssmfn.py)
class ModelNeoModified(nn.Module):
    def __init__(self):
        super(ModelNeoModified, self).__init__()

        # LSTM部分
        self.lstm1 = nn.LSTM(1280, 64, 1, batch_first=True)
        self.lstm2 = nn.LSTM(64, 64, 1, batch_first=True)
        self.batchnormL1 = nn.BatchNorm1d(64)
        self.batchnormL2 = nn.BatchNorm1d(64)
        self.dropL1 = nn.Dropout(p=0.5)
        self.dropL2 = nn.Dropout(p=0.5)
        self.linearL1 = nn.Linear(64, 32)

        # CNN部分
        self.conv1 = nn.Conv2d(1, 64, (3, 1280), padding=(1, 0))  # 输入通道=1, 输出通道=64, 核大小=(3, 1280)
        self.conv2 = nn.Conv2d(64, 64, (3, 1), padding=(1, 0))
        self.conv3 = nn.Conv2d(64, 64, (3, 1), padding=(1, 0))
        self.conv4 = nn.Conv2d(64, 64, (3, 1), padding=(1, 0))
        self.batchnormC1 = nn.BatchNorm2d(64)
        self.batchnormC2 = nn.BatchNorm2d(64)
        self.batchnormC3 = nn.BatchNorm2d(64)
        self.batchnormC4 = nn.BatchNorm2d(64)
        self.dropC1 = nn.Dropout(p=0.5)
        self.dropC2 = nn.Dropout(p=0.5)
        self.dropC3 = nn.Dropout(p=0.5)
        self.dropC4 = nn.Dropout(p=0.5)
        
        self.flatten = nn.Flatten()
        self.linearC1 = nn.Linear(64 * 19, 32)  # CNN输出为64通道，19个时间步

        # 最后线性层
        self.linear2 = nn.Linear(32, 2)
        self.linear3 = nn.Linear(2, 1)

    def forward(self, x):
        # LSTM部分
        lstm_out, _ = self.lstm1(x)  # 输入：(batch_size, 19, 1280)
        lstm_out = self.dropL1(lstm_out)
        lstm_out, _ = self.lstm2(lstm_out)
        lstm_out = self.dropL2(lstm_out)
        lstm_out = self.linearL1(lstm_out[:, -1, :])  # 取最后一个时间步的输出

        # CNN部分
        cnn_in = x.unsqueeze(1)  # 增加通道维度，输入：(batch_size, 1, 19, 1280)
        cnn_out = F.relu(self.conv1(cnn_in))
        cnn_out = self.dropC1(self.batchnormC1(cnn_out))
        cnn_out = F.relu(self.conv2(cnn_out))
        cnn_out = self.dropC2(self.batchnormC2(cnn_out))
        cnn_out = F.relu(self.conv3(cnn_out))
        cnn_out = self.dropC3(self.batchnormC3(cnn_out))
        cnn_out = F.relu(self.conv4(cnn_out))
        cnn_out = self.dropC4(self.batchnormC4(cnn_out))

        cnn_out = self.flatten(cnn_out)  # 展平
        cnn_out = self.linearC1(cnn_out)

        # LSTM和CNN输出结合
        combined = lstm_out + cnn_out

        # 最终输出
        y_pred1 = self.linear2(combined)
        y_pred = self.linear3(y_pred1)
        return y_pred

# 数据加载
for esm_ratio in esm_array:

    # 加载数据
    train_representations = np.load(f'/data/liuyuhuan/zhaijx/data/{type}/{esm_ratio}/train_combined_representations.npy')
    train_labels = np.load(f'/data/liuyuhuan/zhaijx/data/{type}/{esm_ratio}/train_labels.npy')
    test_representations = np.load(f'/data/liuyuhuan/zhaijx/data/{type}/{esm_ratio}/test_combined_representations.npy')
    test_labels = np.load(f'/data/liuyuhuan/zhaijx/data/{type}/{esm_ratio}/test_labels.npy')

    # 转换为张量
    X_train = torch.tensor(train_representations, dtype=torch.float32)
    y_train = torch.tensor(train_labels, dtype=torch.float32)
    X_test = torch.tensor(test_representations, dtype=torch.float32)
    y_test = torch.tensor(test_labels, dtype=torch.float32)

    # 创建数据集
    train_dataset = TensorDataset(X_train, y_train)
    val_size = int(0.1 * len(train_dataset))
    train_size = len(train_dataset) - val_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)

    # 初始化模型
    predict_module = ModelNeoModified().to(device).float()

    class Config:
        def __init__(self):
            self.alpha = 1.0
            self.num_class = 2

    parameters = Config()

    def get_cond_entropy(probs):
        cond_ent = -(probs * torch.log(probs + 1e-12)).sum(1).mean(0, keepdim=True)
        return cond_ent

    # 自定义损失
    def get_val_loss(logits, label, criterion, lambda_):
        bce_loss = criterion(logits, label.view(-1, 1))  # 修改标签维度以匹配logits
        logits = torch.sigmoid(logits)
        ent = -(logits * torch.log(logits + 1e-12) + (1 - logits) * torch.log(1 - logits + 1e-12)).mean()
        cond_ent = -(logits * torch.log(logits + 1e-12)).mean()
        regularization_loss = parameters.alpha * ent - parameters.alpha * cond_ent

        weighted_loss = lambda_ * bce_loss + (1 - lambda_) * regularization_loss
        return weighted_loss

    criterion = nn.BCEWithLogitsLoss().to(device)
    optimizer = optim.Adam(predict_module.parameters(), lr=learning_rate)
    torch.backends.cudnn.benchmark = True

    # 计算评估指标
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
    save_dir = f'/data/liuyuhuan/zhaijx/model/{type}'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 训练和验证循环
    for lambda_ in lambda_array:
        predict_module = ModelNeoModified().to(device).float()
        optimizer = optim.Adam(predict_module.parameters(), lr=learning_rate)
        best_metrics = {
            'accuracy': {'value': 0}, 'auc': {'value': 0}, 'bacc': {'value': 0},
            'sn': {'value': 0}, 'sp': {'value': 0}, 'mcc': {'value': 0}, 'pr_auc': {'value': 0}
        }
        best_epoch = 0
        early_stopping_counter = 0

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
                early_stopping_counter = 0

                # 保存模型权重及超参数
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': predict_module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_metrics': best_metrics,
                    'lambda': lambda_,
                    'lr': learning_rate,
                    'early_stopping_patience': early_stopping_patience,
                    'esm_ratio': esm_ratio,
                }, os.path.join(save_dir, f'{type}_RE_{lambda_}_esm_{esm_ratio}.pth'))

            else:
                early_stopping_counter += 1

            if early_stopping_counter >= early_stopping_patience:
                break