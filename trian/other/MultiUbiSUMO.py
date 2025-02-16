import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from glob import glob
import torch.optim as optim

# 设置GPU设备
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 超参数设置
esm_array = [0.9,1] 
lambda_array = [0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 1]
type = "MultiUbiSUMO"
learning_rate = 0.001
batch_size = 32
num_epochs = 100
early_stopping_patience = 20

for esm in esm_array:
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

    # 定义子网络
    class OneHotCNN(nn.Module):
        def __init__(self):
            super(OneHotCNN, self).__init__()
            self.conv1 = nn.Conv1d(in_channels=1280, out_channels=64, kernel_size=3, padding=1)
            self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
            self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
            self.fc1 = nn.Linear(128 * 16, 128)  # 输入长度为33，池化后为16

        def forward(self, x):
            x = x.permute(0, 2, 1)
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = self.pool(x)
            x = x.view(x.size(0), -1)
            x = F.relu(self.fc1(x))
            return x

    class OtherPropDNN(nn.Module):
        def __init__(self):
            super(OtherPropDNN, self).__init__()
            self.fc1 = nn.Linear(1280 * 33, 128)
            self.fc2 = nn.Linear(128, 128)

        def forward(self, x):
            x = x.view(x.size(0), -1)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            return x

    class PhysicoDNN(nn.Module):
        def __init__(self):
            super(PhysicoDNN, self).__init__()
            self.fc1 = nn.Linear(1280 * 33, 128)
            self.fc2 = nn.Linear(128, 128)

        def forward(self, x):
            x = x.view(x.size(0), -1)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            return x

    class HydroDNN(nn.Module):
        def __init__(self):
            super(HydroDNN, self).__init__()
            self.fc1 = nn.Linear(1280 * 33, 128)
            self.fc2 = nn.Linear(128, 128)

        def forward(self, x):
            x = x.view(x.size(0), -1)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            return x

    class AlphaTurnCNN(nn.Module):
        def __init__(self):
            super(AlphaTurnCNN, self).__init__()
            self.conv1 = nn.Conv1d(in_channels=1280, out_channels=64, kernel_size=3, padding=1)
            self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
            self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
            self.fc1 = nn.Linear(128 * 16, 128)  # 输入长度为33，池化后为16

        def forward(self, x):
            x = x.permute(0, 2, 1)
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = self.pool(x)
            x = x.view(x.size(0), -1)
            x = F.relu(self.fc1(x))
            return x

    class BetaPropCNN(nn.Module):
        def __init__(self):
            super(BetaPropCNN, self).__init__()
            self.conv1 = nn.Conv1d(in_channels=1280, out_channels=64, kernel_size=3, padding=1)
            self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
            self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
            self.fc1 = nn.Linear(128 * 16, 128)

        def forward(self, x):
            x = x.permute(0, 2, 1)
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = self.pool(x)
            x = x.view(x.size(0), -1)
            x = F.relu(self.fc1(x))
            return x

    class CompositionCNN(nn.Module):
        def __init__(self):
            super(CompositionCNN, self).__init__()
            self.conv1 = nn.Conv1d(in_channels=1280, out_channels=64, kernel_size=3, padding=1)
            self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
            self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
            self.fc1 = nn.Linear(128 * 16, 128)

        def forward(self, x):
            x = x.permute(0, 2, 1)
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = self.pool(x)
            x = x.view(x.size(0), -1)
            x = F.relu(self.fc1(x))
            return x

    # 定义主模型，包含上述所有子网络
    # 主模型不再需要7个单独的输入，而是接受一个三维张量
    class MainModel(nn.Module):
        def __init__(self):
            super(MainModel, self).__init__()
            self.one_hot_net = OneHotCNN()
            self.other_prop_net = OtherPropDNN()
            self.physico_net = PhysicoDNN()
            self.hydro_net = HydroDNN()
            self.alpha_turn_net = AlphaTurnCNN()
            self.beta_prop_net = BetaPropCNN()
            self.composition_net = CompositionCNN()

            # 最终的全连接层，将所有子网络的输出拼接
            self.fc_ensemble = nn.Sequential(
                nn.Linear(128 * 7, 256),  # 7个子网络，每个输出128维
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Linear(256, 1)  # 二分类输出
            )

        def forward(self, X):
            # 假设输入 X 是三维张量 (batch_size, 33, 1280)
            one_hot_out = self.one_hot_net(X)
            other_prop_out = self.other_prop_net(X)
            physico_out = self.physico_net(X)
            hydro_out = self.hydro_net(X)
            alpha_turn_out = self.alpha_turn_net(X)
            beta_prop_out = self.beta_prop_net(X)
            composition_out = self.composition_net(X)

            # 将所有子网络输出拼接
            combined_features = torch.cat([
                one_hot_out, other_prop_out, physico_out, hydro_out, 
                alpha_turn_out, beta_prop_out, composition_out
            ], dim=1)

            output = self.fc_ensemble(combined_features)
            return output

    # 创建数据集和数据加载器
    train_dataset = CustomDataset(
        f'/data/liuyuhuan/zhaijx/data/{type}/{esm}/train_combined_representations',
        f'/data/liuyuhuan/zhaijx/data/{type}/{esm}/train_labels'
    )

    test_dataset = CustomDataset(
        f'/data/liuyuhuan/zhaijx/data/{type}/{esm}/test_combined_representations',
        f'/data/liuyuhuan/zhaijx/data/{type}/{esm}/test_labels'
    )

    # 训练集和验证集划分
    train_size = int(0.9 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    # 初始化模型
    model = MainModel().to(device)
    class Config:
        def __init__(self):
            self.alpha = 1.0 
            self.num_class = 2
            
    parameters = Config()

    def get_cond_entropy(probs):
        cond_ent = -(probs * torch.log(probs + 1e-12)).sum(1).mean(0, keepdim=True)
        return cond_ent

    def get_val_loss(logits, label, criterion, lambda_):
        label = label.view(-1, 1)
        bce_loss = criterion(logits, label)
        logits = torch.sigmoid(logits)
        ent = -(logits * torch.log(logits + 1e-12) + (1 - logits) * torch.log(1 - logits + 1e-12)).mean()
        cond_ent = -(logits * torch.log(logits + 1e-12)).mean()
        regularization_loss = parameters.alpha * ent - parameters.alpha * cond_ent

        weighted_loss = lambda_ * bce_loss + (1 - lambda_) * regularization_loss
        return weighted_loss

    criterion = nn.BCEWithLogitsLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
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
    save_dir = f'/data/liuyuhuan/zhaijx/model/{type}'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    
    for lambda_ in lambda_array:

            model = MainModel().to(device)
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            best_metrics = {
                'accuracy': {'value': 0}, 'auc': {'value': 0}, 'bacc': {'value': 0}, 
                'sn': {'value': 0}, 'sp': {'value': 0}, 'mcc': {'value': 0}, 'pr_auc': {'value': 0}
            }
            best_epoch = 0
            early_stopping_counter = 0

            # 训练和验证循环
            for epoch in range(num_epochs):
                # 训练阶段
                model.train()
                train_loss = 0.0

                for inputs, labels in train_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    optimizer.zero_grad()

                    # 直接传递整个 inputs
                    outputs = model(inputs)
                    loss = get_val_loss(outputs, labels, criterion, lambda_)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item() * inputs.size(0)
                    
                train_loss /= len(train_loader.dataset)

                # 验证阶段
                with torch.no_grad():
                    model.eval()  # 切换到评估模式
                    val_loss = 0.0
                    all_val_logits = []
                    all_val_labels = []

                    for inputs, labels in val_loader:
                        inputs, labels = inputs.to(device), labels.to(device)

                        # 直接传递整个 inputs 给模型
                        outputs = model(inputs)
                        
                        # 计算验证损失
                        loss = get_val_loss(outputs, labels, criterion, lambda_)
                        val_loss += loss.item() * inputs.size(0)

                        # 存储输出和标签，用于在后续步骤计算指标
                        all_val_logits.append(outputs)
                        all_val_labels.append(labels)

                    # 在所有数据上计算平均损失
                    val_loss /= len(val_loader.dataset)

                    # 将所有验证集的 logits 和 labels 拼接在一起
                    all_val_logits = torch.cat(all_val_logits)
                    all_val_labels = torch.cat(all_val_labels)

                    # 计算验证集的各种指标
                    val_acc, val_auc_score, val_bacc, val_sn, val_sp, val_mcc, val_pr_auc = calculate_metrics(all_val_logits, all_val_labels)
                # 更新最佳验证指标
                if val_acc > best_metrics['accuracy']['value']:
                    best_metrics['accuracy']['value'] = val_acc
                    best_epoch = epoch
                    early_stopping_counter = 0  # 重置早停计数器

                    # 保存模型权重及超参数
                    torch.save({
                        'epoch': epoch + 1,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'best_metrics': best_metrics,
                        'lambda': lambda_,
                        'lr' : learning_rate,
                        'early_stopping_patience' : early_stopping_patience,
                        'esm': esm,
                    }, os.path.join(save_dir, f'{type}_RE_{lambda_}_esm_{esm}.pth'))

                else:
                    early_stopping_counter += 1

                if early_stopping_counter >= early_stopping_patience:
                    break
