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

ptm_name = "S-Palmitoylation"
esm_ratio = 1

# 定义 Attention 层
class Attention(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Attention, self).__init__()
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.W0 = nn.Linear(input_dim, hidden_dim)
        self.W = nn.Linear(hidden_dim, 1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        energy = F.relu(self.W0(x))
        energy = self.W(energy)
        energy = energy.view(x.size(0), -1)
        attention = self.softmax(energy)
        attention = attention.unsqueeze(1)
        output = torch.bmm(attention, x).squeeze(1)
        return output

# 定义 MultiCNN 模型
class MultiCNN(nn.Module):
    def __init__(self, input_row, input_col):
        super(MultiCNN, self).__init__()
        filter1 = 200
        filter2 = 150
        filter3 = 200
        dropout1 = 0.75
        dropout2 = 0.75
        dropout4 = 0.75
        dropout5 = 0.75
        dropout6 = 0
        dense_size1 = 149
        dense_size2 = 8
        nb_classes = 1

        self.conv1 = nn.Conv1d(input_col, filter1, kernel_size=1, padding='same')
        self.conv2 = nn.Conv1d(filter1, filter2, kernel_size=9, padding='same')
        self.conv3 = nn.Conv1d(filter2, filter3, kernel_size=10, padding='same')

        self.dropout1 = nn.Dropout(dropout1)
        self.dropout2 = nn.Dropout(dropout2)
        self.dropout4 = nn.Dropout(dropout4)
        self.dropout5 = nn.Dropout(dropout5)
        self.dropout6 = nn.Dropout(dropout6)

        self.attention_x = Attention(input_dim=filter3, hidden_dim=10)
        self.attention_xr = Attention(input_dim=filter3, hidden_dim=8)

        self.dense1 = nn.Linear(filter3 + filter3, dense_size1)
        self.dense2 = nn.Linear(dense_size1, dense_size2)
        self.out = nn.Linear(dense_size2, nb_classes)

    def forward(self, x):
        x = x.transpose(1, 2)  # 调整维度以适应 Conv1d
        x = F.relu(self.conv1(x))
        x = self.dropout1(x)

        x = F.relu(self.conv2(x))
        x = self.dropout2(x)

        x = F.relu(self.conv3(x))
        x = self.dropout4(x)
        
        x_reshape = x.transpose(1, 2)  # 调整维度以适应 Attention

        x_attention = self.attention_x(x_reshape)
        xr_attention = self.attention_xr(x_reshape)

        x_concat = torch.cat((x_attention, xr_attention), dim=1)

        x = F.relu(self.dense1(x_concat))
        x = self.dropout5(x)
        x = F.relu(self.dense2(x))
        x = self.dropout6(x)
        x = self.out(x).squeeze(1)  # 添加 squeeze(1) 以移除多余的维度
        return x

# 数据加载
train_representations = np.load(f'/data/liuyuhuan/zhaijx/weight/musite/{ptm_name}/1/train_combined_representations.npy')
train_labels = np.load(f'/data/liuyuhuan/zhaijx/data/weight/musite/{ptm_name}/1/train_labels.npy')
test_representations = np.load(f'/data/liuyuhuan/zhaijx/weight/musite/{ptm_name}/1/test_combined_representations.npy')
test_labels = np.load(f'/data/liuyuhuan/zhaijx/weight/musite/{ptm_name}/1/test_labels.npy')

# 转换为张量
X_train = torch.tensor(train_representations, dtype=torch.float32)
y_train = torch.tensor(train_labels, dtype=torch.float32)
X_test = torch.tensor(test_representations, dtype=torch.float32)
y_test = torch.tensor(test_labels, dtype=torch.float32)

# 数据集划分
train_dataset = TensorDataset(X_train, y_train)
val_size = int(0.1 * len(train_dataset))
train_size = len(train_dataset) - val_size
train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
test_dataset = TensorDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)

# 获取样本数据的尺寸
sample_inputs, _ = next(iter(train_loader))
input_row = sample_inputs.size(1)  # 通常是序列长度
input_col = sample_inputs.size(2)  # 通常是特征数量

# 初始化模型
predict_module = MultiCNN(input_row=input_row, input_col=input_col).to(device).float()

# 超参数设置
learning_rate = 0.001
num_epochs = 100
early_stopping_patience = 20  # 设置早停的耐心值

lambda_array = [0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 1]
print(f"esm_ratio: {esm_ratio}")

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
save_dir = f'/data/liuyuhuan/zhaijx/model/Musitedeep/{ptm_name}'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# 动态生成文件路径
output_save_dir = f'/data/liuyuhuan/zhaijx/code/Timloss/output/musite/{ptm_name}/musite_1.txt'

# 提取目录路径
output_directory = os.path.dirname(output_save_dir)

# 确保父目录存在
if not os.path.exists(output_directory):
    os.makedirs(output_directory)
    
with open(output_save_dir, 'a') as file:
    file.write(f"esm_ratio: {esm_ratio}\n")
    
with open(output_save_dir, 'a') as file:
    for lambda_ in lambda_array:
        # 初始化权重保存路径
        save_path = os.path.join(save_dir, f'{ptm_name}_RE_{lambda_}_esm_{esm_ratio}.pth')
        
        # 重置模型和优化器
        predict_module = MultiCNN(input_row=input_row, input_col=input_col).to(device).float()
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
                }, save_path)

            else:
                early_stopping_counter += 1

            if early_stopping_counter >= early_stopping_patience:
                # print(f"早停机制触发，在第 {epoch+1} 轮停止训练。")
                break

        # 测试阶段
        predict_module.eval()
        test_loss = 0.0
        all_test_logits = []
        all_test_labels = []

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = predict_module(inputs)
                loss = get_val_loss(outputs, labels, criterion, lambda_)
                test_loss += loss.item() * inputs.size(0)
                all_test_logits.append(outputs)
                all_test_labels.append(labels)

        test_loss /= len(test_loader.dataset)
        all_test_logits = torch.cat(all_test_logits)
        all_test_labels = torch.cat(all_test_labels)

        test_acc, test_auc_score, test_bacc, test_sn, test_sp, test_mcc, test_pr_auc = calculate_metrics(all_test_logits, all_test_labels)
        file.write(f"lambda: {lambda_}\n")
        file.write(f"{ptm_name}:ACC: {test_acc:.4f}, AUC: {test_auc_score:.4f}, BACC: {test_bacc:.4f}, SN: {test_sn:.4f}, SP: {test_sp:.4f}, MCC: {test_mcc:.4f}, PR AUC: {test_pr_auc:.4f}\n")
        file.write("\n")
    

print("save")