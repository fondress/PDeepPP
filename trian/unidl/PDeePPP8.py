import pandas as pd
import torch
import torch.nn as nn
import os
import warnings
import torch.nn.functional as F
from sklearn.metrics import roc_curve, roc_auc_score, auc, precision_recall_curve
from torch.utils.data import TensorDataset, DataLoader, random_split
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

ptm_name = "Antiparasitic"
esm_array = [0.9,0.95,1] 
lambda_array = [0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 1]
learning_rate = 0.001
batch_size = 128
num_epochs = 100
early_stopping_patience = 20
warnings.filterwarnings('ignore', category=UserWarning)

for esm_ratio in esm_array:
    train_representations = np.load(f'/data/liuyuhuan/zhaijx/data/unidl/{ptm_name}/{esm_ratio}/train_combined_representations.npy')
    train_labels = np.load(f'/data/liuyuhuan/zhaijx/data/unidl/{ptm_name}/{esm_ratio}/train_labels.npy')
    test_representations = np.load(f'/data/liuyuhuan/zhaijx/data/unidl/{ptm_name}/{esm_ratio}/test_combined_representations.npy')
    test_labels = np.load(f'/data/liuyuhuan/zhaijx/data/unidl/{ptm_name}/{esm_ratio}/test_labels.npy')

    # 转换为张量
    X_train = torch.tensor(train_representations, dtype=torch.float32)
    y_train = torch.tensor(train_labels, dtype=torch.float32)
    X_test = torch.tensor(test_representations, dtype=torch.float32)
    y_test = torch.tensor(test_labels, dtype=torch.float32)

    # 假设 esm_ratio 已经定义

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
                self.position_encoding = nn.Parameter(torch.zeros(64, input_size))  # Positional encoding parameter

        def forward(self, x):
            x = x.permute(0, 2, 1)  # Reshape to (batch_size, input_size, seq_length)
            x = self.conv1d(x)
            x = self.relu(x)

            if self.use_position_encoding:
                seq_len = x.size(2)
                pos_encoding = self.position_encoding[:, :seq_len].unsqueeze(0)  # Broadcast to batch size
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
                nn.Conv1d(in_channels=output_size * 2, out_channels=32, kernel_size=3, stride=1, padding=1),
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
            combined_output = combined_output.permute(0, 2, 1)  # 调整维度顺序为(batch_size, channels, sequence_length)
            output = self.cnn_layers(combined_output)
            output = output.squeeze(1)
            return output

    device = torch.device("cuda")

    input_size = 1280
    output_size = 128

    predict_module = PredictModule(input_size, output_size).to(device)

    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    # Split train_dataset into train and val datasets
    val_size = int(0.1 * len(train_dataset))
    train_size = len(train_dataset) - val_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

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


    def calculate_metrics(logits, labels):
        preds = torch.round(torch.sigmoid(logits))
        labels = labels.int()

        tp = (preds * labels).sum().to(torch.float32)
        tn = ((1 - preds) * (1 - labels)).sum().to(torch.float32)
        fp = (preds * (1 - labels)).sum().to(torch.float32)
        fn = ((1 - preds) * labels).sum().to(torch.float32)

        accuracy = (tp + tn) / (tp + tn + fp + fn)
        auc_score = roc_auc_score(labels.cpu().numpy(), torch.sigmoid(logits).cpu().numpy())
        precision, recall, _ = precision_recall_curve(labels.cpu().numpy(), torch.sigmoid(logits).cpu().numpy())
        pr_auc = auc(recall, precision)
        bacc = (tp / (tp + fn) + tn / (tn + fp)) / 2
        sn = tp / (tp + fn)
        sp = tn / (tn + fp)
        mcc = (tp * tn - fp * fn) / torch.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

        return accuracy.item(), auc_score, bacc.item(), sn.item(), sp.item(), mcc.item(), pr_auc

    # 模型保存目录
    save_dir = f'/data/liuyuhuan/zhaijx/model/{ptm_name}'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for lambda_ in lambda_array:
        save_path = os.path.join(save_dir, f'{ptm_name}_{lambda_}_esm_{esm_ratio}.pth')
        best_metrics = {
            'accuracy': {'value': 0}, 'auc': {'value': 0}, 'bacc': {'value': 0}, 'sn': {'value': 0}, 'sp': {'value': 0}, 'mcc': {'value': 0}, 'pr_auc': {'value': 0}
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
                    'lr': learning_rate,
                    'early_stopping_patience': early_stopping_patience,
                    'esm_ratio': esm_ratio,  # 保存 esm_ratio
                    'pos_embedding': True,
                }, save_path)

            else:
                early_stopping_counter += 1

            if early_stopping_counter >= early_stopping_patience:
                break

