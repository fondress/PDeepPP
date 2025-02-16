import torch

# 定义模型
class CNNModel(torch.nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        # 第一层卷积层
        self.conv1 = torch.nn.Conv1d(in_channels=1280, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn1 = torch.nn.BatchNorm1d(128)
        self.pool1 = torch.nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        self.dropout1 = torch.nn.Dropout(0.15)
        
        # 第二层卷积层
        self.conv2 = torch.nn.Conv1d(in_channels=128, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn2 = torch.nn.BatchNorm1d(32)
        self.pool2 = torch.nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        self.dropout2 = torch.nn.Dropout(0.15)
        
        # 全连接层1
        self.fc1 = torch.nn.Linear(32 * 81, 64)  # 32通道，输入长度经过两次池化后为81
        self.dropout3 = torch.nn.Dropout(0.15)
        
        # 全连接层2
        self.fc2 = torch.nn.Linear(64, 2)
    
    def forward(self, x):
        # 输入 x 的形状为 (batch_size, seq_len, 1280)
        # PyTorch 中 Conv1d 期望输入形状为 (batch_size, in_channels, seq_len)，因此需要转置
        x = x.transpose(1, 2)
        
        # 第一层卷积、BN、激活、池化、Dropout
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.nn.functional.relu(x)
        x = self.pool1(x)
        x = self.dropout1(x)
        
        # 第二层卷积、BN、激活、池化、Dropout
        x = self.conv2(x)
        x = self.bn2(x)
        x = torch.nn.functional.relu(x)
        x = self.pool2(x)
        x = self.dropout2(x)
        
        # 展平
        x = x.view(x.size(0), -1)  # (batch_size, 32 * 81)
        
        # 全连接层1，激活和Dropout
        x = self.fc1(x)
        x = torch.nn.functional.relu(x)
        x = self.dropout3(x)
        
        # 全连接层2，softmax
        x = self.fc2(x)
        x = torch.nn.functional.softmax(x, dim=1)
        
        return x

# 创建模型实例
model = CNNModel()

# 假设 batch_size 为 32
batch_size = 32
sequence_length = 81
feature_size = 1280

# 生成形状为 (batch_size, 81, 1280) 的随机张量
input_tensor = torch.randn(batch_size, sequence_length, feature_size)

# 将输入张量传递给模型
output = model(input_tensor)

# 打印输出张量的形状
print("Output shape:", output.shape)