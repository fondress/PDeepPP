import torch
import torch.nn as nn
import torch.nn.functional as F

# Attention Layer
class Attention(nn.Module):
    def __init__(self, hidden):
        super(Attention, self).__init__()
        self.W0 = nn.Parameter(torch.randn(hidden, requires_grad=True))
        self.b0 = nn.Parameter(torch.zeros(hidden, requires_grad=True))
        self.W = nn.Parameter(torch.randn(1, requires_grad=True))
        self.b = nn.Parameter(torch.zeros(1, requires_grad=True))

    def forward(self, x):
        # energy = x * W0 + b0
        energy = torch.matmul(x, self.W0) + self.b0
        # energy = energy * W + b
        energy = torch.matmul(energy, self.W) + self.b
        # Apply softmax to get attention weights
        energy = torch.softmax(energy, dim=1)
        # Weighted sum of input with attention weights
        xx = torch.bmm(energy.unsqueeze(1), x).squeeze(1)
        # Concatenate attention output with attention weights
        return torch.cat([xx, energy], dim=1)

# MultiCNN Model
class MultiCNN(nn.Module):
    def __init__(self, input_row, input_col, nb_classes=2, dropout1=0.75, dropout2=0.75, dense_size1=149, dense_size2=8):
        super(MultiCNN, self).__init__()
        # Define convolutional layers
        self.conv1 = nn.Conv1d(in_channels=input_row, out_channels=200, kernel_size=1, padding='same')
        self.conv2 = nn.Conv1d(in_channels=200, out_channels=150, kernel_size=9, padding='same')
        self.conv3 = nn.Conv1d(in_channels=150, out_channels=200, kernel_size=10, padding='same')

        # Dropouts
        self.dropout1 = nn.Dropout(dropout1)
        self.dropout2 = nn.Dropout(dropout2)

        # Attention layers
        self.attention_x = Attention(hidden=10)
        self.attention_xr = Attention(hidden=8)

        # Fully connected layers
        self.fc1 = nn.Linear(200 * input_col * 2, dense_size1)
        self.fc2 = nn.Linear(dense_size1, dense_size2)
        self.fc3 = nn.Linear(dense_size2, nb_classes)

    def forward(self, x):
        # Apply convolutional layers with relu and dropout
        x = F.relu(self.conv1(x))
        x = self.dropout1(x)
        x = F.relu(self.conv2(x))
        x = self.dropout2(x)
        x = F.relu(self.conv3(x))

        # Reshape x for the attention layer
        x_reshape = x.permute(0, 2, 1)

        # Apply attention layers
        output_x = self.attention_x(x)
        output_xr = self.attention_xr(x_reshape)

        # Concatenate attention outputs
        output = torch.cat([output_x, output_xr], dim=1)

        # Apply dropout
        output = F.dropout(output, p=0.75)

        # Fully connected layers
        output = F.relu(self.fc1(output))
        output = F.dropout(output, p=0.298224)
        output = F.relu(self.fc2(output))
        output = self.fc3(output)

        # Softmax output
        return F.softmax(output, dim=1)

# Example usage of the combined model
if __name__ == '__main__':
    # 输入数据的维度为 [batch_size, input_row, input_col]
    input_row = 21  # 输入的通道数
    input_col = 33  # 序列的长度

    # 生成一些随机输入数据，形状为 [batch_size, input_row, input_col]
    input_data = torch.randn(32, input_col, input_row)

    # 模型期望输入的形状是 [batch_size, input_row, input_col]，但是 Attention 层期望 [batch_size, seq_len, input_dim]
    # 因此在输入 Attention 层之前，我们需要将输入数据的维度排列为 [batch_size, input_col, input_row]，即 [batch_size, seq_len, input_dim]

    # 调整输入维度，使其与模型的预期输入相匹配
    input_data = input_data.permute(0, 2, 1)  # 交换维度，将输入调整为 [batch_size, seq_len, input_row]
    model = MultiCNN(input_row=input_row, input_col=input_col)
    # 然后将调整后的输入数据传给模型
    output = model(input_data)

    # 检查输出形状
    print("Output shape:", output.shape)