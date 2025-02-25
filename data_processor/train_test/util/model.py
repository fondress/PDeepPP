import torch.nn as nn
import torch

class SelfAttentionGlobalFeatures(nn.Module):
    def __init__(self, input_size, output_size, num_heads=8):
        super().__init__()
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
        super().__init__()
        self.self_attention_global_features = SelfAttentionGlobalFeatures(input_size, output_size)
        self.transformer_encoder = nn.TransformerEncoderLayer(
            d_model=output_size, nhead=8, dim_feedforward=512, dropout=0.3, batch_first=True)
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
        super().__init__()
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
        super().__init__()
        self.transformer = TransConv1d(input_size, output_size)
        self.cnn = PosCNN(input_size, output_size)
        self.cnn_layers = nn.Sequential(
            nn.Conv1d(output_size*2, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1),
            nn.Dropout(0.15),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
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
        combined = torch.cat([transformer_output, cnn_output], dim=2)
        combined = combined.permute(0, 2, 1)
        return self.cnn_layers(combined).squeeze(1)