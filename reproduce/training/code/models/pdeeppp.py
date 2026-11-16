from __future__ import annotations

import torch
import torch.nn as nn


class SelfAttentionGlobalFeatures(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        num_heads: int = 8,
        hidden_size: int = 256,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.self_attention = nn.MultiheadAttention(embed_dim=input_size, num_heads=num_heads, batch_first=True)
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.layer_norm = nn.LayerNorm(input_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_output, _ = self.self_attention(x, x, x)
        x = self.layer_norm(x + attn_output)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class TransConv1d(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        num_heads: int = 8,
        hidden_size: int = 256,
        dropout: float = 0.3,
        num_layers: int = 4,
        ff_dim: int = 512,
    ):
        super().__init__()
        self.self_attention_global_features = SelfAttentionGlobalFeatures(
            input_size,
            output_size,
            num_heads,
            hidden_size,
            dropout,
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=output_size,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc1 = nn.Linear(output_size, output_size)
        self.fc2 = nn.Linear(output_size, output_size)
        self.layer_norm = nn.LayerNorm(output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.self_attention_global_features(x)
        x = self.transformer(x)
        residual = self.fc1(x)
        x = self.fc2(residual)
        return self.layer_norm(x + residual)


class PosCNN(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        use_position_encoding: bool = True,
        conv_channels: int = 64,
        max_len: int = 4096,
    ):
        super().__init__()
        self.use_position_encoding = use_position_encoding
        self.conv1d = nn.Conv1d(in_channels=input_size, out_channels=conv_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.global_pooling = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(conv_channels, output_size)
        if self.use_position_encoding:
            self.position_encoding = nn.Parameter(torch.zeros(conv_channels, max_len))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        x = self.relu(self.conv1d(x))
        if self.use_position_encoding:
            seq_len = x.size(2)
            x = x + self.position_encoding[:, :seq_len].unsqueeze(0)
        x = self.global_pooling(x).squeeze(-1)
        return self.fc(x)


class PDeepPPClassifier(nn.Module):
    def __init__(
        self,
        input_size: int = 1280,
        output_size: int = 128,
        use_position_encoding: bool = True,
        transformer_heads: int = 8,
        transformer_hidden: int = 256,
        transformer_layers: int = 4,
        transformer_ff_dim: int = 512,
        cnn_channels: int = 64,
        fusion_channels: int = 32,
        dropout: float = 0.15,
    ):
        super().__init__()
        self.transformer = TransConv1d(
            input_size,
            output_size,
            num_heads=transformer_heads,
            hidden_size=transformer_hidden,
            num_layers=transformer_layers,
            ff_dim=transformer_ff_dim,
        )
        self.cnn = PosCNN(
            input_size,
            output_size,
            use_position_encoding=use_position_encoding,
            conv_channels=cnn_channels,
        )
        self.cnn_layers = nn.Sequential(
            nn.Conv1d(in_channels=output_size * 2, out_channels=fusion_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1),
            nn.Dropout(dropout),
            nn.Conv1d(in_channels=fusion_channels, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1),
            nn.Dropout(dropout),
            nn.Flatten(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        transformer_output = self.transformer(x)
        cnn_output = self.cnn(x)
        cnn_output = cnn_output.unsqueeze(1).expand(-1, transformer_output.size(1), -1)
        combined_output = torch.cat([transformer_output, cnn_output], dim=2)
        combined_output = combined_output.permute(0, 2, 1)
        return self.cnn_layers(combined_output).squeeze(1)
