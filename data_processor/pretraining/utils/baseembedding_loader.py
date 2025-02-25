import torch
import torch.nn as nn

class EmbeddingPretrainedModel(nn.Module):
    """嵌入预训练模型"""
    def __init__(self, vocab_size, embedding_dim):
        super(EmbeddingPretrainedModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.fc = nn.Linear(embedding_dim, 1280)
        
    def forward(self, x):
        x = self.embedding(x)
        x = self.fc(x)
        return x