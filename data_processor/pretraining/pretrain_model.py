import os
import torch
import esm
import argparse
import numpy as np
import pandas as pd
from utils.data_loader import load_data, DataIterator, seq_to_indices, pad_sequences
from utils.esm_loader import load_esm_model, extract_representations
from utils.baseembedding_loader import EmbeddingPretrainedModel
from utils.file_storing import save_representations

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main(args):
    data_path = "./data"
    weight_path = f"../pretraining/weights/{args.ptm_type}/"
    os.makedirs(weight_path, exist_ok=True)

    # 加载数据
    train_df, test_df = load_data(data_path, args.ptm_type)
    train_data = list(zip(train_df['label'], train_df['sequence']))
    test_data = list(zip(test_df['label'], test_df['sequence']))

    # 加载 ESM 预训练模型
    model, batch_converter = load_esm_model(device)
    train_iterator = DataIterator(train_data, args.batch_size)
    test_iterator = DataIterator(test_data, args.batch_size)

    # 计算 ESM 预训练表示
    train_representations = extract_representations(model, batch_converter, train_iterator, device)
    test_representations = extract_representations(model, batch_converter, test_iterator, device)

    # 处理词汇表
    vocab_dict = {char: i for i, char in enumerate(set(char for _, seq in train_data + test_data for char in seq))}
    train_indices = [seq_to_indices(seq, vocab_dict) for _, seq in train_data]
    test_indices = [seq_to_indices(seq, vocab_dict) for _, seq in test_data]

    # 进行序列填充
    max_len = max(len(seq) for _, seq in train_data + test_data)
    train_padded = pad_sequences(train_indices, max_len)
    test_padded = pad_sequences(test_indices, max_len)

    # 生成嵌入模型
    embedding_model = EmbeddingPretrainedModel(len(vocab_dict), 128).to(device)
    with torch.no_grad():
        train_embedding_output = embedding_model(train_padded.to(device))
        test_embedding_output = embedding_model(test_padded.to(device))

    # 保存训练结果
    save_representations(args.esm_ratio, weight_path, train_representations, test_representations, 
                         train_embedding_output, test_embedding_output, train_df, test_df, device)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pretrain ESM Model with Custom Settings")
    parser.add_argument("--esm_ratio", type=float, required=True, help="ESM pretraining ratio (e.g., 0.9)")
    parser.add_argument("--batch_size", type=int, required=True, help="Batch size for training (e.g., 16)")
    parser.add_argument("--ptm_type", type=str, required=True, help="Type of PTM (e.g., Hydroxyproline_P)")
    args = parser.parse_args()

    main(args)