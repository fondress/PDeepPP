import pandas as pd
import torch

def load_data(data_path, ptm_type):
    """加载训练和测试数据"""
    train_df = pd.read_excel(f"{data_path}/{ptm_type}/train.xlsx")
    test_df = pd.read_excel(f"{data_path}/{ptm_type}/test.xlsx")
    return train_df, test_df

class DataIterator:
    """数据迭代器，按批次处理数据"""
    def __init__(self, data, batch_size):
        self.data = data
        self.batch_size = batch_size
        self.num_batches = (len(data) + batch_size - 1) // batch_size

    def __iter__(self):
        for i in range(self.num_batches):
            yield self.data[i * self.batch_size : (i + 1) * self.batch_size]

def seq_to_indices(seq, vocab_dict):
    """将序列转换为索引"""
    return [vocab_dict[char] for char in seq]

def pad_sequences(sequences, max_len, pad_value=0):
    """对序列进行填充"""
    padded_sequences = torch.zeros((len(sequences), max_len), dtype=torch.long)
    for i, seq in enumerate(sequences):
        padded_sequences[i, :len(seq)] = torch.tensor(seq)
    return padded_sequences