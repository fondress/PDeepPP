import os
import pandas as pd

def pad_sequence(seq, target_length=33, pad_char='X'):
    """确保序列长度为 target_length，不足的部分用 pad_char 填充"""
    if len(seq) < target_length:
        seq += pad_char * (target_length - len(seq))
    return seq[:target_length]

def extract_ptm_sequences(file_path, output_excel, target_length=33):
    """处理 PTM 数据，确保目标氨基酸（S、T、Y）位于序列中心"""
    sequences = []

    with open(file_path, 'r') as f:
        lines = f.readlines()

    current_sequence = ''
    
    for line in lines:
        line = line.strip()
        if line.startswith(">"):  # 处理FASTA的标题行
            if current_sequence:
                sequences.append(current_sequence)
            current_sequence = ''
        else:
            current_sequence += line  # 读取序列内容

    if current_sequence:
        sequences.append(current_sequence)

    # 提取 PTM 相关序列
    ptm_data = []
    for seq in sequences:
        for i in range(len(seq)):
            if seq[i] in {'S', 'T', 'Y'}:  # 仅提取 S、T、Y 作为中心的片段
                start = max(0, i - target_length // 2)
                end = min(len(seq), start + target_length)
                padded_seq = pad_sequence(seq[start:end], target_length)
                ptm_data.append(padded_seq)

    # 存储数据到 Excel
    df_ptm = pd.DataFrame(ptm_data, columns=['PTM Sequence'])
    df_ptm.to_excel(output_excel, sheet_name='PTM Data', index=False)
    print(f"PTM 数据已保存至 {output_excel}")

# 示例用法
if __name__ == "__main__":
    input_fasta = "../data/input.fasta"
    output_excel = "../data/ptm_output.xlsx"
    extract_ptm_sequences(input_fasta, output_excel)