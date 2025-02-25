import os
import pandas as pd

def pad_sequence(seq, target_length=33, pad_char='X'):
    """确保序列长度为 target_length，不足的部分用 pad_char 填充"""
    if len(seq) < target_length:
        seq += pad_char * (target_length - len(seq))
    return seq[:target_length]

def extract_bps_sequences(file_path, output_excel, overlapping=True, step_size=5, target_length=33):
    """处理生物活性数据（BPS），关注整个序列，可重叠"""
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

    # 提取 BPS 相关序列
    bioactive_data = []
    for seq in sequences:
        for i in range(0, len(seq) - target_length + 1, step_size if overlapping else target_length):
            bioactive_data.append(pad_sequence(seq[i:i + target_length], target_length))

    # 存储数据到 Excel
    df_bioactive = pd.DataFrame(bioactive_data, columns=['BPS Sequence'])
    df_bioactive.to_excel(output_excel, sheet_name='BPS Data', index=False)
    print(f"BPS 数据已保存至 {output_excel}")

# 示例用法
if __name__ == "__main__":
    input_fasta = "../data/input.fasta"
    output_excel = "../data/bps_output.xlsx"
    extract_bps_sequences(input_fasta, output_excel)