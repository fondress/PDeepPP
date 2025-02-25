import os
import numpy as np
import torch

def save_representations(esm_ratios, weight_path, train_esm, test_esm, train_embed, test_embed, train_df, test_df, device):
    """保存预训练的表示向量"""
    for esm_ratio in esm_ratios:
        save_path = os.path.join(weight_path, str(esm_ratio))
        os.makedirs(save_path, exist_ok=True)

        train_combined = esm_ratio * torch.stack(train_esm).to(device) + (1 - esm_ratio) * train_embed
        test_combined = esm_ratio * torch.stack(test_esm).to(device) + (1 - esm_ratio) * test_embed

        np.save(os.path.join(save_path, 'train_representations.npy'), train_combined.cpu().numpy())
        np.save(os.path.join(save_path, 'test_representations.npy'), test_combined.cpu().numpy())
        np.save(os.path.join(save_path, 'train_labels.npy'), np.array(train_df['label']))
        np.save(os.path.join(save_path, 'test_labels.npy'), np.array(test_df['label']))

        print(f"Saved pretrained representations at {save_path}")