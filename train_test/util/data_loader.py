import torch
from torch.utils.data import TensorDataset, DataLoader, random_split
import numpy as np

def get_dataloaders(batch_size, args):
    train_data = np.load(f"./data_processor/data_storage/{args.ptm_type}/train_representations.npy")
    train_labels = np.load(f"./data_processor/data_storage/{args.ptm_type}/test_representations.npy")
    test_data = np.load(f"./data_processor/data_storage/{args.ptm_type}/train_labels.npy")
    test_labels = np.load(f"./data_processor/data_storage/{args.ptm_type}/test_labels.npy")

    X_train = torch.tensor(train_data, dtype=torch.float32)
    y_train = torch.tensor(train_labels, dtype=torch.float32)
    X_test = torch.tensor(test_data, dtype=torch.float32)
    y_test = torch.tensor(test_labels, dtype=torch.float32)

    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    val_size = int(0.1 * len(train_dataset))
    train_size = len(train_dataset) - val_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, X_train.shape[1], 128