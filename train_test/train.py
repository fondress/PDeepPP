import torch
import torch.nn as nn
import argparse
from util.model import PredictModule
from util.data_loader import get_dataloaders
from util.metrics import calculate_metrics
from util.loss_function import get_val_loss

# Parse user input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--esm_ratio', type=float, required=True, help='ESM ratio value')
parser.add_argument('--lambda_', type=float, required=True, help='Lambda value')
parser.add_argument('--ptm_name', type=str, required=True, help='PTM type')
args = parser.parse_args()

# Training parameters
learning_rate = 0.001
batch_size = 32
num_epochs = 100
early_stopping_patience = 20
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Get data loader
train_loader, val_loader, test_loader, input_size, output_size = get_dataloaders(batch_size)

# Initialize model
predict_module = PredictModule(input_size, output_size).to(device)
if torch.cuda.device_count() > 1:
    predict_module = nn.DataParallel(predict_module)

criterion = nn.BCEWithLogitsLoss().to(device)
optimizer = torch.optim.Adam(predict_module.parameters(), lr=learning_rate)

# Train and validate
best_val_acc = 0.0
best_epoch = 0
save_path = f"{args.ptm_name}_{args.lambda_}_esm_{args.esm_ratio}.pth"

for epoch in range(num_epochs):
    predict_module.train()
    train_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = predict_module(inputs)
        loss = get_val_loss(outputs, labels, criterion, args.lambda_)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * inputs.size(0)
    train_loss /= len(train_loader.dataset)

    # Validate
    predict_module.eval()
    val_loss = 0.0
    all_val_logits, all_val_labels = [], []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = predict_module(inputs)
            loss = get_val_loss(outputs, labels, criterion, args.lambda_)
            val_loss += loss.item() * inputs.size(0)
            all_val_logits.append(outputs)
            all_val_labels.append(labels)
    val_loss /= len(val_loader.dataset)

    val_acc, *_ = calculate_metrics(torch.cat(all_val_logits), torch.cat(all_val_labels))

    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_epoch = epoch
        torch.save({'epoch': epoch + 1, 'model_state_dict': predict_module.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, save_path)

    print(f"Epoch {epoch+1}: Train Loss {train_loss:.4f}, Val Loss {val_loss:.4f}, Val Acc {val_acc:.4f}")

    if epoch - best_epoch >= early_stopping_patience:
        print("Early stopping triggered.")
        break
