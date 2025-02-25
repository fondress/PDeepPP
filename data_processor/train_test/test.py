import torch
import os
import pandas as pd
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

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Get data loaders
_, _, test_loader, input_size, output_size = get_dataloaders(batch_size=32)

# Initialize model
predict_module = PredictModule(input_size, output_size).to(device)

# Load model checkpoint
checkpoint_path = f'../train_test/models/{args.ptm_name}_{args.lambda_}_esm_{args.esm_ratio}.pth'
checkpoint = torch.load(checkpoint_path)
predict_module.load_state_dict(checkpoint['model_state_dict'])
predict_module.eval()

# Evaluate model
test_loss = 0.0
all_test_logits, all_test_labels = [], []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = predict_module(inputs)
        loss = get_val_loss(outputs, labels, torch.nn.BCEWithLogitsLoss().to(device), args.lambda_)
        test_loss += loss.item() * inputs.size(0)
        all_test_logits.append(outputs)
        all_test_labels.append(labels)

test_loss /= len(test_loader.dataset)
all_test_logits = torch.cat(all_test_logits)
all_test_labels = torch.cat(all_test_labels)

# Compute metrics
test_acc, test_auc_score, test_bacc, test_sn, test_sp, test_mcc, test_pr_auc = calculate_metrics(all_test_logits, all_test_labels)

print(f"{args.ptm_name}_{args.esm_ratio}_Lambda: {args.lambda_}, ACC: {test_acc:.4f}, AUC: {test_auc_score:.4f}, BACC: {test_bacc:.4f}, SN: {test_sn:.4f}, SP: {test_sp:.4f}, MCC: {test_mcc:.4f}, PR: {test_pr_auc:.4f}")

# Save prediction results
test_probs = torch.sigmoid(all_test_logits).cpu().numpy()
test_labels_np = all_test_labels.cpu().numpy()
predicted_labels_np = (test_probs >= 0.5).astype(int)

results_df = pd.DataFrame({
    'PredictedLabel': predicted_labels_np.flatten(),
    'PredictedProb': test_probs.flatten(),
    'OriginalLabel': test_labels_np.flatten()
})

save_dir = os.path.join(f'../train_test/results/{args.ptm_name}')
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, f'{args.ptm_name}_{args.esm_ratio}_lambda_{args.lambda_}.csv')
results_df.to_csv(save_path, index=False)
print(f"Prediction results saved to: {save_path}")
