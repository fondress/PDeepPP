import esm
import torch

def load_esm_model(device):
    """加载 ESM 预训练模型"""
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    return model.to(device), alphabet.get_batch_converter()

def extract_representations(model, batch_converter, data_iterator, device):
    model.eval()
    representations = []
    for batch_data in data_iterator:
        _, _, batch_tokens = batch_converter(batch_data)
        batch_tokens = batch_tokens.to(device)
        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[33])
        token_representations = results["representations"][33]

        for i, (_, seq) in enumerate(batch_data):
            seq_representation = token_representations[i, 1 : len(seq) + 1]
            representations.append(seq_representation)

    return representations