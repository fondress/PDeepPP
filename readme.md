# PDeepPP

## Deep Learning FASTA Processor

This project processes **FASTA format** protein sequences and prepares them for deep learning models.  
It includes:
- **Data Processing**: Extracts PTM and BPS sequences from raw data.
- **Pretraining**: Uses ESM-2 and an embedding model to generate sequence representations.

---
### ⚙️ **Installation**

    pip install -r requirements.txt

### 🚀 Usage
1️⃣ Run Data Processing

    python data_processing/process_ptm.py
    python data_processing/process_bps.py

2️⃣ Run Pretraining

    python pretraining/pretrain_model.py --esm_ratio  --batch_size  --ptm_type 

--esm_ratio: The ratio of ESM pretrained output (e.g., 0.9).  
--batch_size: The batch size for training (e.g., 16).  
--ptm_type: The type of PTM (e.g., Hydroxyproline_P).  

Example:

    python pretraining/pretrain_model.py --esm_ratio 0.9 --batch_size 16 --ptm_type Hydroxyproline_P
    python pretraining/pretrain_model.py --esm_ratio 0.95 --batch_size 32 --ptm_type Phosphorylation