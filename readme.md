# PDeepPP:A general language model for peptide identification

To install dependencies, run:

    pip install -r requirements.txt

## 📋 Deep Learning FASTA Processor

This project processes **FASTA format** protein sequences and prepares them for deep learning models.  
It includes:
- **Data Processing**: Extracts PTM and BPS sequences from raw data.
- **Pretraining**: Uses ESM-2 and an embedding model to generate sequence representations.

---
###  **Installation**

    pip install -r requirements.txt

###  Usage
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

## 🎯 Model Training & Testing

1️⃣ Training the Model (train.py)

This script trains the deep learning model on processed PTM data.
Usage:

    python train.py --esm_ratio <value> --lambda_ <value> --ptm_name <PTM_type>

Arguments:  
--esm_ratio: The ratio of ESM pretrained output (e.g., 0.9).  
--lambda_: The weight factor for the loss function (e.g., 0.1).  
--ptm_name: The name of the PTM type (e.g., Hydroxyproline_P).  

The trained model will be saved as:

    <PTM_name>_<lambda_>_esm_<esm_ratio>.pth

2️⃣ Testing the Model (test.py)

This script evaluates the trained model on test data and generates prediction results.
Usage:

    python test.py --esm_ratio <value> --lambda_ <value> --ptm_name <PTM_type>  

Output Metrics:After running the script, it prints evaluation metrics:

    <PTM_name>_<esm_ratio>_Lambda: <lambda_>, ACC: <accuracy>, AUC: <auc_score>, BACC: <balanced_accuracy>, SN: <sensitivity>, SP: <specificity>, MCC: <mcc_score>, PR: <precision_recall_auc>

The prediction results are saved in:

    ../train_test/results/<PTM_name>/<PTM_name>_<esm_ratio>_lambda_<lambda_>.csv

## ✈️ Using a Pre-trained Model for Prediction

Pre-trained models are stored exclusively on Hugging Face at the following repository(main repository):  
[https://huggingface.co/fondress/PDeppPP](https://huggingface.co/fondress/PDeppPP).  

Each dataset's corresponding Hugging Face repository contains the specific model in the correct format.  
You can access these dataset-specific repositories directly from the main repository

To use a pre-trained model, update the model loading path in `test.py` to the Hugging Face download location:  

    checkpoint_path = '<HuggingFace_model_download_path>'

Then, run the test script:

    python test.py --esm_ratio <value> --lambda_ <value> --ptm_name <PTM_type>

This will use the pre-trained model downloaded from the Hugging Face repository for prediction without retraining.

