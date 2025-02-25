import os
import json
import torch
from datetime import datetime

def setup_directories(ptm_name):
    """创建按ptm_name组织的目录"""
    model_dir = f'../train_test/models/{ptm_name}'
    result_dir = f'../train_test/results/{ptm_name}'
    
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)
    
    return model_dir, result_dir

def save_artifacts(model_dir, result_dir, model, metrics, params):
    """统一保存模型和结果"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 保存模型
    model_path = os.path.join(model_dir, f"model_{timestamp}.pt")
    torch.save(model.state_dict(), model_path)
    
    # 保存结果
    result_path = os.path.join(result_dir, f"result_{timestamp}.json")
    with open(result_path, 'w') as f:
        json.dump({
            "params": params,
            "metrics": metrics,
            "model_path": model_path
        }, f, indent=2)
    
    return model_path, result_path