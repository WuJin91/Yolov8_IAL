# generate_splits.py
import json
import os
import random
import yaml
import glob
from collections import defaultdict
from sklearn.model_selection import KFold

def get_patient_id(filename):
    """從檔名提取病患 ID，避免同一病患影像跨 Fold 洩漏"""
    parts = os.path.basename(filename).split('_')
    # 假設檔名格式為: PatientID_Sequence_...
    if len(parts) >= 3:
        return "_".join(parts[:3]) # 視情況調整，例如 parts[0]
    return parts[0]

def main():
    config_path = 'config.yaml'
    output_path = 'fixed_splits.json'
    
    if not os.path.exists(config_path):
        print(f"[Error] 找不到 {config_path}")
        return

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
        
    source_dir = config['dataset']['source_dir']
    img_dir = os.path.join(source_dir, 'images')
    
    # 支援 jpg 與 png
    all_images = glob.glob(os.path.join(img_dir, "*.jpg")) + \
                 glob.glob(os.path.join(img_dir, "*.png"))
    
    if not all_images:
        print(f"[Error] 在 {img_dir} 找不到任何影像")
        return

    # 1. 根據 Patient ID 分組
    patient_map = defaultdict(list)
    for img_path in all_images:
        pid = get_patient_id(img_path)
        patient_map[pid].append(os.path.abspath(img_path))
    
    patients = list(patient_map.keys())
    random.seed(42) # 固定種子，確保可重現
    random.shuffle(patients)
    
    print(f"[Info] 找到 {len(all_images)} 張影像，來自 {len(patients)} 位病患。")

    # 2. 執行 K-Fold
    k_folds = config['automation']['k_folds']
    kf = KFold(n_splits=k_folds, shuffle=False)
   
    
    splits_data = {}
    for fold_idx, (_, test_indices) in enumerate(kf.split(patients)):
        fold_patients = [patients[i] for i in test_indices]
        fold_images = []
        for p in fold_patients:
            fold_images.extend(patient_map[p])
            
        # 由 0 開始改為從 1 開始，並調整字首大小寫一致性
        splits_data[f"Fold_{fold_idx + 1}"] = fold_images
        print(f"  - Fold {fold_idx + 1}: {len(fold_images)} images")
        
    # 3. 儲存結果    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(splits_data, f, indent=4, ensure_ascii=False)
        
    print(f"\n[Success] 固定分割檔已儲存至: {output_path}")
    print("接下來請使用 main_new.py 進行實驗。")

if __name__ == "__main__":
    main()
