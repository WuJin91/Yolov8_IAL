# utils/data_manager.py

import os
import glob
import random
import yaml
import pandas as pd
import numpy as np
from collections import defaultdict, Counter
from sklearn.model_selection import KFold
from datetime import datetime
import shutil

def get_patient_id(filename: str) -> str:
    """從檔名中提取 Patient ID"""
    parts = os.path.basename(filename).split('_')
    if len(parts) >= 4:
        return "_".join(parts[:3])
    return parts[0]

def create_k_fold_patient_splits(all_images_list: list, k_folds: int) -> list:
    """
    根據 Patient ID 將所有影像劃分為 K-Folds。
    回傳: [[fold1_img_paths], [fold2_img_paths], ...]
    """
    print(f"\n[Data] 正在根據 Patient ID 將 {len(all_images_list)} 張影像劃分為 {k_folds} Folds...")
    
    patient_to_images = defaultdict(list)
    for img_path in all_images_list:
        patient_id = get_patient_id(img_path)
        patient_to_images[patient_id].append(img_path)
    
    all_patients = list(patient_to_images.keys())
    random.shuffle(all_patients)
    
    # 使用 KFold 來切分病患列表
    kf = KFold(n_splits=k_folds, shuffle=False) 
    
    fold_image_lists = []
    
    for fold_idx, (train_indices, test_indices) in enumerate(kf.split(all_patients)):
        # KFold 的 test_indices 在此代表 "當前 Fold" 的病患
        fold_patients = [all_patients[i] for i in test_indices]
        
        current_fold_images = []
        for patient in fold_patients:
            current_fold_images.extend(patient_to_images[patient])
        
        fold_image_lists.append(current_fold_images)
        print(f"  Fold {fold_idx+1}: {len(fold_patients)} 位病患, {len(current_fold_images)} 張影像")
        
    print("[Data] K-Fold 劃分完成。")
    return fold_image_lists

def _write_paths_to_file(filepath: str, paths: list):
    """輔助函數：將路徑列表寫入 .txt"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        for p in paths:
            f.write(f"{os.path.abspath(p)}\n")

def prepare_iteration_data_files(
    iter_save_dir: str, 
    iter_name: str, 
    config: dict, 
    train_image_paths: list, 
    test_image_paths: list, 
    pool_image_paths: list
) -> (str, str):
    """
    為單次迭代準備 train/val/test/pool 的 .txt 和 .yaml 檔案。
    """
    
    # 1. 劃分 train/val
    random.shuffle(train_image_paths)
    val_split_idx = int(len(train_image_paths) * 0.1) # 10% for validation
    
    # 確保 val 至少有 1 張，train 至少有 1 張
    if val_split_idx == 0 and len(train_image_paths) > 1:
        val_split_idx = 1
    elif val_split_idx == len(train_image_paths) and val_split_idx > 0:
        val_split_idx -= 1
        
    val_paths = train_image_paths[:val_split_idx]
    train_paths = train_image_paths[val_split_idx:]

    # 2. 定義所有 .txt 檔案的路徑
    paths_dict = {
        'train': os.path.join(iter_save_dir, 'data_splits', 'train.txt'),
        'val': os.path.join(iter_save_dir, 'data_splits', 'val.txt'),
        'test': os.path.join(iter_save_dir, 'data_splits', 'test.txt'),
        'pool': os.path.join(iter_save_dir, 'data_splits', 'pool.txt')
    }
    
    # 3. 寫入 .txt 檔案
    _write_paths_to_file(paths_dict['train'], train_paths)
    _write_paths_to_file(paths_dict['val'], val_paths)
    _write_paths_to_file(paths_dict['test'], test_image_paths)
    
    pool_txt_path = None
    if pool_image_paths:
        _write_paths_to_file(paths_dict['pool'], pool_image_paths)
        pool_txt_path = os.path.abspath(paths_dict['pool'])

    # 4. 建立 data.yaml 檔案
    yaml_path = os.path.join(iter_save_dir, f"{iter_name}.yaml")
    source_dir_abs = os.path.abspath(config['dataset']['source_dir'])
    
    yaml_content = {
        'path': source_dir_abs, # 指向原始資料夾 (YOLOv8 需要)
        'train': os.path.abspath(paths_dict['train']),
        'val': os.path.abspath(paths_dict['val']),
        'test': os.path.abspath(paths_dict['test']),
        'nc': config['dataset']['nc'],
        'names': config['dataset']['names']
    }
    
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_content, f, default_flow_style=False, sort_keys=False)
        
    print(f"[Data] 已為 {iter_name} 生成資料檔案:")
    print(f"  Train: {len(train_paths)} | Val: {len(val_paths)} | Test: {len(test_image_paths)} | Pool: {len(pool_image_paths)}")
    
    return os.path.abspath(yaml_path), pool_txt_path
