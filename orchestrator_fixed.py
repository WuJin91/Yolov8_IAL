# utils/orchestrator_fixed.py

import os
import copy
import time
import math
import random
import json
import numpy as np
from .data_manager import prepare_iteration_data_files, get_patient_id
from .trainer import run_single_training
from .tester import run_final_test
from .hard_case_miner import find_hard_cases
from .error_analyzer import analyze_image_errors_on_test_set, generate_fold_error_report

def run_fixed_experiment(config: dict, split_file: str, mining_ratio: float) -> tuple:
    """
    執行基於固定分割與指定挖掘比例的主動學習實驗。
    """
    project_name = config['project']['name']
    k_folds = config['automation']['k_folds']
    al_config = config['active_learning']
    num_iterations = al_config['iterations']
    base_model_weights = config['model']['variant']
    num_classes = config['dataset']['nc']
    
    print(f"[System] 讀取固定分割檔: {split_file}")
    with open(split_file, 'r', encoding='utf-8') as f:
        fixed_splits = json.load(f)
    
    if len(fixed_splits) != k_folds:
        raise ValueError(f"分割檔包含 {len(fixed_splits)} folds，但 config 要求 {k_folds} folds")

    # 結果容器
    all_fold_final_results = []
    all_iteration_metrics = []
    kfold_split_stats = []
    mining_efficiency_stats = []
    
    overall_start_time = time.time()

    # --- K-Fold Loop ---
    for k in range(k_folds):
        print(f"\n{'='*60}")
        print(f"   開始 Fold {k+1} / {k_folds}")
        print(f"{'='*60}")
        
        # 1. 分配 Train/Test/Pool
        fold_indices = list(range(k_folds))
        test_fold_idx = k
        train_fold_indices = [(test_fold_idx + 1 + i) % k_folds for i in range(al_config['initial_train_folds'])]
        pool_fold_indices = [i for i in fold_indices if i != test_fold_idx and i not in train_fold_indices]
        
        test_paths = fixed_splits[f"Fold_{test_fold_idx+1}"]
        active_train_paths = []
        for idx in train_fold_indices:
            active_train_paths.extend(fixed_splits[f"Fold_{idx+1}"])
            
        candidate_pool_paths = []
        for idx in pool_fold_indices:
            candidate_pool_paths.extend(fixed_splits[f"Fold_{idx+1}"])
        
        # 計算病患人數 (Patient Counts)
        train_patients = set(get_patient_id(p) for p in active_train_paths)
        pool_patients = set(get_patient_id(p) for p in candidate_pool_paths)
        test_patients = set(get_patient_id(p) for p in test_paths)

        # 紀錄分割統計
        kfold_split_stats.append({
            'Fold': k+1,
            'Train_Images': len(active_train_paths),
            'Train_Patients': len(train_patients),
            'Pool_Images': len(candidate_pool_paths),
            'Pool_Patients': len(pool_patients),
            'Test_Images': len(test_paths),
            'Test_Patients': len(test_patients)
        })
        
        # 準備 Test 的 YAML
        fold_dir = os.path.join(project_name, f"fold_{k+1}")
        os.makedirs(fold_dir, exist_ok=True)
        test_data_yaml, _ = prepare_iteration_data_files(
            fold_dir, "test_only", config, [], test_paths, []
        )
        label_dir = os.path.join(config['dataset']['source_dir'], 'labels')
        
        best_iter_map50 = 0.0
        best_iter_results = None
        fold_error_reports = []
        previous_weights = None
        
        # 用於計算 Gain 與 Reduction 的變數
        prev_map50 = 0.0
        prev_fn = 0
        
        # 記錄上一輪"挖掘"的資訊
        last_mining_info = {'count': 0, 'avg_hardness': 0.0}

        # --- Iteration Loop ---
        for i in range(num_iterations):
            iter_name = f"fold_{k+1}_iter_{i+1}"
            iter_save_dir = os.path.join(project_name, iter_name)
            is_final = (i == num_iterations - 1)
            
            print(f"\n--- {iter_name} (Mining Ratio: {mining_ratio}) ---")
            
            # 準備訓練檔
            train_yaml, pool_txt = prepare_iteration_data_files(
                iter_save_dir, iter_name, config,
                active_train_paths, test_paths, 
                [] if is_final else candidate_pool_paths
            )
            
            # 設定訓練參數
            iter_config = copy.deepcopy(config)
            if i == 0:
                load_weights = base_model_weights
                iter_config['training'] = config['training']
                print(f"[Strategy] Iter 1: From Scratch")
            else:
                load_weights = previous_weights
                iter_config['training'] = config['finetune_training']
                print(f"[Strategy] Iter {i+1}: Finetune")

            # A. 訓練
            _, save_dir, duration_str = run_single_training(iter_config, iter_name, train_yaml, load_weights)
            current_weights = os.path.join(save_dir, 'weights', 'best.pt')
            previous_weights = current_weights
            
            # B. 測試 (確保取得 Confusion Matrix)
            test_res = run_final_test(current_weights, test_data_yaml, config)
            
            # 補全數據
            test_res['Duration'] = duration_str
            test_res['Model Path'] = current_weights
            test_res['iteration'] = i + 1
            test_res['fold'] = k + 1
            
            if 'per_class' in test_res:
                for cls, mets in test_res['per_class'].items():
                    for mk, mv in mets.items(): 
                        test_res[f"{cls}_{mk}"] = mv
            
            # C. 計算 Total FN (從 Confusion Matrix 計算，這是最準確的方式)
            current_total_fn = 0
            if 'confusion_matrix' in test_res and test_res['confusion_matrix']:
                try:
                    # 取得矩陣 (通常是 N+1 x N+1，最後一列/行是 Background)
                    matrix = np.array(test_res['confusion_matrix']['matrix'])
                    # 在 Ultralytics 中，Matrix[GT_Class, Pred_Class]
                    # Pred_Class 的最後一個 index (num_classes) 代表 Background (FN)
                    # 只計算有效類別的 FN (不包含 Background 的 FN，那是無意義的)
                    fn_column_index = num_classes 
                    
                    # 加總所有物件類別 (0 ~ nc-1) 被預測為 Background 的數量
                    current_total_fn = np.sum(matrix[0:num_classes, fn_column_index])
                    print(f"   [Metrics] 從 Confusion Matrix 計算 Total FN: {current_total_fn}")
                except Exception as e:
                    print(f"[Warning] 無法從 Confusion Matrix 計算 FN: {e}")
            
            # D. 詳細錯誤分析 (僅用於生成詳細報告圖片清單，不影響統計數據)
            print(f"   [Analysis] 正在生成詳細錯誤報告 (影像層級)...")
            try:
                image_errors_list = analyze_image_errors_on_test_set(
                    model_path=current_weights,
                    test_data_yaml_path=test_data_yaml,
                    label_dir=label_dir,
                    config=config
                )
                
                # 計算 S1 (臨床) 指標並寫入 test_res (用於學習曲線)
                # S1: polyp_P, polyp_R (Clinical NMS)
                # S2: polyp_precision, polyp_recall (Standard NMS - already in test_res)
                s1_class_names = config['dataset']['names']
                s1_stats = {name: {'gt': 0, 'fn': 0, 'fp': 0, 'ce': 0} for name in s1_class_names}
                
                for record in image_errors_list:
                    for name in s1_class_names:
                        s1_stats[name]['gt'] += record.get(f'gt_count_{name}', 0)
                        s1_stats[name]['fn'] += record.get(f's1_fn_count_{name}', 0)
                        s1_stats[name]['fp'] += record.get(f's1_fp_count_{name}', 0)
                        s1_stats[name]['ce'] += record.get(f's1_class_error_count_{name}', 0)
                
                for name in s1_class_names:
                    gt = s1_stats[name]['gt']
                    fn = s1_stats[name]['fn']
                    fp = s1_stats[name]['fp']
                    ce = s1_stats[name]['ce']
                    tp = gt - fn - ce
                    epsilon = 1e-6
                    test_res[f"{name}_P"] = tp / (tp + fp + epsilon)
                    test_res[f"{name}_R"] = tp / (gt + epsilon)

                iteration_summary = {
                    'iteration': i + 1,
                    'model_path': current_weights,
                    'overall': test_res.get('overall', {}),
                    'per_class': test_res.get('per_class', {}),
                    'error_image_list': image_errors_list
                }
                fold_error_reports.append(iteration_summary)
                
            except Exception as e:
                print(f"[Warning] 詳細錯誤分析失敗: {e}")

            # E. 計算效益指標 (Delta)
            current_map50 = test_res['overall']['metrics/mAP50(B)']
            
            if i == 0:
                perf_gain = 0.0
                fn_reduction_rate = 0.0
                fn_reduction_str = "-"
            else:
                perf_gain = current_map50 - prev_map50
                if prev_fn > 0:
                    reduction = (prev_fn - current_total_fn) / prev_fn
                    fn_reduction_rate = reduction
                    fn_reduction_str = f"{reduction:.2%}"
                elif prev_fn == 0 and current_total_fn == 0:
                    fn_reduction_str = "0.00% (Stable)"
                else:
                    fn_reduction_str = "Increase (Bad)" # 避免分母為 0

            # 記錄挖掘效益
            mining_efficiency_stats.append({
                'Fold': k+1,
                'Iteration': i+1,
                'mAP50': current_map50,
                'Performance Gain': perf_gain,
                'Total_FN': int(current_total_fn), # 確保是整數
                'FN Reduction Rate': fn_reduction_str,
                'Added_Samples': last_mining_info['count'],
                'Avg_Hardness_Score': last_mining_info['avg_hardness'],
                'Duration': duration_str
            })

            # 更新歷史數據
            prev_map50 = current_map50
            prev_fn = current_total_fn
            
            if current_map50 > best_iter_map50:
                best_iter_map50 = current_map50
                best_iter_results = test_res
            
            all_iteration_metrics.append(test_res)
            
            # F. 執行挖掘 (為下一輪做準備)
            if not is_final and candidate_pool_paths:
                remaining_iters = (num_iterations - 1) - i
                total_pool_size = len(candidate_pool_paths)
                budget_size = math.ceil(total_pool_size / max(remaining_iters, 1))
                
                sorted_candidates = find_hard_cases(current_weights, pool_txt, label_dir, config)
                
                if not sorted_candidates:
                    random.shuffle(candidate_pool_paths)
                    selected_paths = candidate_pool_paths[:budget_size]
                    avg_hardness = 0
                else:
                    available_count = len(sorted_candidates)
                    budget_size = min(budget_size, available_count)
                    
                    target_hard_count = int(budget_size * mining_ratio)
                    target_random_count = budget_size - target_hard_count
                    
                    hard_selection = sorted_candidates[:target_hard_count]
                    remaining_pool_candidates = sorted_candidates[target_hard_count:]
                    random.shuffle(remaining_pool_candidates)
                    random_selection = remaining_pool_candidates[:target_random_count]
                    
                    final_selection = hard_selection + random_selection
                    selected_paths = [p for p, s in final_selection]
                    avg_hardness = np.mean([s for p, s in final_selection]) if final_selection else 0
                    
                    print(f"[Mining] Ratio {mining_ratio}: Hard={len(hard_selection)}, Random={len(random_selection)}")

                # 更新資料池
                selected_paths = [os.path.abspath(p) for p in selected_paths]
                active_train_paths.extend(selected_paths)
                sel_set = set(selected_paths)
                candidate_pool_paths = [p for p in candidate_pool_paths if os.path.abspath(p) not in sel_set]
                
                last_mining_info = {'count': len(selected_paths), 'avg_hardness': avg_hardness}

        # --- End of Iterations ---
        if best_iter_results:
            all_fold_final_results.append(best_iter_results)
        
        # 生成 Fold 詳細報告
        if fold_error_reports:
            print(f"   [Report] 生成 Fold {k+1} 詳細迭待報告...")
            try:
                generate_fold_error_report(
                    fold_error_reports, 
                    k+1, 
                    project_name, 
                    config['dataset']['names'], 
                    test_paths
                )
            except Exception as e:
                print(f"[Warning] Fold {k+1} 錯誤報告生成失敗: {e}")

    return all_fold_final_results, all_iteration_metrics, kfold_split_stats, mining_efficiency_stats
