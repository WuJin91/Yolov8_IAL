# utils/reporter.py

import pandas as pd
from openpyxl import Workbook
import os
import numpy as np
from datetime import datetime

def generate_kfold_report(
    all_fold_final_results: list, 
    all_iteration_results: list, 
    config: dict,
    kfold_split_stats: list = None,
    mining_efficiency_stats: list = None
):
    """
    生成 K-Fold 報告。
    """
    if not all_fold_final_results and not all_iteration_results:
        print("[報告] 無結果數據，跳過生成。")
        return

    project_name = config.get('project', {}).get('name', 'default_project')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(project_name, f"KFold_AL_Report_{timestamp}.xlsx")
    
    class_names = config['dataset']['names']
    main_metrics = ['metrics/precision(B)', 'metrics/recall(B)', 'metrics/mAP50(B)', 'metrics/mAP50-95(B)']
    
    # --- Sheet 1: K-Fold Summary (總結) ---
    summary_data = []
    # 定義總結的欄位順序
    cols_summary = ['Fold'] + main_metrics + ['Iteration', 'Duration']
    for name in class_names:
        cols_summary.extend([f'{name}_P', f'{name}_R', f'{name}_mAP50'])
        
    for res in all_fold_final_results:
        row = {
            'Fold': res.get('fold'),
            'Iteration': res.get('iteration'),
            'Duration': res.get('Duration', '-')
        }
        
        # 填入整體指標
        for m in main_metrics:
            row[m] = res['overall'].get(m, 0)
            
        # 填入類別指標
        if 'per_class' in res:
            for cls, metrics in res['per_class'].items():
                row[f'{cls}_P'] = metrics.get('precision', 0)
                row[f'{cls}_R'] = metrics.get('recall', 0)
                row[f'{cls}_mAP50'] = metrics.get('mAP50', 0)
        
        summary_data.append(row)
        
    df_summary = pd.DataFrame(summary_data)
    # 重新排序欄位
    final_summary_cols = [c for c in cols_summary if c in df_summary.columns]
    df_summary = df_summary[final_summary_cols]

    # --- Sheet 2: Learning Curve (學習曲線) ---
    curve_data = []
    for res in all_iteration_results:
        row = res.get('overall', {}).copy()
        row['Fold'] = res.get('fold')
        row['Iteration'] = res.get('iteration')
        row['Duration'] = res.get('Duration', '-')
        
        # 攤平類別指標
        for k, v in res.items():
            if k not in ['overall', 'per_class', 'fold', 'iteration', 'confusion_matrix', 'Duration', 'Model Path']:
                row[k] = v
        
        # 確保加入詳細類別指標
        if 'per_class' in res:
            for cls, metrics in res['per_class'].items():
                row[f'{cls}_precision'] = metrics.get('precision', 0)
                row[f'{cls}_recall'] = metrics.get('recall', 0)
                row[f'{cls}_mAP50'] = metrics.get('mAP50', 0)
                row[f'{cls}_mAP50-95'] = metrics.get('mAP50-95', 0)

        curve_data.append(row)
    
    df_curve = pd.DataFrame(curve_data)
    
    # 強制欄位排序：Fold, Iteration, Duration 在最前面
    front_cols = ['Fold', 'Iteration', 'Duration']
    remaining_cols = [c for c in df_curve.columns if c not in front_cols]
    # 將剩餘欄位依照字母或邏輯排序，或保持原樣
    metric_cols = [c for c in remaining_cols if 'metrics' in c]
    other_cols = [c for c in remaining_cols if c not in metric_cols]
    
    final_curve_cols = front_cols + metric_cols + other_cols
    # 確保只選取存在的欄位
    final_curve_cols = [c for c in final_curve_cols if c in df_curve.columns]
    df_curve = df_curve[final_curve_cols]

    # --- Sheet 3: Mining Efficiency (挖掘效益) ---
    if mining_efficiency_stats:
        df_mining = pd.DataFrame(mining_efficiency_stats)
        mining_cols = [
            'Fold', 'Iteration', 'mAP50', 
            'Performance Gain', 'Total_FN', 'FN Reduction Rate', 
            'Added_Samples', 'Avg_Hardness_Score', 'Duration'
        ]
        exist_cols = [c for c in mining_cols if c in df_mining.columns]
        df_mining = df_mining[exist_cols]
    else:
        df_mining = pd.DataFrame()

    # --- Sheet 4: Data Split Stats (資料劃分) ---
    if kfold_split_stats:
        df_split = pd.DataFrame(kfold_split_stats)
        split_cols = [
            'Fold', 
            'Train_Images', 'Train_Patients', 
            'Pool_Images', 'Pool_Patients', 
            'Test_Images', 'Test_Patients'
        ]
        exist_split_cols = [c for c in split_cols if c in df_split.columns]
        df_split = df_split[exist_split_cols]
    else:
        df_split = pd.DataFrame()

    # --- 寫入 Excel ---
    with pd.ExcelWriter(report_path, engine='openpyxl') as writer:
        if not df_summary.empty:
            df_summary.to_excel(writer, sheet_name='K-Fold 總結', index=False)
        
        if not df_curve.empty:
            df_curve.to_excel(writer, sheet_name='學習曲線', index=False)
            
        if not df_mining.empty:
            df_mining.to_excel(writer, sheet_name='挖掘效益', index=False)
            
        if not df_split.empty:
            df_split.to_excel(writer, sheet_name='資料劃分', index=False)
            
        # Confusion Matrices
        for res in all_fold_final_results:
            fold_num = res.get('fold')
            cm_info = res.get('confusion_matrix')
            if cm_info:
                matrix = cm_info['matrix']
                base_names = cm_info['class_names']
                
                # 建構標籤
                num_rows = len(matrix)
                num_cols = len(matrix[0]) if num_rows > 0 else 0
                
                row_labels = base_names[:]
                if len(row_labels) < num_rows:
                    row_labels.append('background (FN)')
                elif len(row_labels) > num_rows:
                    row_labels = row_labels[:num_rows]
                    
                col_labels = base_names[:]
                if len(col_labels) < num_cols:
                    col_labels.append('background (FP)')
                elif len(col_labels) > num_cols:
                    col_labels = col_labels[:num_cols]
                
                pd.DataFrame(matrix, index=row_labels, columns=col_labels).to_excel(
                    writer, sheet_name=f'CM Fold {fold_num}'
                )

        # Config
        pd.json_normalize(config).T.to_excel(writer, sheet_name='實驗設定', header=False)
        
    print(f"✅ 完整報告已生成: {report_path}")
