# utils/error_analyzer.py

import os
import torch
import numpy as np
import pandas as pd
import yaml
import torchvision  
from ultralytics import YOLO
from torchvision.ops import box_iou
from tqdm import tqdm
from collections import defaultdict
from ultralytics.engine.results import Boxes

# --- 輔助函數：計算 A 包含 B 的面積百分比 ---
def _box_area(boxes: torch.Tensor) -> torch.Tensor:
    """計算 (xyxy) 格式框的面積"""
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

def calculate_containment(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    計算 containment_matrix[i, j] = Area(Intersection(i, j)) / Area(j)
    即：box_i 包含了 box_j 的面積百分比。
    """
    inter_tl = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    inter_br = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    inter_wh = (inter_br - inter_tl).clamp(min=0)
    inter_area = inter_wh[:, :, 0] * inter_wh[:, :, 1] # Shape: (N, M)
    
    area2 = _box_area(boxes2) # Shape: (M)
    
    # 避免除以零
    return inter_area / (area2[None, :] + 1e-6)


# --- 方案一 (臨床) 邏輯：3 階段 NMS ---
def apply_complex_nms(boxes_obj: Boxes, config: dict) -> Boxes:
    """
    保護高風險類別與大框，避免 Recall 下降。
    """
    nms_config = config.get('clinical_nms', {})
    SAME_CLASS_IOU = nms_config.get('same_class_iou_thresh', 0.45)
    CROSS_CLASS_IOU = nms_config.get('cross_class_iou_thresh', 0.7)
    CONTAINMENT_THRESH = nms_config.get('containment_thresh', 0.8)
    
    # 獲取類別名稱以定義優先級 (需確保 config 中有 names)
    class_names = config['dataset']['names'] 
    
    # 定義類別優先權 (數值越大越重要，不易被抑制)
    priority_map = {'tumor': 2, 'polyp': 1} 

    boxes_xyxy = boxes_obj.xyxy
    boxes_conf = boxes_obj.conf
    boxes_cls = boxes_obj.cls
    n = len(boxes_xyxy)
    
    if n == 0:
        return boxes_obj

    iou_matrix = torchvision.ops.box_iou(boxes_xyxy, boxes_xyxy)
    containment_matrix = calculate_containment(boxes_xyxy, boxes_xyxy)

    # 依照信心度排序 (預設邏輯)
    sorted_indices = torch.argsort(boxes_conf, descending=True)
    suppressed_mask = torch.zeros(n, dtype=torch.bool, device=boxes_xyxy.device)

    for i_idx_val in range(n):
        i = sorted_indices[i_idx_val].item() # 高信心框 i
        if suppressed_mask[i]: continue
            
        for j_idx_val in range(i_idx_val + 1, n):
            j = sorted_indices[j_idx_val].item() # 低信心框 j
            if suppressed_mask[j]: continue
                
            cls_i_name = class_names[int(boxes_cls[i])]
            cls_j_name = class_names[int(boxes_cls[j])]
            
            # 取得優先級 (預設為 0)
            prio_i = priority_map.get(cls_i_name, 0)
            prio_j = priority_map.get(cls_j_name, 0)

            is_same_class = (boxes_cls[i] == boxes_cls[j])
            iou = iou_matrix[i, j]
            
            # i 包含 j (Big High Conf contains Small Low Conf) -> 通常刪除 j (重複的小框)
            containment_i_on_j = containment_matrix[i, j] 
            
            # j 包含 i (Big Low Conf contains Small High Conf) -> 這是導致 Recall 下降的主因
            containment_j_on_i = containment_matrix[j, i] 
            
            suppress = False
            
            # --- 邏輯修正開始 ---

            # 規則 1: 同類別 IoU (標準 NMS)
            if is_same_class and iou > SAME_CLASS_IOU:
                suppress = True
                
            # 規則 2: 跨類別 IoU (加入優先權保護)
            elif (not is_same_class) and iou > CROSS_CLASS_IOU:
                # 只有當 j 的優先權 <= i 的優先權時，才抑制 j
                # 如果 j 是 tumor (2) 而 i 是 polyp (1)，即使 i 信心高，也不刪除 j
                if prio_j <= prio_i:
                    suppress = True
            
            # 規則 3: 包含率 (Big High Conf covers Small Low Conf)
            # 大框信心高，包含小框 -> 刪除小框 (合理)
            if (not suppress) and (containment_i_on_j > CONTAINMENT_THRESH):
                 if prio_j <= prio_i: # 同樣保護重要類別
                    suppress = True

            # 規則 4: 反向包含 (Big Low Conf covers Small High Conf)
            if (not suppress) and (containment_j_on_i > CONTAINMENT_THRESH):
                if is_same_class:
                    suppress = True
                else:
                    # 跨類別且 j 包住 i (如 Tumor 包住 Polyp) -> 保留 j (Tumor)
                    # 除非 j 的優先權很低
                    if prio_j < prio_i:
                        suppress = True

            if suppress:
                suppressed_mask[j] = True
                
    final_keep_indices = torch.where(~suppressed_mask)[0]
    return boxes_obj[final_keep_indices]

# --- 來自 hard_case_miner.py 的輔助函數 (保持一致性) ---
def _xywhn_to_xyxyn(boxes_xywhn, w=1, h=1):
    if boxes_xywhn.numel() == 0:
        return torch.empty((0, 4), device=boxes_xywhn.device)
    xy = boxes_xywhn[:, :2]
    wh = boxes_xywhn[:, 2:]
    xy1 = xy - wh / 2
    xy2 = xy + wh / 2
    return torch.cat([xy1, xy2], dim=1)

def _load_gts(label_path: str, device):
    if not os.path.exists(label_path):
        return torch.empty((0), device=device, dtype=torch.long), \
               torch.empty((0, 4), device=device)
               
    labels = np.loadtxt(label_path).reshape(-1, 5)
    gt_classes = torch.tensor(labels[:, 0], device=device, dtype=torch.long)
    gt_boxes_xywhn = torch.tensor(labels[:, 1:], device=device, dtype=torch.float32)
    gt_boxes_xyxyn = _xywhn_to_xyxyn(gt_boxes_xywhn)
    return gt_classes, gt_boxes_xyxyn

# --- 錯誤分析核心邏輯 (S2-分析) ---
#
def _analyze_single_image_errors(preds: Boxes, gts, criteria, class_names):
    """
    分析單張影像的 FN, FP, Class Errors。
    回傳: {fn_count_dict, fp_count_dict, class_error_count_dict, fn_details, ...}
    """
    
    pred_classes = preds.cls
    pred_confs = preds.conf
    pred_boxes_xyxyn = preds.xyxyn
    
    gt_classes, gt_boxes_xyxyn = gts
    
    # 初始化逐類別計數器
    fn_count = {name: 0 for name in class_names}
    fp_count = {name: 0 for name in class_names}
    class_error_count = {name: 0 for name in class_names} # 意義: GT 是 'name', 但被標成 'other'
    
    fn_details = []
    fp_details = []
    class_error_details = []
    
    #
    match_iou_thresh = criteria['mining_criteria']['match_iou_threshold']
    fp_conf_thresh = criteria['mining_criteria']['fp_confidence_threshold']

    num_preds = len(pred_classes)
    num_gts = len(gt_classes)
    
    # --- 處理邊緣情況 ---
    if num_gts == 0 and num_preds == 0:
        return {'fn_count': fn_count, 'fp_count': fp_count, 'class_error_count': class_error_count, 
                'fn_details': '', 'fp_details': '', 'class_error_details': ''}
    
    if num_gts > 0 and num_preds == 0:
        for i in range(num_gts):
            gt_class_name = class_names[gt_classes[i].item()]
            fn_count[gt_class_name] += 1 
            fn_details.append(f"Missed '{gt_class_name}' [GT_ID: {i}]")
        
    elif num_gts == 0 and num_preds > 0:
        for i in range(num_preds):
            if pred_confs[i] >= fp_conf_thresh:
                pred_class_name = class_names[int(pred_classes[i].item())]
                fp_count[pred_class_name] += 1
                fp_details.append(f"FP: '{pred_class_name}' (Conf: {pred_confs[i]:.2f}) [Pred_ID: {i}]")
                
    elif num_gts > 0 and num_preds > 0:
        # --- 主要匹配邏輯 ---
        iou_matrix = box_iou(pred_boxes_xyxyn, gt_boxes_xyxyn)
        best_pred_iou, best_pred_idx = iou_matrix.max(dim=0)
        matched_pred_indices = set()
        
        # 1. 遍歷 GTs (檢查 FN 和分類錯誤)
        for gt_i in range(num_gts):
            gt_class_name = class_names[gt_classes[gt_i].item()]
            
            if best_pred_iou[gt_i] >= match_iou_thresh:
                pred_idx = best_pred_idx[gt_i].item()
                
                if pred_classes[pred_idx] != gt_classes[gt_i]:
                    pred_class_name = class_names[int(pred_classes[pred_idx].item())]
                    class_error_count[gt_class_name] += 1 
                    class_error_details.append(f"Classified '{gt_class_name}' as '{pred_class_name}' [GT_ID: {gt_i} <-> Pred_ID: {pred_idx}]")
                    
                matched_pred_indices.add(pred_idx)
            else:
                fn_count[gt_class_name] += 1 
                fn_details.append(f"Missed '{gt_class_name}' [GT_ID: {gt_i}]")
                
        # 2. 遍歷 Predictions (檢查 FP)
        for pred_i in range(num_preds):
            if pred_i not in matched_pred_indices:
                if pred_confs[pred_i] >= fp_conf_thresh:
                    pred_class_name = class_names[int(pred_classes[pred_i].item())]
                    fp_count[pred_class_name] += 1 
                    fp_details.append(f"FP: '{pred_class_name}' (Conf: {pred_confs[pred_i]:.2f}) [Pred_ID: {pred_i}]")
    
    return {
        'fn_count': fn_count, 
        'fp_count': fp_count, 
        'class_error_count': class_error_count, 
        'fn_details': "; ".join(fn_details),
        'fp_details': "; ".join(fp_details),
        'class_error_details': "; ".join(class_error_details),
    }


def analyze_image_errors_on_test_set(model_path: str, test_data_yaml_path: str, label_dir: str, config: dict) -> list:
    """
    載入模型，對測試集逐張影像進行 *雙方案* 錯誤分析。
    
    - 方案 1 (臨床): 應用 v1.4 複雜 3 階段 NMS，模擬臨床顯示。
    - 方案 2 (分析): 使用原始 (同類別 NMS) 預測，分析所有錯誤。
    
    回傳: [ {image_path, gt_count_polyp, s1_fn_count_polyp, ...}, ... ]
    """
    device = 0 if torch.cuda.is_available() else 'cpu'
    model = YOLO(model_path)
    
    # 1. 讀取 test_data.yaml 找到 test.txt
    try:
        with open(test_data_yaml_path, 'r') as f:
            data_yaml = yaml.safe_load(f)
        test_txt_path = data_yaml.get('test')
        if not test_txt_path or not os.path.exists(test_txt_path):
            print(f"[Error Analysis] 找不到 test.txt 檔案: {test_txt_path}")
            return []
    except Exception as e:
        print(f"[Error Analysis] 讀取 {test_data_yaml_path} 失敗: {e}")
        return []

    # 2. 讀取 test.txt 獲取影像路徑
    with open(test_txt_path, 'r') as f:
        image_paths = [line.strip() for line in f if line.strip()]
        
    criteria = config # _analyze_single_image_errors 需要完整的 config
    class_names = config['dataset']['names'] #
    all_image_results = []
    
    for img_path in tqdm(image_paths, desc="[Error Analysis v1.6] 分析測試集"):
        if not os.path.exists(img_path):
            continue
            
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        
        # 1. 執行預測 (僅執行一次)
        try:
            raw_preds_results = model.predict(img_path, verbose=False, device=device)
            raw_preds_boxes = raw_preds_results[0].boxes
        except Exception as e:
            print(f"[Error Analysis] 預測失敗 {img_path}: {e}")
            continue
            
        # 2. 載入 Ground Truth
        label_path = os.path.join(label_dir, f"{base_name}.txt")
        gt_classes, gt_boxes_xyxyn = _load_gts(label_path, device=device)
        
        # --- [NEW v1.6] 獲取逐類別 GT 總數 ---
        gt_counts_per_class = {name: 0 for name in class_names}
        for gt_cls_idx in gt_classes:
            gt_counts_per_class[class_names[gt_cls_idx.item()]] += 1
        gt_count_total = len(gt_classes)
        gts = (gt_classes, gt_boxes_xyxyn)
        
        # --- 3. 執行方案一 (臨床) 分析 ---
        clinical_preds_boxes = apply_complex_nms(raw_preds_boxes, config)
        error_info_s1 = _analyze_single_image_errors(clinical_preds_boxes, gts, criteria, class_names)
        
        # --- 4. 執行方案二 (分析) 分析 ---
        error_info_s2 = _analyze_single_image_errors(raw_preds_boxes, gts, criteria, class_names)

        # --- 5. 組合結果 ---
        s1_total_errors = sum(error_info_s1['fn_count'].values()) + sum(error_info_s1['fp_count'].values()) + sum(error_info_s1['class_error_count'].values())
        s2_total_errors = sum(error_info_s2['fn_count'].values()) + sum(error_info_s2['fp_count'].values()) + sum(error_info_s2['class_error_count'].values())
        
        s1_status = "Correct (TP)" if s1_total_errors == 0 else "Error"
        s2_status = "Correct (TP)" if s2_total_errors == 0 else "Error"

        # 建立一個扁平的字典，用於 Excel
        combined_info = {
            'image_path': img_path,
            'image_basename': base_name,
            'gt_count_total': gt_count_total, # 總 GT 數
            
            # 狀態與詳細
            's1_status': s1_status,
            's1_fn_details': error_info_s1['fn_details'],
            's1_fp_details': error_info_s1['fp_details'],
            's1_class_error_details': error_info_s1['class_error_details'],

            's2_status': s2_status,
            's2_fn_details': error_info_s2['fn_details'],
            's2_fp_details': error_info_s2['fp_details'],
            's2_class_error_details': error_info_s2['class_error_details'],
        }
        
        # 動態新增所有逐類別的 GT/S1/S2 計數
        for name in class_names:
            # GT
            combined_info[f'gt_count_{name}'] = gt_counts_per_class.get(name, 0)
            
            # S1 (臨床)
            combined_info[f's1_fn_count_{name}'] = error_info_s1['fn_count'].get(name, 0)
            combined_info[f's1_fp_count_{name}'] = error_info_s1['fp_count'].get(name, 0)
            combined_info[f's1_class_error_count_{name}'] = error_info_s1['class_error_count'].get(name, 0)
            
            # S2 (分析)
            combined_info[f's2_fn_count_{name}'] = error_info_s2['fn_count'].get(name, 0)
            combined_info[f's2_fp_count_{name}'] = error_info_s2['fp_count'].get(name, 0)
            combined_info[f's2_class_error_count_{name}'] = error_info_s2['class_error_count'].get(name, 0)
        
        all_image_results.append(combined_info)
            
    del model
    torch.cuda.empty_cache()
    
    return all_image_results

# --- Excel 報告生成器 ---
#
def generate_fold_error_report(
    all_iter_results_with_errors: list, 
    fold_num: int, 
    project_name: str, 
    class_names: list, 
    test_image_paths: list
) -> str:
    """
    為單一 Fold 產生一個包含 *雙方案* 分析的 Excel 報告。
    1. Sheet 1 ('迭代效能總覽') 新增 S1 (臨床) 的 *逐類別* 宏觀 P/R 指標。
    2. Sheet N+ ('Iter_X_詳情') 新增所有 *逐類別* 的微觀計數欄位。
    """
    
    report_path = os.path.join(project_name, f"Fold_{fold_num}_Error_Analysis.xlsx")
    
    with pd.ExcelWriter(report_path, engine='openpyxl') as writer:
        
        # --- Sheet 1: 迭代效能總覽 ---
        overview_data = []
        
        # S2 (分析) 欄位
        metric_cols = ['metrics/mAP50(B)', 'metrics/precision(B)', 'metrics/recall(B)', 'metrics/mAP50-95(B)']
        class_metric_cols = ['precision', 'recall', 'mAP50', 'mAP50-95']
        
        for res in all_iter_results_with_errors:
            # S2 (分析) 宏觀指標 (來自 model.val())
            row = {
                'Iteration': res.get('iteration', 'N/A'),
                'Model Path': res.get('model_path', 'N/A')
            }
            overall = res.get('overall', {})
            for m in metric_cols:
                row[m] = overall.get(m, np.nan)
            
            # S2 (分析) 逐類別指標 (來自 model.val())
            per_class = res.get('per_class', {})
            for name in class_names:
                class_data = per_class.get(name, {})
                for cm in class_metric_cols:
                    row[f'S2_{name}_{cm} (分析)'] = class_data.get(cm, np.nan) # 加上 S2 前綴
                    
            # --- S1 (臨床) 宏觀指標計算 (Per-Class) ---
            total_gt_per_class = {name: 0 for name in class_names}
            total_s1_fn_per_class = {name: 0 for name in class_names}
            total_s1_fp_per_class = {name: 0 for name in class_names}
            total_s1_class_error_per_class = {name: 0 for name in class_names}
            
            image_list = res.get('error_image_list', [])
            for record in image_list:
                for name in class_names:
                    total_gt_per_class[name] += record.get(f'gt_count_{name}', 0)
                    total_s1_fn_per_class[name] += record.get(f's1_fn_count_{name}', 0)
                    total_s1_fp_per_class[name] += record.get(f's1_fp_count_{name}', 0)
                    total_s1_class_error_per_class[name] += record.get(f's1_class_error_count_{name}', 0)

            epsilon = 1e-6
            
            # 計算 S1 逐類別指標並加入 row
            for name in class_names:
                gt = total_gt_per_class[name]
                fn = total_s1_fn_per_class[name]
                fp = total_s1_fp_per_class[name]
                ce = total_s1_class_error_per_class[name] # Class Error (GT=name, Pred=other)

                # TP = Total GTs - FNs - ClassErrors
                tp = gt - fn - ce
                
                s1_recall = (tp) / (gt + epsilon)
                s1_precision = (tp) / (tp + fp + epsilon) # TP / (TP + FP)

                row[f'S1_Recall_{name} (臨床)'] = s1_recall
                row[f'S1_Precision_{name} (臨床)'] = s1_precision
                row[f'S1_TP_{name} (臨床)'] = tp
                row[f'S1_FN_{name} (臨床)'] = fn
                row[f'S1_FP_{name} (臨床)'] = fp
                row[f'S1_CE_{name} (臨床)'] = ce # Class Error (GT=name, Pred=other)
                row[f'S1_GT_{name} (臨床)'] = gt
            
            # 計算 S1 總體 (Overall) 指標
            total_gt = sum(total_gt_per_class.values())
            total_s1_fn = sum(total_s1_fn_per_class.values())
            total_s1_fp = sum(total_s1_fp_per_class.values())
            total_s1_ce = sum(total_s1_class_error_per_class.values())
            
            total_s1_tp = total_gt - total_s1_fn - total_s1_ce

            row['S1_Recall (臨床)'] = (total_s1_tp) / (total_gt + epsilon)
            row['S1_Precision (臨床)'] = (total_s1_tp) / (total_s1_tp + total_s1_fp + epsilon)
            row['S1_Total_TPs (臨床)'] = total_s1_tp
            row['S1_Total_FNs (臨床)'] = total_s1_fn
            row['S1_Total_FPs (臨床)'] = total_s1_fp
            row['S1_Total_GTs (臨床)'] = total_gt
            # ---S1 計算結束 ---
            
            overview_data.append(row)
            
        overview_df = pd.DataFrame(overview_data)
        # [MODIFIED v1.5] 重新命名工作表
        overview_df.to_excel(writer, sheet_name='迭代效能總覽 (S1+S2)', index=False, float_format="%.5f")

        # --- [NEW] 建立 Pivot 表的原始數據 (S1 和 S2) ---
        pivot_data_s1 = []
        pivot_data_s2 = []
        for res in all_iter_results_with_errors:
            iter_num = res.get('iteration', 'N/A')
            for record in res.get('error_image_list', []):
                
                # S1 總錯誤
                total_err_s1 = 0
                for name in class_names:
                    total_err_s1 += record.get(f's1_fn_count_{name}', 0)
                    total_err_s1 += record.get(f's1_fp_count_{name}', 0)
                    total_err_s1 += record.get(f's1_class_error_count_{name}', 0)

                if total_err_s1 > 0:
                    pivot_data_s1.append({
                        'image_basename': record['image_basename'],
                        'Iteration': iter_num,
                        'Total_Errors_S1': total_err_s1
                    })
                
                # S2 總錯誤
                total_err_s2 = 0
                for name in class_names:
                    total_err_s2 += record.get(f's2_fn_count_{name}', 0)
                    total_err_s2 += record.get(f's2_fp_count_{name}', 0)
                    total_err_s2 += record.get(f's2_class_error_count_{name}', 0)
                    
                if total_err_s2 > 0:
                    pivot_data_s2.append({
                        'image_basename': record['image_basename'],
                        'Iteration': iter_num,
                        'Total_Errors_S2': total_err_s2
                    })

        # --- Sheet 2: 錯誤影像追蹤 (S1 - 臨床) ---
        if pivot_data_s1:
            pivot_df_s1 = pd.DataFrame(pivot_data_s1)
            try:
                pivot_table_s1 = pivot_df_s1.pivot_table(
                    index='image_basename', 
                    columns='Iteration', 
                    values='Total_Errors_S1', 
                    fill_value=0, 
                    aggfunc='sum'
                )
                pivot_table_s1.to_excel(writer, sheet_name='錯誤追蹤 (S1-臨床)')
            except Exception as e:
                pd.DataFrame(pivot_data_s1).to_excel(writer, sheet_name='錯誤追蹤 (S1-Raw)')
        else:
            pd.DataFrame([{'Message': '方案1 (臨床) 在測試集上未發現任何錯誤'}]).to_excel(writer, sheet_name='錯誤追蹤 (S1-臨床)')

        # --- Sheet 3: 錯誤影像追蹤 (S2 - 分析) ---
        if pivot_data_s2:
            pivot_df_s2 = pd.DataFrame(pivot_data_s2)
            try:
                pivot_table_s2 = pivot_df_s2.pivot_table(
                    index='image_basename', 
                    columns='Iteration', 
                    values='Total_Errors_S2', 
                    fill_value=0, 
                    aggfunc='sum'
                )
                pivot_table_s2.to_excel(writer, sheet_name='錯誤追蹤 (S2-分析)')
            except Exception as e:
                pd.DataFrame(pivot_data_s2).to_excel(writer, sheet_name='錯誤追蹤 (S2-Raw)')
        else:
            pd.DataFrame([{'Message': '方案2 (分析) 在測試集上未發現任何錯誤'}]).to_excel(writer, sheet_name='錯誤追蹤 (S2-分析)')


        # --- Sheet 4...N: Iter_X_詳情 (Combined) ---
        for res in all_iter_results_with_errors:
            iter_num = res.get('iteration', 'N/A')
            sheet_name = f'Iter_{iter_num}_詳情 (Combined)'
            
            full_log_list = res.get('error_image_list', [])
            
            if full_log_list:
                iter_df = pd.DataFrame(full_log_list)
                
                # 動態產生欄位順序
                cols_order = ['image_basename', 'gt_count_total']
                for name in class_names:
                    cols_order.append(f'gt_count_{name}')
                
                cols_order.extend(['s1_status', 's2_status'])
                
                # 動態加入所有微觀計數
                for prefix in ['s1_fn_count', 's1_fp_count', 's1_class_error_count', 
                               's2_fn_count', 's2_fp_count', 's2_class_error_count']:
                    for name in class_names:
                        cols_order.append(f'{prefix}_{name}')
                        
                cols_order.extend([
                    's1_fn_details', 's1_fp_details', 's1_class_error_details',
                    's2_fn_details', 's2_fp_details', 's2_class_error_details', 
                    'image_path'
                ])
                
                final_cols = [c for c in cols_order if c in iter_df.columns]
                iter_df[final_cols].to_excel(writer, sheet_name=sheet_name, index=False)
            else:
                pd.DataFrame([{'Message': f'Iteration {iter_num} 沒有可分析的日誌'}]).to_excel(writer, sheet_name=sheet_name, index=False)

        # --- Sheet N+1: 測試集影像列表 (Test Set Image List) ---
        if test_image_paths:
            test_set_df = pd.DataFrame(test_image_paths, columns=['image_path'])
            test_set_df['image_basename'] = test_set_df['image_path'].apply(os.path.basename)
            test_set_df[['image_basename', 'image_path']].to_excel(writer, sheet_name='測試集影像列表', index=False)
            
    return report_path
