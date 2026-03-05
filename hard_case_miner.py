# utils/hard_case_miner.py

import os
import torch
import numpy as np
from ultralytics import YOLO
from torchvision.ops import box_iou
from tqdm import tqdm

def _xywhn_to_xyxyn(boxes_xywhn, w=1, h=1):
    """將 [x_center, y_center, w, h] (normalized) 轉為 [x1, y1, x2, y2] (normalized)"""
    if boxes_xywhn.numel() == 0:
        return torch.empty((0, 4), device=boxes_xywhn.device)
    xy = boxes_xywhn[:, :2]
    wh = boxes_xywhn[:, 2:]
    xy1 = xy - wh / 2
    xy2 = xy + wh / 2
    return torch.cat([xy1, xy2], dim=1)

def _load_gts(label_path: str, device):
    """從 .txt 載入 ground truths"""
    if not os.path.exists(label_path):
        return torch.empty((0), device=device, dtype=torch.long), \
               torch.empty((0, 4), device=device)
               
    try:
        # 處理空檔案或讀取錯誤
        labels = np.loadtxt(label_path).reshape(-1, 5)
    except Exception:
        return torch.empty((0), device=device, dtype=torch.long), \
               torch.empty((0, 4), device=device)

    gt_classes = torch.tensor(labels[:, 0], device=device, dtype=torch.long)
    gt_boxes_xywhn = torch.tensor(labels[:, 1:], device=device, dtype=torch.float32)
    
    gt_boxes_xyxyn = _xywhn_to_xyxyn(gt_boxes_xywhn)
    
    return gt_classes, gt_boxes_xyxyn

def _calculate_hardness_score(preds, gts, criteria):
    """
    [Recall-Optimized] 計算單張影像的困難度分數。
    重點：加入信心度懲罰，讓模型對 GT 的信心不足也被視為困難案例。
    """
    pred_classes = preds.cls
    pred_confs = preds.conf
    pred_boxes_xyxyn = preds.xyxyn
    
    gt_classes, gt_boxes_xyxyn = gts
    
    score = 0.0
    
    # 讀取權重
    fn_weight = criteria['score_weights']['fn_weight']
    fp_weight = criteria['score_weights']['fp_weight']
    class_error_weight = criteria['score_weights']['class_error_weight']
    
    match_iou_thresh = criteria['match_iou_threshold']
    # 注意：使用較低的 conf 閾值來計算 FP，避免過度懲罰背景雜訊
    fp_conf_thresh = criteria.get('fp_confidence_threshold', 0.25) 

    num_preds = len(pred_classes)
    num_gts = len(gt_classes)
    
    # --- 1. 處理 GT 為空的情況 (純背景圖) ---
    if num_gts == 0:
        if num_preds == 0:
            return 0.0 # TN (True Negative), 分數為 0
        else:
            # 計算高信心 FP 的數量
            fp_count = torch.sum(pred_confs >= fp_conf_thresh).item()
            return fp_count * fp_weight

    # --- 2. 處理 Pred 為空的情況 (嚴重漏檢) ---
    if num_preds == 0:
        return num_gts * fn_weight * 1.5 # 加重全漏檢的懲罰

    # --- 3. 計算 IoU 與匹配 ---
    # iou_matrix: [num_preds, num_gts]
    iou_matrix = box_iou(pred_boxes_xyxyn, gt_boxes_xyxyn) 
    
    # 為每個 GT 找到最佳匹配 (Best Recall perspective)
    # best_pred_iou: [num_gts]
    if iou_matrix.numel() > 0:
        best_pred_iou, best_pred_idx = iou_matrix.max(dim=0)
    else:
        # 防禦性編碼：如果沒有預測框，理論上上面 num_preds==0 已攔截，但以防萬一
        return num_gts * fn_weight

    matched_pred_indices = set()
    
    # --- 4. 評估每個 Ground Truth (Recall 關鍵) ---
    for gt_i in range(num_gts):
        max_iou = best_pred_iou[gt_i]
        
        if max_iou < match_iou_thresh:
            # [Hard FN] 完全沒框到，或者框得太偏
            score += fn_weight * 1.5
        else:
            # [Soft FN] 框到了，但檢查品質
            pred_idx = best_pred_idx[gt_i]
            matched_pred_indices.add(pred_idx.item())
            
            curr_conf = pred_confs[pred_idx].item()
            curr_cls = pred_classes[pred_idx].item()
            gt_cls = gt_classes[gt_i].item()
            
            # (A) 分類錯誤懲罰
            if curr_cls != gt_cls:
                score += class_error_weight
            
            # (B) [關鍵] 信心度不足懲罰 (Confidence Penalty)
            # 即使框對了，如果信心只有 0.4，要加上 (1-0.4)*Weight 的懲罰
            # 這能強迫模型在下一次訓練中對此物件更有信心
            confidence_gap = 1.0 - curr_conf
            score += confidence_gap * fn_weight * 0.5 

    # --- 5. 評估 False Positives (Precision 視角) ---
    for pred_i in range(num_preds):
        if pred_i not in matched_pred_indices:
            # 只有當誤檢的信心度夠高時才懲罰，避免因為我們降低了 infer conf 而導致分數虛高
            if pred_confs[pred_i] >= fp_conf_thresh:
                score += fp_weight

    return score

def find_hard_cases(model_path: str, pool_data_txt: str, label_dir: str, config: dict) -> list:
    """
    (主函數) 載入模型，對挖掘池進行預測，並計算困難度分數。
    回傳: [(img_path, score), ...] (由高至低排序)
    """
    print(f"\n[Miner] 載入模型 {model_path} 開始挖掘難例...")
    
    device = 0 if torch.cuda.is_available() else 'cpu'
    
    # 載入模型
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"[Miner Critical] 無法載入模型: {e}")
        return []
    
    # 讀取影像列表
    if not os.path.exists(pool_data_txt):
        print(f"[Miner Critical] 找不到 Pool 列表: {pool_data_txt}")
        return []

    with open(pool_data_txt, 'r') as f:
        image_paths = [line.strip() for line in f if line.strip()]
        
    criteria = config['mining_criteria']
    image_scores = {}
    
    # 開始挖掘
    for img_path in tqdm(image_paths, desc="[Miner] 挖掘候選池"):
        if not os.path.exists(img_path):
            continue
            
        # 1. 執行預測 (使用極低閾值以捕捉所有潛在特徵)
        try:
            # [關鍵] conf=0.01 
            # 我們需要看到模型所有"微弱"的預測，才能計算信心懲罰
            results = model.predict(img_path, verbose=False, device=device, conf=0.01)
            preds = results[0].boxes
        except Exception as e:
            print(f"[Miner Error] 預測失敗 {img_path}: {e}")
            preds = torch.empty(0) # 視為沒有預測
            
        # 2. 載入 Ground Truth
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(label_dir, f"{base_name}.txt")
        gt_classes, gt_boxes_xyxyn = _load_gts(label_path, device=device)
        
        # 3. 計算分數
        score = _calculate_hardness_score(preds, (gt_classes, gt_boxes_xyxyn), criteria)
        image_scores[img_path] = score
        
    del model
    torch.cuda.empty_cache()
    
    # 4. 排序 (分數高者排前面 -> 越難越前面)
    sorted_cases = sorted(image_scores.items(), key=lambda item: item[1], reverse=True)
    
    if len(sorted_cases) > 0:
        print(f"[Miner] 挖掘完成。共分析 {len(sorted_cases)} 張影像。")
        print(f"  最難影像分數: {sorted_cases[0][1]:.2f} | 最簡單影像分數: {sorted_cases[-1][1]:.2f}")
    else:
        print("[Miner] 警告：沒有分析到任何影像。")
    
    return sorted_cases
