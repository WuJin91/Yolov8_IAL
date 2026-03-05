# utils/tester.py

from ultralytics import YOLO
import torch
import pandas as pd

def run_final_test(model_path: str, test_data_yaml: str, config: dict) -> dict:
    """
    使用最佳模型在獨立測試集上執行最終評估。
    """
    print(f"\n{'='*30} 開始最終測試評估 {'='*30}")
    print(f"載入模型: {model_path}")
    print(f"使用測試設定檔: {test_data_yaml}")

    try:
        model = YOLO(model_path)
        
        test_results = model.val(
            data=test_data_yaml,
            split='test',
            imgsz=config['training']['img_size'],
            batch=config['training']['batch_size'],
            project=config['project']['name'],
            name='final_test_evaluation',
            device=0
        )
        
        print(f"--- 最終測試完成 ---")
        
        class_names = test_results.names
        overall_metrics = test_results.results_dict
        box_metrics = test_results.box # Metric 物件
        
        per_class_metrics = {}
        for i, name in class_names.items():
            per_class_metrics[name] = {
                'precision': box_metrics.p[i],
                'recall': box_metrics.r[i],
                'mAP50': box_metrics.ap50[i],
                'mAP50-95': box_metrics.maps[i]
            }

        confusion_matrix = None
        if hasattr(test_results, 'confusion_matrix') and test_results.confusion_matrix is not None:
            cm_data = test_results.confusion_matrix.matrix
            confusion_matrix = {
                'matrix': cm_data.tolist(),
                'class_names': [class_names[i] for i in sorted(class_names.keys())]
            }

        final_report = {
            'overall': overall_metrics,
            'per_class': per_class_metrics,
            'confusion_matrix': confusion_matrix,
        }

        del model
        torch.cuda.empty_cache()

        return final_report

    except Exception as e:
        print(f"[ERROR] 在最終測試過程中發生錯誤: {e}")
        return None
