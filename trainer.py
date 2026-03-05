# utils/trainer.py 

from ultralytics import YOLO
import torch
import time
import os  

def run_single_training(config: dict, run_name: str, data_path: str, model_weights: str = None) -> (dict, str, str):
    """
    執行單一回合的YOLOv8訓練。
    """
    try:
        print(f"--- 開始訓練: {run_name} ---")
        
        start_time = time.time()

        if model_weights and os.path.exists(model_weights):
            model_variant = model_weights
            print(f"  [Trainer] 載入權重進行 Finetune: {model_weights}")
        else:
            model_variant = config['model']['variant']
            print(f"  [Trainer] 從頭訓練模型: {model_variant}")
        
        project_name = config['project']['name']
        
        training_params = config['training'].copy()
        
        if 'img_size' in training_params:
            training_params['imgsz'] = training_params.pop('img_size')
        if 'batch_size' in training_params:
            training_params['batch'] = training_params.pop('batch_size')

        if 'augmentations' in training_params:
            training_params.update(training_params.pop('augmentations'))

        model = YOLO(model_variant)

        results = model.train(
            data=data_path,
            project=project_name,
            name=run_name,
            device=0,
            **training_params
        )
        
        end_time = time.time()
        duration_seconds = int(end_time - start_time)
        duration_str = time.strftime('%H:%M:%S', time.gmtime(duration_seconds))
        print(f"--- 訓練完成: {run_name} | 耗時: {duration_str} ---")
        
        final_metrics = results.results_dict
        class_names = results.names
        
        val_metrics = results.box
        
        for i, name in class_names.items():
            final_metrics[f'val/{name}_P'] = val_metrics.p[i]
            final_metrics[f'val/{name}_R'] = val_metrics.r[i]
            final_metrics[f'val/{name}_mAP50'] = val_metrics.ap50[i]
            final_metrics[f'val/{name}_mAP50-95'] = val_metrics.maps[i]

        del model
        torch.cuda.empty_cache()

        return final_metrics, results.save_dir, duration_str
    
    except Exception as e:
        print(f"[ERROR] 在訓練 '{run_name}' 過程中發生錯誤: {e}")
        error_metrics = {'error': str(e)}
        return error_metrics, None, "00:00:00"
