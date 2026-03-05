# main_v2.py
import argparse
import yaml
import os
import sys
from utils.orchestrator_fixed import run_fixed_experiment
from utils.reporter import generate_kfold_report

def main():
    parser = argparse.ArgumentParser(description="YOLOv8 Active Learning Experiment Runner")
    parser.add_argument('--ratio', type=float, required=True, 
                        help='Hard Case Mining Ratio (0.0 - 1.0). e.g., 0.8 for 8:2 split')
    parser.add_argument('--name', type=str, default=None,
                        help='Custom project name suffix')
    args = parser.parse_args()
    
    config_path = 'config.yaml'
    split_file = 'fixed_splits.json'
    
    # 檢查必要檔案
    if not os.path.exists(config_path):
        print(f"[Error] Config file not found: {config_path}")
        return
    if not os.path.exists(split_file):
        print(f"[Error] Split file not found: {split_file}. Please run 'generate_splits.py' first.")
        return

    # 讀取設定
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 動態調整專案名稱
    ratio_str = f"{int(args.ratio*10)}:{int((1-args.ratio)*10)}"
    base_name = config['project']['name']
    suffix = args.name if args.name else f"Ratio_{ratio_str}"
    
    # 更新 Config 中的專案名稱，讓結果存在不同資料夾
    config['project']['name'] = f"{base_name}_{suffix}"
    
    print(f"\n{'='*50}")
    print(f"🚀 開始實驗: {config['project']['name']}")
    print(f"🎯 挖掘比例: Hard {args.ratio*100}% | Random {(1-args.ratio)*100}%")
    print(f"{'='*50}\n")
    
    # 執行實驗
    results = run_fixed_experiment(config, split_file, mining_ratio=args.ratio)
    
    if len(results) == 4:
        all_fold_final, all_iter_metrics, split_stats, mining_stats = results
        
        # 生成報告
        print("\n[Main] 生成詳細實驗報告...")
        generate_kfold_report(
            all_fold_final,
            all_iter_metrics,
            config,
            split_stats,
            mining_stats
        )
    else:
        print("[Error] 實驗未正確回傳結果。")

if __name__ == "__main__":
    main()
