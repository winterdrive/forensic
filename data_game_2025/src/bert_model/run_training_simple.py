#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BERT 分類器訓練執行腳本 - 簡化版（使用絕對路徑）
"""

import os
import sys
import argparse

# 添加當前目錄到 Python 路徑
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# 添加 src 目錄到 Python 路徑
src_dir = os.path.abspath(os.path.join(current_dir, '..'))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from bert_model.bert_travel_classifier import TravelBertClassifier
from bert_model.bert_name_classifier import NameBertClassifier
from bert_model.bert_config import BertConfig


def train_travel_classifier(train_files, output_dir, use_config=True, **kwargs):
    """訓練旅遊分類器"""
    print("=== BERT 旅遊簡訊分類器訓練 ===")
    
    # 載入配置檔案
    if use_config:
        config = BertConfig()
        training_config = config.get_training_config()
        print("✅ 載入配置檔案中的訓練參數...")
        for key, value in training_config.items():
            print(f"  {key}: {value}")
        
        # 將配置檔案的參數與命令列參數合併（命令列參數優先）
        for key, value in training_config.items():
            if key not in kwargs:
                kwargs[key] = value
    
    # 檢查檔案是否存在
    for file_path in train_files:
        if not os.path.exists(file_path):
            print(f"錯誤: 數據檔案不存在 - {file_path}")
            return None
    
    print(f"數據檔案: {train_files}")
    print(f"輸出目錄: {output_dir}")
    print()
    
    try:
        # 初始化分類器
        classifier = TravelBertClassifier()
        
        # 執行訓練
        print("開始訓練...")
        model_path = classifier.train(
            data_files=train_files,
            output_dir=output_dir,
            **kwargs
        )
        
        print(f"\n✅ 旅遊分類器訓練完成！模型保存於: {model_path}")
        return model_path
        
    except Exception as e:
        print(f"❌ 旅遊分類器訓練過程中發生錯誤: {e}")
        import traceback
        traceback.print_exc()
        return None


def train_name_classifier(train_files, output_dir, use_config=True, **kwargs):
    """訓練姓名分類器"""
    print("=== BERT 姓名簡訊分類器訓練 ===")
    
    # 載入配置檔案
    if use_config:
        config = BertConfig()
        training_config = config.get_training_config()
        print("✅ 載入配置檔案中的訓練參數...")
        for key, value in training_config.items():
            print(f"  {key}: {value}")
        
        # 將配置檔案的參數與命令列參數合併（命令列參數優先）
        for key, value in training_config.items():
            if key not in kwargs:
                kwargs[key] = value
    
    # 檢查檔案是否存在
    for file_path in train_files:
        if not os.path.exists(file_path):
            print(f"錯誤: 數據檔案不存在 - {file_path}")
            return None
    
    print(f"數據檔案: {train_files}")
    print(f"輸出目錄: {output_dir}")
    print()
    
    try:
        # 初始化分類器
        classifier = NameBertClassifier()
        
        # 執行訓練
        print("開始訓練...")
        model_path = classifier.train(
            data_files=train_files,
            output_dir=output_dir,
            **kwargs
        )
        
        print(f"\n✅ 姓名分類器訓練完成！模型保存於: {model_path}")
        return model_path
        
    except Exception as e:
        print(f"❌ 姓名分類器訓練過程中發生錯誤: {e}")
        import traceback
        traceback.print_exc()
        return None


def get_default_file_paths():
    """獲取默認的絕對文件路徑（從配置檔案讀取）"""
    config = BertConfig()
    
    return {
        'travel': {
            'train': config.config.get('paths', 'travel_train'),
            'val': config.config.get('paths', 'travel_val'),
            'test': config.config.get('paths', 'travel_test'),
            'output': config.config.get('paths', 'travel_model_dir')
        },
        'name': {
            'train': config.config.get('paths', 'name_train'),
            'val': config.config.get('paths', 'name_val'),
            'test': config.config.get('paths', 'name_test'),
            'output': config.config.get('paths', 'name_model_dir')
        }
    }


def main():
    """主函數"""
    parser = argparse.ArgumentParser(description='BERT 分類器訓練 - 使用配置檔案優化參數')
    parser.add_argument('--mode', choices=['travel', 'name', 'both'], default='both',
                        help='訓練模式: travel, name, 或 both')
    
    # 配置檔案選項
    parser.add_argument('--use-config', action='store_true', default=True,
                        help='使用配置檔案中的訓練參數（預設啟用）')
    parser.add_argument('--no-config', action='store_true', default=False,
                        help='不使用配置檔案，僅使用命令列參數')
    
    # 旅遊分類器參數
    parser.add_argument('--travel-train', type=str,
                        help='旅遊分類訓練文件絕對路徑')
    parser.add_argument('--travel-val', type=str,
                        help='旅遊分類驗證文件絕對路徑')
    parser.add_argument('--travel-test', type=str,
                        help='旅遊分類測試文件絕對路徑')
    parser.add_argument('--travel-output', type=str,
                        help='旅遊分類器輸出目錄（預設使用配置檔案中的路徑）')
    
    # 姓名分類器參數
    parser.add_argument('--name-train', type=str,
                        help='姓名分類訓練文件絕對路徑')
    parser.add_argument('--name-val', type=str,
                        help='姓名分類驗證文件絕對路徑')
    parser.add_argument('--name-test', type=str,
                        help='姓名分類測試文件絕對路徑')
    parser.add_argument('--name-output', type=str,
                        help='姓名分類器輸出目錄（預設使用配置檔案中的路徑）')
    
    # 訓練參數（如果不使用配置檔案時的預設值）
    parser.add_argument('--batch-size', type=int, help='批次大小')
    parser.add_argument('--learning-rate', type=float, help='學習率')
    parser.add_argument('--num-epochs', type=int, help='訓練輪數')
    parser.add_argument('--max-length', type=int, help='最大序列長度')
    parser.add_argument('--warmup-steps', type=int, help='預熱步數')
    parser.add_argument('--weight-decay', type=float, help='權重衰減')
    
    args = parser.parse_args()
    
    # 決定是否使用配置檔案
    use_config = args.use_config and not args.no_config
    
    if use_config:
        print("✅ 使用配置檔案中的優化訓練參數")
    else:
        print("⚠️ 不使用配置檔案，僅使用命令列參數")
    
    # 如果沒有提供文件路徑，使用默認路徑
    default_paths = get_default_file_paths()
    
    # 獲取訓練參數 - 只傳遞非 None 的參數
    train_kwargs = {}
    if args.batch_size is not None:
        train_kwargs['batch_size'] = args.batch_size
    if args.learning_rate is not None:
        train_kwargs['learning_rate'] = args.learning_rate
    if args.num_epochs is not None:
        train_kwargs['num_epochs'] = args.num_epochs
    if args.max_length is not None:
        train_kwargs['max_length'] = args.max_length
    if args.warmup_steps is not None:
        train_kwargs['warmup_steps'] = args.warmup_steps
    if args.weight_decay is not None:
        train_kwargs['weight_decay'] = args.weight_decay
    
    print("🚀 開始 BERT 分類器訓練流程")
    print("=" * 50)
    
    results = {}
    
    # 訓練旅遊分類器
    if args.mode in ['travel', 'both']:
        travel_files = []
        if args.travel_train:
            travel_files.append(args.travel_train)
        else:
            travel_files.append(default_paths['travel']['train'])
            
        if args.travel_val:
            travel_files.append(args.travel_val)
        else:
            travel_files.append(default_paths['travel']['val'])
            
        # 加入測試集（如果存在的話）
        if args.travel_test:
            travel_files.append(args.travel_test)
        elif default_paths['travel']['test']:
            travel_files.append(default_paths['travel']['test'])
        
        results['travel'] = train_travel_classifier(
            travel_files, 
            args.travel_output or default_paths['travel']['output'], 
            use_config=use_config, 
            **train_kwargs
        )
    
    # 訓練姓名分類器
    if args.mode in ['name', 'both']:
        name_files = []
        if args.name_train:
            name_files.append(args.name_train)
        else:
            name_files.append(default_paths['name']['train'])
            
        if args.name_val:
            name_files.append(args.name_val)
        else:
            name_files.append(default_paths['name']['val'])
            
        # 加入測試集（如果存在的話）
        if args.name_test:
            name_files.append(args.name_test)
        elif default_paths['name']['test']:
            name_files.append(default_paths['name']['test'])
        
        results['name'] = train_name_classifier(
            name_files, 
            args.name_output or default_paths['name']['output'], 
            use_config=use_config, 
            **train_kwargs
        )
    
    # 打印結果總結
    print("\n" + "=" * 50)
    print("📊 訓練結果總結:")
    
    if 'travel' in results:
        if results['travel']:
            print(f"✅ 旅遊分類器: {results['travel']}")
        else:
            print("❌ 旅遊分類器: 訓練失敗")
    
    if 'name' in results:
        if results['name']:
            print(f"✅ 姓名分類器: {results['name']}")
        else:
            print("❌ 姓名分類器: 訓練失敗")
    
    # 檢查是否全部成功
    success_count = sum(1 for result in results.values() if result is not None)
    total_count = len(results)
    
    if success_count == total_count and total_count > 0:
        print("\n🎉 所有分類器訓練完成！")
    elif success_count > 0:
        print(f"\n⚠️  部分分類器訓練成功 ({success_count}/{total_count})")
    else:
        print("\n❌ 所有分類器訓練失敗，請檢查錯誤訊息")


def print_usage_examples():
    """打印使用範例"""
    print("""
使用範例:

1. 使用默認路徑訓練所有分類器:
   python run_training_simple.py

2. 只訓練旅遊分類器:
   python run_training_simple.py --mode travel

3. 使用自定義路徑訓練旅遊分類器:
   python run_training_simple.py --mode travel \\
       --travel-train /path/to/travel_train.csv \\
       --travel-val /path/to/travel_val.csv \\
       --travel-output /path/to/output

4. 使用自定義參數:
   python run_training_simple.py --batch-size 32 --num-epochs 5
""")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("BERT 分類器訓練腳本")
        print_usage_examples()
        print("\n使用 --help 獲取完整參數說明")
        print("\n使用默認設定開始訓練...")
        print("=" * 50)
    
    main()
