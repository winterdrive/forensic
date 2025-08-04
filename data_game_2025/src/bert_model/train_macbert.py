#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MacBERT 模型訓練腳本
專門用於訓練 hfl/chinese-macbert-large 模型
"""

import sys
import os
from pathlib import Path

# 添加 src 目錄到 Python 路徑
src_dir = Path(__file__).parent.parent
sys.path.append(str(src_dir))

from bert_model.bert_travel_trainer import TravelModelTrainer
from bert_model.bert_name_trainer import NameModelTrainer
from bert_model.bert_config import BertConfig


def train_macbert_travel():
    """訓練 MacBERT 旅遊分類模型"""
    print("🚀 開始訓練 MacBERT 旅遊分類模型...")
    
    try:
        # 使用 MacBERT 專用配置
        config = BertConfig(config_path="src/bert_model/config_macbert.ini")
        model_config = config.get_model_config()
        training_config = config.get_training_config()
        wandb_config = config.get_wandb_config()
        
        print(f"📋 配置資訊:")
        print(f"  模型: {model_config['model_name']}")
        print(f"  批次大小: {training_config['batch_size']}")
        print(f"  學習率: {training_config['learning_rate']}")
        print(f"  訓練輪數: {training_config['num_epochs']}")
        
        actual_batch_size = training_config['batch_size']
        
        # 初始化訓練器
        trainer = TravelModelTrainer(
            model_name=model_config['model_name'],
            max_length=model_config['max_length'],
            batch_size=actual_batch_size,
            learning_rate=training_config['learning_rate'],
            num_epochs=training_config['num_epochs'],
            warmup_steps=training_config['warmup_steps'],
            weight_decay=training_config['weight_decay'],
            random_seed=training_config['random_seed'],
            use_wandb=wandb_config['use_wandb'],
            wandb_project=wandb_config['travel_project'],
            wandb_run_name=f"macbert-travel-{training_config['num_epochs']}epochs"
        )
        
        print("✅ 訓練器初始化完成")
        
        # 使用配置中的數據路徑
        data_files = [
            "/Users/winstontang/PycharmProjects/forensic/data_game_2025/data/results/labled/stage2/train_data/travel_train_8000.csv",
            "/Users/winstontang/PycharmProjects/forensic/data_game_2025/data/results/labled/stage2/train_data/travel_val_8000.csv",
            "/Users/winstontang/PycharmProjects/forensic/data_game_2025/data/results/labled/stage2/train_data/travel_test_8000.csv"
        ]
        
        # 檢查數據檔案是否存在
        for file_path in data_files:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"數據檔案不存在: {file_path}")
        
        print("✅ 數據檔案檢查完成")
        
        # 開始訓練
        print("🎯 開始模型訓練...")
        model_path = trainer.train(
            data_files=data_files,
            output_dir="/Users/winstontang/PycharmProjects/forensic/data_game_2025/results/travel_macbert"
        )
        
        print(f"✅ MacBERT 旅遊分類模型訓練完成！")
        print(f"📁 模型路徑: {model_path}")
        return model_path
        
    except Exception as e:
        print(f"❌ 訓練失敗: {str(e)}")
        import traceback
        traceback.print_exc()
        raise


def train_macbert_name():
    """訓練 MacBERT 姓名分類模型"""
    print("🚀 開始訓練 MacBERT 姓名分類模型...")
    
    # 使用 MacBERT 專用配置
    config = BertConfig(config_path="src/bert_model/config_macbert.ini")
    model_config = config.get_model_config()
    training_config = config.get_training_config()
    wandb_config = config.get_wandb_config()
    
    # 初始化訓練器
    trainer = NameModelTrainer(
        model_name=model_config['model_name'],
        max_length=model_config['max_length'],
        batch_size=training_config['batch_size'],
        learning_rate=training_config['learning_rate'],
        num_epochs=training_config['num_epochs'],
        warmup_steps=training_config['warmup_steps'],
        weight_decay=training_config['weight_decay'],
        random_seed=training_config['random_seed'],
        use_wandb=wandb_config['use_wandb'],
        wandb_project=wandb_config['name_project'],
        wandb_run_name=f"macbert-name-{training_config['num_epochs']}epochs"
    )
    
    # 使用配置中的數據路徑
    data_files = [
        "/Users/winstontang/PycharmProjects/forensic/data_game_2025/data/results/labled/stage2/train_data/name_train_8000.csv",
        "/Users/winstontang/PycharmProjects/forensic/data_game_2025/data/results/labled/stage2/train_data/name_val_8000.csv",
        "/Users/winstontang/PycharmProjects/forensic/data_game_2025/data/results/labled/stage2/train_data/name_test_8000.csv"
    ]
    
    # 開始訓練
    model_path = trainer.train(
        data_files=data_files,
        output_dir="/Users/winstontang/PycharmProjects/forensic/data_game_2025/results/name_macbert"
    )
    
    print(f"✅ MacBERT 姓名分類模型訓練完成！")
    print(f"📁 模型路徑: {model_path}")
    return model_path


def main():
    """主函數"""
    import argparse
    
    parser = argparse.ArgumentParser(description="MacBERT 模型訓練")
    parser.add_argument('--task', choices=['travel', 'name', 'both'], default='both',
                       help='要訓練的任務類型')
    
    args = parser.parse_args()
    
    print("🔥 MacBERT Large 模型訓練腳本")
    print("=" * 50)
    
    if args.task in ['travel', 'both']:
        travel_model_path = train_macbert_travel()
        print()
    
    if args.task in ['name', 'both']:
        name_model_path = train_macbert_name()
        print()
    
    print("🎉 所有模型訓練完成！")
    print("\n📋 模型路徑總結:")
    if args.task in ['travel', 'both']:
        print(f"  旅遊分類: {travel_model_path}")
    if args.task in ['name', 'both']:
        print(f"  姓名分類: {name_model_path}")
    
    print("\n💡 使用提示:")
    print("1. 模型已自動保存在指定目錄")
    print("2. 可以使用推論腳本進行測試")
    print("3. wandb 記錄可在 https://wandb.ai 查看")


if __name__ == "__main__":
    main()
