#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BERT 旅遊分類器配置管理模組
"""

import configparser
import os
from typing import Dict, List, Any


class BertConfig:
    """BERT 分類器配置管理類別"""
    
    def __init__(self, config_path: str = None):
        """
        初始化配置管理器
        
        Args:
            config_path: 配置檔案路徑，預設為 config.ini
        """
        if config_path is None:
            config_path = os.path.join(os.path.dirname(__file__), "config.ini")
        
        self.config_path = config_path
        self.config = configparser.ConfigParser()
        
        # 載入配置
        self.config.read(config_path, encoding='utf-8')
    
    def get_model_config(self) -> Dict[str, Any]:
        """取得模型配置"""
        return {
            'model_name': self.config.get('model', 'model_name'),
            'max_length': self.config.getint('model', 'max_length'),
            'num_labels': self.config.getint('model', 'num_labels')
        }
    
    def get_training_config(self) -> Dict[str, Any]:
        """取得訓練配置"""
        return {
            'batch_size': self.config.getint('training', 'batch_size'),
            'learning_rate': self.config.getfloat('training', 'learning_rate'),
            'num_epochs': self.config.getint('training', 'num_epochs'),
            'warmup_steps': self.config.getint('training', 'warmup_steps'),
            'weight_decay': self.config.getfloat('training', 'weight_decay'),
            'random_seed': self.config.getint('training', 'random_seed'),
            'train_ratio': self.config.getfloat('training', 'train_ratio'),
            'val_ratio': self.config.getfloat('training', 'val_ratio'),
            'test_ratio': self.config.getfloat('training', 'test_ratio'),
            'early_stopping_patience': self.config.getint('training', 'early_stopping_patience')
        }
    
    def get_inference_config(self) -> Dict[str, Any]:
        """取得推論配置"""
        return {
            'inference_batch_size': self.config.getint('inference', 'inference_batch_size'),
            'input_file': self.config.get('inference', 'input_file'),
            'travel_model_path': self.config.get('inference', 'travel_model_path'),
            'name_model_path': self.config.get('inference', 'name_model_path'),
            'output_dir': self.config.get('inference', 'output_dir')
        }
    
    def get_paths_config(self) -> Dict[str, str]:
        """取得路徑配置"""
        return {
            'model_output_dir': self.config.get('paths', 'model_output_dir'),
            'results_output_dir': self.config.get('paths', 'results_output_dir')
        }
    
    def get_optimization_config(self) -> Dict[str, List]:
        """取得超參數優化配置"""
        return {
            'learning_rates': eval(self.config.get('optimization', 'learning_rates')),
            'batch_sizes': eval(self.config.get('optimization', 'batch_sizes')),
            'num_epochs_options': eval(self.config.get('optimization', 'num_epochs_options')),
            'warmup_ratios': eval(self.config.get('optimization', 'warmup_ratios')),
            'weight_decays': eval(self.config.get('optimization', 'weight_decays'))
        }
    
    def get_ensemble_config(self) -> Dict[str, Any]:
        """取得集成配置"""
        return {
            'model_weights': eval(self.config.get('ensemble', 'model_weights')),
            'voting_strategy': self.config.get('ensemble', 'voting_strategy'),
            'confidence_threshold': self.config.getfloat('ensemble', 'confidence_threshold')
        }
    
    def get_wandb_config(self) -> Dict[str, Any]:
        """取得 wandb 配置"""
        return {
            'use_wandb': self.config.getboolean('wandb', 'use_wandb'),
            'travel_project': self.config.get('wandb', 'travel_project'),
            'name_project': self.config.get('wandb', 'name_project'),
            'entity': self.config.get('wandb', 'entity'),
            'notes': self.config.get('wandb', 'notes'),
            'tags': self.config.get('wandb', 'tags').split(', ')
        }
    
    def set_wandb_config(self, **kwargs):
        """設定 wandb 配置"""
        for key, value in kwargs.items():
            if key == 'tags' and isinstance(value, list):
                value = ', '.join(value)
            self.update_config('wandb', key, str(value))
    
    def update_config(self, section: str, key: str, value: str):
        """更新配置"""
        if section not in self.config:
            self.config.add_section(section)
        
        self.config.set(section, key, str(value))
    
    def save_config(self):
        """儲存配置到檔案"""
        with open(self.config_path, 'w', encoding='utf-8') as f:
            self.config.write(f)
    
    def print_config(self):
        """列印所有配置"""
        print("=== BERT 旅遊分類器配置 ===")
        
        print("\n[模型配置]")
        model_config = self.get_model_config()
        for key, value in model_config.items():
            print(f"  {key}: {value}")
        
        print("\n[訓練配置]")
        training_config = self.get_training_config()
        for key, value in training_config.items():
            print(f"  {key}: {value}")
        
        print("\n[推論配置]")
        inference_config = self.get_inference_config()
        for key, value in inference_config.items():
            print(f"  {key}: {value}")
        
        print("\n[路徑配置]")
        paths_config = self.get_paths_config()
        for key, value in paths_config.items():
            print(f"  {key}: {value}")


# 全域配置實例
bert_config = BertConfig()


if __name__ == "__main__":
    # 測試配置管理
    config = BertConfig()
    config.print_config()
    
    # 測試配置更新
    config.update_config('training', 'batch_size', '32')
    print(f"\n更新後的批次大小: {config.get_training_config()['batch_size']}")
    
    # 儲存配置
    config.save_config()
    print("\n配置已儲存")
