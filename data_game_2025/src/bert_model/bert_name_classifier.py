#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Distilled Multilingual BERT 姓名簡訊分類器
整合式分類器 - 處理姓名分類任務
"""

import argparse
import os
from typing import Optional, List

try:
    from bert_model.bert_name_trainer import NameModelTrainer
    from bert_model.bert_name_inference import NameBertInference, generate_submission, optimize_threshold_on_validation
except ImportError:
    from bert_name_trainer import NameModelTrainer
    from bert_name_inference import NameBertInference, generate_submission, optimize_threshold_on_validation

class NameBertClassifier:
    """
    Distilled Multilingual BERT 姓名分類器主類別
    整合訓練和推論功能
    """
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        self.trainer = None
        self.inferencer = None
        if model_path and os.path.exists(model_path):
            self.inferencer = NameBertInference(model_path)
    def train(
        self, 
        data_files: List[str] = None,
        output_dir: str = "data_game_2025/results/name_bert",
        use_wandb: bool = None,
        wandb_project: str = None, 
        wandb_run_name: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        訓練姓名分類模型
        
        Args:
            data_files: 數據檔案路徑列表（如果為None，則使用默認路徑）
            output_dir: 模型輸出目錄
            use_wandb: 是否使用 wandb 記錄（如果為None，則從配置檔案讀取）
            wandb_project: wandb 專案名稱（如果為None，則從配置檔案讀取）
            wandb_run_name: wandb 執行名稱
            **kwargs: 其他訓練參數
            
        Returns:
            訓練好的模型路徑
        """
        # 從配置檔案讀取 wandb 設定
        from bert_model.bert_config import BertConfig
        config = BertConfig()
        wandb_config = config.get_wandb_config()
        
        if use_wandb is None:
            use_wandb = wandb_config['use_wandb']
        if wandb_project is None:
            wandb_project = wandb_config['name_project']
        
        # 過濾出訓練器支援的參數
        supported_params = {
            'model_name', 'max_length', 'batch_size', 'learning_rate', 
            'num_epochs', 'warmup_steps', 'weight_decay', 'random_seed'
        }
        trainer_kwargs = {k: v for k, v in kwargs.items() if k in supported_params}
        
        self.trainer = NameModelTrainer(
            use_wandb=use_wandb,
            wandb_project=wandb_project,
            wandb_run_name=wandb_run_name,
            **trainer_kwargs
        )
        model_path = self.trainer.train(data_files, output_dir)
        self.model_path = model_path
        
        # 初始化推論器
        self.inferencer = NameBertInference(model_path)
        
        return model_path
    def predict(self, input_csv_path: str, output_csv_path: Optional[str] = None, batch_size: int = 32) -> str:
        if self.inferencer is None:
            raise ValueError("請先載入模型或進行訓練")
        return self.inferencer.predict_csv(input_csv_path, output_csv_path, batch_size)
    def generate_submission(self, input_csv_path: str, threshold: float = 0.5, max_submissions: int = 30000, output_dir: str = "data_game_2025/data/results") -> str:
        if self.model_path is None:
            raise ValueError("請先載入模型或進行訓練")
        return generate_submission(input_csv_path, self.model_path, threshold, max_submissions, output_dir)
    def optimize_threshold(self, validation_csv: str, input_csv: str) -> tuple:
        if self.model_path is None:
            raise ValueError("請先載入模型或進行訓練")
        return optimize_threshold_on_validation(self.model_path, validation_csv, input_csv)

def main():
    parser = argparse.ArgumentParser(description="BERT 姓名簡訊分類器")
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    # 訓練命令
    train_parser = subparsers.add_parser('train', help='訓練模型')
    train_parser.add_argument('--labeled_csv', required=True, help='標註結果檔案路徑')
    train_parser.add_argument('--input_csv', required=True, help='原始簡訊檔案路徑')
    train_parser.add_argument('--output_dir', default='data_game_2025/data/results/name_bert', help='模型輸出目錄')
    train_parser.add_argument('--batch_size', type=int, default=16, help='批次大小')
    train_parser.add_argument('--learning_rate', type=float, default=2e-5, help='學習率')
    train_parser.add_argument('--num_epochs', type=int, default=5, help='訓練輪數')
    train_parser.add_argument('--max_length', type=int, default=512, help='最大序列長度')
    train_parser.add_argument('--use_wandb', action='store_true', help='是否使用 wandb 記錄訓練過程')
    train_parser.add_argument('--wandb_project', default='bert-name-classifier', help='wandb 專案名稱')
    train_parser.add_argument('--wandb_run_name', help='wandb 執行名稱')
    
    # 預測命令
    predict_parser = subparsers.add_parser('predict', help='進行預測')
    predict_parser.add_argument('--model_path', required=True, help='模型路徑')
    predict_parser.add_argument('--input_csv', required=True, help='輸入 CSV 檔案路徑')
    predict_parser.add_argument('--output_csv', help='輸出 CSV 檔案路徑')
    predict_parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    submit_parser = subparsers.add_parser('submit', help='生成競賽提交檔案')
    submit_parser.add_argument('--model_path', required=True, help='模型路徑')
    submit_parser.add_argument('--input_csv', required=True, help='輸入 CSV 檔案路徑')
    submit_parser.add_argument('--threshold', type=float, default=0.5, help='分類閾值')
    submit_parser.add_argument('--max_submissions', type=int, default=30000, help='最大提交筆數')
    submit_parser.add_argument('--output_dir', default='data_game_2025/data/results', help='輸出目錄')
    threshold_parser = subparsers.add_parser('optimize', help='優化分類閾值')
    threshold_parser.add_argument('--model_path', required=True, help='模型路徑')
    threshold_parser.add_argument('--validation_csv', required=True, help='驗證集標註檔案')
    threshold_parser.add_argument('--input_csv', required=True, help='原始簡訊檔案')
    evaluate_parser = subparsers.add_parser('evaluate', help='推論並與真實標籤比對')
    evaluate_parser.add_argument('--model_path', required=True, help='模型路徑')
    evaluate_parser.add_argument('--input_csv', required=True, help='輸入 CSV 檔案路徑')
    evaluate_parser.add_argument('--label_csv', required=True, help='真實標籤 CSV (含 sms_id, label)')
    evaluate_parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    evaluate_parser.add_argument('--output_csv', help='預測結果輸出路徑 (含真實標籤)')
    args = parser.parse_args()
    if args.command == 'train':
        classifier = NameBertClassifier()
        model_path = classifier.train(
            labeled_csv_path=args.labeled_csv,
            input_csv_path=args.input_csv,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            num_epochs=args.num_epochs,
            max_length=args.max_length,
            use_wandb=args.use_wandb,
            wandb_project=args.wandb_project,
            wandb_run_name=args.wandb_run_name
        )
        print(f"模型訓練完成，保存於: {model_path}")
        
        if args.use_wandb:
            print(f"✅ 訓練記錄已同步至 wandb: {args.wandb_project}")
    
    elif args.command == 'predict':
        classifier = NameBertClassifier(args.model_path)
        output_path = classifier.predict(
            input_csv_path=args.input_csv,
            output_csv_path=args.output_csv,
            batch_size=args.batch_size
        )
        print(f"預測完成，結果保存於: {output_path}")
    elif args.command == 'submit':
        classifier = NameBertClassifier(args.model_path)
        submission_path = classifier.generate_submission(
            input_csv_path=args.input_csv,
            threshold=args.threshold,
            max_submissions=args.max_submissions,
            output_dir=args.output_dir
        )
        print(f"提交檔案生成完成: {submission_path}")
    elif args.command == 'optimize':
        classifier = NameBertClassifier(args.model_path)
        best_threshold, metrics = classifier.optimize_threshold(
            validation_csv=args.validation_csv,
            input_csv=args.input_csv
        )
        print(f"最佳閾值: {best_threshold}")
        print(f"效能指標: {metrics}")
    elif args.command == 'evaluate':
        import pandas as pd
        from sklearn.metrics import classification_report, confusion_matrix
        classifier = NameBertClassifier(args.model_path)
        input_df = pd.read_csv(args.input_csv)
        label_df = pd.read_csv(args.label_csv)
        if 'sms_id' in input_df.columns:
            input_id_col = 'sms_id'
        elif 'id' in input_df.columns:
            input_id_col = 'id'
        else:
            raise ValueError('input_csv 必須包含 sms_id 或 id 欄位')
        if 'sms_body' in input_df.columns:
            text_col = 'sms_body'
        elif 'message' in input_df.columns:
            text_col = 'message'
        else:
            raise ValueError('input_csv 必須包含 sms_body 或 message 欄位')
        merged = pd.merge(input_df, label_df, left_on=input_id_col, right_on='sms_id')
        texts = merged[text_col].tolist()
        y_true = merged['name_flg'].tolist()
        _, y_pred_proba = classifier.inferencer.predict_batch(texts, args.batch_size)
        y_pred = [int(p >= 0.5) for p in y_pred_proba]
        print(classification_report(y_true, y_pred, digits=4))
        print('Confusion matrix:')
        print(confusion_matrix(y_true, y_pred))
        if args.output_csv:
            merged['pred_name_flg'] = y_pred
            merged['pred_prob'] = y_pred_proba
            merged.to_csv(args.output_csv, index=False)
            print(f'詳細預測結果已輸出至: {args.output_csv}')
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
