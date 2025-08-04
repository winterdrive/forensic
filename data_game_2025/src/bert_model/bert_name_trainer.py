#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BERT 姓名簡訊分類器訓練模組
使用 distilbert-base-multilingual-cased 進行姓名相關簡訊二元分類
"""

import os
import torch
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DistilBertTokenizer, 
    DistilBertForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)

from utils import load_input_csv

# 嘗試載入 wandb，如果沒有安裝則不使用
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("⚠️  wandb 未安裝，訓練過程將不記錄到 Weights & Biases")


class NameMessageDataset(Dataset):
    """姓名簡訊資料集類別"""
    
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 512, sms_ids: List[str] = None):
        """
        初始化資料集
        
        Args:
            texts: 簡訊文本列表
            labels: 對應的標籤列表 (0: 非姓名, 1: 姓名)
            tokenizer: BERT 分詞器
            max_length: 最大序列長度
            sms_ids: SMS ID 列表（可選）
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.sms_ids = sms_ids if sms_ids is not None else [str(i) for i in range(len(texts))]
    
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = str(self.texts[idx])
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class NameModelTrainer:
    """姓名分類模型訓練管理器"""
    def __init__(
        self,
        model_name: str = "distilbert-base-multilingual-cased",
        max_length: int = 512,
        batch_size: int = 16,
        learning_rate: float = 2e-5,
        num_epochs: int = 5,
        warmup_steps: int = 500,
        weight_decay: float = 0.01,
        random_seed: int = 42,
        use_wandb: bool = False,
        wandb_project: str = "bert-name-classifier",
        wandb_run_name: Optional[str] = None
    ):
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.warmup_steps = warmup_steps
        self.weight_decay = weight_decay
        self.random_seed = random_seed
        self.use_wandb = use_wandb
        self.wandb_project = wandb_project
        self.wandb_run_name = wandb_run_name
        
        # 設定隨機種子確保可重現性
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        
        # 設定設備，支援 mps (Apple Silicon)
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        print(f"使用設備: {self.device}")
        
        # 初始化分詞器和模型
        self.tokenizer = self._get_tokenizer(model_name)
        self.model = None
        
        # 檢查 wandb 配置
        if self.use_wandb and not WANDB_AVAILABLE:
            print("⚠️  wandb 未安裝，將忽略 wandb 紀錄功能")
            self.use_wandb = False
    
    def _get_tokenizer(self, model_name: str):
        """根據模型名稱獲取對應的分詞器"""
        try:
            return AutoTokenizer.from_pretrained(model_name)
        except Exception as e:
            print(f"⚠️  無法使用 AutoTokenizer 載入 {model_name}，嘗試使用 DistilBertTokenizer: {e}")
            return DistilBertTokenizer.from_pretrained(model_name)
    
    def _get_model(self, model_name: str, num_labels: int = 2):
        """根據模型名稱獲取對應的模型"""
        try:
            return AutoModelForSequenceClassification.from_pretrained(
                model_name, num_labels=num_labels
            )
        except Exception as e:
            print(f"⚠️  無法使用 AutoModel 載入 {model_name}，嘗試使用 DistilBertForSequenceClassification: {e}")
            return DistilBertForSequenceClassification.from_pretrained(
                model_name, num_labels=num_labels
            )
    
    def _get_model_tag(self, model_name: str) -> str:
        """根據模型名稱生成識別標籤"""
        model_tags = {
            "distilbert-base-multilingual-cased": "distilbert",
            "hfl/chinese-macbert-large": "macbert",
            "hfl/chinese-macbert-base": "macbert-base",
            "ckiplab/bert-base-chinese": "ckipbert",
            "bert-base-multilingual-cased": "bert-multi",
            "google/muril-large-cased": "muril"
        }
        
        if model_name in model_tags:
            return model_tags[model_name]
        
        if "/" in model_name:
            parts = model_name.split("/")[-1].split("-")
            if len(parts) >= 2:
                return f"{parts[0]}-{parts[1]}"
            else:
                return parts[0]
        else:
            return model_name.replace("-", "_")[:10]
    def load_labeled_data(self, data_files: List[str]) -> Tuple[List[str], List[int], List[str]]:
        """
        載入新格式的標註數據
        Args:
            data_files: 數據檔案路徑列表
        Returns:
            (texts, labels, sms_ids) 元組
        """
        from utils import load_multiple_datasets_simple_with_ids, clean_text_for_bert
        
        # 載入並合併所有數據檔案
        texts, labels, sms_ids = load_multiple_datasets_simple_with_ids(data_files, "name")
        
        # 清理文本
        texts = [clean_text_for_bert(text) for text in texts]
        
        # 打印數據統計
        label_counts = pd.Series(labels).value_counts()
        print(f"姓名分類數據統計:")
        print(f"  總數據量: {len(texts)} 筆")
        print(f"  標籤分佈: \n{label_counts}")
        print(f"  正樣本比例: {label_counts.get(1, 0) / len(labels):.3f}")
        
        return texts, labels, sms_ids
    
    def load_single_file(self, file_path: str, category_type: str) -> Tuple[List[str], List[int], List[str]]:
        """
        載入單個數據檔案
        Args:
            file_path: 數據檔案路徑
            category_type: 分類類型 ("travel" 或 "name")
        Returns:
            (texts, labels, sms_ids) 元組
        """
        from utils import load_labeled_data_simple_with_ids, clean_text_for_bert
        
        # 載入單個檔案
        texts, labels, sms_ids = load_labeled_data_simple_with_ids(file_path, category_type)
        
        # 清理文本
        texts = [clean_text_for_bert(text) for text in texts]
        
        return texts, labels, sms_ids
        
    def split_data(
        self, 
        texts: List[str], 
        labels: List[int], 
        sms_ids: List[str],
        train_ratio: float = 0.8, 
        val_ratio: float = 0.1, 
        test_ratio: float = 0.1
    ) -> Tuple[List[str], List[str], List[str], List[int], List[int], List[int], List[str], List[str], List[str]]:
        """
        分割資料集
        
        Args:
            texts: 文本列表
            labels: 標籤列表
            sms_ids: SMS ID 列表
            train_ratio: 訓練集比例
            val_ratio: 驗證集比例
            test_ratio: 測試集比例
            
        Returns:
            (train_texts, val_texts, test_texts, train_labels, val_labels, test_labels, train_sms_ids, val_sms_ids, test_sms_ids)
        """
        # 先分出訓練集和其餘部分
        train_texts, temp_texts, train_labels, temp_labels, train_sms_ids, temp_sms_ids = train_test_split(
            texts, labels, sms_ids,
            test_size=(1 - train_ratio),
            random_state=self.random_seed,
            stratify=labels
        )
        
        # 再將其餘部分分為驗證集和測試集
        val_size = val_ratio / (val_ratio + test_ratio)
        val_texts, test_texts, val_labels, test_labels, val_sms_ids, test_sms_ids = train_test_split(
            temp_texts, temp_labels, temp_sms_ids,
            test_size=(1 - val_size),
            random_state=self.random_seed,
            stratify=temp_labels
        )
        
        print(f"資料集劃分:")
        print(f"  訓練集: {len(train_texts)} 筆")
        print(f"  驗證集: {len(val_texts)} 筆")
        print(f"  測試集: {len(test_texts)} 筆")
        
        return train_texts, val_texts, test_texts, train_labels, val_labels, test_labels, train_sms_ids, val_sms_ids, test_sms_ids
    def compute_metrics(self, eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
        accuracy = accuracy_score(labels, predictions)
        return {
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }
    def train(self, data_files: List[str] = None, output_dir: str = "data_game_2025/results/name_bert") -> str:
        """
        執行模型訓練
        
        Args:
            data_files: 數據檔案路徑列表（如果為None，則使用配置檔案中的路徑）
            output_dir: 模型輸出目錄
            
        Returns:
            訓練好的模型路徑
        """
        # 如果沒有提供數據檔案，從配置檔案載入路徑
        if data_files is None:
            from bert_model.bert_config import BertConfig
            config = BertConfig()
            data_files = [
                config.config.get('paths', 'name_train'),
                config.config.get('paths', 'name_val'),
                config.config.get('paths', 'name_test')
            ]
        
        # 1. 載入與預處理資料
        train_texts, train_labels, train_sms_ids = self.load_single_file(data_files[0], "name")
        val_texts, val_labels, val_sms_ids = self.load_single_file(data_files[1], "name") 
        test_texts, test_labels, test_sms_ids = self.load_single_file(data_files[2], "name")
        
        print(f"使用預分割資料集:")
        print(f"  訓練集: {len(train_texts)} 筆")
        print(f"  驗證集: {len(val_texts)} 筆") 
        print(f"  測試集: {len(test_texts)} 筆")
        
        # 2. 建立資料集
        train_dataset = NameMessageDataset(train_texts, train_labels, self.tokenizer, self.max_length, train_sms_ids)
        val_dataset = NameMessageDataset(val_texts, val_labels, self.tokenizer, self.max_length, val_sms_ids)
        test_dataset = NameMessageDataset(test_texts, test_labels, self.tokenizer, self.max_length, test_sms_ids)
        
        # 3. 初始化模型
        self.model = self._get_model(self.model_name, num_labels=2)
        self.model.to(self.device)
        
        # 4. 設定訓練參數
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        # 根據模型名稱生成識別標籤
        model_tag = self._get_model_tag(self.model_name)
        final_output_dir = f"{output_dir}_{model_tag}_{timestamp}"
        
        # 5. 初始化 wandb (如果啟用)
        if self.use_wandb and WANDB_AVAILABLE:
            run_name = self.wandb_run_name or f"name-{model_tag}-{timestamp}"
            wandb.init(
                project=self.wandb_project,
                name=run_name,
                config={
                    "model_name": self.model_name,
                    "max_length": self.max_length,
                    "batch_size": self.batch_size,
                    "learning_rate": self.learning_rate,
                    "num_epochs": self.num_epochs,
                    "warmup_steps": self.warmup_steps,
                    "weight_decay": self.weight_decay,
                    "random_seed": self.random_seed,
                    "device": str(self.device),
                    "train_samples": len(train_texts),
                    "val_samples": len(val_texts),
                    "test_samples": len(test_texts)
                }
            )
            print(f"✅ wandb 已初始化 - 專案: {self.wandb_project}, 執行名稱: {run_name}")
        
        # 6. 設定訓練參數
        training_args = TrainingArguments(
            output_dir=final_output_dir,
            num_train_epochs=self.num_epochs,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            warmup_steps=self.warmup_steps,
            weight_decay=self.weight_decay,
            learning_rate=self.learning_rate,
            logging_dir=f"{final_output_dir}/logs",
            logging_steps=100,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            save_total_limit=3,
            report_to=["wandb"] if (self.use_wandb and WANDB_AVAILABLE) else None
        )
        
        # 7. 初始化訓練器
        print("初始化訓練器...")
        print(f'使用模型: {self.model_name}')
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )
        
        # 8. 執行訓練
        print("開始訓練...")
        trainer.train()
        
        # 9. 保存模型
        trainer.save_model()
        self.tokenizer.save_pretrained(final_output_dir)
        
        # 10. 在測試集上評估
        print("在測試集上評估...")
        test_results = trainer.evaluate(test_dataset)
        print(f"測試集結果: {test_results}")
        
        # 11. 記錄測試結果到 wandb
        if self.use_wandb and WANDB_AVAILABLE:
            wandb.log({
                "test_accuracy": test_results.get("eval_accuracy", 0),
                "test_f1": test_results.get("eval_f1", 0),
                "test_precision": test_results.get("eval_precision", 0),
                "test_recall": test_results.get("eval_recall", 0)
            })
        
        # 12. 生成評估報告
        self._generate_evaluation_report(trainer, test_dataset, test_labels, final_output_dir)
        self._save_training_config(final_output_dir)
        
        # 13. 關閉 wandb
        if self.use_wandb and WANDB_AVAILABLE:
            wandb.finish()
            print("✅ wandb 紀錄完成")
        
        print(f"訓練完成，模型保存至: {final_output_dir}")
        return final_output_dir
    def _generate_evaluation_report(self, trainer, test_dataset, test_labels: List[int], output_dir: str):
        predictions = trainer.predict(test_dataset)
        pred_probs = torch.softmax(torch.tensor(predictions.predictions), dim=-1)
        pred_labels = np.argmax(predictions.predictions, axis=1)
        accuracy = accuracy_score(test_labels, pred_labels)
        precision, recall, f1, _ = precision_recall_fscore_support(test_labels, pred_labels, average='binary')
        try:
            roc_auc = roc_auc_score(test_labels, pred_probs[:, 1])
        except:
            roc_auc = 0.0
        cm = confusion_matrix(test_labels, pred_labels)
        
        # 分析預測錯誤的樣本
        error_analysis = self._analyze_prediction_errors(test_dataset, test_labels, pred_labels, pred_probs)
        
        report_path = os.path.join(output_dir, "evaluation_report.txt")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=== Name SMS Classifier - Evaluation Report ===\n\n")
            f.write(f"Model: {self.model_name}\n")
            f.write(f"Training time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("=== Main Metrics ===\n")
            f.write(f"Accuracy:  {accuracy:.4f}\n")
            f.write(f"Precision: {precision:.4f}\n")
            f.write(f"Recall:    {recall:.4f}\n")
            f.write(f"F1-Score:  {f1:.4f}\n")
            f.write(f"ROC-AUC:   {roc_auc:.4f}\n\n")
            f.write("=== Confusion Matrix ===\n")
            f.write(f"True\\Pred   0(Not Name)  1(Name)\n")
            f.write(f"0(Not Name)     {cm[0,0]:6d}    {cm[0,1]:6d}\n")
            f.write(f"1(Name)         {cm[1,0]:6d}    {cm[1,1]:6d}\n\n")
            
            # 添加錯誤分析
            f.write("=== 預測錯誤分析 ===\n")
            f.write(f"總錯誤數: {error_analysis['total_errors']}\n")
            f.write(f"False Positive (誤判為姓名): {error_analysis['false_positives']}\n")
            f.write(f"False Negative (漏判姓名): {error_analysis['false_negatives']}\n\n")
            
            if error_analysis['fp_examples']:
                f.write("--- False Positive 錯誤樣本 (前10個) ---\n")
                for i, example in enumerate(error_analysis['fp_examples'][:10], 1):
                    f.write(f"{i}. 真實標籤: 0(非姓名), 預測標籤: 1(姓名), 信心度: {example['confidence']:.3f}\n")
                    f.write(f"   簡訊ID: {example['sms_id']} 簡訊內容: {example['text']}\n\n")
            
            if error_analysis['fn_examples']:
                f.write("--- False Negative 錯誤樣本 (前10個) ---\n")
                for i, example in enumerate(error_analysis['fn_examples'][:10], 1):
                    f.write(f"{i}. 真實標籤: 1(姓名), 預測標籤: 0(非姓名), 信心度: {example['confidence']:.3f}\n")
                    f.write(f"   簡訊ID: {example['sms_id']} 簡訊內容: {example['text']}\n\n")
                    
        # 保存詳細錯誤列表到 CSV
        self._save_error_details_csv(error_analysis, output_dir)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Not Name', 'Name'], 
                   yticklabels=['Not Name', 'Name'])
        plt.title('Name SMS Classification - Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
        plt.close()
        print(f"Evaluation report saved to: {report_path}")
        print(f"錯誤分析詳情請查看: {output_dir}/prediction_errors.csv")
    
    def _analyze_prediction_errors(
        self, 
        test_dataset, 
        test_labels: List[int], 
        pred_labels: List[int], 
        pred_probs: torch.Tensor
    ) -> Dict:
        """分析預測錯誤的樣本"""
        errors = {
            'total_errors': 0,
            'false_positives': 0,
            'false_negatives': 0,
            'fp_examples': [],
            'fn_examples': []
        }
        
        # 獲取原始文本和 SMS ID
        test_texts = test_dataset.texts
        test_sms_ids = test_dataset.sms_ids
        
        for i, (true_label, pred_label) in enumerate(zip(test_labels, pred_labels)):
            if true_label != pred_label:
                errors['total_errors'] += 1
                confidence = float(pred_probs[i][pred_label])
                
                error_info = {
                    'index': i,
                    'sms_id': test_sms_ids[i],
                    'text': test_texts[i],
                    'true_label': true_label,
                    'pred_label': pred_label,
                    'confidence': confidence,
                    'prob_0': float(pred_probs[i][0]),
                    'prob_1': float(pred_probs[i][1])
                }
                
                if true_label == 0 and pred_label == 1:
                    # False Positive: 非姓名被誤判為姓名
                    errors['false_positives'] += 1
                    errors['fp_examples'].append(error_info)
                elif true_label == 1 and pred_label == 0:
                    # False Negative: 姓名被誤判為非姓名
                    errors['false_negatives'] += 1
                    errors['fn_examples'].append(error_info)
        
        # 按信心度排序（信心度高的錯誤更值得關注）
        errors['fp_examples'].sort(key=lambda x: x['confidence'], reverse=True)
        errors['fn_examples'].sort(key=lambda x: x['confidence'], reverse=True)
        
        return errors
    
    def _save_error_details_csv(self, error_analysis: Dict, output_dir: str):
        """將錯誤分析詳情保存到 CSV 檔案"""
        all_errors = []
        
        # 收集所有錯誤樣本
        for fp in error_analysis['fp_examples']:
            all_errors.append({
                'sms_id': fp['sms_id'],
                'index': fp['index'],
                'text': fp['text'],
                'true_label': fp['true_label'],
                'pred_label': fp['pred_label'],
                'error_type': 'False Positive',
                'confidence': fp['confidence'],
                'prob_not_name': fp['prob_0'],
                'prob_name': fp['prob_1']
            })
        
        for fn in error_analysis['fn_examples']:
            all_errors.append({
                'sms_id': fn['sms_id'],
                'index': fn['index'],
                'text': fn['text'],
                'true_label': fn['true_label'],
                'pred_label': fn['pred_label'],
                'error_type': 'False Negative',
                'confidence': fn['confidence'],
                'prob_not_name': fn['prob_0'],
                'prob_name': fn['prob_1']
            })
        
        # 保存到 CSV
        if all_errors:
            error_df = pd.DataFrame(all_errors)
            error_df = error_df.sort_values('confidence', ascending=False)
            csv_path = os.path.join(output_dir, "prediction_errors.csv")
            error_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
            print(f"錯誤詳情已保存至: {csv_path}")
    def _save_training_config(self, output_dir: str):
        """保存訓練配置"""
        config = {
            'model_name': self.model_name,
            'max_length': self.max_length,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'num_epochs': self.num_epochs,
            'warmup_steps': self.warmup_steps,
            'weight_decay': self.weight_decay,
            'random_seed': self.random_seed,
            'device': str(self.device),
            'use_wandb': self.use_wandb,
            'wandb_project': self.wandb_project,
            'wandb_run_name': self.wandb_run_name,
            'training_time': datetime.now().isoformat()
        }
        
        config_path = os.path.join(output_dir, "training_config.json")
        import json
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        print(f"Training config saved to: {config_path}")

def optimize_threshold(y_true: List[int], y_pred_proba: List[float]) -> Tuple[float, float]:
    from sklearn.metrics import f1_score
    thresholds = np.arange(0.1, 0.9, 0.05)
    best_f1 = 0
    best_threshold = 0.5
    for threshold in thresholds:
        y_pred = (np.array(y_pred_proba) >= threshold).astype(int)
        f1 = f1_score(y_true, y_pred)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    return best_threshold, best_f1

if __name__ == "__main__":
    # 使用範例
    trainer = NameModelTrainer(
        model_name="distilbert-base-multilingual-cased",
        batch_size=16,
        learning_rate=2e-5,
        num_epochs=5,
        use_wandb=True,  # 啟用 wandb 紀錄
        wandb_project="forensic-name-classifier",
        wandb_run_name="name-bert-test"
    )
    
    # 訓練模型 (使用新格式數據)
    from utils import get_dataset_paths
    
    paths = get_dataset_paths("name")
    data_files = [paths['train'], paths['val']]  # 使用訓練和驗證數據
    
    model_path = trainer.train(data_files=data_files)
    
    print(f"模型訓練完成，保存於: {model_path}")
