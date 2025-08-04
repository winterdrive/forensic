#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BERT 旅遊簡訊分類器推論模組
載入訓練好的 distilbert-base-multilingual-cased 模型進行推論
"""

import os
import torch
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import json

from transformers import AutoTokenizer, AutoModelForSequenceClassification, DistilBertTokenizer, DistilBertForSequenceClassification

from utils import load_input_csv, save_results_csv


class TravelBertInference:
    """BERT 旅遊分類推論器"""
    
    def __init__(self, model_path: str):
        """
        初始化推論器
        
        Args:
            model_path: 訓練好的模型路徑
        """
        self.model_path = model_path
        
        # 支援 mps 裝置 (Apple Silicon)
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        
        # 載入分詞器和模型 - 支援多種模型
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        except Exception as e:
            print(f"⚠️  無法使用 Auto 類別載入模型，嘗試使用 DistilBert: {e}")
            self.tokenizer = DistilBertTokenizer.from_pretrained(model_path)
            self.model = DistilBertForSequenceClassification.from_pretrained(model_path)
        
        self.model.to(self.device)
        self.model.eval()
        
        # 載入訓練配置
        self.config = self._load_training_config()
        self.max_length = self.config.get('max_length', 512)
        
        print(f"推論器初始化完成")
        print(f"使用設備: {self.device}")
        print(f"模型路徑: {model_path}")
        print(f"最大序列長度: {self.max_length}")
    
    def _load_training_config(self) -> Dict:
        """載入訓練配置，支援本地模型和 Hugging Face 模型"""
        config_path = os.path.join(self.model_path, "training_config.json")
        
        # 檢查是否為 Hugging Face 模型 ID
        is_hf_model = "/" in self.model_path and not os.path.exists(self.model_path)
        
        if is_hf_model:
            print(f"使用 Hugging Face 模型，跳過本地訓練配置載入")
            return {}
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"警告: 找不到訓練配置檔案 {config_path}，使用預設值")
            return {}
    
    def predict_single(self, text: str) -> Tuple[int, float]:
        """
        預測單條簡訊的旅遊分類
        
        Args:
            text: 簡訊文本
            
        Returns:
            (預測標籤, 旅遊機率)
        """
        # 文本預處理
        text = str(text).strip()
        
        # 分詞與編碼
        inputs = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        ).to(self.device)
        
        # 模型推論
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            
            # 計算機率
            probabilities = torch.softmax(logits, dim=-1)
            travel_probability = probabilities[0][1].item()  # 取得旅遊類別的機率
            
            # 預測標籤
            predicted_label = torch.argmax(probabilities, dim=-1).item()
        
        return predicted_label, travel_probability
    
    def predict_batch(
        self, 
        texts: List[str], 
        batch_size: int = 32
    ) -> Tuple[List[int], List[float]]:
        """
        批次預測多條簡訊
        
        Args:
            texts: 簡訊文本列表
            batch_size: 批次大小
            
        Returns:
            (預測標籤列表, 旅遊機率列表)
        """
        all_labels = []
        all_probabilities = []
        
        # 分批處理
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # 批次編碼
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            ).to(self.device)
            
            # 批次推論
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                
                # 計算機率
                probabilities = torch.softmax(logits, dim=-1)
                travel_probabilities = probabilities[:, 1].cpu().numpy()  # 旅遊類別機率
                
                # 預測標籤
                predicted_labels = torch.argmax(probabilities, dim=-1).cpu().numpy()
            
            all_labels.extend(predicted_labels.tolist())
            all_probabilities.extend(travel_probabilities.tolist())
            
            # 顯示進度
            if (i // batch_size + 1) % 10 == 0:
                print(f"已處理 {min(i + batch_size, len(texts))}/{len(texts)} 筆資料")
        
        return all_labels, all_probabilities
    
    def predict_csv(
        self, 
        input_csv_path: str, 
        output_csv_path: Optional[str] = None,
        batch_size: int = 32
    ) -> str:
        """
        對 CSV 檔案中的簡訊進行批次預測
        
        Args:
            input_csv_path: 輸入 CSV 檔案路徑
            output_csv_path: 輸出 CSV 檔案路徑（可選）
            batch_size: 批次大小
            
        Returns:
            輸出檔案路徑
        """
        # 載入資料
        messages = load_input_csv(input_csv_path)
        sms_ids = [msg['id'] for msg in messages]
        texts = [msg['message'] for msg in messages]
        
        print(f"載入 {len(texts)} 筆簡訊進行預測")
        
        # 批次預測
        labels, probabilities = self.predict_batch(texts, batch_size)
        
        # 準備結果
        results = {sms_id: label for sms_id, label in zip(sms_ids, labels)}
        
        # 生成輸出檔案路徑
        if output_csv_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M")
            output_csv_path = f"data_game_2025/data/results/travel_bert_predictions_{timestamp}.csv"
        
        # 保存結果
        save_results_csv(results, output_csv_path, classifier_type="travel")
        
        # 統計結果
        positive_count = sum(labels)
        total_count = len(labels)
        print(f"預測完成:")
        print(f"  總筆數: {total_count}")
        print(f"  預測為旅遊: {positive_count} ({positive_count/total_count:.2%})")
        print(f"  預測為非旅遊: {total_count - positive_count} ({(total_count-positive_count)/total_count:.2%})")
        print(f"結果已保存至: {output_csv_path}")
        
        return output_csv_path


def generate_submission(
    input_csv: str,
    model_path: str,
    threshold: float = 0.5,
    max_submissions: int = 30000,
    output_dir: str = "/Users/winstontang/PycharmProjects/forensic/data_game_2025/data/results/vote/candidates"
) -> str:
    """
    生成符合競賽規格的提交檔案
    
    Args:
        input_csv: 輸入簡訊檔案
        model_path: 訓練好的模型路徑
        threshold: 分類閾值
        max_submissions: 最大提交筆數
        output_dir: 輸出目錄
        
    Returns:
        output_path: 輸出檔案路徑
    """
    # 載入推論器
    inferencer = TravelBertInference(model_path)
    
    # 載入資料
    messages = load_input_csv(input_csv)
    sms_ids = [msg['id'] for msg in messages]
    texts = [msg['message'] for msg in messages]
    
    print(f"載入 {len(texts)} 筆簡訊")
    
    # 批次預測
    labels, probabilities = inferencer.predict_batch(texts)
    
    # 建立結果 DataFrame
    df = pd.DataFrame({
        'sms_id': sms_ids,
        'label': labels,
        'travel_probability': probabilities
    })
    
    # 依信心度排序並篩選
    df_positive = df[df['travel_probability'] >= threshold]
    df_sorted = df_positive.sort_values(
        'travel_probability',
        ascending=False
    ).head(max_submissions)
    
    # 生成提交格式 (只保留 sms_id 和 label)
    submission = df_sorted[['sms_id', 'label']].copy()
    
    # 生成輸出檔案路徑
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    output_path = os.path.join(output_dir, f"travel_bert_submission_{timestamp}.csv")
    
    # 保存結果
    os.makedirs(output_dir, exist_ok=True)
    submission.to_csv(output_path, index=False)
    
    print(f"提交檔案生成完成:")
    print(f"  閾值: {threshold}")
    print(f"  符合條件的筆數: {len(df_positive)}")
    print(f"  實際提交筆數: {len(submission)}")
    print(f"  平均信心度: {df_sorted['travel_probability'].mean():.4f}")
    print(f"  信心度範圍: {df_sorted['travel_probability'].min():.4f} - {df_sorted['travel_probability'].max():.4f}")
    print(f"檔案已保存至: {output_path}")
    
    return output_path


def optimize_threshold_on_validation(
    model_path: str,
    validation_csv: str,
    input_csv: str
) -> Tuple[float, Dict[str, float]]:
    """
    在驗證集上最佳化分類閾值
    
    Args:
        model_path: 模型路徑
        validation_csv: 驗證集標註檔案
        input_csv: 原始簡訊檔案
        
    Returns:
        (最佳閾值, 效能指標)
    """
    from sklearn.metrics import f1_score, precision_score, recall_score
    
    # 載入推論器
    inferencer = TravelBertInference(model_path)
    
    # 載入驗證資料
    val_df = pd.read_csv(validation_csv)
    messages = load_input_csv(input_csv)
    input_df = pd.DataFrame(messages)
    input_df.columns = ['sms_id', 'sms_body']
    
    # 合併資料
    merged_df = pd.merge(val_df, input_df, on='sms_id', how='inner')
    
    # 預測
    texts = merged_df['sms_body'].tolist()
    y_true = merged_df['label'].tolist()
    
    _, y_pred_proba = inferencer.predict_batch(texts)
    
    # 尋找最佳閾值
    thresholds = np.arange(0.1, 0.9, 0.05)
    best_f1 = 0
    best_threshold = 0.5
    best_metrics = {}
    
    for threshold in thresholds:
        y_pred = (np.array(y_pred_proba) >= threshold).astype(int)
        
        f1 = f1_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
            best_metrics = {
                'threshold': threshold,
                'f1': f1,
                'precision': precision,
                'recall': recall
            }
    
    print(f"最佳閾值優化結果:")
    print(f"  最佳閾值: {best_threshold}")
    print(f"  F1-Score: {best_metrics['f1']:.4f}")
    print(f"  Precision: {best_metrics['precision']:.4f}")
    print(f"  Recall: {best_metrics['recall']:.4f}")
    
    return best_threshold, best_metrics


if __name__ == "__main__":
    # 使用範例
    
    # 1. 載入模型進行推論
    model_path = "data_game_2025/data/results/travel_bert_20250711_1200"  # 替換為實際模型路徑
    
    if os.path.exists(model_path):
        # 預測單條簡訊
        inferencer = TravelBertInference(model_path)
        text = "恭喜您獲得免費日本旅遊券！請點擊連結領取"
        label, prob = inferencer.predict_single(text)
        print(f"簡訊: {text}")
        print(f"預測: {'旅遊' if label == 1 else '非旅遊'} (信心度: {prob:.4f})")
        
        # 對整個資料集進行預測
        output_path = inferencer.predict_csv("data_game_2025/data/input.csv")
        
        # 生成競賽提交檔案
        submission_path = generate_submission(
            input_csv="data_game_2025/data/input.csv",
            model_path=model_path,
            threshold=0.7,
            max_submissions=30000
        )
        
    else:
        print(f"模型路徑不存在: {model_path}")
        print("請先執行 bert_travel_trainer.py 進行模型訓練")
