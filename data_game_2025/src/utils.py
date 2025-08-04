#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
共用工具函數 (for 簡訊分類任務)
"""
import os
import csv
from typing import Dict, List
import xml.etree.ElementTree as ET

def load_input_csv(file_path: str) -> List[Dict[str, str]]:
    """
    載入輸入 CSV 檔案 (新格式: sms_id, sms_body, label, name_flg)
    Args:
        file_path: CSV 檔案路徑
    Returns:
        簡訊列表，格式為 sms_id, sms_body
    """
    messages = []
    try:
        with open(file_path, 'r', encoding='utf-8-sig') as f:  # 使用 utf-8-sig 處理 BOM
            reader = csv.DictReader(f)
            
            # 調試：檢查欄位名稱
            fieldnames = reader.fieldnames
            print(f"🔍 CSV 欄位名稱: {fieldnames}")
            
            for row in reader:
                # 支援多種 ID 和內容欄位名稱，並處理可能的 BOM 問題
                sms_id = None
                sms_body = None
                
                # 尋找 ID 欄位（處理可能的 BOM 或空白字元）
                for key in row.keys():
                    clean_key = key.strip().lstrip('\ufeff')  # 移除 BOM 和空白
                    if clean_key in ["sms_id", "id"]:
                        sms_id = row[key]
                        break
                
                # 尋找內容欄位
                for key in row.keys():
                    clean_key = key.strip().lstrip('\ufeff')
                    if clean_key in ["sms_body", "content", "message", "body"]:
                        sms_body = row[key]
                        break
                
                if sms_id and sms_body:
                    messages.append({
                        "id": sms_id.strip(),
                        "message": sms_body.strip()
                    })
                    
        print(f"✅ load_input_csv 成功載入 {len(messages)} 筆簡訊")
        
        # 調試：檢查前幾筆資料
        if len(messages) > 0:
            print(f"🔍 load_input_csv 前3筆資料:")
            for i, msg in enumerate(messages[:3]):
                print(f"    {i+1}. ID: '{msg['id']}' (類型: {type(msg['id'])})")
                print(f"       文字: '{msg['message'][:50]}...'")
        else:
            print("⚠️ 沒有載入任何資料，請檢查檔案格式和欄位名稱")
                
    except FileNotFoundError:
        raise FileNotFoundError(f"找不到輸入檔案: {file_path}")
    except Exception as e:
        raise Exception(f"讀取 CSV 檔案時發生錯誤: {e}")
    return messages

def load_labeled_data(file_path: str, category_type: str) -> tuple:
    """
    載入新格式的標註數據
    Args:
        file_path: CSV 檔案路徑
        category_type: 分類類型 ("name" 或 "travel")
    Returns:
        (texts, labels) 元組
    """
    import pandas as pd
    
    try:
        df = pd.read_csv(file_path)
        
        # 根據分類類型選擇對應的標籤欄位
        if category_type == "name":
            # name 分類使用 name_flg 欄位
            df = df[df['name_flg'].notna()].copy()
            texts = df['sms_body'].tolist()
            labels = df['name_flg'].astype(int).tolist()
        elif category_type == "travel":
            # travel 分類使用 label 欄位
            df = df[df['label'].notna()].copy()
            texts = df['sms_body'].tolist()
            labels = df['label'].astype(int).tolist()
        else:
            raise ValueError(f"不支援的分類類型: {category_type}")
        
        return texts, labels
        
    except FileNotFoundError:
        raise FileNotFoundError(f"找不到標註檔案: {file_path}")
    except Exception as e:
        raise Exception(f"讀取標註數據時發生錯誤: {e}")

def load_multiple_datasets(file_paths: List[str], category_type: str) -> tuple:
    """
    載入多個數據集檔案並合併
    Args:
        file_paths: 檔案路徑列表
        category_type: 分類類型 ("name" 或 "travel")
    Returns:
        (all_texts, all_labels) 元組
    """
    all_texts = []
    all_labels = []
    
    for file_path in file_paths:
        if os.path.exists(file_path):
            texts, labels = load_labeled_data(file_path, category_type)
            all_texts.extend(texts)
            all_labels.extend(labels)
            print(f"載入 {file_path}: {len(texts)} 筆數據")
        else:
            print(f"警告: 檔案不存在 {file_path}")
    
    print(f"總計載入 {len(all_texts)} 筆 {category_type} 分類數據")
    return all_texts, all_labels

def load_labeled_data_simple_with_ids(file_path: str, category_type: str) -> tuple:
    """
    簡化版載入標註數據，包含 SMS ID（直接使用絕對路徑）
    Args:
        file_path: 數據檔案絕對路徑
        category_type: 分類類型 ("name" 或 "travel")
    Returns:
        (texts, labels, sms_ids) 元組
    """
    import pandas as pd
    
    try:
        df = pd.read_csv(file_path)
        
        # 根據分類類型選擇對應的標籤欄位
        if category_type == "name":
            # name 分類使用 name_flg 欄位
            df = df[df['name_flg'].notna()].copy()
            texts = df['sms_body'].tolist()
            labels = df['name_flg'].astype(int).tolist()
            sms_ids = df['sms_id'].astype(str).tolist()
        elif category_type == "travel":
            # travel 分類使用 label 欄位
            df = df[df['label'].notna()].copy()
            texts = df['sms_body'].tolist()
            labels = df['label'].astype(int).tolist()
            sms_ids = df['sms_id'].astype(str).tolist()
        else:
            raise ValueError(f"不支援的分類類型: {category_type}")
        
        print(f"從 {file_path} 載入 {len(texts)} 筆 {category_type} 數據")
        return texts, labels, sms_ids
        
    except FileNotFoundError:
        raise FileNotFoundError(f"找不到檔案: {file_path}")
    except Exception as e:
        raise Exception(f"讀取檔案 {file_path} 時發生錯誤: {e}")

def load_multiple_datasets_simple_with_ids(file_paths: List[str], category_type: str) -> tuple:
    """
    載入多個數據集檔案並合併，包含 SMS ID（簡化版）
    Args:
        file_paths: 檔案絕對路徑列表
        category_type: 分類類型 ("name" 或 "travel")
    Returns:
        (all_texts, all_labels, all_sms_ids) 元組
    """
    all_texts = []
    all_labels = []
    all_sms_ids = []
    
    for file_path in file_paths:
        if os.path.exists(file_path):
            texts, labels, sms_ids = load_labeled_data_simple_with_ids(file_path, category_type)
            all_texts.extend(texts)
            all_labels.extend(labels)
            all_sms_ids.extend(sms_ids)
        else:
            print(f"警告: 檔案不存在 {file_path}")
    
    print(f"總計載入 {len(all_texts)} 筆 {category_type} 分類數據")
    return all_texts, all_labels, all_sms_ids

def load_labeled_data_simple(file_path: str, category_type: str) -> tuple:
    """
    簡化版載入標註數據（直接使用絕對路徑）
    Args:
        file_path: 數據檔案絕對路徑
        category_type: 分類類型 ("name" 或 "travel")
    Returns:
        (texts, labels) 元組
    """
    import pandas as pd
    
    try:
        df = pd.read_csv(file_path)
        
        # 根據分類類型選擇對應的標籤欄位
        if category_type == "name":
            # name 分類使用 name_flg 欄位
            df = df[df['name_flg'].notna()].copy()
            texts = df['sms_body'].tolist()
            labels = df['name_flg'].astype(int).tolist()
        elif category_type == "travel":
            # travel 分類使用 label 欄位
            df = df[df['label'].notna()].copy()
            texts = df['sms_body'].tolist()
            labels = df['label'].astype(int).tolist()
        else:
            raise ValueError(f"不支援的分類類型: {category_type}")
        
        print(f"從 {file_path} 載入 {len(texts)} 筆 {category_type} 數據")
        return texts, labels
        
    except FileNotFoundError:
        raise FileNotFoundError(f"找不到檔案: {file_path}")
    except Exception as e:
        raise Exception(f"讀取檔案 {file_path} 時發生錯誤: {e}")

def load_multiple_datasets_simple(file_paths: List[str], category_type: str) -> tuple:
    """
    載入多個數據集檔案並合併（簡化版）
    Args:
        file_paths: 檔案絕對路徑列表
        category_type: 分類類型 ("name" 或 "travel")
    Returns:
        (all_texts, all_labels) 元組
    """
    all_texts = []
    all_labels = []
    
    for file_path in file_paths:
        if os.path.exists(file_path):
            texts, labels = load_labeled_data_simple(file_path, category_type)
            all_texts.extend(texts)
            all_labels.extend(labels)
        else:
            print(f"警告: 檔案不存在 {file_path}")
    
    print(f"總計載入 {len(all_texts)} 筆 {category_type} 分類數據")
    return all_texts, all_labels

def load_multiple_datasets(file_paths: List[str], category_type: str) -> tuple:
    """
    載入多個數據集檔案並合併 (使用絕對路徑)
    Args:
        file_paths: 檔案絕對路徑列表
        category_type: 分類類型 ("name" 或 "travel")
    Returns:
        (all_texts, all_labels) 元組
    """
    all_texts = []
    all_labels = []
    
    for file_path in file_paths:
        if os.path.exists(file_path):
            try:
                texts, labels = load_labeled_data(file_path, category_type)
                all_texts.extend(texts)
                all_labels.extend(labels)
                print(f"載入 {file_path}: {len(texts)} 筆數據")
            except Exception as e:
                print(f"載入失敗 {file_path}: {e}")
        else:
            print(f"警告: 檔案不存在 {file_path}")
    
    print(f"總計載入 {len(all_texts)} 筆 {category_type} 分類數據")
    return all_texts, all_labels

def save_results_csv(results: Dict[str, int], output_path: str, classifier_type: str = "name") -> None:
    """
    儲存分類結果到 CSV 檔案，使用競賽規則的欄位名稱
    Args:
        results: 分類結果字典 {id: result}
        output_path: 輸出檔案路徑
        classifier_type: 分類器類型 ("name" 或 "travel")
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    try:
        with open(output_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            # 根據分類器類型決定欄位名稱
            if classifier_type == "name":
                writer.writerow(['sms_id', 'name_flg'])
            else:  # travel
                writer.writerow(['sms_id', 'label'])
            
            for k, v in results.items():
                writer.writerow([k, v])
        print(f"分類結果已儲存至: {output_path}")
    except Exception as e:
        raise Exception(f"儲存 CSV 檔案時發生錯誤: {e}")

def parse_xml_response(xml_response: str, tag: str) -> Dict[str, int]:
    """
    解析 LLM 返回的 XML 格式回應
    Args:
        xml_response: LLM 返回的 XML 格式字串
        tag: 結果標籤 (如 isName, isTravel)
    Returns:
        字典格式 {id: 0/1}
    """
    results = {}
    try:
        cleaned_response = xml_response.strip()
        if cleaned_response.startswith("```xml"):
            cleaned_response = cleaned_response[6:]
        if cleaned_response.endswith("```"):
            cleaned_response = cleaned_response[:-3]
        cleaned_response = cleaned_response.strip()
        root = ET.fromstring(cleaned_response)
        for message in root.findall('message'):
            mid = message.attrib.get('id')
            result = message.find(tag)
            if mid is not None and result is not None:
                results[mid] = 1 if result.text.strip().lower() == 'yes' else 0
    except ET.ParseError as e:
        print(f"XML 解析錯誤: {e}")
        print(f"回應內容: {xml_response}")
        raise
    except Exception as e:
        print(f"解析回應時發生錯誤: {e}")
        raise
    return results

def estimate_token_count(prompt: str, messages: List[Dict[str, str]]) -> int:
    """
    粗略估算 prompt + messages 的 token 數（以字元數/4 近似）
    """
    total_chars = len(prompt)
    for msg in messages:
        total_chars += len(msg.get('message', ''))
    return total_chars // 4

def escape_xml(text: str) -> str:
    """
    將文字中的 XML 特殊字元進行轉義
    """
    if not text:
        return text
    text = text.replace("&", "&amp;")
    text = text.replace("<", "&lt;")
    text = text.replace(">", "&gt;")
    text = text.replace('"', "&quot;")
    text = text.replace("'", "&apos;")
    return text

def sanitize_llm_xml(xml_str: str) -> str:
    """
    修復 LLM 輸出的常見 XML 格式錯誤
    """
    import re
    if not xml_str:
        return xml_str
    xml_str = re.sub(r'^```xml\s*', '', xml_str, flags=re.MULTILINE)
    xml_str = re.sub(r'\s*```\s*$', '', xml_str, flags=re.MULTILINE)
    seen_ids = set()
    lines = xml_str.split('\n')
    filtered_lines = []
    skip_until_close = False
    for line in lines:
        match = re.search(r'<message id=\"([^\"]+)\">', line)
        if match:
            msg_id = match.group(1)
            if msg_id in seen_ids:
                skip_until_close = True
                continue
            else:
                seen_ids.add(msg_id)
                skip_until_close = False
        if '</message>' in line and skip_until_close:
            skip_until_close = False
            continue
        if not skip_until_close:
            filtered_lines.append(line)
    return '\n'.join(filtered_lines)

def get_existing_classified_ids(output_dir: str, prefix: str, model_name: str) -> set:
    """
    讀取同類型分類結果檔案，取得已分類的 ID 清單
    Args:
        output_dir: 輸出目錄路徑
        prefix: 檔案前綴（如 name 或 travel）
        model_name: 模型名稱
    Returns:
        已分類的 ID 集合
    """
    import glob
    import csv
    from pathlib import Path
    existing_ids = set()
    safe_model_name = model_name.replace("/", "_").replace(":", "_")
    pattern = f"{prefix}_{safe_model_name}_*.csv"
    search_path = Path(output_dir) / pattern
    try:
        matching_files = glob.glob(str(search_path))
        for file_path in matching_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        # 支援多種 id 欄位名稱
                        sms_id = row.get('sms_id') or row.get('id')
                        if sms_id:
                            existing_ids.add(str(sms_id))
            except Exception as e:
                print(f"讀取檔案 {file_path} 時發生錯誤: {e}")
                continue
        if existing_ids:
            print(f"找到 {len(matching_files)} 個已存在的分類檔案，共 {len(existing_ids)} 筆已分類記錄")
    except Exception as e:
        print(f"搜尋已存在檔案時發生錯誤: {e}")
    return existing_ids

def clean_text_for_bert(text: str) -> str:
    """
    為 BERT 模型清理文本
    Args:
        text: 原始文本
    Returns:
        清理後的文本
    """
    if not text or not isinstance(text, str):
        return ""
    
    import re
    
    # 移除 HTML 標籤
    text = re.sub(r'<[^>]+>', '', text)
    
    # 統一換行符號
    text = re.sub(r'\r\n|\r|\n', ' ', text)
    
    # 處理連續空白
    text = re.sub(r'\s+', ' ', text)
    
    # 移除首尾空白
    text = text.strip()
    
    return text


def analyze_text_length_distribution(texts: list, tokenizer=None) -> dict:
    """
    分析文本長度分佈，用於設定 BERT 的 max_length 參數
    Args:
        texts: 文本列表
        tokenizer: BERT 分詞器（可選）
    Returns:
        長度統計資訊
    """
    import numpy as np
    
    if tokenizer:
        # 使用 BERT 分詞器計算 token 長度
        lengths = []
        for text in texts:
            tokens = tokenizer.encode(str(text), add_special_tokens=True)
            lengths.append(len(tokens))
    else:
        # 使用字元長度
        lengths = [len(str(text)) for text in texts]
    
    lengths = np.array(lengths)
    
    stats = {
        'count': len(lengths),
        'mean': float(np.mean(lengths)),
        'std': float(np.std(lengths)),
        'min': float(np.min(lengths)),
        'max': float(np.max(lengths)),
        'percentile_50': float(np.percentile(lengths, 50)),
        'percentile_95': float(np.percentile(lengths, 95)),
        'percentile_99': float(np.percentile(lengths, 99))
    }
    
    return stats


def load_bert_labeled_data(labeled_csv_path: str, input_csv_path: str) -> tuple:
    """
    載入用於 BERT 訓練的標註資料
    Args:
        labeled_csv_path: 標註結果檔案路徑
        input_csv_path: 原始簡訊檔案路徑
    Returns:
        (texts, labels, sms_ids)
    """
    import pandas as pd
    
    # 讀取標註結果
    labeled_df = pd.read_csv(labeled_csv_path)
    
    # 讀取原始簡訊內容
    input_messages = load_input_csv(input_csv_path)
    input_df = pd.DataFrame(input_messages)
    input_df.columns = ['sms_id', 'sms_body']
    
    # 合併資料
    merged_df = pd.merge(labeled_df, input_df, on='sms_id', how='inner')
    
    # 清理文本
    merged_df['sms_body'] = merged_df['sms_body'].apply(clean_text_for_bert)
    
    # 移除空文本
    merged_df = merged_df[merged_df['sms_body'].str.len() > 0]
    
    texts = merged_df['sms_body'].tolist()
    labels = merged_df['label'].tolist()
    sms_ids = merged_df['sms_id'].tolist()
    
    return texts, labels, sms_ids


def save_bert_predictions(predictions: dict, probabilities: dict, output_path: str, classifier_type: str = "travel") -> None:
    """
    儲存 BERT 模型預測結果（包含機率）
    Args:
        predictions: 預測結果字典 {id: label}
        probabilities: 預測機率字典 {id: probability}
        output_path: 輸出檔案路徑
        classifier_type: 分類器類型
    """
    import csv
    import os
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    try:
        with open(output_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            
            # 根據分類器類型決定欄位名稱
            if classifier_type == "name":
                writer.writerow(['sms_id', 'name_flg', 'probability'])
            else:  # travel
                writer.writerow(['sms_id', 'label', 'probability'])
            
            for sms_id in predictions:
                label = predictions[sms_id]
                prob = probabilities.get(sms_id, 0.0)
                writer.writerow([sms_id, label, prob])
        
        print(f"BERT 預測結果（含機率）已儲存至: {output_path}")
    except Exception as e:
        raise Exception(f"儲存 BERT 預測結果時發生錯誤: {e}")


def create_ensemble_submission(
    model_predictions: list,
    weights: list = None,
    max_submissions: int = 30000,
    output_path: str = None
) -> str:
    """
    建立多模型集成的提交檔案
    Args:
        model_predictions: 模型預測結果列表，每個元素為 (sms_id, label, probability)
        weights: 模型權重列表
        max_submissions: 最大提交筆數
        output_path: 輸出檔案路徑
    Returns:
        輸出檔案路徑
    """
    pass  # 保留原有實作

def clean_text_for_bert(text: str) -> str:
    """
    為 BERT 模型清理文本
    Args:
        text: 原始文本
    Returns:
        清理後的文本
    """
    import re
    import pandas as pd
    
    if not text or pd.isna(text):
        return ""
    
    # 轉換為字符串並去除前後空白
    text = str(text).strip()
    
    # 移除HTML標籤
    text = re.sub(r'<[^>]+>', '', text)
    
    # 正規化空白字符
    text = re.sub(r'\s+', ' ', text)
    
    # 移除控制字符
    text = re.sub(r'[\r\n\t]', ' ', text)
    
    return text.strip()

def analyze_text_length_distribution(texts: List[str]) -> Dict:
    """
    分析文本長度分布
    Args:
        texts: 文本列表
    Returns:
        長度統計字典
    """
    import numpy as np
    
    lengths = [len(text) for text in texts]
    
    return {
        'mean': np.mean(lengths),
        'median': np.median(lengths),
        'std': np.std(lengths),
        'min': min(lengths),
        'max': max(lengths),
        'p95': np.percentile(lengths, 95),
        'p99': np.percentile(lengths, 99)
    }

def get_dataset_paths(category_type: str, base_dir: str = "data_game_2025/data/results/labled/stage2/train_data") -> Dict[str, str]:
    """
    獲取指定分類類型的數據集路徑
    Args:
        category_type: 分類類型 ("name" 或 "travel")
        base_dir: 基礎目錄路徑
    Returns:
        包含 train, val, test 路徑的字典
    """
    return {
        'train': os.path.join(base_dir, f"{category_type}_train_8000.csv"),
        'val': os.path.join(base_dir, f"{category_type}_val_8000.csv"),
        'test': os.path.join(base_dir, f"{category_type}_test_8000.csv")
    }
    
    # 收集所有模型的預測結果
    all_predictions = {}
    
    for i, (predictions, model_name) in enumerate(model_predictions):
        weight = weights[i]
        
        for sms_id, label, prob in predictions:
            if sms_id not in all_predictions:
                all_predictions[sms_id] = {'probabilities': [], 'labels': []}
            
            all_predictions[sms_id]['probabilities'].append(prob * weight)
            all_predictions[sms_id]['labels'].append(label)
    
    # 計算加權平均和最終預測
    final_results = []
    for sms_id, data in all_predictions.items():
        avg_prob = np.mean(data['probabilities'])
        # 使用多數投票決定最終標籤
        final_label = 1 if avg_prob > 0.5 else 0
        final_results.append((sms_id, final_label, avg_prob))
    
    # 按機率排序並選取前 N 筆
    final_results.sort(key=lambda x: x[2], reverse=True)
    
    # 只保留預測為正例的結果
    positive_results = [(sms_id, label, prob) for sms_id, label, prob in final_results if label == 1]
    submission_results = positive_results[:max_submissions]
    
    # 建立提交檔案
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        output_path = f"data_game_2025/data/results/ensemble_submission_{timestamp}.csv"
    
    # 儲存結果
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    import csv
    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['sms_id', 'label'])
        
        for sms_id, label, _ in submission_results:
            writer.writerow([sms_id, label])
    
    print(f"集成提交檔案已生成: {output_path}")
    print(f"提交筆數: {len(submission_results)}")
    
    return output_path
