#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MacBERT 模型推論腳本
專門用於運行 MacBERT 模型的推論
"""

import sys
import os
from pathlib import Path
from datetime import datetime
from typing import List, Dict

# 添加 src 目錄到 Python 路徑
src_dir = Path(__file__).parent.parent
sys.path.append(str(src_dir))

from bert_model.bert_config import BertConfig
from bert_model.bert_travel_inference import TravelBertInference
from bert_model.bert_name_inference import NameBertInference
from utils import load_input_csv


def save_dataframe_chunked(df, output_path: str, chunk_size: int = 10000) -> None:
    """
    分批儲存 DataFrame 以避免記憶體問題
    
    Args:
        df: 要儲存的 DataFrame
        output_path: 輸出檔案路徑
        chunk_size: 每批的大小，預設 10000
    """
    import pandas as pd
    
    total_rows = len(df)
    
    if total_rows <= chunk_size:
        # 資料量小，直接儲存
        df.to_csv(output_path, index=False, encoding='utf-8')
        print(f"✅ 結果已保存: {output_path}")
    else:
        # 資料量大，分批儲存
        print(f"🔄 資料量較大 ({total_rows} 筆)，分批儲存中...")
        for i in range(0, total_rows, chunk_size):
            chunk_end = min(i + chunk_size, total_rows)
            chunk_df = df.iloc[i:chunk_end]
            
            # 第一批包含表頭，後續批次不包含表頭
            mode = 'w' if i == 0 else 'a'
            header = i == 0
            
            chunk_df.to_csv(output_path, mode=mode, header=header, index=False, encoding='utf-8')
            print(f"  💾 已儲存第 {i//chunk_size + 1} 批 ({i+1}-{chunk_end}/{total_rows})")
        
        print(f"✅ 結果已分批保存: {output_path}")


def infer_travel_classifier(
    model_dir: str,
    input_csv: str,
    batch_size: int = 32
) -> List[Dict]:
    """
    執行旅遊分類推論
    
    Args:
        model_dir: 旅遊分類模型目錄
        input_csv: 輸入 CSV 檔案路徑
        batch_size: 批次大小
        
    Returns:
        預測結果列表，包含 sms_id, travel_prob, label
    """
    print("=== BERT 旅遊簡訊分類器推論 ===")
    
    # 檢查模型是否存在（如果是本地路徑）或是否為 Hugging Face 模型 ID
    is_local_path = os.path.exists(model_dir) or os.path.isabs(model_dir)
    is_hf_model = not is_local_path and "/" in model_dir
    
    if not is_local_path and not is_hf_model:
        raise FileNotFoundError(f"找不到旅遊分類模型: {model_dir}")
    
    # 載入推論器
    try:
        inferencer = TravelBertInference(model_dir)
        if is_hf_model:
            print(f"✅ 旅遊分類模型從 Hugging Face 載入成功: {model_dir}")
        else:
            print(f"✅ 旅遊分類模型從本地載入成功: {model_dir}")
    except Exception as e:
        raise Exception(f"載入旅遊分類模型失敗: {e}")
    
    # 載入資料
    try:
        messages = load_input_csv(input_csv)
        sms_ids = [msg['id'] for msg in messages]
        texts = [msg['message'] for msg in messages]
        print(f"✅ 載入 {len(texts)} 筆簡訊")
    except Exception as e:
        raise Exception(f"載入輸入檔案失敗: {e}")
    
    # 批次推論
    try:
        print(f"🔄 開始旅遊分類推論（批次大小: {batch_size}）...")
        labels, probabilities = inferencer.predict_batch(texts, batch_size=batch_size)
        print(f"✅ 旅遊分類推論完成")
    except Exception as e:
        raise Exception(f"旅遊分類推論失敗: {e}")
    
    # 整理結果，機率保留到小數點第4位以避免過度四捨五入
    results = []
    for sms_id, prob, label in zip(sms_ids, probabilities, labels):
        results.append({
            'sms_id': sms_id,
            'travel_prob': round(prob, 4),
            'label': label
        })
    return results


def infer_name_classifier(
    model_dir: str,
    input_csv: str,
    batch_size: int = 32
) -> List[Dict]:
    """
    執行姓名分類推論
    
    Args:
        model_dir: 姓名分類模型目錄
        input_csv: 輸入 CSV 檔案路徑
        batch_size: 批次大小
        
    Returns:
        預測結果列表，包含 sms_id, name_prob, name_flg
    """
    print("=== BERT 姓名簡訊分類器推論 ===")
    
    # 檢查模型是否存在（如果是本地路徑）或是否為 Hugging Face 模型 ID
    is_local_path = os.path.exists(model_dir) or os.path.isabs(model_dir)
    is_hf_model = not is_local_path and "/" in model_dir
    
    if not is_local_path and not is_hf_model:
        raise FileNotFoundError(f"找不到姓名分類模型: {model_dir}")
    
    # 載入推論器
    try:
        inferencer = NameBertInference(model_dir)
        if is_hf_model:
            print(f"✅ 姓名分類模型從 Hugging Face 載入成功: {model_dir}")
        else:
            print(f"✅ 姓名分類模型從本地載入成功: {model_dir}")
    except Exception as e:
        raise Exception(f"載入姓名分類模型失敗: {e}")
    
    # 載入資料
    try:
        messages = load_input_csv(input_csv)
        sms_ids = [msg['id'] for msg in messages]
        texts = [msg['message'] for msg in messages]
        print(f"✅ 載入 {len(texts)} 筆簡訊")
    except Exception as e:
        raise Exception(f"載入輸入檔案失敗: {e}")
    
    # 批次推論
    try:
        print(f"🔄 開始姓名分類推論（批次大小: {batch_size}）...")
        labels, probabilities = inferencer.predict_batch(texts, batch_size=batch_size)
        print(f"✅ 姓名分類推論完成")
    except Exception as e:
        raise Exception(f"姓名分類推論失敗: {e}")
    
    # 整理結果，機率保留到小數點第4位以避免過度四捨五入
    results = []
    for sms_id, prob, label in zip(sms_ids, probabilities, labels):
        results.append({
            'sms_id': sms_id,
            'name_prob': round(prob, 4),
            'name_flg': label
        })
    return results


def combine_inference_results(
    travel_results: List[Dict],
    name_results: List[Dict],
    output_csv: str
) -> str:
    """
    合併旅遊與姓名分類結果，並輸出為 CSV
    使用記憶體友善的方式處理大量資料
    
    Args:
        travel_results: 旅遊分類結果
        name_results: 姓名分類結果
        output_csv: 輸出 CSV 檔案路徑
        
    Returns:
        輸出檔案路徑
    """
    import pandas as pd
    import gc
    
    print("=== 合併推論結果 ===")
    print(f"🔄 處理旅遊資料: {len(travel_results)} 筆")
    print(f"🔄 處理姓名資料: {len(name_results)} 筆")
    
    # 分批合併以避免記憶體問題
    chunk_size = 10000
    total_batches = max(len(travel_results), len(name_results))
    
    # 建立輸出目錄
    output_dir = os.path.dirname(output_csv)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # 如果資料量較小，使用原本的方式
    if total_batches <= chunk_size:
        print("📦 資料量較小，使用標準合併方式")
        
        # 將結果轉換為 DataFrame
        travel_df = pd.DataFrame(travel_results)
        name_df = pd.DataFrame(name_results)
        
        # 合併結果（以 sms_id 為鍵）
        merged_df = pd.merge(
            travel_df,
            name_df,
            on='sms_id',
            how='outer'
        )
        print(f"✅ 已合併結果")
        
        # 確保欄位順序符合 PRD 要求: sms_id, travel_prob, label, name_prob, name_flg
        result_df = merged_df[['sms_id', 'travel_prob', 'label', 'name_prob', 'name_flg']]
        print(f"✅ 已確保欄位順序符合 PRD 要求")

        # 修正SMS ID排序問題：將sms_id轉換為整數後排序
        try:
            result_df['sms_id'] = pd.to_numeric(result_df['sms_id'], errors='coerce')
            result_df = result_df.sort_values('sms_id').reset_index(drop=True)
            print(f"✅ 已按SMS ID整數順序排序")
        except Exception as e:
            print(f"⚠️ SMS ID排序警告: {e}")
        
        # 使用分批儲存函數
        save_dataframe_chunked(result_df, output_csv)
        
        # 清理記憶體
        del travel_df, name_df, merged_df, result_df
        gc.collect()
        
    else:
        print(f"📦 資料量較大，使用分批合併方式（每批 {chunk_size} 筆）")
        
        # 建立 sms_id 到索引的映射以加速查找
        print("🔄 建立索引映射...")
        travel_dict = {item['sms_id']: item for item in travel_results}
        name_dict = {item['sms_id']: item for item in name_results}
        
        # 獲取所有 sms_id 並排序
        all_sms_ids = set(travel_dict.keys()) | set(name_dict.keys())
        print(f"📊 總共有 {len(all_sms_ids)} 個唯一的 SMS ID")
        
        # 將 sms_id 轉換為數字並排序
        try:
            sorted_sms_ids = sorted(all_sms_ids, key=lambda x: int(x))
            print("✅ SMS ID 已按數字順序排序")
        except (ValueError, TypeError):
            sorted_sms_ids = sorted(all_sms_ids)
            print("⚠️ SMS ID 無法轉換為數字，使用字串排序")
        
        # 分批處理並直接寫入檔案
        first_batch = True
        total_processed = 0
        
        for i in range(0, len(sorted_sms_ids), chunk_size):
            batch_sms_ids = sorted_sms_ids[i:i + chunk_size]
            batch_results = []
            
            for sms_id in batch_sms_ids:
                # 從兩個字典中獲取資料
                travel_data = travel_dict.get(sms_id, {})
                name_data = name_dict.get(sms_id, {})
                
                # 合併資料
                combined_row = {
                    'sms_id': sms_id,
                    'travel_prob': travel_data.get('travel_prob', 0.0),
                    'label': travel_data.get('label', 0),
                    'name_prob': name_data.get('name_prob', 0.0),
                    'name_flg': name_data.get('name_flg', 0)
                }
                batch_results.append(combined_row)
            
            # 將批次結果轉換為 DataFrame 並寫入
            batch_df = pd.DataFrame(batch_results)
            
            # 寫入檔案
            mode = 'w' if first_batch else 'a'
            header = first_batch
            
            batch_df.to_csv(output_csv, mode=mode, header=header, index=False, encoding='utf-8')
            
            total_processed += len(batch_results)
            batch_num = (i // chunk_size) + 1
            total_batches_calc = (len(sorted_sms_ids) + chunk_size - 1) // chunk_size
            
            print(f"  � 已處理第 {batch_num}/{total_batches_calc} 批 ({total_processed}/{len(sorted_sms_ids)})")
            
            # 清理記憶體
            del batch_df, batch_results
            gc.collect()
            
            first_batch = False
        
        print(f"✅ 分批合併完成，總共處理 {total_processed} 筆資料")
        
        # 清理記憶體
        del travel_dict, name_dict, all_sms_ids, sorted_sms_ids
        gc.collect()
    
    print(f"📊 總共處理 {len(travel_results)} 筆簡訊")
    
    # 計算統計數據（避免重新載入整個檔案）
    travel_positive = sum(1 for r in travel_results if r.get('label', 0) == 1)
    name_positive = sum(1 for r in name_results if r.get('name_flg', 0) == 1)
    
    print(f"📊 旅遊相關簡訊: {travel_positive} 筆")
    print(f"📊 姓名相關簡訊: {name_positive} 筆")
    print(f"✅ 合併結果已保存: {output_csv}")
    
    return output_csv


def run_macbert_inference(input_file=None, output_dir=None, travel_model=None, name_model=None, mode='both'):
    """
    執行 MacBERT 模型推論
    
    Args:
        input_file: 輸入 CSV 檔案路徑
        output_dir: 輸出目錄路徑
        travel_model: 旅遊分類模型路徑或 Hugging Face 模型 ID
        name_model: 姓名分類模型路徑或 Hugging Face 模型 ID
        mode: 推論模式 ('travel', 'name', 'both')
    """
    print("🚀 開始執行 MacBERT 模型推論...")
    
    # 使用 MacBERT 專用配置，並處理配置檔案載入錯誤
    try:
        config_path = os.path.join(os.path.dirname(__file__), "config_macbert.ini")
        config = BertConfig(config_path)
        inference_config = config.get_inference_config()
        
        print(f"📋 成功載入 MacBERT 配置檔案: {config_path}")
        print(f"📦 MacBERT 配置的批次大小: {inference_config['inference_batch_size']}")
        
        # 直接從配置檔案讀取 output_dir（在 inference 區段）
        config_output_dir = config.config.get('inference', 'output_dir')
        
    except Exception as e:
        print(f"⚠️ MacBERT 配置檔案載入失敗: {e}")
        print("🔄 使用 MacBERT 預設配置...")
        # 使用預設配置
        inference_config = {
            'input_file': '/Users/winstontang/PycharmProjects/forensic/data_game_2025/data/raw/datagame_sms_stage1_raw_TEXT_ONLY.csv',
            'inference_batch_size': 32,  # MacBERT 預設較小的批次大小
        }
        config_output_dir = '/Users/winstontang/PycharmProjects/forensic/data_game_2025/data/results/vote/candidates'
    
    # 設定輸入檔案
    if input_file is None:
        input_file = inference_config['input_file']
    
    # 設定輸出目錄
    if output_dir is None:
        output_dir = Path(config_output_dir)
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 處理模型路徑
    def get_model_path(model_arg, model_type):
        """獲取模型路徑，支援本地路徑和 Hugging Face 模型 ID"""
        if model_arg:
            # 用戶指定了模型
            is_local_path = os.path.exists(model_arg) or os.path.isabs(model_arg)
            is_hf_model = not is_local_path and "/" in model_arg
            
            if is_local_path or is_hf_model:
                return model_arg
            else:
                raise FileNotFoundError(f"找不到{model_type}分類模型: {model_arg}")
        else:
            # 使用預設邏輯搜索本地 MacBERT 模型
            base_dir = Path("/Users/winstontang/PycharmProjects/forensic/data_game_2025/results")
            models = list(base_dir.glob(f"{model_type}_macbert_macbert_*"))
            
            if not models:
                raise FileNotFoundError(f"找不到本地 MacBERT {model_type}分類模型，請先訓練模型或指定 Hugging Face 模型 ID！")
            
            # 選擇最新的模型
            return str(max(models, key=lambda x: x.stat().st_mtime))
    
    results = {}
    
    # 處理旅遊分類
    if mode in ['travel', 'both']:
        travel_model_path = get_model_path(travel_model, 'travel')
        print(f"📁 旅遊分類模型: {travel_model_path}")
        
        print("\n🔍 開始旅遊分類推論...")
        travel_results = infer_travel_classifier(
            model_dir=travel_model_path,
            input_csv=input_file,
            batch_size=inference_config['inference_batch_size']
        )
        results['travel'] = travel_results
    
    # 處理姓名分類
    if mode in ['name', 'both']:
        name_model_path = get_model_path(name_model, 'name')
        print(f"📁 姓名分類模型: {name_model_path}")
        
        print("\n🔍 開始姓名分類推論...")
        name_results = infer_name_classifier(
            model_dir=name_model_path,
            input_csv=input_file,
            batch_size=inference_config['inference_batch_size']
        )
        results['name'] = name_results
    
    # 生成時間戳記
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    
    # 合併結果（如果兩種分類都執行了）
    if mode == 'both' and 'travel' in results and 'name' in results:
        print("\n🔄 合併推論結果...")
        combined_output = str(output_dir / f"both_macbert_{timestamp}.csv")
        combine_inference_results(results['travel'], results['name'], combined_output)
        print(f"✅ 合併結果完成: {combined_output}")
        results['combined'] = combined_output
    elif mode == 'travel' and 'travel' in results:
        # 只有旅遊分類，直接輸出（支援分批儲存）
        import pandas as pd
        print("\n🔄 儲存旅遊分類結果...")
        travel_output = str(output_dir / f"travel_macbert_{timestamp}.csv")
        travel_df = pd.DataFrame(results['travel'])
        save_dataframe_chunked(travel_df, travel_output)
        results['travel_output'] = travel_output
        print(f"✅ 旅遊分類結果完成: {travel_output}")
    elif mode == 'name' and 'name' in results:
        # 只有姓名分類，直接輸出（支援分批儲存）
        import pandas as pd
        print("\n🔄 儲存姓名分類結果...")
        name_output = str(output_dir / f"name_macbert_{timestamp}.csv")
        name_df = pd.DataFrame(results['name'])
        save_dataframe_chunked(name_df, name_output)
        results['name_output'] = name_output
        print(f"✅ 姓名分類結果完成: {name_output}")
    
    print("\n🎉 MacBERT 推論完成！")
    
    # 構建返回結果
    return_results = {}
    if 'travel' in results:
        return_results['travel'] = f"旅遊分類完成，共 {len(results['travel'])} 筆資料"
    if 'name' in results:
        return_results['name'] = f"姓名分類完成，共 {len(results['name'])} 筆資料"
    if 'combined' in results:
        return_results['combined'] = results['combined']
    if 'travel_output' in results:
        return_results['travel_output'] = results['travel_output']
    if 'name_output' in results:
        return_results['name_output'] = results['name_output']
    
    return return_results


def main():
    """主函數"""
    import argparse
    
    parser = argparse.ArgumentParser(description="MacBERT 模型推論")
    parser.add_argument('--input', help='輸入 CSV 檔案路徑 (可選，使用配置檔案中的預設值)')
    parser.add_argument('--output-dir', help='輸出目錄路徑 (可選)')
    parser.add_argument('--travel-model', help='旅遊分類模型路徑或 Hugging Face 模型 ID')
    parser.add_argument('--name-model', help='姓名分類模型路徑或 Hugging Face 模型 ID')
    parser.add_argument('--mode', choices=['travel', 'name', 'both'], default='both',
                       help='推論模式: travel（旅遊）, name（姓名）, 或 both（兩者）')
    
    args = parser.parse_args()
    
    print("🔥 MacBERT 推論腳本")
    print("=" * 40)
    
    # 顯示配置資訊
    print("🔧 載入 MacBERT 配置...")
    
    # 執行推論
    results = run_macbert_inference(
        input_file=args.input,
        output_dir=args.output_dir,
        travel_model=args.travel_model,
        name_model=args.name_model,
        mode=args.mode
    )
    
    print("\n📋 推論結果總結:")
    for task, path in results.items():
        print(f"  {task}: {path}")
    
    print("\n💡 使用提示:")
    print("1. 推論結果已保存在 candidates 目錄")
    print("2. 可以用於後續的模型比較分析")
    print("3. combined 檔案包含完整的分類結果")


if __name__ == "__main__":
    main()
