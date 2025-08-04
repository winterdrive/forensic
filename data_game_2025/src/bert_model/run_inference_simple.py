#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BERT 分類器推論執行腳本 - 簡化版（使用絕對路徑）
根據 data_game_2025/docs/bert/inference.md PRD 實作
"""

import os
import sys
import argparse
import pandas as pd
from datetime import datetime
from typing import List, Dict

# 添加當前目錄到 Python 路徑
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# 添加 src 目錄到 Python 路徑
src_dir = os.path.abspath(os.path.join(current_dir, '..'))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

try:
    from bert_model.bert_travel_inference import TravelBertInference
    from bert_model.bert_name_inference import NameBertInference
    from bert_model.bert_config import BertConfig
    from utils import load_input_csv
except ImportError as e:
    print(f"⚠️ 無法載入模組: {e}")
    print("請確認所有相關模組已正確安裝")
    sys.exit(1)


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
        
        # 調試：檢查前幾個 sms_id 是否正確
        if len(sms_ids) > 0:
            print(f"🔍 旅遊分類前5個 sms_id: {sms_ids[:5]}")
            print(f"🔍 旅遊分類 sms_id 類型: {type(sms_ids[0])}")
        
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
        
        # 調試：檢查前幾個 sms_id 是否正確
        if len(sms_ids) > 0:
            print(f"🔍 姓名分類前5個 sms_id: {sms_ids[:5]}")
            print(f"🔍 姓名分類 sms_id 類型: {type(sms_ids[0])}")
        
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


def save_dataframe_chunked(df: pd.DataFrame, output_path: str, chunk_size: int = 10000) -> None:
    """
    分批儲存 DataFrame 以避免記憶體問題
    
    Args:
        df: 要儲存的 DataFrame
        output_path: 輸出檔案路徑
        chunk_size: 每批的大小，預設 10000
    """
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
        
        # 調試：檢查每個 DataFrame 的內容
        print(f"🔍 旅遊分類結果 DataFrame 形狀: {travel_df.shape}")
        print(f"🔍 旅遊分類結果前3行:\n{travel_df.head(3)}")
        
        print(f"🔍 姓名分類結果 DataFrame 形狀: {name_df.shape}")
        print(f"🔍 姓名分類結果前3行:\n{name_df.head(3)}")
        
        # 合併結果（以 sms_id 為鍵）
        merged_df = pd.merge(
            travel_df,
            name_df,
            on='sms_id',
            how='outer'
        )
        print(f"✅ 已合併結果")
        
        # 調試：檢查合併後的 DataFrame
        print(f"🔍 合併後 DataFrame 形狀: {merged_df.shape}")
        print(f"🔍 合併後前3行:\n{merged_df.head(3)}")
        
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
        
        # 調試：檢查最終結果
        print(f"🔍 最終結果 DataFrame 前3行:\n{result_df.head(3)}")
        
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


def get_default_paths():
    """獲取默認的絕對文件路徑（從配置檔案或預設值讀取）"""
    try:
        # 確保使用正確的配置檔案路徑
        config_path = os.path.join(os.path.dirname(__file__), "config.ini")
        config = BertConfig(config_path)
        inference_config = config.get_inference_config()
        
        print(f"📋 成功載入配置檔案: {config_path}")
        print(f"📦 配置的批次大小: {inference_config['inference_batch_size']}")
        
        # 直接從配置檔案讀取 output_dir（在 inference 區段）
        output_dir = config.config.get('inference', 'output_dir')
        
        return {
            'input': config.config.get('inference', 'input_file'),
            'travel_model': config.config.get('inference', 'travel_model_path'),
            'name_model': config.config.get('inference', 'name_model_path'),
            'output_dir': output_dir,
            'batch_size': inference_config['inference_batch_size']
        }
    except Exception as e:
        # 如果配置檔案載入失敗，使用預設路徑
        print(f"⚠️ 配置檔案載入失敗: {e}")
        print("🔄 使用預設配置...")
        return {
            'input': 'data_game_2025/data/raw/datagame_sms_stage1_raw_TEXT_ONLY.csv',
            'travel_model': 'data_game_2025/results/travel_bert_20250725_2337',
            'name_model': 'data_game_2025/results/name_bert_20250725_2359',
            'output_dir': 'data_game_2025/data/results',
            'batch_size': 32
        }


def main():
    """主函數"""
    parser = argparse.ArgumentParser(
        description='BERT 分類器推論 - 根據 PRD 文件實作',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用範例:
  # 使用預設路徑進行推論
  python run_inference_simple.py
  
  # 指定模型路徑進行推論
  python run_inference_simple.py \\
    --input data_game_2025/data/raw/datagame_sms_stage1_raw_TEXT_ONLY.csv \\
    --travel-model data_game_2025/results/travel_bert_20250725_2337 \\
    --name-model data_game_2025/results/name_bert_20250725_2359 \\
    --output data_game_2025/data/results/inference_result.csv
  
  # 只執行旅遊分類推論
  python run_inference_simple.py --mode travel \\
    --travel-model data_game_2025/results/travel_bert_20250725_2337
        """
    )
    
    # 模式選擇
    parser.add_argument(
        '--mode', 
        choices=['travel', 'name', 'both'], 
        default='both',
        help='推論模式: travel（旅遊）, name（姓名）, 或 both（兩者）'
    )
    
    # 輸入檔案
    parser.add_argument(
        '--input', 
        type=str,
        help='輸入 CSV 檔案路徑（預設使用配置檔案中的路徑）'
    )
    
    # 模型路徑
    parser.add_argument(
        '--travel-model', 
        type=str,
        help='旅遊分類模型目錄路徑'
    )
    
    parser.add_argument(
        '--name-model', 
        type=str,
        help='姓名分類模型目錄路徑'
    )
    
    # 輸出檔案
    parser.add_argument(
        '--output', 
        type=str,
        help='輸出 CSV 檔案路徑（預設為時間戳命名）'
    )
    
    # 推論參數
    parser.add_argument(
        '--batch-size', 
        type=int, 
        help='批次大小（預設從配置檔案讀取）'
    )
    
    args = parser.parse_args()
    
    # 獲取預設路徑
    print("🔧 載入配置...")
    default_paths = get_default_paths()
    
    # 設定批次大小（優先使用命令列參數，否則使用配置檔案）
    batch_size = args.batch_size if args.batch_size is not None else default_paths['batch_size']
    
    if args.batch_size is not None:
        print(f"📦 使用命令列指定的批次大小: {batch_size}")
    else:
        print(f"📦 使用配置檔案的批次大小: {batch_size}")
    
    # 設定輸入檔案路徑
    input_csv = args.input or default_paths['input']
    if not os.path.exists(input_csv):
        print(f"❌ 找不到輸入檔案: {input_csv}")
        return 1
    
    # 設定輸出檔案路徑
    if args.output:
        output_csv = args.output
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        output_filename = f"both_multiligual_{timestamp}.csv"
        output_csv = os.path.join(default_paths['output_dir'], output_filename)
    
    print("🚀 開始 BERT 分類器推論流程")
    print("=" * 50)
    print(f"📄 輸入檔案: {input_csv}")
    print(f"💾 輸出檔案: {output_csv}")
    print(f"🔧 推論模式: {args.mode}")
    print(f"📦 批次大小: {batch_size}")
    print()
    
    results = {}
    
    try:
        # 執行旅遊分類推論
        if args.mode in ['travel', 'both']:
            travel_model = args.travel_model or default_paths['travel_model']
            
            # 檢查模型是否存在（本地路徑）或是否為 Hugging Face 模型 ID
            is_local_path = os.path.exists(travel_model) or os.path.isabs(travel_model)
            is_hf_model = not is_local_path and "/" in travel_model
            
            if not is_local_path and not is_hf_model:
                print(f"❌ 找不到旅遊分類模型: {travel_model}")
                return 1
            
            travel_results = infer_travel_classifier(
                model_dir=travel_model,
                input_csv=input_csv,
                batch_size=batch_size
            )
            results['travel'] = travel_results
            print(f"✅ 旅遊分類推論成功，處理 {len(travel_results)} 筆簡訊")
            print()
        
        # 執行姓名分類推論
        if args.mode in ['name', 'both']:
            name_model = args.name_model or default_paths['name_model']
            
            # 檢查模型是否存在（本地路徑）或是否為 Hugging Face 模型 ID
            is_local_path = os.path.exists(name_model) or os.path.isabs(name_model)
            is_hf_model = not is_local_path and "/" in name_model
            
            if not is_local_path and not is_hf_model:
                print(f"❌ 找不到姓名分類模型: {name_model}")
                return 1
            
            name_results = infer_name_classifier(
                model_dir=name_model,
                input_csv=input_csv,
                batch_size=batch_size
            )
            results['name'] = name_results
            print(f"✅ 姓名分類推論成功，處理 {len(name_results)} 筆簡訊")
            print()
        
        # 合併並輸出結果
        if args.mode == 'both':
            # 兩種分類都執行，合併結果
            combine_inference_results(
                travel_results=results['travel'],
                name_results=results['name'],
                output_csv=output_csv
            )
        elif args.mode == 'travel':
            # 只有旅遊分類，直接輸出（支援分批儲存）
            travel_df = pd.DataFrame(results['travel'])
            os.makedirs(os.path.dirname(output_csv), exist_ok=True)
            save_dataframe_chunked(travel_df, output_csv)
            
        elif args.mode == 'name':
            # 只有姓名分類，直接輸出（支援分批儲存）
            name_df = pd.DataFrame(results['name'])
            os.makedirs(os.path.dirname(output_csv), exist_ok=True)
            save_dataframe_chunked(name_df, output_csv)
        
        # 打印結果總結
        print("\n" + "=" * 50)
        print("📊 推論結果總結:")
        
        if 'travel' in results:
            travel_positive = sum(1 for r in results['travel'] if r['label'] == 1)
            print(f"  🧳 旅遊分類: {travel_positive}/{len(results['travel'])} 筆預測為旅遊相關")
        
        if 'name' in results:
            name_positive = sum(1 for r in results['name'] if r['name_flg'] == 1)
            print(f"  👤 姓名分類: {name_positive}/{len(results['name'])} 筆預測為姓名相關")
        
        print(f"  💾 結果檔案: {output_csv}")
        print("🎉 推論完成！")
        
        return 0
        
    except Exception as e:
        print(f"❌ 推論過程中發生錯誤: {e}")
        return 1


def print_usage_examples():
    """打印使用範例"""
    print("BERT 分類器推論腳本使用範例:")
    print()
    print("1. 使用預設路徑進行雙模型推論:")
    print("   python run_inference_simple.py")
    print()
    print("2. 指定模型路徑進行推論:")
    print("   python run_inference_simple.py \\")
    print("     --input data_game_2025/data/raw/datagame_sms_stage1_raw_TEXT_ONLY.csv \\")
    print("     --travel-model data_game_2025/results/travel_bert_20250725_2337 \\")
    print("     --name-model data_game_2025/results/name_bert_20250725_2359 \\")
    print("     --output data_game_2025/data/results/inference_result.csv")
    print()
    print("3. 只執行旅遊分類推論:")
    print("   python run_inference_simple.py --mode travel \\")
    print("     --travel-model data_game_2025/results/travel_bert_20250725_2337")
    print()
    print("4. 只執行姓名分類推論:")
    print("   python run_inference_simple.py --mode name \\")
    print("     --name-model data_game_2025/results/name_bert_20250725_2359")
    print()
    print("5. 指定批次大小:")
    print("   python run_inference_simple.py --batch-size 16")


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
