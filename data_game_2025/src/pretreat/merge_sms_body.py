#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
合併SMS簡訊內容腳本
將分類結果檔案與原始簡訊內容進行合併

功能：
- 讀取姓名和旅遊分類結果檔案
- 透過 sms_id 與原始簡訊檔案進行 JOIN
- 填入 sms_body 欄位，其他欄位保持不變
- 輸出合併後的檔案
"""

import pandas as pd
import argparse
import os
from pathlib import Path

def load_csv_with_bom(file_path: str) -> pd.DataFrame:
    """
    載入可能包含 BOM 的 CSV 檔案
    Args:
        file_path: CSV 檔案路徑
    Returns:
        DataFrame
    """
    try:
        # 先嘗試使用 utf-8-sig 處理 BOM
        df = pd.read_csv(file_path, encoding='utf-8-sig')
        print(f"✅ 成功載入檔案: {file_path} ({len(df)} 筆記錄)")
        return df
    except Exception as e:
        # 如果失敗，嘗試使用 utf-8
        try:
            df = pd.read_csv(file_path, encoding='utf-8')
            print(f"✅ 成功載入檔案: {file_path} ({len(df)} 筆記錄)")
            return df
        except Exception as e2:
            raise Exception(f"無法載入檔案 {file_path}: {e2}")

def merge_sms_body(result_file: str, sms_file: str, output_file: str):
    """
    合併分類結果檔案與簡訊內容
    Args:
        result_file: 分類結果檔案路徑
        sms_file: 原始簡訊檔案路徑
        output_file: 輸出檔案路徑
    """
    print(f"📝 開始合併檔案...")
    print(f"   分類結果檔案: {result_file}")
    print(f"   簡訊內容檔案: {sms_file}")
    print(f"   輸出檔案: {output_file}")
    print("=" * 50)
    
    # 載入分類結果檔案
    print("📂 載入分類結果檔案...")
    result_df = load_csv_with_bom(result_file)
    print(f"   欄位: {list(result_df.columns)}")
    print(f"   前3筆 sms_id: {result_df['sms_id'].head(3).tolist()}")
    
    # 載入簡訊內容檔案
    print("\n📂 載入簡訊內容檔案...")
    sms_df = load_csv_with_bom(sms_file)
    print(f"   欄位: {list(sms_df.columns)}")
    print(f"   總簡訊數: {len(sms_df)}")
    
    # 檢查 sms_id 類型並轉換
    print("\n🔄 處理資料類型...")
    result_df['sms_id'] = result_df['sms_id'].astype(str)
    sms_df['sms_id'] = sms_df['sms_id'].astype(str)
    
    # 合併前統計
    original_sms_body_count = result_df['sms_body'].notna().sum()
    print(f"   原本已有 sms_body 的記錄: {original_sms_body_count}")
    print(f"   需要填入 sms_body 的記錄: {len(result_df) - original_sms_body_count}")
    
    # 進行 LEFT JOIN 合併
    print("\n🔗 合併資料...")
    merged_df = result_df.merge(
        sms_df[['sms_id', 'sms_body']], 
        on='sms_id', 
        how='left',
        suffixes=('', '_new')
    )
    
    # 填入空白的 sms_body 欄位
    print("📝 填入 sms_body 欄位...")
    if 'sms_body_new' in merged_df.columns:
        # 如果原本的 sms_body 是空的，就用新的填入
        mask = merged_df['sms_body'].isna() | (merged_df['sms_body'] == '')
        merged_df.loc[mask, 'sms_body'] = merged_df.loc[mask, 'sms_body_new']
        
        # 移除臨時欄位
        merged_df = merged_df.drop('sms_body_new', axis=1)
    
    # 統計結果
    final_sms_body_count = merged_df['sms_body'].notna().sum()
    filled_count = final_sms_body_count - original_sms_body_count
    missing_count = len(merged_df) - final_sms_body_count
    
    print(f"   成功填入 {filled_count} 筆 sms_body")
    print(f"   最終有 sms_body 的記錄: {final_sms_body_count}")
    print(f"   仍缺少 sms_body 的記錄: {missing_count}")
    
    if missing_count > 0:
        print(f"⚠️  缺少 sms_body 的 sms_id:")
        missing_ids = merged_df[merged_df['sms_body'].isna()]['sms_id'].head(10).tolist()
        print(f"   前10個: {missing_ids}")
    
    # 確保輸出目錄存在
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # 保存結果
    print(f"\n💾 保存合併結果...")
    merged_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"✅ 合併完成！檔案已保存至: {output_file}")
    
    # 顯示合併後的樣本
    print(f"\n📋 合併後的前3筆資料:")
    print(merged_df.head(3).to_string())

def main():
    parser = argparse.ArgumentParser(description='合併SMS分類結果與簡訊內容')
    parser.add_argument('--result', required=True, help='分類結果檔案路徑')
    parser.add_argument('--sms', required=True, help='原始簡訊檔案路徑')
    parser.add_argument('--output', required=True, help='輸出檔案路徑')
    
    args = parser.parse_args()
    
    # 檢查輸入檔案是否存在
    if not os.path.exists(args.result):
        print(f"❌ 分類結果檔案不存在: {args.result}")
        return
    
    if not os.path.exists(args.sms):
        print(f"❌ 簡訊檔案不存在: {args.sms}")
        return
    
    try:
        merge_sms_body(args.result, args.sms, args.output)
    except Exception as e:
        print(f"❌ 合併過程中發生錯誤: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    main()
