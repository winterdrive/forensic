#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批次合併SMS簡訊內容腳本
一次處理多個分類結果檔案，將它們與原始簡訊內容進行合併
"""

import os
import sys
from pathlib import Path

# 添加項目根目錄到 Python 路徑
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.pretreat.merge_sms_body import merge_sms_body

def batch_merge():
    """
    批次合併分類結果檔案與簡訊內容
    """
    # 基礎路徑設定
    base_dir = Path(__file__).parent.parent.parent
    sms_file = base_dir / "data" / "raw" / "datagame_sms_stage2.csv"
    output_dir = base_dir / "data" / "results"
    
    # 要處理的檔案清單
    files_to_process = [
        {
            "result_file": base_dir / "data" / "total_result_name+(2).csv",
            "output_file": output_dir / "merged_name_results.csv",
            "description": "姓名分類結果"
        },
        {
            "result_file": base_dir / "data" / "total_result_travel+(1).csv", 
            "output_file": output_dir / "merged_travel_results.csv",
            "description": "旅遊分類結果"
        }
    ]
    
    print("🚀 開始批次合併SMS簡訊內容")
    print("=" * 60)
    
    # 檢查簡訊檔案是否存在
    if not sms_file.exists():
        print(f"❌ 找不到簡訊檔案: {sms_file}")
        return 1
    
    success_count = 0
    total_count = len(files_to_process)
    
    for i, file_info in enumerate(files_to_process, 1):
        result_file = file_info["result_file"]
        output_file = file_info["output_file"]
        description = file_info["description"]
        
        print(f"\n📝 處理第 {i}/{total_count} 個檔案: {description}")
        print(f"   輸入: {result_file.name}")
        print(f"   輸出: {output_file.name}")
        
        # 檢查輸入檔案是否存在
        if not result_file.exists():
            print(f"❌ 找不到檔案: {result_file}")
            continue
        
        try:
            # 執行合併
            merge_sms_body(str(result_file), str(sms_file), str(output_file))
            success_count += 1
            print(f"✅ {description} 合併完成")
            
        except Exception as e:
            print(f"❌ {description} 合併失敗: {e}")
    
    print("\n" + "=" * 60)
    print(f"📊 批次處理完成: {success_count}/{total_count} 個檔案成功合併")
    
    if success_count == total_count:
        print("🎉 所有檔案都已成功合併！")
        return 0
    else:
        print("⚠️ 部分檔案合併失敗，請檢查錯誤訊息")
        return 1

if __name__ == "__main__":
    exit_code = batch_merge()
    sys.exit(exit_code)
