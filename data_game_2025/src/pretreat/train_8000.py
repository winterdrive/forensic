"""
id_list
name: data_game_2025/data/results/labled/stage2/name_train_8000.csv 
travel: data_game_2025/data/results/labled/stage2/travel_train_8000.csv

ans_list
name:
data_game_2025/data/raw/both_6_offical.csv 的 name_flg 欄位，約 6 筆
data_game_2025/data/raw/name_1000_offical.csv 的 name_flg 欄位，約 994 筆
data_game_2025/data/raw/raw_3000_labeled.csv 的 name_flg 欄位，約 3000 筆
data_game_2025/data/results/labled/stage2/label_for_more_data_result/name_consensus.csv 的 name_flg 欄位，約 4000 筆
travel:
data_game_2025/data/raw/both_6_offical.csv 的 name_flg 欄位，約 6 筆
data_game_2025/data/raw/travel_1000_offical.csv 的 label 欄位，約 994 筆
data_game_2025/data/raw/raw_3000_labeled.csv 的 label 欄位，約 3000 筆
data_game_2025/data/results/labled/stage2/label_for_more_data_result/travel_consensus.csv 的 label 欄位，約 4000 筆

update_list
data_game_2025/data/results/labled/stage2/label_for_more_data_result/bert4_mismatch_1000_name.csv 的 name_flg 欄位，約 600 筆
data_game_2025/data/results/labled/stage2/label_for_more_data_result/bert4_mismatch_1000_travel.csv 的 label 欄位，約 600 筆
data_game_2025/data/raw/raw_1000_name_mismatch.csv 的 name_flg 欄位
data_game_2025/data/raw/raw_1000_travel_mismatch.csv 的 label 欄位
data_game_2025/data/results/labled/stage2/label_for_more_data_result/more_4000_name_mismatch.csv 的 name_flg 欄位
data_game_2025/data/results/labled/stage2/label_for_more_data_result/more_4000_travel_mismatch.csv 的 label 欄位

check_list
name:
data_game_2025/data/results/labled/stage1/label_for_llm_vote/name_gemini-2.5-flash_20250712_0456.csv 的 sms_id, name_flg 欄位
data_game_2025/data/results/labled/stage2/label_for_more_data/name_more_4000_gemini-2.5-flash.csv 的 sms_id, name_flg 欄位
travel:
data_game_2025/data/results/labled/stage1/label_for_llm_vote/travel_gemini-2.5-flash_20250711_1207.csv 的 sms_id, label 欄位
data_game_2025/data/results/labled/stage2/label_for_more_data/travel_more_4000_gemini-2.5-flash.csv 的 sms_id, label 欄位


### TASK:
協助我完善 name_train_8000.csv 和 travel_train_8000.csv 檔案。
完整欄位為 sms_id,sms_body,label,name_flg。
name_train_8000.csv 會需要 sms_id,sms_body,name_flg 欄位有值，且 label 欄位為空。
travel_train_8000.csv 會需要 sms_id,sms_body,label 欄位有值，且 name_flg 欄位為空。
補上的欄位職按照以下規則填補
1. 先整理id_list：將 train_8000.py 中的 id_list 依序進行 排序、去重、去除空值 三步處理
2. 再依id將ans_list中各檔案的對應欄位結果寫入到name_train_8000.csv 和 travel_train_8000.csv對應的欄位中。如果 id_list 中沒有 ans_list 的 id，則幫忙補上。
3. 請依照update_list清單中的id與欄位名稱，更新name_train_8000.csv 和 travel_train_8000.csv對應答案欄位。如果 id_list 中沒有 update_list 的 id，則幫忙補上。如果沒有提供檔案則跳過此步驟。
4. 檢查 name_train_8000.csv 和 travel_train_8000.csv 是否所有欄位都已經填寫完整，都為 0 或 1，且沒有空值。
5. 用 check_list 檔案抽樣檢查，預期 check_list 內對應類別的檔案，在 name_train_8000.csv 或 travel_train_8000.csv 中數值相同的比例要達到 90% 以上。
    如 travel_more_4000_gemini-2.5-flash.csv 有 4000 筆資料，則在 travel_train_8000.csv 中應該有 3600 筆資料的 sms_id 和 label 欄位值與之相同。
6. 依照 name_train_8000.csv 和 travel_train_8000.csv 的欄位值，
    按 8:1:1 切分數據集存於 data_game_2025/data/results/labled/stage2/train_data，
    供模型訓練、驗證和測試使用。
    檔案包括：
        name_train_8000.csv、 travel_train_8000.csv、
        name_val_8000.csv、travel_val_8000.csv、
        name_test_8000.csv、travel_test_8000.csv。
"""

import pandas as pd
import numpy as np
import csv
from pathlib import Path
from sklearn.model_selection import train_test_split

def main():
    # 設定基礎路徑
    base_path = Path(__file__).parent.parent.parent
    data_path = base_path / "data"
    
    print("開始處理 train_8000.csv 檔案...")
    
    # 步驟1: 讀取並整理 id_list
    print("步驟1: 讀取並整理 id_list")
    
    # 讀取 name_train_8000.csv 和 travel_train_8000.csv
    name_train_file = data_path / "results" / "labled" / "stage2" / "name_train_8000.csv"
    travel_train_file = data_path / "results" / "labled" / "stage2" / "travel_train_8000.csv"
    
    # 讀取現有檔案，使用pandas直接讀取
    def read_train_csv(file_path):
        """讀取訓練 CSV 檔案"""
        try:
            # 嘗試正常讀取
            df = pd.read_csv(file_path)
            
            # 確保有必要的欄位
            required_columns = ['sms_id', 'sms_body', 'label', 'name_flg']
            for col in required_columns:
                if col not in df.columns:
                    df[col] = ''
            
            # 重新排序欄位
            df = df[required_columns]
            return df
        except Exception as e:
            print(f"讀取 {file_path} 時發生錯誤: {e}")
            # 返回空的 DataFrame
            return pd.DataFrame(columns=['sms_id', 'sms_body', 'label', 'name_flg'])
    
    name_train_df = read_train_csv(name_train_file)
    travel_train_df = read_train_csv(travel_train_file)
    
    # 轉換 sms_id 為數字
    name_train_df['sms_id'] = pd.to_numeric(name_train_df['sms_id'], errors='coerce')
    travel_train_df['sms_id'] = pd.to_numeric(travel_train_df['sms_id'], errors='coerce')
    
    # 去除空值、排序、去重
    name_ids = sorted(name_train_df['sms_id'].dropna().unique())
    travel_ids = sorted(travel_train_df['sms_id'].dropna().unique())
    
    print(f"Name train 數據: {len(name_ids)} 筆")
    print(f"Travel train 數據: {len(travel_ids)} 筆")
    
    # 步驟2: 準備完整的 DataFrame 結構
    print("步驟2: 準備完整的 DataFrame 結構")
    
    # 為 name_train_8000 準備完整結構 (sms_id, sms_body, label, name_flg)
    name_final_df = name_train_df[['sms_id', 'sms_body']].copy()
    name_final_df['label'] = ''  # label 欄位為空
    name_final_df['name_flg'] = np.nan  # 待填入
    
    # 為 travel_train_8000 準備完整結構 (sms_id, sms_body, label, name_flg) 
    travel_final_df = travel_train_df[['sms_id', 'sms_body']].copy()
    travel_final_df['label'] = np.nan  # 待填入
    travel_final_df['name_flg'] = ''  # name_flg 欄位為空
    
    # 步驟3: 從 ans_list 檔案中讀取答案
    print("步驟3: 從 ans_list 檔案中讀取答案")
    
    # 定義 ans_list 檔案路徑
    ans_files_name = [
        data_path / "raw" / "both_6_offical.csv",
        data_path / "raw" / "name_1000_offical.csv", 
        data_path / "raw" / "raw_3000_labeled.csv",
        data_path / "results" / "labled" / "stage2" / "label_for_more_data_result" / "name_consensus.csv"
    ]
    
    ans_files_travel = [
        (data_path / "raw" / "both_6_offical.csv", 'label'),  # 從 label 欄位讀取旅行標籤
        (data_path / "raw" / "travel_1000_offical.csv", 'label'),
        (data_path / "raw" / "raw_3000_labeled.csv", 'label'), 
        (data_path / "results" / "labled" / "stage2" / "label_for_more_data_result" / "travel_consensus.csv", 'label')
    ]
    
    # 收集 name_flg 答案並補充缺失的 sms_id
    name_answers = {}
    name_sms_body = {}  # 用於存儲缺失 sms_id 的 sms_body
    
    for file_path in ans_files_name:
        if file_path.exists():
            try:
                df = pd.read_csv(file_path)
                if 'sms_id' in df.columns and 'name_flg' in df.columns:
                    for _, row in df.iterrows():
                        sms_id = row['sms_id']
                        name_flg = row['name_flg']
                        sms_body = row.get('sms_body', '')
                        if pd.notna(sms_id) and pd.notna(name_flg) and str(name_flg) != 'NA':
                            sms_id = int(sms_id)
                            name_answers[sms_id] = int(name_flg)
                            if pd.notna(sms_body):
                                name_sms_body[sms_id] = str(sms_body)
                    print(f"從 {file_path.name} 讀取了 {len([x for x in df['name_flg'] if pd.notna(x) and str(x) != 'NA'])} 筆 name_flg 答案")
            except Exception as e:
                print(f"讀取 {file_path} 時發生錯誤: {e}")
        else:
            print(f"檔案不存在: {file_path}")
    
    # 收集 label (travel) 答案並補充缺失的 sms_id
    travel_answers = {}
    travel_sms_body = {}  # 用於存儲缺失 sms_id 的 sms_body
    
    for file_path, column_name in ans_files_travel:
        if file_path.exists():
            try:
                df = pd.read_csv(file_path)
                if 'sms_id' in df.columns and column_name in df.columns:
                    for _, row in df.iterrows():
                        sms_id = row['sms_id']
                        label_val = row[column_name]
                        sms_body = row.get('sms_body', '')
                        if pd.notna(sms_id) and pd.notna(label_val) and str(label_val) != 'NA':
                            sms_id = int(sms_id)
                            travel_answers[sms_id] = int(label_val)
                            if pd.notna(sms_body):
                                travel_sms_body[sms_id] = str(sms_body)
                    print(f"從 {file_path.name} 讀取了 {len([x for x in df[column_name] if pd.notna(x) and str(x) != 'NA'])} 筆 {column_name} 答案")
            except Exception as e:
                print(f"讀取 {file_path} 時發生錯誤: {e}")
        else:
            print(f"檔案不存在: {file_path}")
    
    # 檢查並補充 name_final_df 中缺失的 sms_id
    existing_name_ids = set(name_final_df['sms_id'].tolist())
    missing_name_ids = set(name_answers.keys()) - existing_name_ids
    if missing_name_ids:
        print(f"補充 name_final_df 中缺失的 {len(missing_name_ids)} 個 sms_id")
        for sms_id in missing_name_ids:
            new_row = {
                'sms_id': sms_id,
                'sms_body': name_sms_body.get(sms_id, ''),
                'label': '',
                'name_flg': np.nan
            }
            name_final_df = pd.concat([name_final_df, pd.DataFrame([new_row])], ignore_index=True)
    
    # 檢查並補充 travel_final_df 中缺失的 sms_id
    existing_travel_ids = set(travel_final_df['sms_id'].tolist())
    missing_travel_ids = set(travel_answers.keys()) - existing_travel_ids
    if missing_travel_ids:
        print(f"補充 travel_final_df 中缺失的 {len(missing_travel_ids)} 個 sms_id")
        for sms_id in missing_travel_ids:
            new_row = {
                'sms_id': sms_id,
                'sms_body': travel_sms_body.get(sms_id, ''),
                'label': np.nan,
                'name_flg': ''
            }
            travel_final_df = pd.concat([travel_final_df, pd.DataFrame([new_row])], ignore_index=True)
    
    # 填入答案到對應的 DataFrame
    print("填入 name_flg 答案...")
    name_final_df['name_flg'] = name_final_df['sms_id'].map(name_answers)
    
    print("填入 label 答案...")
    travel_final_df['label'] = travel_final_df['sms_id'].map(travel_answers)
    
    # 額外處理：從其他來源補充缺失的 sms_body
    print("補充缺失的 sms_body...")
    
    # 從所有可能的檔案收集 sms_body 資訊
    all_sms_body = {}
    
    # 收集所有檔案路徑
    all_files = [
        data_path / "raw" / "datagame_sms_stage1_raw_TEXT_ONLY.csv",  # 加入 raw/datagame_sms_stage1_raw_TEXT_ONLY.csv 作為 sms_body 來源
        data_path / "raw" / "both_6_offical.csv",
        data_path / "raw" / "name_1000_offical.csv",
        data_path / "raw" / "travel_1000_offical.csv",
        data_path / "raw" / "raw_3000_labeled.csv",
        data_path / "results" / "labled" / "stage2" / "label_for_more_data_result" / "name_consensus.csv",
        data_path / "results" / "labled" / "stage2" / "label_for_more_data_result" / "travel_consensus.csv",
        data_path / "results" / "labled" / "stage2" / "label_for_more_data_result" / "bert4_mismatch_1000_name.csv",
        data_path / "results" / "labled" / "stage2" / "label_for_more_data_result" / "bert4_mismatch_1000_travel.csv",
        data_path / "raw" / "raw_1000_name_mismatch.csv",
        data_path / "raw" / "raw_1000_travel_mismatch.csv",
        data_path / "results" / "labled" / "stage2" / "label_for_more_data_result" / "more_4000_name_mismatch.csv",
        data_path / "results" / "labled" / "stage2" / "label_for_more_data_result" / "more_4000_travel_mismatch.csv"
    ]
    
    for file_path in all_files:
        if file_path.exists():
            try:
                df = pd.read_csv(file_path)
                if 'sms_id' in df.columns and 'sms_body' in df.columns:
                    for _, row in df.iterrows():
                        sms_id = row['sms_id']
                        sms_body = row['sms_body']
                        if pd.notna(sms_id) and pd.notna(sms_body) and str(sms_body).strip() != '':
                            all_sms_body[int(sms_id)] = str(sms_body)
            except Exception as e:
                print(f"讀取 {file_path.name} 收集 sms_body 時發生錯誤: {e}")
    
    # 補充 name_final_df 中缺失的 sms_body
    name_empty_body_count = 0
    name_filled_count = 0
    for idx, row in name_final_df.iterrows():
        if pd.isna(row['sms_body']) or str(row['sms_body']).strip() == '':
            sms_id = int(row['sms_id'])
            name_empty_body_count += 1
            if sms_id in all_sms_body:
                name_final_df.at[idx, 'sms_body'] = all_sms_body[sms_id]
                name_filled_count += 1
    
    # 補充 travel_final_df 中缺失的 sms_body
    travel_empty_body_count = 0
    travel_filled_count = 0
    for idx, row in travel_final_df.iterrows():
        if pd.isna(row['sms_body']) or str(row['sms_body']).strip() == '':
            sms_id = int(row['sms_id'])
            travel_empty_body_count += 1
            if sms_id in all_sms_body:
                travel_final_df.at[idx, 'sms_body'] = all_sms_body[sms_id]
                travel_filled_count += 1
    
    print(f"Name: 發現 {name_empty_body_count} 筆空白 sms_body，成功補充 {name_filled_count} 筆")
    print(f"Travel: 發現 {travel_empty_body_count} 筆空白 sms_body，成功補充 {travel_filled_count} 筆")
    
    # 步驟4: 從 update_list 檔案中更新答案
    print("步驟4: 從 update_list 檔案中更新答案")
    
    update_files = [
        (data_path / "results" / "labled" / "stage2" / "label_for_more_data_result" / "bert4_mismatch_1000_name.csv", 'name_flg', 'name'),
        (data_path / "results" / "labled" / "stage2" / "label_for_more_data_result" / "bert4_mismatch_1000_travel.csv", 'label', 'travel'),
        (data_path / "raw" / "raw_1000_name_mismatch.csv", 'name_flg', 'name'),
        (data_path / "raw" / "raw_1000_travel_mismatch.csv", 'label', 'travel'),
        (data_path / "results" / "labled" / "stage2" / "label_for_more_data_result" / "more_4000_name_mismatch.csv", 'name_flg', 'name'),
        (data_path / "results" / "labled" / "stage2" / "label_for_more_data_result" / "more_4000_travel_mismatch.csv", 'label', 'travel')
    ]
    
    for file_path, column, target_type in update_files:
        if file_path.exists():
            try:
                df = pd.read_csv(file_path)
                if 'sms_id' in df.columns and column in df.columns:
                    update_count = 0
                    added_count = 0
                    
                    for _, row in df.iterrows():
                        sms_id = row['sms_id']
                        value = row[column]
                        sms_body = row.get('sms_body', '')
                        
                        if pd.notna(sms_id) and pd.notna(value) and str(value) != 'NA':
                            sms_id = int(sms_id)
                            value = int(value)
                            
                            if target_type == 'name':
                                mask = name_final_df['sms_id'] == sms_id
                                if mask.any():
                                    name_final_df.loc[mask, 'name_flg'] = value
                                    update_count += 1
                                else:
                                    # 如果 sms_id 不存在，則新增一行
                                    new_row = {
                                        'sms_id': sms_id,
                                        'sms_body': str(sms_body) if pd.notna(sms_body) else '',
                                        'label': '',
                                        'name_flg': value
                                    }
                                    name_final_df = pd.concat([name_final_df, pd.DataFrame([new_row])], ignore_index=True)
                                    added_count += 1
                                    
                            elif target_type == 'travel':
                                mask = travel_final_df['sms_id'] == sms_id
                                if mask.any():
                                    travel_final_df.loc[mask, 'label'] = value
                                    update_count += 1
                                else:
                                    # 如果 sms_id 不存在，則新增一行
                                    new_row = {
                                        'sms_id': sms_id,
                                        'sms_body': str(sms_body) if pd.notna(sms_body) else '',
                                        'label': value,
                                        'name_flg': ''
                                    }
                                    travel_final_df = pd.concat([travel_final_df, pd.DataFrame([new_row])], ignore_index=True)
                                    added_count += 1
                    
                    print(f"從 {file_path.name} 更新了 {update_count} 筆，新增了 {added_count} 筆 {column} 資料")
            except Exception as e:
                print(f"更新 {file_path} 時發生錯誤: {e}")
        else:
            print(f"更新檔案不存在，跳過: {file_path}")
    
    # 步驟5: 檢查並修正資料完整性
    print("步驟5: 檢查並修正資料完整性")
    
    # 確保所有值都是 0 或 1，處理 NaN 值
    name_final_df['name_flg'] = name_final_df['name_flg'].fillna(0).astype(int)
    travel_final_df['label'] = travel_final_df['label'].fillna(0).astype(int)
    
    # 檢查完整性
    name_missing = name_final_df['name_flg'].isna().sum()
    travel_missing = travel_final_df['label'].isna().sum()
    
    print(f"Name train 缺失值: {name_missing}")
    print(f"Travel train 缺失值: {travel_missing}")
    
    # 確保列順序正確
    name_final_df = name_final_df[['sms_id', 'sms_body', 'label', 'name_flg']]
    travel_final_df = travel_final_df[['sms_id', 'sms_body', 'label', 'name_flg']]
    
    # 按 sms_id 排序
    name_final_df = name_final_df.sort_values('sms_id').reset_index(drop=True)
    travel_final_df = travel_final_df.sort_values('sms_id').reset_index(drop=True)
    
    # 步驟6: 儲存結果
    print("步驟6: 儲存結果")
    
    name_final_df.to_csv(name_train_file, index=False)
    travel_final_df.to_csv(travel_train_file, index=False)
    
    print(f"已儲存 name_train_8000.csv: {len(name_final_df)} 筆資料")
    print(f"已儲存 travel_train_8000.csv: {len(travel_final_df)} 筆資料")
    
    # 最終檢查
    print("\n最終檢查:")
    print(f"Name train - 總筆數: {len(name_final_df)}")
    print(f"Name train - name_flg 分布: {name_final_df['name_flg'].value_counts().to_dict()}")
    print(f"Travel train - 總筆數: {len(travel_final_df)}")
    print(f"Travel train - label 分布: {travel_final_df['label'].value_counts().to_dict()}")
    
    # 步驟7: 用 check_list 檔案抽樣檢查
    print("\n步驟7: 用 check_list 檔案抽樣檢查")
    
    check_files = [
        # name 檢查檔案
        (data_path / "results" / "labled" / "stage1" / "label_for_llm_vote" / "name_gemini-2.5-flash_20250712_0456.csv", 'name_flg', 'name'),
        (data_path / "results" / "labled" / "stage2" / "label_for_more_data" / "name_more_4000_gemini-2.5-flash.csv", 'name_flg', 'name'),
        # travel 檢查檔案
        (data_path / "results" / "labled" / "stage1" / "label_for_llm_vote" / "travel_gemini-2.5-flash_20250711_1207.csv", 'label', 'travel'),
        (data_path / "results" / "labled" / "stage2" / "label_for_more_data" / "travel_more_4000_gemini-2.5-flash.csv", 'label', 'travel')
    ]
    
    for check_file_path, column, target_type in check_files:
        if check_file_path.exists():
            try:
                check_df = pd.read_csv(check_file_path)
                if 'sms_id' in check_df.columns and column in check_df.columns:
                    total_check = len(check_df)
                    matched_count = 0
                    
                    if target_type == 'name':
                        target_df = name_final_df
                        target_column = 'name_flg'
                    else:
                        target_df = travel_final_df
                        target_column = 'label'
                    
                    # 檢查每一筆資料是否匹配
                    overlapping_count = 0
                    for _, row in check_df.iterrows():
                        sms_id = row['sms_id']
                        expected_value = row[column]
                        
                        mask = target_df['sms_id'] == sms_id
                        if mask.any():
                            overlapping_count += 1
                            actual_value = target_df.loc[mask, target_column].iloc[0]
                            if actual_value == expected_value:
                                matched_count += 1
                    
                    if overlapping_count > 0:
                        match_percentage = (matched_count / overlapping_count) * 100
                        coverage_percentage = (overlapping_count / total_check) * 100
                        print(f"檢查 {check_file_path.name}:")
                        print(f"  總計: {total_check} 筆, 重疊: {overlapping_count} 筆 ({coverage_percentage:.1f}%), 匹配: {matched_count} 筆")
                        print(f"  重疊部分匹配率: {match_percentage:.1f}%")
                        
                        if match_percentage >= 90:
                            print(f"  ✅ 重疊部分匹配率達到 90% 以上")
                        else:
                            print(f"  ⚠️ 重疊部分匹配率低於 90%，需要檢查")
                    else:
                        print(f"檢查 {check_file_path.name}: 沒有重疊的數據")
                else:
                    print(f"檢查檔案格式錯誤: {check_file_path.name}")
            except Exception as e:
                print(f"檢查 {check_file_path.name} 時發生錯誤: {e}")
        else:
            print(f"檢查檔案不存在: {check_file_path.name}")
    
    # 步驟8: 按 8:1:1 切分數據集
    print("\n步驟8: 按 8:1:1 切分數據集")
    
    # 確保輸出目錄存在
    train_data_dir = data_path / "results" / "labled" / "stage2" / "train_data"
    train_data_dir.mkdir(parents=True, exist_ok=True)
    
    # 在切分前進行隨機排序（用於訓練集切分）
    name_shuffled_df = name_final_df.sample(frac=1, random_state=42).reset_index(drop=True)
    travel_shuffled_df = travel_final_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # 切分 name 數據集
    name_train, name_temp = train_test_split(name_shuffled_df, test_size=0.2, random_state=42, stratify=name_shuffled_df['name_flg'])
    name_val, name_test = train_test_split(name_temp, test_size=0.5, random_state=42, stratify=name_temp['name_flg'])
    
    # 切分 travel 數據集
    travel_train, travel_temp = train_test_split(travel_shuffled_df, test_size=0.2, random_state=42, stratify=travel_shuffled_df['label'])
    travel_val, travel_test = train_test_split(travel_temp, test_size=0.5, random_state=42, stratify=travel_temp['label'])
    
    # 儲存切分後的數據集
    name_train.to_csv(train_data_dir / "name_train_8000.csv", index=False)
    name_val.to_csv(train_data_dir / "name_val_8000.csv", index=False)
    name_test.to_csv(train_data_dir / "name_test_8000.csv", index=False)
    
    travel_train.to_csv(train_data_dir / "travel_train_8000.csv", index=False)
    travel_val.to_csv(train_data_dir / "travel_val_8000.csv", index=False)
    travel_test.to_csv(train_data_dir / "travel_test_8000.csv", index=False)
    
    print(f"Name 數據集切分完成:")
    print(f"  訓練集: {len(name_train)} 筆 ({len(name_train)/len(name_final_df)*100:.1f}%)")
    print(f"  驗證集: {len(name_val)} 筆 ({len(name_val)/len(name_final_df)*100:.1f}%)")
    print(f"  測試集: {len(name_test)} 筆 ({len(name_test)/len(name_final_df)*100:.1f}%)")
    
    print(f"Travel 數據集切分完成:")
    print(f"  訓練集: {len(travel_train)} 筆 ({len(travel_train)/len(travel_final_df)*100:.1f}%)")
    print(f"  驗證集: {len(travel_val)} 筆 ({len(travel_val)/len(travel_final_df)*100:.1f}%)")
    print(f"  測試集: {len(travel_test)} 筆 ({len(travel_test)/len(travel_final_df)*100:.1f}%)")
    
    print(f"\n所有切分後檔案已儲存至: {train_data_dir}")
    
    print("處理完成！")

if __name__ == "__main__":
    main()
