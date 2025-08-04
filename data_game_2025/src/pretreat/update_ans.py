"""
根據 update_list 檔案，更新 name_train_8000.csv 和 travel_train_8000.csv 的答案欄位。
可指定 input/output 路徑。
用法範例：
python update_ans.py --name-input <name_train_8000.csv> --travel-input <travel_train_8000.csv> --name-output <name_train_8000.csv> --travel-output <travel_train_8000.csv>
"""

import pandas as pd
import argparse
from pathlib import Path


def update_train_files(
    name_input, travel_input, name_output, travel_output, data_path=None
):
    # 讀取原始檔案
    name_df = pd.read_csv(name_input)
    travel_df = pd.read_csv(travel_input)

    # 轉換 sms_id 為數字
    name_df["sms_id"] = pd.to_numeric(name_df["sms_id"], errors="coerce")
    travel_df["sms_id"] = pd.to_numeric(travel_df["sms_id"], errors="coerce")

    # 定義 update_list 檔案
    if data_path is None:
        data_path = Path(name_input).parent.parent.parent.parent / "data"
    else:
        data_path = Path(data_path)
    update_files = [
        (data_path / "raw" / "raw_1000_name_mismatch.csv", "name_flg", "name"),
        (data_path / "raw" / "raw_1000_travel_mismatch.csv", "label", "travel"),
        (
            data_path
            / "results"
            / "labled"
            / "stage2"
            / "label_for_more_data_result"
            / "more_4000_name_mismatch.csv",
            "name_flg",
            "name",
        ),
        (
            data_path
            / "results"
            / "labled"
            / "stage2"
            / "label_for_more_data_result"
            / "more_4000_travel_mismatch.csv",
            "label",
            "travel",
        ),
    ]

    for file_path, column, target_type in update_files:
        if file_path.exists():
            try:
                df = pd.read_csv(file_path)
                if "sms_id" in df.columns and column in df.columns:
                    update_count = 0
                    for _, row in df.iterrows():
                        sms_id = row["sms_id"]
                        value = row[column]
                        if pd.notna(sms_id) and pd.notna(value) and str(value) != "NA":
                            sms_id = int(sms_id)
                            value = int(value)
                            if target_type == "name":
                                mask = name_df["sms_id"] == sms_id
                                if mask.any():
                                    name_df.loc[mask, "name_flg"] = value
                                    update_count += 1
                            elif target_type == "travel":
                                mask = travel_df["sms_id"] == sms_id
                                if mask.any():
                                    travel_df.loc[mask, "label"] = value
                                    update_count += 1
                    print(f"從 {file_path.name} 更新了 {update_count} 筆 {column} 資料")
            except Exception as e:
                print(f"更新 {file_path} 時發生錯誤: {e}")
        else:
            print(f"更新檔案不存在，跳過: {file_path}")

    # 儲存結果
    name_df.to_csv(name_output, index=False)
    travel_df.to_csv(travel_output, index=False)
    print(f"已更新並儲存: {name_output}, {travel_output}")


def main():
    parser = argparse.ArgumentParser(
        description="根據 update_list 檔案，更新任意 name/travel csv 檔案"
    )
    parser.add_argument("--name-input", required=True, help="name 任意 CSV 輸入路徑")
    parser.add_argument(
        "--travel-input", required=True, help="travel 任意 CSV 輸入路徑"
    )
    parser.add_argument("--name-output", required=True, help="name 任意 CSV 輸出路徑")
    parser.add_argument(
        "--travel-output", required=True, help="travel 任意 CSV 輸出路徑"
    )
    parser.add_argument(
        "--data-path", default=None, help="data 目錄路徑（預設自動推斷）"
    )
    args = parser.parse_args()
    update_train_files(
        args.name_input,
        args.travel_input,
        args.name_output,
        args.travel_output,
        args.data_path,
    )


if __name__ == "__main__":
    main()
