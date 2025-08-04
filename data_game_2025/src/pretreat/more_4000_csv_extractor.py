"""
source:
data_game_2025/data/results/labled/stage2/magistral_more_10000/name_mistralai_magistral-small_20250721_1855.csv
data_game_2025/data/results/labled/stage2/magistral_more_10000/travel_mistralai_magistral-small_20250710_2231.csv

destination path:
data_game_2025/data/results/labled/stage2/label_for_more_data

id list:
data_game_2025/data/raw/more_4000.csv


幫我依照 id list
從 source path 中提取出 id list 中的 id 與其他欄位，並存至 destination path 中。
name_mistralai_magistral-small_20250721_1855.csv 命名為 name_mistralai_magistral-small_20250721_1855_more_4000.csv
travel_mistralai_magistral-small_20250710_2231.csv 命名 as travel_mistralai_magistral-small_20250710_2231_more_4000.csv
"""

import os
import pandas as pd
import argparse

def filter_and_save(src_path, dst_path, id_set):
    df = pd.read_csv(src_path)
    possible_id_cols = ["id", "ID", "Id"] + list(df.columns)
    for col in possible_id_cols:
        if col in df.columns:
            df["_id_col"] = df[col].astype(str)
            filtered = df[df["_id_col"].isin(id_set)].drop(columns=["_id_col"])
            filtered.to_csv(dst_path, index=False)
            print(f"Saved {len(filtered)} rows to {dst_path}")
            return
    raise ValueError(f"No id column found in {src_path}")

def main():
    parser = argparse.ArgumentParser(description="Extract rows by id from source CSVs and save to destination.")
    parser.add_argument('--id_list', type=str, default="data_game_2025/data/raw/more_4000.csv", help='Path to id list CSV')
    parser.add_argument('--src_name', type=str, default="data_game_2025/data/results/labled/stage2/magistral_more_10000/name_mistralai_magistral-small_20250721_1855.csv", help='Source name CSV')
    parser.add_argument('--src_travel', type=str, default="data_game_2025/data/results/labled/stage2/magistral_more_10000/travel_mistralai_magistral-small_20250710_2231.csv", help='Source travel CSV')
    parser.add_argument('--dst_dir', type=str, default="data_game_2025/data/results/labled/stage2/label_for_more_data", help='Destination directory')
    args = parser.parse_args()

    dst_name = os.path.join(args.dst_dir, "name_mistralai_magistral-small_20250721_1855_more_4000.csv")
    dst_travel = os.path.join(args.dst_dir, "travel_mistralai_magistral-small_20250710_2231_more_4000.csv")

    os.makedirs(args.dst_dir, exist_ok=True)

    id_df = pd.read_csv(args.id_list)
    id_col = id_df.columns[0]
    id_set = set(id_df[id_col].astype(str))

    filter_and_save(args.src_name, dst_name, id_set)
    filter_and_save(args.src_travel, dst_travel, id_set)

if __name__ == "__main__":
    main()
