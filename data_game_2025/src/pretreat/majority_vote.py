import pandas as pd
import argparse
from typing import List
import os


def consensus_and_disagreement(
    pred_dfs: List[pd.DataFrame], model_names: List[str], sms_body_path: str = None
) -> pd.DataFrame:
    # 取得所有預測的 sms_id 交集
    common_ids = set(pred_dfs[0]["sms_id"])
    for df in pred_dfs[1:]:
        common_ids &= set(df["sms_id"])
    # 只保留交集 id
    pred_dfs = [df[df["sms_id"].isin(common_ids)].copy() for df in pred_dfs]
    # 自動偵測比較欄位
    possible_cols = ["name_flg", "label"]
    col = None
    for c in possible_cols:
        if c in pred_dfs[0].columns:
            col = c
            break
    if not col:
        # 若都沒有，則選第一個非 sms_id 欄位
        col = [c for c in pred_dfs[0].columns if c != "sms_id"][0]
    # 合併，欄位名稱用模型名
    result_df = pred_dfs[0][["sms_id", col]].rename(columns={col: model_names[0]})
    for i, df in enumerate(pred_dfs[1:]):
        result_df = result_df.merge(
            df[["sms_id", col]].rename(columns={col: model_names[i + 1]}),
            on="sms_id",
            how="left",
        )
    # 判斷是否一致
    result_df["consensus"] = result_df.apply(
        lambda row: len(set([row[model_names[i]] for i in range(len(pred_dfs))])) == 1,
        axis=1,
    )
    consensus_df = result_df[result_df["consensus"]].copy()
    mismatch_df = result_df[~result_df["consensus"]].copy()
    print(f"共識決（全數一致）: {len(consensus_df)} 筆")
    print(f"不一致(mismatch): {len(mismatch_df)} 筆")
    # 若有提供 sms body csv，補上內容
    if sms_body_path:
        try:
            sms_df = pd.read_csv(sms_body_path, dtype={"sms_id": int}, encoding="cp950")
        except UnicodeDecodeError:
            sms_df = pd.read_csv(sms_body_path, dtype={"sms_id": int}, encoding="utf-8")
        sms_body_map = sms_df.set_index("sms_id")["sms_body"]
        consensus_df["sms_body"] = consensus_df["sms_id"].map(sms_body_map)
        mismatch_df["sms_body"] = mismatch_df["sms_id"].map(sms_body_map)
    return consensus_df, mismatch_df


def main():
    import os

    parser = argparse.ArgumentParser(
        description="Consensus and mismatch among multiple prediction CSVs."
    )
    parser.add_argument(
        "--input_dir", required=True, help="Directory containing prediction CSV files"
    )
    parser.add_argument(
        "--sms_body", help="Optional: Original SMS data CSV (cp950) to attach sms_body"
    )
    args = parser.parse_args()

    # 自動尋找 name/travel 檔案
    files = [
        os.path.join(args.input_dir, f)
        for f in os.listdir(args.input_dir)
        if f.endswith(".csv")
    ]
    name_files = [f for f in files if os.path.basename(f).startswith("name")]
    travel_files = [f for f in files if os.path.basename(f).startswith("travel")]

    import re
    def get_model_name(path):
        fname = os.path.basename(path)
        m = re.search(r'(deepseek_chat|gemini-2\.5-flash|mistralai_magistral-small)', fname)
        return m.group(1) if m else f"pred_{fname}"

    # name 共識決
    if name_files:
        print(f"🔎 Found name files: {name_files}")
        name_dfs = [pd.read_csv(f) for f in name_files]
        name_model_names = [get_model_name(f) for f in name_files]
        consensus_df, mismatch_df = consensus_and_disagreement(name_dfs, name_model_names, args.sms_body)
        # 合併 name_flg 與 label 欄位
        out_cols = ["sms_id", "sms_body", "label", "name_flg"]
        if "sms_body" not in consensus_df.columns and args.sms_body:
            try:
                sms_df = pd.read_csv(
                    args.sms_body, dtype={"sms_id": int}, encoding="cp950"
                )
            except UnicodeDecodeError:
                sms_df = pd.read_csv(
                    args.sms_body, dtype={"sms_id": int}, encoding="utf-8"
                )
            sms_body_map = sms_df.set_index("sms_id")["sms_body"]
            consensus_df["sms_body"] = consensus_df["sms_id"].map(sms_body_map)
        for colname in ["label", "name_flg"]:
            if colname not in consensus_df.columns:
                for df in name_dfs:
                    if colname in df.columns:
                        consensus_df[colname] = consensus_df["sms_id"].map(
                            df.set_index("sms_id")[colname]
                        )
                        break
        consensus_out = consensus_df[[c for c in out_cols if c in consensus_df.columns]]
        consensus_out.to_csv("name_consensus.csv", index=False)
        print(f"💾 Consensus saved to: name_consensus.csv")
        mismatch_df.to_csv("name_mismatch.csv", index=False)
        print(f"💾 Mismatch saved to: name_mismatch.csv")

    # travel 共識決
    if travel_files:
        print(f"🔎 Found travel files: {travel_files}")
        travel_dfs = [pd.read_csv(f) for f in travel_files]
        travel_model_names = [get_model_name(f) for f in travel_files]
        consensus_df, mismatch_df = consensus_and_disagreement(travel_dfs, travel_model_names, args.sms_body)
        out_cols = ["sms_id", "sms_body", "label", "name_flg"]
        if "sms_body" not in consensus_df.columns and args.sms_body:
            try:
                sms_df = pd.read_csv(
                    args.sms_body, dtype={"sms_id": int}, encoding="cp950"
                )
            except UnicodeDecodeError:
                sms_df = pd.read_csv(
                    args.sms_body, dtype={"sms_id": int}, encoding="utf-8"
                )
            sms_body_map = sms_df.set_index("sms_id")["sms_body"]
            consensus_df["sms_body"] = consensus_df["sms_id"].map(sms_body_map)
        for colname in ["label", "name_flg"]:
            if colname not in consensus_df.columns:
                for df in travel_dfs:
                    if colname in df.columns:
                        consensus_df[colname] = consensus_df["sms_id"].map(
                            df.set_index("sms_id")[colname]
                        )
                        break
        consensus_out = consensus_df[[c for c in out_cols if c in consensus_df.columns]]
        consensus_out.to_csv("travel_consensus.csv", index=False)
        print(f"💾 Consensus saved to: travel_consensus.csv")
        mismatch_df.to_csv("travel_mismatch.csv", index=False)
        print(f"💾 Mismatch saved to: travel_mismatch.csv")


if __name__ == "__main__":
    main()
