"""
多模型分類結果加權整合與正類排序機制

本腳本用來對使用者在指定目錄 data_game_2025/data/results/vote/candidates 下，
放置的所有檔案進行機率的加權平均計算。

根據競賽規則，分別選出旅遊分類和姓名分類機率最高的前 30,000 筆正類預測，
作為本次競賽的提交結果。

input_csv 檔案的格式為：
| 欄位 | 說明 |
|------|------|
| `sms_id` | 簡訊 ID |
| `travel_prob` | 旅遊分類預測機率 |
| `label` | 旅遊分類預測結果（0: 非旅遊, 1: 旅遊）|
| `name_prob` | 姓名分類預測機率 |
| `name_flg` | 姓名分類預測結果（0: 非姓名, 1: 姓名）|

output_csv 檔案會產生兩個：
1. category_result.csv - 旅遊分類前30,000筆正類 sms_id
2. name_result.csv - 姓名分類前30,000筆正類 sms_id
"""

import pandas as pd
import numpy as np
import configparser
from pathlib import Path
import logging
from typing import List

# 設定日誌
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class VoteEnsemble:
    """多模型投票集成類別"""

    def __init__(self, config_path: str = None):
        """
        初始化投票集成器

        Args:
            config_path: 配置檔案路徑，預設為相對路徑
        """
        if config_path is None:
            # 根據當前檔案位置計算相對路徑
            current_dir = Path(__file__).parent
            config_path = current_dir / "../../bert_model/config.ini"

        self.config = configparser.ConfigParser()
        self.config.read(config_path)

        # 從 config 讀取設定
        self.model_weights = eval(
            self.config.get("ensemble", "model_weights", fallback="[1, 1, 1]")
        )
        self.voting_strategy = self.config.get(
            "ensemble", "voting_strategy", fallback="weighted_average"
        )
        self.confidence_threshold = float(
            self.config.get("ensemble", "confidence_threshold", fallback="0.5")
        )

        # 設定路徑 - 使用絕對路徑確保準確性
        self.candidates_dir = Path(
            "/Users/winstontang/PycharmProjects/forensic/data_game_2025/data/results/vote/candidates"
        )
        self.output_dir = Path(
            "/Users/winstontang/PycharmProjects/forensic/data_game_2025/data/results/vote/output"
        )

        # 確保輸出目錄存在
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"初始化完成 - 模型權重: {self.model_weights}")
        logger.info(f"候選檔案目錄: {self.candidates_dir}")
        logger.info(f"輸出目錄: {self.output_dir}")

    def load_candidate_files(self) -> List[pd.DataFrame]:
        """載入候選目錄下的所有 CSV 檔案"""
        csv_files = list(self.candidates_dir.glob("*.csv"))

        if not csv_files:
            raise FileNotFoundError(f"在 {self.candidates_dir} 中未找到任何 CSV 檔案")

        logger.info(f"找到 {len(csv_files)} 個候選檔案")

        dataframes = []
        for file_path in csv_files:
            logger.info(f"載入檔案: {file_path.name}")
            df = pd.read_csv(file_path)

            # 檢查必要欄位
            required_columns = ["sms_id", "travel_prob", "name_prob"]
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(
                    f"檔案 {file_path.name} 缺少必要欄位: {missing_columns}"
                )

            dataframes.append(df)

        return dataframes

    def validate_consistency(self, dataframes: List[pd.DataFrame]) -> None:
        """驗證所有模型的 sms_id 是否一致"""
        if len(dataframes) < 2:
            logger.warning("只有一個模型檔案，無需驗證一致性")
            return

        reference_ids = set(dataframes[0]["sms_id"])

        for i, df in enumerate(dataframes[1:], 1):
            current_ids = set(df["sms_id"])
            if reference_ids != current_ids:
                missing_in_current = reference_ids - current_ids
                extra_in_current = current_ids - reference_ids

                error_msg = f"模型 {i+1} 的 sms_id 與第一個模型不一致\n"
                if missing_in_current:
                    error_msg += f"缺少的 ID: {list(missing_in_current)[:10]}...\n"
                if extra_in_current:
                    error_msg += f"多餘的 ID: {list(extra_in_current)[:10]}...\n"

                raise ValueError(error_msg)

        logger.info("所有模型的 sms_id 一致性驗證通過")

    def weighted_average(self, dataframes: List[pd.DataFrame]) -> pd.DataFrame:
        """計算加權平均"""
        if len(dataframes) == 1:
            logger.info("只有一個模型，直接使用該模型結果")
            return dataframes[0].copy()

        # 調整權重數量以匹配模型數量
        weights = self.model_weights[: len(dataframes)]
        if len(weights) < len(dataframes):
            weights.extend([1] * (len(dataframes) - len(weights)))

        # 正規化權重
        weights = np.array(weights)
        weights = weights / weights.sum()

        logger.info(f"使用權重: {weights}")

        # 基於第一個 dataframe 建立結果
        result_df = dataframes[0][["sms_id"]].copy()

        # 計算加權平均
        travel_probs = np.zeros(len(result_df))
        name_probs = np.zeros(len(result_df))

        for i, (df, weight) in enumerate(zip(dataframes, weights)):
            # 確保順序一致
            df_sorted = df.set_index("sms_id").loc[result_df["sms_id"]].reset_index()

            travel_probs += df_sorted["travel_prob"].values * weight
            name_probs += df_sorted["name_prob"].values * weight

        result_df["travel_prob"] = travel_probs
        result_df["name_prob"] = name_probs

        return result_df

    def calculate_labels_and_confidence(self, df: pd.DataFrame) -> pd.DataFrame:
        """計算標籤"""
        result_df = df.copy()

        # 計算標籤
        result_df["label"] = (result_df["travel_prob"] > 0.5).astype(int)
        result_df["name_flg"] = (result_df["name_prob"] > 0.5).astype(int)

        return result_df

    def select_top_positive_samples(self, df: pd.DataFrame, top_k: int = 30000) -> tuple:
        """分別根據旅遊和姓名機率選擇前 K 個正類樣本"""
        
        # 旅遊分類：先篩選正類，再按機率排序
        travel_positive = df[df["label"] == 1]
        if len(travel_positive) >= top_k:
            travel_top = travel_positive.nlargest(top_k, "travel_prob")
        else:
            # 如果正類不足，按機率排序取前 top_k
            travel_top = df.nlargest(top_k, "travel_prob")
            logger.warning(f"旅遊正類樣本不足 {top_k} 筆，改為按機率取前 {top_k} 筆")
        
        # 姓名分類：先篩選正類，再按機率排序
        name_positive = df[df["name_flg"] == 1]
        if len(name_positive) >= top_k:
            name_top = name_positive.nlargest(top_k, "name_prob")
        else:
            # 如果正類不足，按機率排序取前 top_k
            name_top = df.nlargest(top_k, "name_prob")
            logger.warning(f"姓名正類樣本不足 {top_k} 筆，改為按機率取前 {top_k} 筆")

        # 統計信息
        logger.info(f"旅遊分類：總正類 {len(travel_positive)} 筆，選擇前 {len(travel_top)} 筆")
        travel_selected_positive = (travel_top["label"] == 1).sum()
        logger.info(f"旅遊分類選中樣本中正類: {travel_selected_positive} ({travel_selected_positive/len(travel_top)*100:.1f}%)")
        
        logger.info(f"姓名分類：總正類 {len(name_positive)} 筆，選擇前 {len(name_top)} 筆")
        name_selected_positive = (name_top["name_flg"] == 1).sum()
        logger.info(f"姓名分類選中樣本中正類: {name_selected_positive} ({name_selected_positive/len(name_top)*100:.1f}%)")

        return travel_top, name_top

    def run_ensemble(self, top_k: int = 30000) -> dict:
        """執行完整的集成流程"""
        logger.info("開始執行多模型投票集成")

        # 1. 載入候選檔案
        dataframes = self.load_candidate_files()

        # 2. 驗證一致性
        self.validate_consistency(dataframes)

        # 3. 計算加權平均
        ensemble_df = self.weighted_average(dataframes)

        # 4. 計算標籤和信心度
        result_df = self.calculate_labels_and_confidence(ensemble_df)

        # 5. 分別選擇前 K 個正類樣本
        travel_top, name_top = self.select_top_positive_samples(result_df, top_k)

        # 6. 儲存結果
        output_files = {}
        
        # 6.1 旅遊分類結果（用於提交 category.csv）
        category_file = self.output_dir / "category_result.csv"
        travel_top[["sms_id"]].to_csv(category_file, index=False)
        output_files["category"] = str(category_file)
        logger.info(f"旅遊分類結果已儲存至: {category_file}")

        # 6.2 姓名分類結果（用於提交 name.csv）
        name_file = self.output_dir / "name_result.csv"
        name_top[["sms_id"]].to_csv(name_file, index=False)
        output_files["name"] = str(name_file)
        logger.info(f"姓名分類結果已儲存至: {name_file}")

        logger.info(f"旅遊分類輸出樣本數: {len(travel_top)}")
        logger.info(f"姓名分類輸出樣本數: {len(name_top)}")

        return output_files


def main():
    """主函數"""
    try:
        ensemble = VoteEnsemble()
        output_files = ensemble.run_ensemble()
        
        print(f"✅ 投票集成完成！")
        print(f"📁 旅遊分類結果: {output_files['category']}")
        print(f"📁 姓名分類結果: {output_files['name']}")

    except Exception as e:
        logger.error(f"執行失敗: {str(e)}")
        raise


if __name__ == "__main__":
    main()
