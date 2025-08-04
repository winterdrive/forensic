# 📊 2025 資料科學組競賽 - 分類器訓練方法說明

## 🎯 專案概述

本專案聚焦於構建一套混合式文本分類框架，同時處理兩個二元子任務：其一辨識簡訊是否屬於「旅遊行程與服務通知」；其二偵測內容中是否包含中文姓名。為因應這兩項需求，團隊採行「大型語言模型弱監督標註」與「BERT
精調」相結合的策略，透過層層把關，兼顧資料規模與標註品質。
專案首先以多個大型語言模型進行 zero-/few-shot
推斷，並以多模型共識及置信度閾值篩選結果，迅速為二十萬筆原始簡訊建立初始標籤。此階段雖可大幅降低人工成本，但模型彼此之間的判斷歧異仍不可避免，因此我們在下一步導入人工複核機制，只針對置信度不足或模型意見分歧的樣本進行審閱。經過數輪迭代後，團隊凝練出八千筆高一致性、涵蓋率佳的金標資料，作為後續深度學習模型的核心訓練集。
在模型端，框架採用多模型投票策略：Multilingual BERT、MacBERT 以及 CKIPLab-BERT 共同參與訓練，每個子模型透過專屬超參數優化後輸出各自的後驗機率；最終由
soft-voting 漏斗彙整，交叉比較平均置信度，選取信心最高的結果作為預測。此設計兼顧語言多樣性與領域適配，並透過模型間互補性提升召回率與精度，同時降低單一模型過擬合風險。

---

## 🏗️ 整體架構

### 📊 數據流程圖

```text
原始數據 (200,000+ SMS)
     ↓
[人工標注 3000 筆 ＋ LLM 標註 4000 筆 階段] - 使用人工與多個LLM進行初步分類
     ↓
[人工校驗] - 處理模型間歧異，確保標註品質
     ↓
[數據整合] - 整合標注資料與官方提供 ground truth 為 8,000 筆高品質訓練數據
     ↓
[BERT 訓練] - 訓練多個BERT變體模型
     ↓
[模型融合] - 投票決策與歧異檢測
     ↓
選出最佳的模型組合用於正式資料集
```

---

## 📝 資料標註流程

### 第一階段：LLM 多模型標註

我們選用了表現優異的四個 LLM 模型進行初步標註：

#### 🤖 使用的 LLM 模型

- **Gemini 2.5 Flash** (Google)
- **Llama-3.3-70B-Instruct** (Meta)
- **DeepSeek-R1-Distill-Llama-70B** (DeepSeek)
- **Magistral-Small-2506** (Mistral AI)

#### 📋 標註策略

1. **結構化提示詞設計**：
    - 姓名分類：明確定義中文姓名的識別規則，包含標準姓名、暱稱小名、稱謂用法等
    - 旅遊分類：聚焦於旅遊行程通知、旅行社發送、相關關鍵字等判斷依據
    - 提示詞詳見：https://github.com/winterdrive/forensic/tree/main/data_game_2025/prompt


2. **XML 格式化輸入**：
    - 使用 XML 結構化輸入簡訊內容，提升 LLM 解析準確度
    - 實作 XML 轉義處理，避免特殊字元影響解析

3. **批次處理優化**：
    - 每次處理 20-50 則簡訊，平衡效率與準確度
    - 實作 token 數量估算，控制 API 呼叫成本

### 第二階段：人工校驗與品質控制

#### 🔍 歧異檢測機制

- **四模型投票制**：當任一模型預測結果與其他不同時，標記為需人工檢查，並依人工確認結果淘汰錯誤率較高的 *
  *Llama-3.3-70B-Instruct** (Meta)
- **共識決策**：只有當模型間達成共識的預測才自動採納
- **分歧處理**：所有分歧案例都經過人工逐一檢視和標註

#### 📊 品質檢查流程

1. **mismatch 識別**：使用 `majority_vote.py` 腳本自動識別模型間分歧
2. **Google Sheets 協作**：建立線上表單供團隊成員協作標註
3. **交叉驗證**：重要案例由多人獨立標註後討論確認

### 第三階段：數據集構建

#### 📚 訓練數據來源

最終的訓練集 (`train_8000.csv`) 整合了以下數據源：

1. **官方提供數據** (1,000 筆)：
    - `both_6_offical.csv` - 6 筆
    - `name_1000_offical.csv` - 994 筆姓名標註
    - `travel_1000_offical.csv` - 994 筆旅遊標註

2. **第一輪人工標註** (3,000 筆)：
    - `raw_3000_labeled.csv` - 經過四個 LLM 投票和人工校驗

3. **擴充標註數據** (4,000 筆)：
    - `name_consensus.csv` / `travel_consensus.csv` - 由最佳的三個 LLM 額外擴充的高品質標註
    - 前三可信之 LLM 模型：
        - **Gemini 2.5 Flash** (Google)
        - **DeepSeek-R1-Distill-Llama-70B** (DeepSeek)
        - **Magistral-Small-2506** (Mistral AI)

#### 🔧 數據預處理 (`train_8000.py`)

1. **ID 標準化**：排序、依 sms_body 去重、去除空值
2. **標籤整合**：合併來自不同來源的標註結果
3. **品質驗證**：確保所有標籤為 0/1 二元值，無缺失值
4. **數據分割**：按 8:1:1 比例分割為訓練/驗證/測試集

---

## 🤖 BERT 模型訓練

### 模型選擇策略

我們訓練了多個 BERT 變體，針對中文文本特性進行優化：

#### 🏆 主要模型架構

| 模型                                   | 特點與設定                        |
|--------------------------------------|------------------------------|
| `distilbert-base-multilingual-cased` | 中英混合、快速推論、lr=2e-5、bs=16      |
| `hfl/chinese-macbert-large`          | 中文語義強、lr=1e-5、bs=8（大模型記憶體需求） |
| `ckiplab/bert-base-chinese`          | 中研院中文 BERT、效能平衡              |
| `hfl/chinese-macbert-base`           | MacBERT 基礎版、適合快速迭代           |

### 🛠️ 訓練配置

以 **distilbert-base-multilingual-cased** 與 **hfl/chinese-macbert-large** 為例，以下是訓練配置：

#### 訓練參數

```ini
max_length = 512          # 最大序列長度
num_epochs = 5            # 訓練輪數
warmup_steps = 500        # 預熱步數
weight_decay = 0.01       # 權重衰減
early_stopping_patience = 3  # 早停機制
random_seed = 42          # 隨機種子
```

#### 分別訓練策略

- **姓名分類器** (`bert_name_trainer.py`)：專注於識別中文姓名模式
- **旅遊分類器** (`bert_travel_trainer.py`)：專注於旅遊相關語義理解

### 📊 訓練監控

#### Weights & Biases 整合

- 實時監控訓練 loss 和驗證準確率
- 自動保存最佳模型權重
- 訓練過程可視化和分析

#### 評估指標

- **準確率 (Accuracy)**
- **精確率 (Precision)**

---

## 🔄 推論實驗

### 多模型推論

#### BERT 推論 (`bert_*_inference.py`)，用於完整測試資料及與正式資料集

- GPU 加速推論
- 批次處理優化
- 機率輸出與閾值調整
- 模型權重自動載入

---

## 📈 實驗結果

### 模型表現對比

#### 訓練數據迭代

- **train_8000**：初版訓練集，建立基準表現
- **train_9000**：加入 mismatch 重新標註數據，提升模型穩健性

#### 模型評估結果

- **姓名分類**：目標準確率 >98%，如未達標考慮 NER 模型
- **旅遊分類**：通過多輪測試和人工驗證優化

### 最佳實驗組別

1. 姓名分類於測試資料集中，透過 train_9000.csv 訓練三模型
   ckiplab/bert-base-chinese、distilbert-base-multilingual-cased、hfl/chinese-macbert-base 在第一階段達到 100% 的準確率。
2. 旅遊分類於測試資料集中，透過 train_8000.csv 訓練三模型
   ckiplab/bert-base-chinese、distilbert-base-multilingual-cased、hfl/chinese-macbert-base 在第一階段達到 >99.7% 的準確率。

---

## 🛠️ 技術實現細節

### 核心模組架構

#### 數據處理模組 (`src/pretreat/`)

- `train_8000.py`：數據集構建與標準化
- `majority_vote.py`：多模型投票與共識決策
- `llm_compare.py`：LLM 模型效能比較

#### 模型訓練模組 (`src/bert_model/`)

- `bert_*_trainer.py`：BERT 模型訓練腳本
- `bert_*_inference.py`：推論執行腳本
- `config*.ini`：模型配置檔案

#### LLM 分類模組 (`src/`)

- `llm_name_classifier.py`：LLM 姓名分類器
- `llm_travel_classifier.py`：LLM 旅遊分類器
- `utils.py`：共用工具函數

### 🔧 工程化特性

1. **模組化設計**：各功能獨立封裝，便於維護和測試
2. **配置驅動**：使用 INI 檔案管理模型參數和路徑
3. **錯誤處理**：完整的異常處理和重試機制
4. **日誌記錄**：詳細的訓練和推論過程記錄
5. **版本控制**：模型權重和結果檔案自動版本標記

---

## 🚀 部署與使用

### 環境需求

```bash
# 核心依賴
torch >= 1.9.0
transformers >= 4.20.0
pandas >= 1.3.0
scikit-learn >= 1.0.0

# 可選依賴
wandb  # 訓練監控
openai  # LLM API 調用
requests  # HTTP 請求處理
```

### 快速開始

1. **環境設置**：

   ```bash
   pip install -r requirements_bert.txt
   ```

2. **模型訓練**：

   ```bash
   # 訓練姓名分類器
   python src/bert_model/bert_name_trainer.py
   
   # 訓練旅遊分類器
   python src/bert_model/bert_travel_trainer.py
   ```

3. **模型推論**：

   ```bash
   # BERT 推論
   python src/bert_model/run_inference_simple.py
   
   # LLM 推論
   python src/llm_name_classifier.py
   ```

4. **結果融合**：

   ```bash
   python src/pretreat/majority_vote.py --input_dir results/
   ```

---

## 🎯 競賽策略總結

1. **資料集建立**：使用官方提供的 ground truth 數據作為基礎，結合 LLM 標註和人工校驗建立高品質訓練集
2. **資料品質控制**：嚴格的人工校驗和歧異處理機制
3. **模型迭代優化**：基於驗證結果持續改進模型和數據
4. **多模型融合**：結合多種 BERT 模型進行特徵學習

---

## 正賽

### 🗳️ 信心分數加總機制 (在 [Colab](https://colab.research.google.com/drive/14LA2AULo9yjiDkcsrYvjTYCSPNVNDyq3?usp=sharing) 上進行)

#### 決策方式 

1. **多模型預測整合**：收集所有模型的預測結果並依照模型信心分數加總進行排序
2. **歧異檢測**：標記模型間預測不一致的案例後進行人工審查
3. **數據清理**：刪除所有含有 "contentReference" 字樣之 placeholder 所對應的資料
4. **最終決策**：結合模型共識和數據清理的最終結果

#### 上傳策略

- **第一次**：將先推論好的兩個模型進行決策後上傳 (`ckiplab/bert-base-chinese` & `hfl/chinese-macbert-base` )
- **第二次**：將最後一個推論好的模型加入決策後上傳 (`distilbert-base-multilingual-cased`)
