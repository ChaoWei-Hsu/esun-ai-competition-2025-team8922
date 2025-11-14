# AI CUP 2025 玉山人工智慧公開挑戰賽 - Team 8922

本專案使用圖神經網路 (GraphSAGE) 處理玉山銀行競賽的交易數據，以偵測可疑帳戶。

專案透過 `main.py` 協調執行三個主要階段：數據預處理、模型訓練和最終預測。

## 專案結構

├ data \
│ └ (請將 CSV 檔案放於此處) \
├ Preprocess \
│ └ Preprocess.py \
├ Model \
│ └ Model.py \
├ Prediction \
│ └ Prediction.py \
├ main.py <- (主執行檔) \
├ requirements.txt <- (專案依賴) \
└ README.md <- (您正在閱讀此檔案)

## 安裝與設定

### 步驟 1: 安裝依賴
建議在虛擬環境中安裝：
```bash
pip install -r requirements.txt
```

### 步驟 2: 下載數據
由於競賽數據無法上傳至 Git，您必須手動下載。
請至[ 玉山競賽的官方網站連結 ](https://tbrain.trendmicro.com.tw/Competitions/Details/40)下載，並將 acct_transaction.csv, acct_alert.csv, acct_predict.csv 放入 data 資料夾。

## 如何執行
本管線由 main.py 統一調度。

1. 執行完整管線
(包含預處理、訓練、預測)

```bash
python main.py
```

2. (可選) 跳過已完成的步驟   
如果您已經有 processed_features_with_interactions.pt 或 best_model.pt，可以使用參數跳過特定階段：
  - 僅執行預測 (假設已完成預處理和訓練):
  ```bash
  python main.py --skip-preprocess --skip-train
  ```

  - 僅執行訓練和預測 (假設已完成預處理):
  ```bash
  python main.py --skip-preprocess
  ```

## 輸出檔案
執行 main.py 後，以下檔案將會生成在專案根目錄：

1. processed_features_with_interactions.pt: (由 Preprocess 產出) 包含特徵張量和帳戶索引。
2. best_model.pt: (由 Model 產出) 儲存了訓練好的 GNN 模型權重。
3. submission.csv: (由 Prediction 產出) 最終的預測提交檔案。
