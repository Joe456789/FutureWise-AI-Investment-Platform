# FutureWise AI Investment Platform｜未來智策AI投資平台

**AI 驅動台股量化分析系統**

> 畢業專題 | 世新大學資管系 | 2025–2026

---

## 專案簡介

以「從數據到可驗證策略」為核心，整合台股歷史數據爬取、機器學習預測模型與 FastAPI 雲端部署，打造一個涵蓋看盤、選股、AI 深度分析、社群討論、投資教學的端到端台股量化分析系統，並包裝成 iOS App（WKWebView）與網頁版雙形式提供服務。

**我的負責範圍（與組員合作，負責資料管線、預測模型與後端 API）**
- 台股歷史數據自動化抓取與清洗 Pipeline（上市＋上櫃，約 2,300+ 檔）
- XGBoost 次日漲跌方向預測模型 ＋ AutoGluon 7 天走勢預測模型
- FastAPI RESTful API 設計與 Oracle Cloud 雲端部署

> 前端網頁與 iOS App 由組員負責。

---

## 技術棧

| 類別 | 技術 |
|------|------|
| 語言 | Python |
| 模型 | XGBoost、AutoGluon TimeSeries |
| 後端 | FastAPI + Uvicorn（多 worker） |
| 資料庫 | MySQL |
| 前端 | 原生 HTML / CSS / JavaScript（無框架）、Chart.js |
| App 封裝 | SwiftUI + WKWebView |
| 部署 | Oracle Cloud VM、pm2、zrok（穩定保留網址） |
| 排程 | crontab（平日盤後爬蟲、週六模型重訓） |

---

## 功能總覽

- **看盤**：大盤加權指數／櫃買指數走勢、個股K線（可縮放、Y軸自動校正）、三大法人籌碼圖、個股深度財報
- **選股**：9 種技術型態策略選擇器、AI精選（全市場 XGBoost 排名／我的關注股評分）、自選股清單（可拖曳排序）
- **AI分析**：XGBoost 次日漲跌信心指數、AutoGluon 7天走勢預測（含信賴區間）、模型特徵重要性、預測命中率回測橫幅、AI資金流向提示（三大法人籌碼解讀）、重大訊息公告（證交所OpenAPI）、輿情分析（Gemini）、ATR 停損停利建議
- **社群**：發文／回覆／按讚、檢舉與封鎖、我的貼文管理（編輯／刪除、回覆通知彙整）
- **教學**：73 篇投資知識文章（基本面／籌碼面／技術分析／總體經濟／交易心法），個人化閱讀進度追蹤
- **帳號系統**：JWT 登入驗證、個人資料、通知設定、自選股同步

---

## 模型效能（XGBoost 次日漲跌方向預測）

| 指標 | 數值 |
|------|------|
| 預測目標 | 次日收盤漲跌方向（二元分類） |
| 測試集 | 最後 200,000 筆（時序切分，模型未見過；約85個交易日） |
| **方向預測準確率** | **56.18%** |
| 測試集中「隔日上漲」比例 | 43.76% |
| **基準線（多數類別：永遠猜「跌」）** | **56.24%** |
| 超越多數類別基準線 | −0.06 個百分點 |
| 精確率（預測漲、實際也漲的比例） | 49.80% |
| 召回率（真的上漲的股票，抓到幾成） | 16.46% |
| 看漲信心度 ≥65% 時的精確率 | **60.60%** |
| 驗證方式 | 時序切分（依日期排序，最後一段為測試集；訓練／驗證不打亂，避免未來資料洩漏），詳見 `Python/evaluate_model_db.py` |

> 標籤定義：次日收盤價高於今日為「漲」，其餘（下跌或持平）皆為「跌」。
>
> **準確率跟基準線幾乎相同（56.18% vs 56.24%），代表準確率本身無法證明模型有預測能力**——這段期間本來就有超過五成的股票下跌，「永遠猜跌」也能拿到差不多的分數。真正有意義的訊號在**精確率隨信心度提高而上升**：信心度 50~55% 時精確率 48.0%，55~60% 為 53.8%，60~65% 為 57.7%，≥65% 為 60.6%，且每一級距都高於測試集整體的上漲比例（43.76%），顯示模型的排序能力（信心度越高、越可能真的上漲）優於其單點分類準確率。這也是系統採用「依信心度排序」而非單純二元判斷來呈現 AI 精選結果的原因。
>
> 先前版本用來評估準確率的腳本（`evaluate_model.py`）讀取的是本機一份較舊的 CSV 快照、且標籤計算有 bug（每檔股票最後一個交易日被誤標成「跌」），量出的 66.76% 偏高且不可靠；上表數字改用 `evaluate_model_db.py` 直接對資料庫做時序切分驗證，並修正了標籤 bug。

**特徵工程**：28 個特徵，涵蓋技術面（RSI、MACD、KD、ATR、乖離率、跳空缺口）、籌碼面（外資／投信／自營商買賣超）、總經連動（大盤漲跌幅、費半漲跌、台幣匯率）、基本面（EPS、本益比、股價淨值比、營收YoY）。

特徵重要性 Top 3（目前部署中的模型）：量能比（12.07%）、D值（9.87%）、漲跌幅_1日（7.24%）。

---

## 系統架構

```
證交所 OpenAPI / Yahoo Finance
        ↓
資料爬蟲（cloud_crawler_TWSE.py，crontab 平日盤後執行）
        ↓
        MySQL（StockPrice 等資料表）
        ↓
每週六重訓：Universal Trainer.py（XGBoost）／train_autogluon_7days.py（AutoGluon）
        ↓
FastAPI 後端（stock_api_server.py，多 worker）
        ↓
   ┌────────┴────────┐
MyWebApp（iOS App／WKWebView）   網頁版（同一份後端）
```

---

## 資料夾結構

```
Python/       後端：API 伺服器、爬蟲、模型訓練、SQL 遷移檔
MyWebApp/     前端：13 個頁面（看盤／選股／AI分析／社群／教學／帳號相關）
```

> 註：本 repo 不包含訓練好的模型二進位檔（`model_universal.json`、`autogluon_7days_model/`），可透過 `Universal Trainer.py` / `train_autogluon_7days.py` 重新訓練產生。

---

## 快速開始

```bash
# 1. 安裝依賴
pip install -r Python/requirements.txt

# 2. 建立環境變數檔（複製範本後填入自己的資料庫密碼、Gemini金鑰、JWT密鑰）
cp Python/.env.example Python/.env

# 3. 依序執行 Python/ 底下的 migrate_*.sql / SQLQuery*.sql 到你的 MySQL

# 4. 執行數據爬取
python Python/cloud_crawler_TWSE.py

# 5. 訓練模型
python "Python/Universal Trainer.py"
python Python/train_autogluon_7days.py

# 6. 啟動 API 伺服器
python Python/stock_api_server.py
```

前端 `MyWebApp/` 內的 HTML 檔案透過 `API_BASE_URL` 常數指向後端網址，本機測試可用 `python -m http.server` 於該資料夾下啟動靜態伺服器。

---

## 相關連結

- [AWS Certified AI Practitioner 認證](https://www.credly.com/badges/7975a528-0eba-4f45-8b2d-1656872941de/public_url)
