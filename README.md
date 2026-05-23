# ProQuant｜AI 驅動台股量化分析系統

> 畢業專題 | 世新大學資管系 | 2025–2026

---

## 專案簡介

以「從數據到可驗證策略」為核心，整合台股歷史數據爬取、XGBoost 趨勢預測模型與 FastAPI 雲端部署，實現端到端的台股量化分析系統。

**本人負責範圍（後端核心開發者）**
- 台股歷史數據自動化抓取與清洗 Pipeline
- XGBoost 趨勢預測模型訓練與超參數調優
- FastAPI RESTful API 設計與 Render 雲端部署

---

## 技術棧

| 類別 | 技術 |
|------|------|
| 語言 | Python |
| 模型 | XGBoost |
| 後端 | FastAPI |
| 資料庫 | MySQL |
| 部署 | Render |

---

## 模型效能

| 指標 | 數值 |
|------|------|
| 預測目標 | 次日收盤漲跌方向（二元分類） |
| 測試集 | 最後 13,958 筆（時序切分，模型未見過） |
| **方向預測準確率** | **66.76%** |
| 基準線（永遠猜漲） | 41.90% |
| **超越基準線** | **+24.86 個百分點** |
| 驗證方式 | TimeSeriesSplit（防止未來資料洩漏） |

---

## 特徵工程

使用 28 個特徵，涵蓋四大維度：

- **技術面**：RSI、MACD、KD、ATR、乖離率、跳空缺口
- **籌碼面**：外資買賣超、投信買賣超、自營商買賣超
- **總經連動**：大盤漲跌幅、費半漲跌、台幣匯率
- **基本面**：EPS、本益比、股價淨值比、營收 YoY

**特徵重要性 Top 3：**
1. 台幣匯率（14.14%）
2. 大盤漲跌幅（10.59%）
3. 費半漲跌（8.34%）

---

## 系統架構

```
資料爬蟲（cloud_crawler_TWSE.py）
    ↓
特徵工程 + 標註（Universal Trainer.py）
    ↓
XGBoost 模型訓練 → model_universal.json
    ↓
FastAPI 後端（stock_api_server.py）
    ↓
Render 雲端部署
```

---

## 快速開始

```bash
# 安裝依賴
pip install -r requirements.txt

# 執行數據爬取
python cloud_crawler_TWSE.py

# 訓練模型
python "Universal Trainer.py"

# 啟動 API 伺服器
python stock_api_server.py
```

---

## 相關連結

- [AWS Certified AI Practitioner 認證](https://www.credly.com/badges/7975a528-0eba-4f45-8b2d-1656872941de/public_url)
