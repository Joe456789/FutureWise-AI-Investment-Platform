# 程式名稱：FutureWise 終極通用模型訓練大腦 (V3.1 籌碼核心版)
# 修改重點：
# 1. 正式納入籌碼面：在特徵清單中加入 "外資買賣超", "投信買賣超", "自營商買賣超"。
# 2. 欄位名稱完全對齊：將英文特徵名修改為與 CSV 一致的中文標題。
# 3. 特徵工程升級：修正均線斜率計算邏輯，與爬蟲欄位命名匹配。

import json
from datetime import datetime
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sqlalchemy import create_engine, text
from config import MYSQL_CONN_STR

# ==========================================
# 1. 設定終極特徵清單 (必須與 CSV 欄位標題完全一致)
# ==========================================
UNIVERSAL_FEATURES = [
    # --- 技術面 ---
    "RSI_6", "RSI_14", 
    "MACD_快線", "MACD_慢線", "MACD_柱狀", 
    "K值", "D值", "ATR", 
    "量能比", "5日乖離", "20日乖離", 
    "漲跌幅_1日", "漲跌幅_3日", "漲跌幅_5日", "跳空缺口",
    
    # --- 基本面 ---
    "營收YoY", "EPS", "本益比", "股價淨值比",
    
    # --- 總經連動 ---
    "大盤漲跌幅",   
    "費半漲跌",     
    "台幣匯率",     
    
    # --- ★ 籌碼面 (V3.1 新增：讓 AI 學習主力動向) ---
    "外資買賣超", 
    "投信買賣超", 
    "自營商買賣超",
    
    # --- 另類數據 ---
    "情緒分數",     
    
    # --- 趨勢斜率 (由下方函式產生) ---
    "5日均線_斜率", "20日均線_斜率", "60日均線_斜率"
]

def perform_feature_engineering(df):
    """
    執行進階特徵加工
    配合爬蟲產出的『n日均線』欄位計算斜率
    """
    df = df.copy()
    ma_mapping = {
        "5日均線": "5日均線_斜率",
        "20日均線": "20日均線_斜率",
        "60日均線": "60日均線_斜率"
    }
    
    for ma_col, slope_name in ma_mapping.items():
        if ma_col in df.columns:
            # 計算百分比變化作為趨勢力道
            df[slope_name] = df.groupby("股票代碼")[ma_col].pct_change(fill_method=None)
            
    return df

def train_brain():
    # 改成直接讀 MySQL，不再依賴 cloud_crawler_TWSE.py 額外匯出的 CSV 備份檔
    # （CSV 曾經發生沒寫成功、訓練資料卡在舊日期的問題，MySQL 才是每天真正在更新的資料源）
    model_output = "model_universal.json"

    print(f"📂 正在從 MySQL 載入海量數據...")
    engine = create_engine(MYSQL_CONN_STR)
    query = """
        SELECT
            ticker AS 股票代碼, trade_date AS 交易日期, close_price AS 收盤價,
            RSI_6, RSI_14,
            MACD_DIF AS MACD_快線, MACD_Signal AS MACD_慢線, MACD_Hist AS MACD_柱狀,
            KD_K AS K值, KD_D AS D值, ATR,
            Vol_Ratio AS 量能比,
            Bias_5 AS `5日乖離`, Bias_20 AS `20日乖離`,
            Change_1D AS 漲跌幅_1日, Change_3D AS 漲跌幅_3日, Change_5D AS 漲跌幅_5日,
            Gap AS 跳空缺口,
            Revenue_YoY AS 營收YoY, EPS, PE_Ratio AS 本益比, PB_Ratio AS 股價淨值比,
            Market_Return AS 大盤漲跌幅, SOX_Return AS 費半漲跌, TWD_Exchange AS 台幣匯率,
            Foreign_Buy AS 外資買賣超, Trust_Buy AS 投信買賣超, Dealer_Buy AS 自營商買賣超,
            Sentiment_Score AS 情緒分數,
            MA_5 AS `5日均線`, MA_20 AS `20日均線`, MA_60 AS `60日均線`
        FROM StockPrice
        ORDER BY 交易日期, 股票代碼
    """
    with engine.connect() as conn:
        df = pd.read_sql(text(query), conn)

    if df.empty:
        print("❌ 查無資料，中止訓練")
        return

    print(f"✅ 成功載入 {len(df)} 筆交易紀錄。")

    print(f"⚙️ 執行進階特徵工程 (斜率計算)...")
    df = perform_feature_engineering(df)

    print(f"🏷️ 正在標註明日漲跌目標...")
    # 預測目標：明天收盤 > 今天收盤
    # 每檔股票最後一個交易日還沒有「明天」，不能標成「沒漲」：pandas 裡 NaN > x 是 False，
    # 舊寫法會把每檔股票的最後一天全標成 0(跌)，而測試集正好取最新的資料，等於測試集混進一批假標籤
    next_close = df.groupby('股票代碼')['收盤價'].shift(-1)
    df['target'] = (next_close > df['收盤價']).astype(int).where(next_close.notna())

    # 檢查特徵完整性
    available_features = [c for c in UNIVERSAL_FEATURES if c in df.columns]
    missing = set(UNIVERSAL_FEATURES) - set(available_features)
    if missing:
        print(f"⚠️ 警告：資料中遺漏特徵 {missing}")
    
    print(f"🛠️ 數據清洗與極端值處理...")
    df_clean = df.dropna(subset=['target']).copy()
    
    # 關鍵：處理斜率計算產生的 Inf 與 NaN
    X = df_clean[available_features].replace([np.inf, -np.inf], np.nan).fillna(0)
    y = df_clean['target'].astype(int)

    # --- 時間序列分割 ---
    test_size = 200000
    if len(X) < test_size * 2:
        test_size = int(len(X) * 0.1) # 若資料不夠大，自動調整測試集比例
        
    X_train_full = X.iloc[:-test_size]
    y_train_full = y.iloc[:-test_size]
    X_test = X.iloc[-test_size:]
    y_test = y.iloc[-test_size:]

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.1, random_state=42, shuffle=False
    )

    print(f"🧠 開始訓練深度 XGBoost 森林大腦...")
    print(f"   [訓練特徵數: {len(available_features)} | 總訓練樣本: {len(X_train)}]")
    
    model = xgb.XGBClassifier(
        n_estimators=1200,      
        learning_rate=0.02,     
        max_depth=9,            
        tree_method='hist',     
        subsample=0.85,
        colsample_bytree=0.8,
        random_state=42, 
        eval_metric=['logloss'],
        early_stopping_rounds=100, 
        n_jobs=-1
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=50 
    )

    # 儲存模型
    model.save_model(model_output)
    print(f"\n✨ 訓練完成！最佳迭代次數: {model.best_iteration}")
    print(f"💾 終極模型已存至: {model_output}")

    # --- 效能測試報告 ---
    print(f"\n--- 🏁 最終效能盲測 (未來 {test_size} 筆數據) ---")
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    print(f"🎯 預測準確率: {acc:.2%}")

    # 準確率有盲點（如果多數天數都漲，永遠猜漲也有不錯的準確率），所以一併算精確率、召回率、F1，
    # 並跟「永遠猜多數類別」的基準線比較。標籤：1=隔日上漲、0=隔日未上漲
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, labels=[0, 1], zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
    up_rate = float(y_test.mean())
    majority_baseline = max(up_rate, 1 - up_rate)
    print(classification_report(y_test, y_pred, target_names=["Down", "Up"], zero_division=0))
    print(f"混淆矩陣 TN={tn} FP={fp} FN={fn} TP={tp}；測試集上漲比例 {up_rate:.2%}；多數類別基準線 {majority_baseline:.2%}")
    
    importance = pd.Series(model.feature_importances_, index=available_features).sort_values(ascending=False)
    print("\n🔥 特徵決策貢獻排行 Top 15 (請確認三大法人是否上榜)：")
    print(importance.head(15))

    # 把特徵重要性存成 JSON，讓 API / 前端可以做「模型可解釋性」視覺化，不再只印在 log 裡
    importance_output = {
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "accuracy": round(float(acc), 4),
        "metrics": {
            "test_size": int(test_size),
            "up_rate": round(up_rate, 4),
            "majority_baseline": round(majority_baseline, 4),
            "edge_over_majority": round(float(acc) - majority_baseline, 4),
            "precision_up": round(float(prec[1]), 4), "recall_up": round(float(rec[1]), 4), "f1_up": round(float(f1[1]), 4),
            "precision_down": round(float(prec[0]), 4), "recall_down": round(float(rec[0]), 4), "f1_down": round(float(f1[0]), 4),
            "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        },
        "features": [
            {"name": name, "importance": round(float(val), 6)}
            for name, val in importance.head(15).items()
        ]
    }
    with open("feature_importance.json", "w", encoding="utf-8") as f:
        json.dump(importance_output, f, ensure_ascii=False, indent=2)
    print("💾 特徵重要性已存至: feature_importance.json")

if __name__ == "__main__":
    train_brain()