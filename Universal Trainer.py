# 程式名稱：ProQuant 終極通用模型訓練大腦 (V3.1 籌碼核心版)
# 修改重點：
# 1. 正式納入籌碼面：在特徵清單中加入 "外資買賣超", "投信買賣超", "自營商買賣超"。
# 2. 欄位名稱完全對齊：將英文特徵名修改為與 CSV 一致的中文標題。
# 3. 特徵工程升級：修正均線斜率計算邏輯，與爬蟲欄位命名匹配。

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import os

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
    csv_file = "股票清單_Cloud.csv"
    model_output = "model_universal.json"

    if not os.path.exists(csv_file):
        print(f"❌ 找不到 {csv_file}")
        return

    print(f"📂 正在載入海量數據... ")
    # 指定 dtype 確保股票代碼不丟失前導零
    df = pd.read_csv(csv_file, dtype={'股票代碼': str})
    
    # 清理股票代碼格式
    df["股票代碼"] = df["股票代碼"].astype(str).str.replace('="', '').str.replace('"', '')
    df = df.sort_values(["股票代碼", "交易日期"])

    print(f"✅ 成功載入 {len(df)} 筆交易紀錄。")

    print(f"⚙️ 執行進階特徵工程 (斜率計算)...")
    df = perform_feature_engineering(df)

    print(f"🏷️ 正在標註明日漲跌目標...")
    # 預測目標：明天收盤 > 今天收盤
    df['target'] = df.groupby('股票代碼')['收盤價'].apply(
        lambda x: (x.shift(-1) > x).astype(int)
    ).reset_index(level=0, drop=True)

    # 檢查特徵完整性
    available_features = [c for c in UNIVERSAL_FEATURES if c in df.columns]
    missing = set(UNIVERSAL_FEATURES) - set(available_features)
    if missing:
        print(f"⚠️ 警告：資料中遺漏特徵 {missing}")
    
    print(f"🛠️ 數據清洗與極端值處理...")
    df_clean = df.dropna(subset=['target']).copy()
    
    # 關鍵：處理斜率計算產生的 Inf 與 NaN
    X = df_clean[available_features].replace([np.inf, -np.inf], np.nan).fillna(0)
    y = df_clean['target']

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

    # ==========================================
    # ★ 新增：未來 1~7 天軌跡趨勢預測大腦 (Regressor)
    # ==========================================
    print(f"\n📈 開始訓練未來 1~7 天走勢回歸大腦...")
    for day in range(1, 8):
        print(f"   ► 正在建構 T+{day} 趨勢大腦...")
        # 標註未來 N 日的報酬率
        df_trend = df.copy()
        df_trend['target_reg'] = df_trend.groupby('股票代碼')['收盤價'].transform(lambda x: (x.shift(-day) - x) / x)
        
        # 過濾掉可能因為除以 0 產生的 inf，並丟棄缺失值
        df_trend['target_reg'] = df_trend['target_reg'].replace([np.inf, -np.inf], np.nan)
        df_reg_clean = df_trend.dropna(subset=['target_reg']).copy()
        X_reg_full = df_reg_clean[available_features].replace([np.inf, -np.inf], np.nan).fillna(0)
        y_reg_full = df_reg_clean['target_reg']
        
        # 簡單切分最近的資料集
        test_size_reg = int(len(X_reg_full) * 0.1) if len(X_reg_full) < test_size * 2 else test_size
        X_train_reg = X_reg_full.iloc[:-test_size_reg]
        y_train_reg = y_reg_full.iloc[:-test_size_reg]
        X_val_reg = X_reg_full.iloc[-test_size_reg:]
        y_val_reg = y_reg_full.iloc[-test_size_reg:]

        reg_model = xgb.XGBRegressor(
            n_estimators=300,      
            learning_rate=0.05,     
            max_depth=6,            
            tree_method='hist',     
            random_state=42, 
            n_jobs=-1
        )
        
        reg_model.fit(X_train_reg, y_train_reg, eval_set=[(X_val_reg, y_val_reg)], verbose=False)
        trend_model_path = f"model_trend_step_{day}.json"
        reg_model.save_model(trend_model_path)
    print(f"💾 7天走勢預測模型全數儲存完成！")

    # --- 效能測試報告 ---
    print(f"\n--- 🏁 最終效能盲測 (未來 {test_size} 筆數據) ---")
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    print(f"🎯 預測準確率: {acc:.2%}")
    
    importance = pd.Series(model.feature_importances_, index=available_features).sort_values(ascending=False)
    print("\n🔥 特徵決策貢獻排行 Top 15 (請確認三大法人是否上榜)：")
    print(importance.head(15))

if __name__ == "__main__":
    train_brain()