# =============================================
# FutureWise 模型快速評估腳本
# ⚠️ 這支讀的是本機的「股票清單_Cloud.csv」(舊的快照)，不是資料庫。
#    模型是用資料庫的全部資料訓練的，CSV 涵蓋的期間很可能已經在訓練集裡，
#    這樣量出來的準確率是「用訓練過的資料考試」，會偏高，不能拿來寫報告。
#    要量測報告用的準確率，請改用 evaluate_model_db.py
# =============================================
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report

UNIVERSAL_FEATURES = [
    "RSI_6", "RSI_14",
    "MACD_快線", "MACD_慢線", "MACD_柱狀",
    "K值", "D值", "ATR",
    "量能比", "5日乖離", "20日乖離",
    "漲跌幅_1日", "漲跌幅_3日", "漲跌幅_5日", "跳空缺口",
    "營收YoY", "EPS", "本益比", "股價淨值比",
    "大盤漲跌幅", "費半漲跌", "台幣匯率",
    "外資買賣超", "投信買賣超", "自營商買賣超",
    "情緒分數",
    "5日均線_斜率", "20日均線_斜率", "60日均線_斜率"
]

def perform_feature_engineering(df):
    df = df.copy()
    ma_mapping = {
        "5日均線": "5日均線_斜率",
        "20日均線": "20日均線_斜率",
        "60日均線": "60日均線_斜率"
    }
    for ma_col, slope_name in ma_mapping.items():
        if ma_col in df.columns:
            df[slope_name] = df.groupby("股票代碼")[ma_col].pct_change(fill_method=None)
    return df

print("[1/5] Loading data...")
df = pd.read_csv("股票清單_Cloud.csv", dtype={'股票代碼': str})
df["股票代碼"] = df["股票代碼"].astype(str).str.replace('="', '').str.replace('"', '')
# 排序要先照日期再照股票代碼，不然下面用iloc[-test_size:]切測試集時，
# 切到的會是「代碼排序最後面那幾檔股票的全部歷史」而不是「全市場最近一段期間」，
# 等於訓練集看得到測試集同期間其他股票的大盤/總經特徵，回測分數會失真（跟Universal Trainer.py同一種bug）
df = df.sort_values(["交易日期", "股票代碼"])
print(f"      Loaded {len(df):,} rows")

print("[2/5] Feature engineering...")
df = perform_feature_engineering(df)

print("[3/5] Labeling target (next day up/down)...")
# 每檔股票最後一個交易日還沒有「明天」，不能標成「沒漲」：pandas 裡 NaN > x 是 False，
# 舊寫法會把每檔股票的最後一天全標成 0(跌)，而測試集正好取最新的資料，等於測試集混進一批假標籤
next_close = df.groupby('股票代碼')['收盤價'].shift(-1)
df['target'] = (next_close > df['收盤價']).astype(int).where(next_close.notna())

available_features = [c for c in UNIVERSAL_FEATURES if c in df.columns]
missing = set(UNIVERSAL_FEATURES) - set(available_features)
if missing:
    print(f"      Missing features: {missing}")

df_clean = df.dropna(subset=['target']).copy()
X = df_clean[available_features].replace([np.inf, -np.inf], np.nan).fillna(0)
y = df_clean['target'].astype(int)

test_size = 200000
if len(X) < test_size * 2:
    test_size = int(len(X) * 0.1)

X_test = X.iloc[-test_size:]
y_test = y.iloc[-test_size:]
print(f"      Test set size: {test_size:,} rows")

print("[4/5] Loading model...")
model = xgb.XGBClassifier()
model.load_model("model_universal.json")

print("[5/5] Predicting...")
y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
baseline = y_test.mean()

print("\n" + "="*55)
print(f"  Direction Accuracy : {acc:.2%}")
print(f"  Baseline (always up): {baseline:.2%}")
print(f"  Edge over baseline : {acc - baseline:+.2%}")
majority = max(baseline, 1 - baseline)
print(f"  Baseline (majority): {majority:.2%}")
print(f"  Edge over majority : {acc - majority:+.2%}")
print("="*55)

print("\nDetailed Report:")
print(classification_report(y_test, y_pred, target_names=["Down", "Up"]))

importance = pd.Series(model.feature_importances_, index=available_features).sort_values(ascending=False)
print("Top 10 Feature Importance:")
for i, (feat, val) in enumerate(importance.head(10).items(), 1):
    print(f"  {i:2}. {feat:<20} {val:.4f}")

print("\nDone! Use the accuracy number above for your resume.")
