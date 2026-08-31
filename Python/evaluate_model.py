# =============================================
# ProQuant 模型快速評估腳本
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
df = df.sort_values(["股票代碼", "交易日期"])
print(f"      Loaded {len(df):,} rows")

print("[2/5] Feature engineering...")
df = perform_feature_engineering(df)

print("[3/5] Labeling target (next day up/down)...")
df['target'] = df.groupby('股票代碼')['收盤價'].apply(
    lambda x: (x.shift(-1) > x).astype(int)
).reset_index(level=0, drop=True)

available_features = [c for c in UNIVERSAL_FEATURES if c in df.columns]
missing = set(UNIVERSAL_FEATURES) - set(available_features)
if missing:
    print(f"      Missing features: {missing}")

df_clean = df.dropna(subset=['target']).copy()
X = df_clean[available_features].replace([np.inf, -np.inf], np.nan).fillna(0)
y = df_clean['target']

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
print("="*55)

print("\nDetailed Report:")
print(classification_report(y_test, y_pred, target_names=["Down", "Up"]))

importance = pd.Series(model.feature_importances_, index=available_features).sort_values(ascending=False)
print("Top 10 Feature Importance:")
for i, (feat, val) in enumerate(importance.head(10).items(), 1):
    print(f"  {i:2}. {feat:<20} {val:.4f}")

print("\nDone! Use the accuracy number above for your resume.")
