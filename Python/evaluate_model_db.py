# =============================================
# FutureWise 模型評估（報告用）— 從資料庫讀資料，不重新訓練、不會動到 model_universal.json
#
# 跟 evaluate_model.py 的差別：那支讀的是舊的 CSV 快照，期間很可能已經在模型的訓練集裡；
# 這支讀資料庫，切測試集的方式跟 Universal Trainer.py 完全一樣（依日期排序，取最後
# test_size 筆；資料不足 40 萬筆時取最後 10%），所以量到的是模型「沒看過的資料」上的表現。
#
# 用法（在伺服器上，跟 config.py 同一個資料夾）：
#     python3 evaluate_model_db.py                     # 預設：跟訓練時相同的測試集
#     python3 evaluate_model_db.py --since 2026-09-14  # 只評估這天(含)之後的資料，
#                                                      # 給「模型訓練完之後才出現的資料」用，最嚴格，
#                                                      # 而且只載入近期資料，記憶體用量小很多
# 不加 --since 會把整張 StockPrice（300 多萬筆）讀進記憶體，跟每週六重新訓練時一樣，
# 會花一些時間，建議避開排程時段
# =============================================
import sys
import numpy as np
import pandas as pd
import xgboost as xgb
from sqlalchemy import create_engine, text
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support

from config import MYSQL_CONN_STR

UNIVERSAL_FEATURES = [
    "RSI_6", "RSI_14", "MACD_快線", "MACD_慢線", "MACD_柱狀", "K值", "D值", "ATR",
    "量能比", "5日乖離", "20日乖離", "漲跌幅_1日", "漲跌幅_3日", "漲跌幅_5日", "跳空缺口",
    "營收YoY", "EPS", "本益比", "股價淨值比", "大盤漲跌幅", "費半漲跌", "台幣匯率",
    "外資買賣超", "投信買賣超", "自營商買賣超", "情緒分數",
    "5日均線_斜率", "20日均線_斜率", "60日均線_斜率",
]

# 跟 Universal Trainer.py 同一份查詢（欄位別名要一致，特徵才對得上模型）
QUERY = """
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
    {where}
    ORDER BY 交易日期, 股票代碼
"""


def main():
    since = None
    if "--since" in sys.argv:
        since = pd.to_datetime(sys.argv[sys.argv.index("--since") + 1])

    print("[1/4] 從資料庫載入資料...")
    if since is not None:
        # 只載入 --since 之前 10 天起的資料（算均線斜率要前一天、標籤要隔天），比載入整張表快很多
        where = "WHERE trade_date >= DATE_SUB(:since, INTERVAL 10 DAY)"
        params = {"since": since.strftime("%Y-%m-%d")}
    else:
        where, params = "", {}
    with create_engine(MYSQL_CONN_STR).connect() as conn:
        df = pd.read_sql(text(QUERY.format(where=where)), conn, params=params)
    print(f"      共 {len(df):,} 筆，日期 {df['交易日期'].min()} ~ {df['交易日期'].max()}")

    print("[2/4] 特徵與標籤...")
    for ma_col, slope in {"5日均線": "5日均線_斜率", "20日均線": "20日均線_斜率", "60日均線": "60日均線_斜率"}.items():
        df[slope] = df.groupby("股票代碼")[ma_col].pct_change(fill_method=None)
    # 每檔股票最後一天還沒有「明天」，不能當成「沒漲」，要排除
    next_close = df.groupby("股票代碼")["收盤價"].shift(-1)
    df["target"] = (next_close > df["收盤價"]).astype(int).where(next_close.notna())
    df = df.dropna(subset=["target"])
    df["交易日期"] = pd.to_datetime(df["交易日期"])

    X = df[UNIVERSAL_FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0)
    y = df["target"].astype(int)

    if since is not None:
        mask = (df["交易日期"] >= since).values
        X_test, y_test = X[mask], y[mask]
        print(f"      只評估 {since.date()} 之後的資料：{len(y_test):,} 筆")
    else:
        test_size = 200000
        if len(X) < test_size * 2:
            test_size = int(len(X) * 0.1)
        X_test, y_test = X.iloc[-test_size:], y.iloc[-test_size:]
        d = df["交易日期"].iloc[-test_size:]
        print(f"      測試集（跟訓練時相同的切法）：最後 {test_size:,} 筆，日期 {d.min().date()} ~ {d.max().date()}")

    if len(y_test) == 0:
        print("❌ 測試集是空的（--since 的日期太晚？）")
        return

    print("[3/4] 載入模型並預測...")
    model = xgb.XGBClassifier()
    model.load_model("model_universal.json")
    y_pred = model.predict(X_test)

    print("[4/4] 指標")
    acc = accuracy_score(y_test, y_pred)
    up_rate = float(y_test.mean())
    majority = max(up_rate, 1 - up_rate)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, labels=[0, 1], zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()

    print("\n" + "=" * 58)
    print(f"  準確率 Accuracy                 : {acc:.2%}")
    print(f"  測試集中「隔日上漲」的比例       : {up_rate:.2%}")
    print(f"  基準線（永遠猜多數類別）         : {majority:.2%}")
    print(f"  超過基準線的幅度                 : {acc - majority:+.2%}")
    print("=" * 58)
    print(f"  精確率 Precision（預測漲的裡面真的漲）: {prec[1]:.2%}")
    print(f"  召回率 Recall   （真的漲的被抓到多少）: {rec[1]:.2%}")
    print(f"  F1（漲）: {f1[1]:.3f}    F1（跌）: {f1[0]:.3f}")
    print(f"  混淆矩陣  預測跌/實際跌 TN={tn:,}  預測漲/實際跌 FP={fp:,}")
    print(f"            預測跌/實際漲 FN={fn:,}  預測漲/實際漲 TP={tp:,}")
    print("=" * 58)

    # 信心度高時準不準？（AI 精選挑的就是高信心看漲股）
    proba_up = model.predict_proba(X_test)[:, 1]
    print("\n依「模型看漲信心度」分組，看漲時實際上漲的比例（精確率）：")
    y_arr = np.asarray(y_test)
    for lo, hi in [(0.5, 0.55), (0.55, 0.60), (0.60, 0.65), (0.65, 1.01)]:
        m = (proba_up >= lo) & (proba_up < hi)
        n = int(m.sum())
        if n:
            print(f"  信心度 {lo:.0%}~{min(hi, 1):.0%}: {n:>8,} 筆，實際上漲 {y_arr[m].mean():.2%}")
        else:
            print(f"  信心度 {lo:.0%}~{min(hi, 1):.0%}: 沒有資料")

    print("\n完整報告：")
    print(classification_report(y_test, y_pred, target_names=["Down", "Up"], zero_division=0))


if __name__ == "__main__":
    main()
