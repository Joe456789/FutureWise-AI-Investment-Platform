# 程式名稱：FutureWise AI 精選批次評分器
# 功能：一次性掃描全市場最新資料，用 XGBoost 對每檔股票算出漲跌信心分數，
#      取排名前 N 名（與最後 N 名）寫入快取檔，給 /api/ai_top_picks 讀取。
# 用意：避免選股頁「AI精選」為了做全市場排名，對後端發出 2000+ 次即時預測請求。
# 排程：跟 cloud_crawler_TWSE.py 一樣，建議掛在 pm2 排程裡，資料更新完之後跑一次。

import os
import json
from datetime import datetime

import pandas as pd
from sqlalchemy import create_engine, text
import xgboost as xgb

from config import MYSQL_CONN_STR

current_dir = os.path.dirname(os.path.abspath(__file__))
TOP_N = 5

# 必須與 stock_api_server.py / Universal Trainer.py 完全一致
UNIVERSAL_FEATURES = [
    "RSI_6", "RSI_14", "MACD_快線", "MACD_慢線", "MACD_柱狀",
    "K值", "D值", "ATR", "量能比", "5日乖離", "20日乖離",
    "漲跌幅_1日", "漲跌幅_3日", "漲跌幅_5日", "跳空缺口",
    "營收YoY", "EPS", "本益比", "股價淨值比",
    "大盤漲跌幅", "費半漲跌", "台幣匯率",
    "外資買賣超", "投信買賣超", "自營商買賣超",
    "情緒分數",
    "5日均線_斜率", "20日均線_斜率", "60日均線_斜率"
]

COLUMN_MAPPING = {
    "Foreign_Buy": "外資買賣超",
    "Trust_Buy": "投信買賣超",
    "Dealer_Buy": "自營商買賣超",
    "MACD_DIF": "MACD_快線",
    "MACD_Signal": "MACD_慢線",
    "MACD_Hist": "MACD_柱狀",
    "KD_K": "K值",
    "KD_D": "D值",
    "Vol_Ratio": "量能比",
    "Bias_5": "5日乖離",
    "Bias_20": "20日乖離",
    "Change_1D": "漲跌幅_1日",
    "Change_3D": "漲跌幅_3日",
    "Change_5D": "漲跌幅_5日",
    "Gap": "跳空缺口",
    "Revenue_YoY": "營收YoY",
    "PE_Ratio": "本益比",
    "PB_Ratio": "股價淨值比",
    "Market_Return": "大盤漲跌幅",
    "SOX_Return": "費半漲跌",
    "TWD_Exchange": "台幣匯率",
    "Sentiment_Score": "情緒分數",
}

# 排除大盤/櫃買指數等非個股代碼，避免混進排名
EXCLUDED_TICKERS = {"TSE", "OTC"}


def load_model():
    model = xgb.XGBClassifier()
    model.load_model(os.path.join(current_dir, "model_universal.json"))
    return model


def compute_row_features(ticker_df: pd.DataFrame):
    """輸入單一股票依日期排序後的近幾日資料，回傳(補齊特徵欄位的最新一列, 前一日收盤價或None)"""
    g = ticker_df.sort_values("trade_date").copy()

    prev_close = None
    if len(g) >= 2 and "close_price" in g.columns:
        prev_val = g["close_price"].iloc[-2]
        prev_close = float(prev_val) if pd.notna(prev_val) else None

    if "MA_5" in g.columns:
        g["5日均線_斜率"] = g["MA_5"].pct_change()
    if "MA_20" in g.columns:
        g["20日均線_斜率"] = g["MA_20"].pct_change()
    if "MA_60" in g.columns:
        g["60日均線_斜率"] = g["MA_60"].pct_change()

    latest = g.tail(1).copy()
    latest = latest.rename(columns=COLUMN_MAPPING)

    for col in UNIVERSAL_FEATURES:
        if col not in latest.columns:
            latest[col] = 0.0
        latest[col] = pd.to_numeric(latest[col], errors="coerce").fillna(0.0)

    return latest, prev_close


def resolve_pending_predictions(engine, df_all: pd.DataFrame):
    """核對之前記錄但還沒驗證過的預測：如果現在資料庫已經有更新的交易日資料，
    就比對「預測當下收盤價」vs「下一個交易日收盤價」，判斷猜對了沒，回填 PredictionLog。"""
    with engine.connect() as conn:
        pending = conn.execute(text(
            "SELECT id, ticker, predicted_at, close_at_predict, predicted_direction FROM PredictionLog WHERE resolved = 0"
        )).fetchall()

    if not pending:
        print("ℹ️ 沒有待驗證的歷史預測")
        return

    resolved_count = 0
    for row in pending:
        pid, ticker, predicted_at, close_at_predict, predicted_direction = row
        ticker_rows = df_all[(df_all["ticker"] == ticker) & (pd.to_datetime(df_all["trade_date"]) > pd.to_datetime(predicted_at))]
        if ticker_rows.empty:
            continue  # 還沒有更新的交易日資料，下次排程再驗證

        next_row = ticker_rows.sort_values("trade_date").iloc[0]
        next_close = next_row.get("close_price")
        if pd.isna(next_close):
            continue

        actual_direction = "up" if float(next_close) > float(close_at_predict) else "down"
        is_correct = 1 if actual_direction == predicted_direction else 0

        with engine.connect() as conn:
            conn.execute(text("""
                UPDATE PredictionLog
                SET resolved = 1, actual_direction = :ad, is_correct = :ic, resolved_at = NOW()
                WHERE id = :id
            """), {"ad": actual_direction, "ic": is_correct, "id": pid})
            conn.commit()
        resolved_count += 1

    print(f"✅ 驗證完成 {resolved_count} 筆歷史預測（{len(pending) - resolved_count} 筆仍在等待下一個交易日資料）")


def log_predictions(engine, results: list):
    """把這次掃描全市場的預測結果記錄下來，之後用來計算命中率"""
    with engine.connect() as conn:
        for r in results:
            if r["price"] is None or r["predicted_at"] is None:
                continue
            direction = "up" if r["confidence"] >= 50 else "down"
            conn.execute(text("""
                INSERT INTO PredictionLog (ticker, predicted_at, close_at_predict, predicted_direction, confidence)
                VALUES (:t, :d, :p, :dir, :c)
                ON DUPLICATE KEY UPDATE close_at_predict = VALUES(close_at_predict),
                                        predicted_direction = VALUES(predicted_direction),
                                        confidence = VALUES(confidence)
            """), {"t": r["ticker"], "d": r["predicted_at"], "p": r["price"], "dir": direction, "c": r["confidence"]})
        conn.commit()
    print(f"💾 已記錄 {len(results)} 筆預測到 PredictionLog，供之後計算命中率")


def run_scan(top_n: int = TOP_N):
    print("📡 正在連線至 MySQL 雲端資料庫...")
    engine = create_engine(MYSQL_CONN_STR)
    model = load_model()

    # 只抓最近幾個交易日的資料（算均線斜率需要至少 2 天），避免整張 2G 的表都撈進記憶體
    query = """
        SELECT * FROM StockPrice
        WHERE trade_date >= (SELECT DATE_SUB(MAX(trade_date), INTERVAL 5 DAY) FROM StockPrice)
    """
    with engine.connect() as conn:
        df_all = pd.read_sql(text(query), conn)

    if df_all.empty:
        print("⚠️ 查無近期資料，中止掃描")
        return

    # 先驗證之前記錄的預測，準不準都要老實記錄下來，不能只挑對的秀
    resolve_pending_predictions(engine, df_all)

    results = []
    total = 0
    for ticker, group in df_all.groupby("ticker"):
        if ticker in EXCLUDED_TICKERS or ticker.startswith("00"):
            continue
        total += 1
        try:
            latest, prev_close = compute_row_features(group)
            X_pred = latest[UNIVERSAL_FEATURES]

            probabilities = model.predict_proba(X_pred)[0]
            base_up_confidence = float(probabilities[1])  # numpy.float64 不能被 json.dump 序列化，要轉成原生 float

            sentiment = latest["情緒分數"].values[0] if "情緒分數" in latest.columns else 0.0
            sentiment = 0.0 if pd.isna(sentiment) else float(sentiment)
            adjusted_up_confidence = max(0.01, min(0.99, base_up_confidence + sentiment * 0.1))
            up_confidence = round(adjusted_up_confidence * 100, 2)

            stock_name = latest["stock_name"].values[0] if "stock_name" in latest.columns else ticker
            close_price = latest["close_price"].values[0] if "close_price" in latest.columns else None
            close_price = float(close_price) if pd.notna(close_price) else None
            predicted_at = latest["trade_date"].values[0] if "trade_date" in latest.columns else None
            predicted_at = pd.to_datetime(predicted_at).strftime("%Y-%m-%d") if pd.notna(predicted_at) else None

            diff = (close_price - prev_close) if (close_price is not None and prev_close is not None) else None
            percent = (diff / prev_close * 100) if (diff is not None and prev_close) else None

            results.append({
                "ticker": ticker,
                "name": stock_name if pd.notna(stock_name) else ticker,
                "price": close_price,
                "predicted_at": predicted_at,
                "diff": round(diff, 2) if diff is not None else None,
                "percent": round(percent, 2) if percent is not None else None,
                "confidence": up_confidence,
                "prediction": "看漲 🚀" if up_confidence >= 50 else "看跌 📉"
            })
        except Exception as e:
            print(f"⚠️ {ticker} 評分失敗，略過：{e}")
            continue

    if not results:
        print("⚠️ 沒有任何股票評分成功，中止寫入快取")
        return

    log_predictions(engine, results)

    # 快取檔不需要 predicted_at 這個內部欄位，清掉避免前端多餘資料
    for r in results:
        r.pop("predicted_at", None)

    results.sort(key=lambda x: x["confidence"], reverse=True)
    top_bullish = results[:top_n]
    top_bearish = sorted(results, key=lambda x: x["confidence"])[:top_n]

    cache = {
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "scanned_count": total,
        "top_bullish": top_bullish,
        "top_bearish": top_bearish
    }

    cache_path = os.path.join(current_dir, "ai_top_picks_cache.json")
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)

    print(f"🎉 AI精選掃描完成！共評分 {total} 檔股票，結果已寫入 {cache_path}")


if __name__ == "__main__":
    run_scan()
