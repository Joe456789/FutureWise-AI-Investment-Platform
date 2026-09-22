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


def fill_pending_pick_entries(engine, df_all: pd.DataFrame):
    """把之前AI精選挑出、還沒有隔天開盤價的模擬交易，用現在資料庫已經有的最新資料補上進場價"""
    with engine.connect() as conn:
        pending = conn.execute(text(
            "SELECT id, ticker, pick_date FROM AiPickTrades WHERE entry_price IS NULL"
        )).fetchall()

    if not pending:
        return

    filled_count = 0
    for pid, ticker, pick_date in pending:
        ticker_rows = df_all[(df_all["ticker"] == ticker) & (pd.to_datetime(df_all["trade_date"]).dt.date > pick_date)]
        if ticker_rows.empty:
            continue
        next_row = ticker_rows.sort_values("trade_date").iloc[0]
        entry_price = next_row.get("open_price")
        if pd.isna(entry_price):
            continue
        entry_date = pd.to_datetime(next_row["trade_date"]).date()

        with engine.connect() as conn:
            conn.execute(text(
                "UPDATE AiPickTrades SET entry_price = :p, entry_date = :d WHERE id = :id"
            ), {"p": float(entry_price), "d": entry_date, "id": pid})
            conn.commit()
        filled_count += 1

    if filled_count:
        print(f"📌 補上 {filled_count} 筆AI精選模擬交易的隔天開盤進場價")


def record_pick_trades(engine, momentum_list, emerging_list, bearish_list, pick_date):
    """把今天AI精選挑出的股票，記錄成待補進場價的模擬交易（進場價要等明天開盤才知道）。
    bearish(AI看跌名單)模擬「隔天開盤放空」，成效算法跟看漲的兩類相反（見compute_trade_return）"""
    rows = [(r, "momentum") for r in momentum_list] + [(r, "potential") for r in emerging_list] \
        + [(r, "bearish") for r in bearish_list]
    if not rows:
        return

    with engine.begin() as conn:
        for r, signal_type in rows:
            conn.execute(text("""
                INSERT IGNORE INTO AiPickTrades (ticker, stock_name, signal_type, pick_date, confidence)
                VALUES (:ticker, :name, :signal_type, :pick_date, :confidence)
            """), {
                "ticker": r["ticker"], "name": r["name"], "signal_type": signal_type,
                "pick_date": pick_date, "confidence": r["confidence"],
            })
    print(f"📝 記錄 {len(rows)} 筆今日AI精選為模擬交易，待隔天開盤補上進場價")


def compute_trade_return(signal_type: str, entry_price: float, latest_close: float) -> float:
    """算單筆模擬交易目前的浮動報酬率(%)。bearish類是「模擬放空」，股價跌越多報酬越高，
    所以方向跟momentum/potential(模擬做多)相反"""
    raw = (latest_close - entry_price) / entry_price * 100
    return -raw if signal_type == "bearish" else raw


def snapshot_performance(engine, snapshot_date):
    """把AiPickTrades目前累積(所有已進場)的浮動績效，依日期存一筆快照到AiPickPerformanceHistory，
    用來畫「近30天報酬率變化」趨勢圖——只看累積總平均，看不出績效是在變好還變差"""
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT signal_type, entry_price,
                   (SELECT close_price FROM StockPrice sp WHERE sp.ticker = t.ticker ORDER BY sp.trade_date DESC LIMIT 1) AS latest_close
            FROM AiPickTrades t WHERE entry_price IS NOT NULL
        """)).fetchall()

    stats = {}
    for signal_type, entry_price, latest_close in rows:
        if entry_price is None or latest_close is None or float(entry_price) == 0:
            continue
        ret = compute_trade_return(signal_type, float(entry_price), float(latest_close))
        s = stats.setdefault(signal_type, {"returns": [], "wins": 0})
        s["returns"].append(ret)
        if ret > 0:
            s["wins"] += 1

    if not stats:
        return

    with engine.begin() as conn:
        for signal_type, s in stats.items():
            count = len(s["returns"])
            conn.execute(text("""
                INSERT INTO AiPickPerformanceHistory (snapshot_date, signal_type, avg_return, win_rate, trade_count)
                VALUES (:d, :st, :ar, :wr, :c)
                ON DUPLICATE KEY UPDATE avg_return = VALUES(avg_return), win_rate = VALUES(win_rate), trade_count = VALUES(trade_count)
            """), {
                "d": snapshot_date, "st": signal_type,
                "ar": round(sum(s["returns"]) / count, 2),
                "wr": round(s["wins"] / count * 100, 2),
                "c": count,
            })
    print(f"📈 已記錄 {snapshot_date} 的AI精選績效快照")


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
    # 順便把之前AI精選挑出、還在等隔天開盤價的模擬交易補上進場價
    fill_pending_pick_entries(engine, df_all)

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

    # AI精選拆成兩類：今天漲幅已經很大的股票，模型看漲的信心分數常常只是反映
    # 「量能比/K值/D值/漲跌幅_1日」這些技術指標同時衝到極端值，比較像是「動能延續」，
    # 追價風險較高；分開列出，讓使用者自己判斷要哪一種，而不是全部混在一起當成同一種推薦
    MOMENTUM_THRESHOLD = 5.0  # 今日漲幅超過 5% 歸類為動能延續
    bullish_candidates = [r for r in results if r["confidence"] >= 50]
    top_bullish_momentum = [r for r in bullish_candidates if r["percent"] is not None and r["percent"] >= MOMENTUM_THRESHOLD][:top_n]
    top_bullish_emerging = [r for r in bullish_candidates if r["percent"] is None or r["percent"] < MOMENTUM_THRESHOLD][:top_n]
    top_bearish = sorted(results, key=lambda x: x["confidence"])[:top_n]

    # 把今天精選出來的股票記錄成模擬交易，長期追蹤「精選隔天開盤買進/放空」實際績效如何
    pick_date = pd.to_datetime(df_all["trade_date"]).max().date()
    record_pick_trades(engine, top_bullish_momentum, top_bullish_emerging, top_bearish, pick_date)
    # 順便存一筆今天的績效快照，之後才有辦法在前端畫「近30天報酬率變化」趨勢圖
    snapshot_performance(engine, pick_date)

    cache = {
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "scanned_count": total,
        "top_bullish_momentum": top_bullish_momentum,
        "top_bullish_emerging": top_bullish_emerging,
        "top_bearish": top_bearish
    }

    cache_path = os.path.join(current_dir, "ai_top_picks_cache.json")
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)

    print(f"🎉 AI精選掃描完成！共評分 {total} 檔股票，結果已寫入 {cache_path}")


if __name__ == "__main__":
    run_scan()
