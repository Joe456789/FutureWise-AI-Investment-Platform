# 程式名稱：自選股買賣訊號監控排程
# 功能：對每個使用者的自選股(Favorites)，比對AI信心度是否轉為強烈偏多/偏空，
#      以及 pattern_scanner.py 當天算好的技術面型態訊號，觸發時寫入 WatchlistAlerts，
#      提供 /api/watchlist_alerts 給前端「通知」頁顯示。
# 排程：必須接在 cloud_crawler_TWSE.py 跟 pattern_scanner.py 之後執行——
#      前者要有當天最新股價才能算AI信心度，後者的 strategy_cache.json 要是當天算好的。

import os
import json

import pandas as pd
from sqlalchemy import create_engine, text

from config import MYSQL_CONN_STR
from ai_top_picks_scanner import load_model, compute_row_features, UNIVERSAL_FEATURES
from auth_schema import ensure_alert_payload_column

current_dir = os.path.dirname(os.path.abspath(__file__))
STRATEGY_CACHE_FILE = os.path.join(current_dir, "strategy_cache.json")

AI_BEARISH_THRESHOLD = 35  # confidence <= 此值 → 觸發賣出提醒
AI_BULLISH_THRESHOLD = 65  # confidence >= 此值 → 觸發買進提醒

# pattern_scanner.py 算出的9種型態，分成買進訊號跟賣出/警示訊號兩類
PATTERN_BUY_LABELS = {
    "bullish_ma": "均線呈多頭排列",
    "w_bottom": "形成W底型態",
    "golden_cross": "出現黃金交叉",
    "bollinger_breakout": "布林通道帶量突破",
    "cup_handle": "形成杯柄型態",
    "macd_flip": "MACD由負轉正",
    "kd_golden_cross": "KD低檔黃金交叉",
    "volume_breakout": "量價齊揚",
}
PATTERN_SELL_LABELS = {
    "m_top": "形成M頭型態，留意回檔風險",
}


def load_pattern_signals():
    """讀取 pattern_scanner.py 算好的快取，回傳 {ticker: [(signal_type, detail), ...]}"""
    if not os.path.exists(STRATEGY_CACHE_FILE):
        print("⚠️ 找不到 strategy_cache.json，跳過技術面訊號比對（可能 pattern_scanner.py 還沒跑過）")
        return {}
    with open(STRATEGY_CACHE_FILE, "r", encoding="utf-8") as f:
        cache = json.load(f)
    strategies = cache.get("strategies", {})

    signals = {}
    for key, label in {**PATTERN_BUY_LABELS, **PATTERN_SELL_LABELS}.items():
        signal_type = "pattern_sell" if key in PATTERN_SELL_LABELS else "pattern_buy"
        for item in strategies.get(key, []):
            ticker = item.get("ticker")
            if not ticker:
                continue
            signals.setdefault(ticker, []).append((signal_type, label, item.get("reason") or label))
    return signals


def _num(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if f != f else f  # NaN 視為沒有資料


def compute_grade(ai_hit, ai_conf, n_patterns, vol_ratio):
    """通知等級（AA/A/B）：依「同方向訊號的數量與強度」自訂的簡單計分，不是模型輸出，僅供排序參考。
    AI訊號 +1（信心度極端 >=75 或 <=25 再 +1）；同方向技術型態每個 +1（最多計2個）；量能比 >=2 倍 +1。
    4分以上 AA、3分 A、其餘 B"""
    points = 0
    if ai_hit:
        points += 1
        if ai_conf is not None and (ai_conf >= 75 or ai_conf <= 25):
            points += 1
    points += min(n_patterns, 2)
    if vol_ratio is not None and vol_ratio >= 2:
        points += 1
    return "AA" if points >= 4 else "A" if points == 3 else "B"


def build_alert_payload(signal_type, strategy, own_reason, row, confidence, ai_signal_type, same_dir_patterns, data_date):
    """組出通知卡片要顯示的結構化內容。
    ai_signal_type：同方向的 AI 訊號類型，沒有就是 None；
    same_dir_patterns：這檔股票今天同方向的所有技術型態 [(label, reason), ...]（包含這則通知自己）"""
    direction = "buy" if signal_type in ("ai_bullish", "pattern_buy") else "sell"
    vol_ratio = _num(row.get("量能比"))
    change = _num(row.get("漲跌幅_1日"))
    rsi = _num(row.get("RSI_14"))
    macd = _num(row.get("MACD_柱狀"))
    k, d = _num(row.get("K值")), _num(row.get("D值"))
    foreign, trust, dealer = _num(row.get("外資買賣超")), _num(row.get("投信買賣超")), _num(row.get("自營商買賣超"))

    reasons = [own_reason]
    if vol_ratio is not None and vol_ratio >= 1.5:
        reasons.append(f"量能放大至均量的 {vol_ratio:.2f} 倍")
    if signal_type.startswith("pattern") and ai_signal_type and confidence is not None:
        reasons.append(f"AI 模型同向：預測隔日上漲機率 {confidence:.1f}%")
    for label, reason in same_dir_patterns:
        if reason != own_reason:
            reasons.append(f"同時出現：{label}")

    metrics = []
    if confidence is not None:
        metrics.append({"label": "AI信心度（隔日上漲機率）", "value": f"{confidence:.1f}%"})
    if vol_ratio is not None:
        metrics.append({"label": "量能比", "value": f"{vol_ratio:.2f}x"})
    if change is not None:
        metrics.append({"label": "今日漲跌", "value": f"{change * 100:+.2f}%"})
    if rsi is not None:
        metrics.append({"label": "RSI(14)", "value": f"{rsi:.1f}"})
    if k is not None and d is not None:
        metrics.append({"label": "KD", "value": f"K {k:.1f} / D {d:.1f}"})
    if macd is not None:
        metrics.append({"label": "MACD柱狀", "value": f"{macd:+.3f}"})
    if any(v for v in (foreign, trust, dealer)):
        metrics.append({"label": "外資／投信／自營買賣超",
                        "value": " / ".join(f"{(v or 0):+,.0f}" for v in (foreign, trust, dealer))})

    return {
        "version": 1,
        "direction": direction,
        "strategy": strategy,
        "grade": compute_grade(ai_signal_type is not None, confidence if ai_signal_type else None,
                               len(same_dir_patterns), vol_ratio),
        "price": _num(row.get("close_price")),
        "change_pct": round(change * 100, 2) if change is not None else None,
        "reasons": reasons,
        "metrics": metrics,
        "data_date": str(data_date),
    }


def fill_pending_alert_entries(engine, df_all: pd.DataFrame):
    """把之前觸發、還沒有隔天開盤價的買賣訊號，用現在資料庫已經有的最新資料補上進場價，
    比照 ai_top_picks_scanner.py 的 fill_pending_pick_entries 做法"""
    with engine.connect() as conn:
        pending = conn.execute(text(
            "SELECT id, ticker, trade_date FROM WatchlistAlerts WHERE entry_price IS NULL"
        )).fetchall()

    if not pending:
        return

    filled_count = 0
    for aid, ticker, trade_date in pending:
        ticker_rows = df_all[(df_all["ticker"] == ticker) & (pd.to_datetime(df_all["trade_date"]).dt.date > trade_date)]
        if ticker_rows.empty:
            continue
        next_row = ticker_rows.sort_values("trade_date").iloc[0]
        entry_price = next_row.get("open_price")
        if pd.isna(entry_price):
            continue
        entry_date = pd.to_datetime(next_row["trade_date"]).date()

        with engine.connect() as conn:
            conn.execute(text(
                "UPDATE WatchlistAlerts SET entry_price = :p, entry_date = :d WHERE id = :id"
            ), {"p": float(entry_price), "d": entry_date, "id": aid})
            conn.commit()
        filled_count += 1

    if filled_count:
        print(f"📌 補上 {filled_count} 筆自選股買賣訊號的隔天開盤進場價")


def run_scan():
    print("🔔 開始掃描自選股買賣訊號...")
    engine = create_engine(MYSQL_CONN_STR)
    ensure_alert_payload_column(engine)

    with engine.connect() as conn:
        # LEFT JOIN NotificationSettings：使用者從沒設定過的話 ai_alert 是 NULL，
        # 要當成預設值 TRUE（跟 /api/notifications/settings 讀取時的預設行為一致），
        # 只有明確關掉(ai_alert = 0)的人才跳過，不產生這類通知
        watchlist_rows = conn.execute(text("""
            SELECT f.user_email, f.ticker, f.stock_name
            FROM Favorites f
            LEFT JOIN NotificationSettings ns ON ns.user_email = f.user_email
            WHERE ns.ai_alert IS NULL OR ns.ai_alert = 1
        """)).fetchall()

    if not watchlist_rows:
        print("ℹ️ 目前沒有任何使用者的自選股（或大家都關掉了AI提醒），結束")
        return

    watched_tickers = {r[1] for r in watchlist_rows}
    print(f"📋 共 {len(watchlist_rows)} 筆自選股關注記錄，涉及 {len(watched_tickers)} 檔不重複股票")

    # 只抓最近幾個交易日的資料算AI信心度(跟ai_top_picks_scanner.py同一套做法)，
    # 再用watchlist過濾，避免對整張2G的表下WHERE ticker IN (...)這種動態長度查詢
    with engine.connect() as conn:
        query = """
            SELECT * FROM StockPrice
            WHERE trade_date >= (SELECT DATE_SUB(MAX(trade_date), INTERVAL 5 DAY) FROM StockPrice)
        """
        df_all = pd.read_sql(text(query), conn)

    # 先用全市場資料補上之前觸發訊號的隔天開盤進場價（不能先按目前自選股過濾，
    # 不然使用者事後取消關注的股票，進場價就永遠補不到了）
    fill_pending_alert_entries(engine, df_all)

    df_all = df_all[df_all["ticker"].isin(watched_tickers)]

    # 一次算好每檔股票的AI信心度，避免同一檔股票被多個使用者關注時重複計算
    model = load_model()
    confidence_map = {}
    trade_date_map = {}
    row_map = {}  # 每檔股票最新一列特徵，通知卡片要顯示量能比、RSI 等數字
    for ticker, group in df_all.groupby("ticker"):
        try:
            latest, _ = compute_row_features(group)
            row_map[ticker] = latest.iloc[0].to_dict()
            X_pred = latest[UNIVERSAL_FEATURES]
            probabilities = model.predict_proba(X_pred)[0]
            confidence_map[ticker] = round(float(probabilities[1]) * 100, 2)
            trade_date_map[ticker] = pd.to_datetime(latest["trade_date"].values[0]).date()
        except Exception as e:
            print(f"⚠️ {ticker} AI信心度計算失敗，略過：{e}")

    pattern_signals = load_pattern_signals()

    alerts_to_insert = []
    for user_email, ticker, stock_name in watchlist_rows:
        trade_date = trade_date_map.get(ticker)
        if trade_date is None:
            continue  # 這檔股票今天算不出AI信心度(可能資料不足)，技術面訊號沒有對應日期可歸屬，一併跳過

        row = row_map.get(ticker, {})
        confidence = confidence_map.get(ticker)
        patterns = pattern_signals.get(ticker, [])

        ai_signal_type = None
        if confidence is not None:
            if confidence <= AI_BEARISH_THRESHOLD:
                ai_signal_type = "ai_bearish"
            elif confidence >= AI_BULLISH_THRESHOLD:
                ai_signal_type = "ai_bullish"

        def same_dir(signal_type):
            return [(label, reason) for st, label, reason in patterns if st == signal_type]

        if ai_signal_type:
            bullish = ai_signal_type == "ai_bullish"
            if bullish:
                detail = f"AI轉強烈偏多({confidence}%)，建議留意買進時機"
                strategy = "AI 轉強烈偏多"
                own_reason = f"AI 模型預測隔日上漲機率 {confidence:.1f}%，明顯偏多"
            else:
                detail = f"AI轉強烈偏空({confidence}%)，建議留意賣出時機"
                strategy = "AI 轉強烈偏空"
                own_reason = f"AI 模型預測隔日上漲機率僅 {confidence:.1f}%，明顯偏空"
            alerts_to_insert.append({
                "user_email": user_email, "ticker": ticker, "stock_name": stock_name,
                "signal_type": ai_signal_type, "signal_detail": detail, "trade_date": trade_date,
                "payload": json.dumps(build_alert_payload(
                    ai_signal_type, strategy, own_reason, row, confidence, ai_signal_type,
                    same_dir("pattern_buy" if bullish else "pattern_sell"), trade_date), ensure_ascii=False),
            })

        for signal_type, label, reason in patterns:
            action = "買進" if signal_type == "pattern_buy" else "賣出"
            # 同方向才算「呼應」：買進型態呼應 AI 偏多，賣出型態呼應 AI 偏空
            ai_same = ai_signal_type if (ai_signal_type == "ai_bullish") == (signal_type == "pattern_buy") else None
            alerts_to_insert.append({
                "user_email": user_email, "ticker": ticker, "stock_name": stock_name,
                "signal_type": signal_type, "signal_detail": f"{label}，建議留意{action}時機", "trade_date": trade_date,
                "payload": json.dumps(build_alert_payload(
                    signal_type, label, reason, row, confidence, ai_same,
                    same_dir(signal_type), trade_date), ensure_ascii=False),
            })

    if not alerts_to_insert:
        print("ℹ️ 本次掃描沒有觸發任何買賣訊號")
        return

    # UNIQUE KEY(user_email, ticker, signal_type, trade_date) 避免同一天同一則訊號被重複寫入；
    # 同一天重跑時只更新卡片內容(payload)，不動已讀狀態與進場價
    with engine.begin() as conn:
        for a in alerts_to_insert:
            conn.execute(text("""
                INSERT INTO WatchlistAlerts (user_email, ticker, stock_name, signal_type, signal_detail, trade_date, payload)
                VALUES (:user_email, :ticker, :stock_name, :signal_type, :signal_detail, :trade_date, :payload)
                ON DUPLICATE KEY UPDATE payload = VALUES(payload)
            """), a)

    print(f"✅ 掃描完成，共觸發 {len(alerts_to_insert)} 則買賣訊號通知")


if __name__ == "__main__":
    run_scan()
