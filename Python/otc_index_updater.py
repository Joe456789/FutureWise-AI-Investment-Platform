# 程式名稱：櫃買指數每日更新（獨立腳本）
# 背景：cloud_crawler_TWSE.py 原本用 yfinance 的 ^TWOII 代碼抓櫃買指數，
#       但這個代碼從 2026-07-17 之後就再也抓不到新資料，導致資料庫卡住不動、
#       前端看盤頁一直顯示一個多月前的舊資料當作「最新」。
#       改成用證券櫃檯買賣中心(TPEx)官方 OpenAPI，比依賴 Yahoo Finance 穩定。
# 用法：獨立於主爬蟲之外執行，不影響其他 2000+ 檔個股的抓取流程。

import requests
import pandas as pd
from sqlalchemy import create_engine, text
from datetime import datetime

from config import MYSQL_CONN_STR
from cloud_crawler_TWSE import calculate_all_indicators, save_batch_to_sql

API_URL = "https://www.tpex.org.tw/openapi/v1/tpex_index"


def roc_to_date(date_str: str):
    """把 TPEx API 回傳的日期字串轉成西元 date 物件。
    這個 openapi 實測回傳的是西元年8碼 (例如 '20260901')，
    但保留民國年7碼 (例如 '1150901') 的相容處理以防格式變動。"""
    s = str(date_str).strip()
    try:
        if len(s) == 8:
            return datetime.strptime(s, "%Y%m%d").date()
        elif len(s) == 7:
            year = int(s[:3]) + 1911
            month = int(s[3:5])
            day = int(s[5:7])
            return datetime(year, month, day).date()
        return None
    except Exception:
        return None


def fetch_today_otc():
    resp = requests.get(API_URL, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    if not data:
        raise RuntimeError("TPEx OpenAPI 回傳空資料")
    row = data[-1]
    trade_date = roc_to_date(row.get("Date"))
    if not trade_date:
        raise RuntimeError(f"日期格式無法解析：{row.get('Date')}")
    return {
        "交易日期": trade_date,
        "開盤價": float(row["Open"]),
        "最高價": float(row["High"]),
        "最低價": float(row["Low"]),
        "收盤價": float(row["Close"]),
        "成交量": 0,
        "漲跌": float(row["Change"]),
    }


def run():
    print("📡 正在向櫃買中心 OpenAPI 請求今日櫃買指數...")
    today_row = fetch_today_otc()
    print(f"✅ 取得 {today_row['交易日期']} 櫃買指數收盤：{today_row['收盤價']}")

    engine = create_engine(MYSQL_CONN_STR)

    # 讀取歷史資料，用來正確計算均線等技術指標（需要有足夠天數才能算出 MA_240）
    query = """
        SELECT trade_date AS 交易日期, open_price AS 開盤價, high_price AS 最高價,
               low_price AS 最低價, close_price AS 收盤價, volume AS 成交量
        FROM StockPrice WHERE ticker = 'OTC' ORDER BY trade_date
    """
    with engine.connect() as conn:
        hist_df = pd.read_sql(text(query), conn)

    if hist_df.empty:
        print("⚠️ 資料庫裡沒有任何 OTC 歷史資料，無法計算技術指標，中止")
        return

    hist_df["交易日期"] = pd.to_datetime(hist_df["交易日期"]).dt.date

    # 如果今天的資料已經存在（重複執行），先移除舊的再重算，避免重複
    if today_row["交易日期"] in set(hist_df["交易日期"]):
        print(f"ℹ️ {today_row['交易日期']} 的資料已經存在，將覆蓋更新")
        hist_df = hist_df[hist_df["交易日期"] != today_row["交易日期"]]

    combined = pd.concat([hist_df, pd.DataFrame([today_row])], ignore_index=True)
    combined["股票代碼"] = "OTC"
    combined["股票名稱"] = "櫃買指數"

    combined = calculate_all_indicators(combined)

    latest_row = combined.tail(1).copy()

    # 本機歷史資料在 7/17 之後有一段缺漏（舊版 yfinance 抓取失效期間），
    # 用本地資料算出的「前一筆」不是真正的前一交易日，pct_change 會失真。
    # TPEx API 自己就有算好真正正確的漲跌點數，直接拿來覆蓋，不依賴本地歷史的連續性。
    prev_close = today_row["收盤價"] - today_row["漲跌"]
    if prev_close:
        latest_row["漲跌幅_1日"] = today_row["漲跌"] / prev_close

    save_batch_to_sql([latest_row], engine)

    print("💾 櫃買指數已更新完成")


if __name__ == "__main__":
    run()
