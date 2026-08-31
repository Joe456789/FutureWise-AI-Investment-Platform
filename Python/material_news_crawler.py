# 程式名稱：ProQuant 重大訊息爬蟲
# 資料來源：證交所 OpenAPI（免費、官方、不需金鑰）
#   https://openapi.twse.com.tw/v1/opendata/t187ap04_L
# 這支只回傳「當天」公告的重大訊息，所以要排進 crontab 每天跑一次累積成歷史資料。

import requests
from sqlalchemy import create_engine, text
from datetime import datetime

from config import MYSQL_CONN_STR

API_URL = "https://openapi.twse.com.tw/v1/opendata/t187ap04_L"


def roc_to_date(roc_str: str):
    """把民國年字串 (例如 '1150827') 轉成西元日期字串 'YYYY-MM-DD'"""
    if not roc_str or len(roc_str) < 7:
        return None
    try:
        year = int(roc_str[:3]) + 1911
        month = int(roc_str[3:5])
        day = int(roc_str[5:7])
        return datetime(year, month, day).strftime("%Y-%m-%d")
    except Exception:
        return None


def fetch_material_news():
    print("📡 正在向證交所 OpenAPI 請求今日重大訊息...")
    resp = requests.get(API_URL, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    print(f"✅ 取得 {len(data)} 筆重大訊息公告")
    return data


def run_crawler():
    rows = fetch_material_news()
    if not rows:
        print("⚠️ 今日沒有新的重大訊息公告，結束")
        return

    engine = create_engine(MYSQL_CONN_STR)
    inserted = 0

    with engine.connect() as conn:
        for row in rows:
            ticker = (row.get("公司代號") or "").strip()
            company_name = (row.get("公司名稱") or "").strip()
            announce_date = roc_to_date(row.get("發言日期"))
            announce_time = (row.get("發言時間") or "").strip()
            subject = (row.get("主旨") or row.get("主旨 ") or "").replace("\r\n", " ").strip()
            detail = (row.get("說明") or "").replace("\r\n", "\n").strip()

            if not ticker or not announce_date or not subject:
                continue

            conn.execute(text("""
                INSERT INTO MaterialNews (ticker, company_name, announce_date, announce_time, subject, detail)
                VALUES (:ticker, :company_name, :announce_date, :announce_time, :subject, :detail)
                ON DUPLICATE KEY UPDATE detail = VALUES(detail)
            """), {
                "ticker": ticker,
                "company_name": company_name,
                "announce_date": announce_date,
                "announce_time": announce_time,
                "subject": subject[:500],
                "detail": detail,
            })
            inserted += 1
        conn.commit()

    print(f"💾 已寫入/更新 {inserted} 筆重大訊息到 MaterialNews")


if __name__ == "__main__":
    run_crawler()
