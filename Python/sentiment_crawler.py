# FutureWise 輿情分析系統 V5.0（Gemini 語意判讀版）
#
# 為什麼改版：V4.0 用關鍵字字典比對，測試下來效果不佳，原因：
#   1. PTT/Dcard 只比對「股票代碼數字」有沒有出現在標題裡，但討論通常用暱稱（護國神山等），幾乎抓不到文章
#   2. 關鍵字字典不懂否定句（"不創新高" 還是會被當成正面）
#   3. 訓練時的特徵重要性排行裡，情緒分數完全沒有上榜，代表對模型幾乎沒有貢獻
# 這版改成：抓到的新聞/論壇標題交給 Gemini（免費額度內）做整體語意判讀，而不是自己刻關鍵字表；
# 同時比對股票代碼「與」股票名稱兩種寫法，增加抓到相關文章的機率。

import json
import re
import time
import random

import requests
from bs4 import BeautifulSoup
import pandas as pd
from sqlalchemy import create_engine, text
import google.generativeai as genai

from config import MYSQL_CONN_STR, GEMINI_API_KEY

GEMINI_READY = bool(GEMINI_API_KEY) and GEMINI_API_KEY != "在這裡填入您的金鑰"
if GEMINI_READY:
    genai.configure(api_key=GEMINI_API_KEY)

SENTIMENT_MODEL = "gemini-2.5-flash"


# ==========================================
# 1. 各來源爬蟲：只負責「收集文字」，不再自己算分數
# ==========================================
def fetch_yahoo_news(ticker):
    """Yahoo 個股新聞頁本來就是該股專屬頁面，不需要額外篩選相關性"""
    url = f"https://tw.stock.yahoo.com/quote/{ticker}/news"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        res = requests.get(url, headers=headers, timeout=5)
        soup = BeautifulSoup(res.text, "html.parser")
        return [t.get_text().strip() for t in soup.find_all("h3") if t.get_text().strip()]
    except Exception:
        return []


def fetch_ptt_sentiment(ticker, name=None):
    """同時比對代碼與股票名稱，暱稱抓不到的問題不解，但至少正式名稱能命中"""
    url = "https://www.ptt.cc/bbs/Stock/index.html"
    headers = {"User-Agent": "Mozilla/5.0"}
    keywords = [k for k in [ticker, name] if k]
    try:
        res = requests.get(url, headers=headers, timeout=5)
        soup = BeautifulSoup(res.text, "html.parser")
        posts = soup.find_all("div", class_="title")
        return [p.get_text().strip() for p in posts if any(k in p.get_text() for k in keywords)]
    except Exception:
        return []


def fetch_dcard_sentiment(ticker, name=None):
    url = "https://www.dcard.tw/service/api/v2/forums/stock/posts?limit=50"
    headers = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}
    keywords = [k for k in [ticker, name] if k]
    try:
        res = requests.get(url, headers=headers, timeout=5)
        posts = res.json()
        result = []
        for post in posts:
            combined = f"{post.get('title', '')} {post.get('excerpt', '')}"
            if any(k in combined for k in keywords):
                result.append(combined.strip())
        return result
    except Exception as e:
        print("Dcard error:", e)
        return []


# ==========================================
# 2. 關鍵字備援評分（Gemini 沒設定金鑰或呼叫失敗時才會用到）
# ==========================================
FALLBACK_DICT = {
    "戰爭": -1.5, "崩盤": -1.5, "黑天鵝": -1.5, "降息": 0.8, "救市": 1.0, "升息": -0.5,
    "創高": 0.8, "大漲": 0.8, "獲利增": 0.8, "漲停": 0.8, "跌停": -0.8, "大跌": -0.8,
    "衰退": -0.8, "砍單": -0.8, "看好": 0.4, "買超": 0.4, "利多": 0.4, "看壞": -0.4, "利空": -0.4,
}

def fallback_sentiment(texts):
    if not texts:
        return 0.0
    scores = []
    for t in texts:
        if "？" in t or "?" in t or "嗎" in t:
            continue
        s = sum(w for k, w in FALLBACK_DICT.items() if k in t)
        scores.append(max(-2.0, min(2.0, s)))
    return round(sum(scores) / len(scores), 2) if scores else 0.0


# ==========================================
# 3. 用 Gemini 一次性判讀該股所有收集到的文字，回傳 -1.0 ~ 1.0 分數
# ==========================================
def gemini_sentiment(ticker, texts):
    if not texts:
        return 0.0
    if not GEMINI_READY:
        return fallback_sentiment(texts)

    sample = texts[:20]  # 避免單次 prompt 過長
    joined = "\n".join(f"- {t}" for t in sample)
    prompt = f"""你是台股分析助理。以下是關於股票代碼 {ticker} 的新聞標題與論壇貼文標題：
{joined}

請整體判斷這些內容對該股票的市場情緒，回傳一個 -1.0（極度負面）到 1.0（極度正面）之間的數字，0 代表中性或內容與該股無明顯關聯。
只回傳 JSON，格式：{{"score": 數字}}，不要有其他文字。"""

    try:
        model = genai.GenerativeModel(SENTIMENT_MODEL)
        response = model.generate_content(prompt)
        raw = response.text.strip()

        match = re.search(r"-?\d+\.?\d*", raw)
        if match:
            score = float(match.group())
            return round(max(-1.0, min(1.0, score)), 2)
        return 0.0
    except Exception as e:
        print(f"⚠️ Gemini 判讀失敗（{ticker}），改用關鍵字備援：{e}")
        return fallback_sentiment(texts)


def get_final_sentiment(ticker, name=None):
    news = fetch_yahoo_news(ticker)
    ptt = fetch_ptt_sentiment(ticker, name)
    dcard = fetch_dcard_sentiment(ticker, name)

    all_texts = news + ptt + dcard
    final_score = gemini_sentiment(ticker, all_texts)

    print(f"{ticker} → 蒐集 新聞:{len(news)} PTT:{len(ptt)} Dcard:{len(dcard)} 篇 => 情緒分數:{final_score}")
    return final_score


# ==========================================
# 4. 主執行（整合原本 DB 更新邏輯）
# ==========================================
def run_sentiment_crawler(target_ticker=None):
    print("🚀 啟動輿情分析系統 V5.0（Gemini 判讀版）...")
    if not GEMINI_READY:
        print("⚠️ 未設定 Gemini API Key，將全程使用關鍵字備援評分（準確度較低）")

    try:
        df = pd.read_csv("股票清單_Cloud.csv", dtype={'股票代碼': str})
        df["股票代碼"] = df["股票代碼"].str.replace('="', '').str.replace('"', '')

        # 建立代碼 -> 名稱對照，讓 PTT/Dcard 也能用正式股名比對
        name_col = "股票名稱" if "股票名稱" in df.columns else None
        name_map = {}
        if name_col:
            name_map = df.drop_duplicates("股票代碼").set_index("股票代碼")[name_col].to_dict()

        if target_ticker:
            tickers = [str(target_ticker)]
        else:
            # 比關鍵字版可以掃更多檔，因為爬蟲成本沒變，只有評分方式換了
            tickers = df["股票代碼"].unique()[:150]

        engine = create_engine(MYSQL_CONN_STR)
        sentiment_map = {}

        for t in tickers:
            score = get_final_sentiment(t, name_map.get(t))
            sentiment_map[t] = score
            time.sleep(random.uniform(1, 2))

        print("📊 更新資料中...")

        # 更新 CSV（維持跟爬蟲一致的備份用途）
        for t, s in sentiment_map.items():
            idx = df[df['股票代碼'] == t].last_valid_index()
            if idx is not None:
                df.at[idx, '情緒分數'] = s

        df["股票代碼"] = df["股票代碼"].apply(lambda x: f'="{x}"')
        df.to_csv("股票清單_Cloud.csv", index=False, encoding='utf-8-sig')

        # 更新 MySQL（維持原本安全語法，避免鎖表）
        with engine.begin() as conn:
            for t, s in sentiment_map.items():
                query = text(f"""
                    UPDATE StockPrice
                    SET Sentiment_Score = {s}
                    WHERE ticker = '{t}'
                    AND trade_date = (
                        SELECT max_date FROM (
                            SELECT MAX(trade_date) AS max_date FROM StockPrice WHERE ticker = '{t}'
                        ) AS tmp
                    )
                """)
                conn.execute(query)

        print("✅ 完成！")
        return {"status": "success"}

    except Exception as e:
        print("❌ 錯誤:", e)
        return {"status": "error", "message": str(e)}


if __name__ == "__main__":
    run_sentiment_crawler()
