# ProQuant 輿情分析系統 V4.0 (PTT + Dcard + 新聞融合版)

import requests
from bs4 import BeautifulSoup
import pandas as pd
from sqlalchemy import create_engine, text
import time
import random
from config import MYSQL_CONN_STR

# ==========================================
# 1. 情緒權重字典（保留我們稍早擴充的進化版）
# ==========================================
WEIGHTED_DICT = {
    # 第四級：系統性極端風險 / 極端總經 (-1.5 到 +1.0)
    "戰爭": -1.5, "開戰": -1.5, "空襲": -1.5, "制裁": -1.5, "崩盤": -1.5, 
    "血洗": -1.5, "黑天鵝": -1.5, "震災": -1.5, "斷電": -1.0, "降息": 0.8,
    "救市": 1.0, "護盤": 0.8, "停戰": 1.0, "升息": -0.5, "通膨": -0.5, "衰退風險": -1.0,

    # 第一級：已發生的硬事實 (+0.8 / -0.8)
    "創高": 0.8, "創新高": 0.8, "營收增": 0.8, "大漲": 0.8, "飆升": 0.8, "獲利增": 0.8, "優於預期": 0.8, "漲停": 0.8, "爆發": 0.8, "雙增": 0.8, "創紀錄": 0.8,
    "跌停": -0.8, "大跌": -0.8, "營收減": -0.8, "衰退": -0.8, "砍單": -0.8, "腰斬": -0.8, "低於預期": -0.8, "重挫": -0.8, "跳水": -0.8, "不如預期": -0.8,

    # 第二級：主觀分析與法人動向 (+0.4 / -0.4)
    "看好": 0.4, "買超": 0.4, "強勢": 0.4, "受惠": 0.4, "商機": 0.4, "利多": 0.4, "紅盤": 0.4, "加碼": 0.4, "目標價升": 0.4, "拉貨": 0.4,
    "看壞": -0.4, "賣超": -0.4, "弱勢": -0.4, "受害": -0.4, "利空": -0.4, "跌破": -0.4, "降評": -0.4, "減碼": -0.4, "目標價降": -0.4, "出貨": -0.4, "探底": -0.4,

    # 第三級：市場傳言與不確定臆測 (+0.1 / -0.1)
    "有望": 0.1, "預估": 0.1, "上看": 0.1, "看旺": 0.1, "轉機": 0.1, "可望": 0.1,
    "恐": -0.1, "傳言": -0.1, "傳出": -0.1, "據悉": -0.1, "疑": -0.1, "隱憂": -0.1, "雜音": -0.1
}

def analyze_sentiment(text):
    if not text:
        return 0
    
    # 保留我們剛剛新增的「嗎」問句防呆
    if "？" in text or "?" in text or "嗎" in text:
        return 0

    score = 0
    text_lower = text.lower()
    for word, weight in WEIGHTED_DICT.items():
        if word in text_lower:
            score += weight

    return max(-2.0, min(2.0, score))


# ==========================================
# 2. Yahoo 新聞
# ==========================================
def fetch_yahoo_news(ticker):
    url = f"https://tw.stock.yahoo.com/quote/{ticker}/news"
    headers = {"User-Agent": "Mozilla/5.0"}

    try:
        res = requests.get(url, headers=headers, timeout=5)
        soup = BeautifulSoup(res.text, "html.parser")
        titles = soup.find_all("h3")

        scores = []
        for t in titles:
            text = t.get_text()
            scores.append(analyze_sentiment(text))

        return sum(scores)/len(scores) if scores else 0

    except:
        return 0


# ==========================================
# 3. PTT 股版爬蟲
# ==========================================
def fetch_ptt_sentiment(ticker):
    url = "https://www.ptt.cc/bbs/Stock/index.html"
    headers = {"User-Agent": "Mozilla/5.0"}

    try:
        res = requests.get(url, headers=headers)
        soup = BeautifulSoup(res.text, "html.parser")

        posts = soup.find_all("div", class_="title")

        scores = []
        for p in posts:
            title = p.get_text()

            # 篩選有股票代碼或名稱的文章
            if ticker in title:
                scores.append(analyze_sentiment(title))

        return sum(scores)/len(scores) if scores else 0

    except:
        return 0


# ==========================================
# 4. Dcard 投資版 (簡化版 API)
# ==========================================
def fetch_dcard_sentiment(ticker):
    url = "https://www.dcard.tw/service/api/v2/forums/stock/posts?limit=50"
    
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json"
    }

    try:
        res = requests.get(url, headers=headers, timeout=5)
        posts = res.json()

        scores = []

        for post in posts:
            title = post.get("title", "")
            excerpt = post.get("excerpt", "")

            text = title + " " + excerpt

            # 🔥 關鍵：同時支援股票代碼 + 名稱
            if ticker in text:
                scores.append(analyze_sentiment(text))

        return sum(scores) / len(scores) if scores else 0

    except Exception as e:
        print("Dcard error:", e)
        return 0


# ==========================================
# 5. 多來源融合（重點）
# ==========================================
def get_final_sentiment(ticker):
    news = fetch_yahoo_news(ticker)
    ptt = fetch_ptt_sentiment(ticker)
    dcard = fetch_dcard_sentiment(ticker)

    # 👉 權重設計（論壇 > 新聞）
    final_score = (
        news * 0.3 +
        ptt * 0.4 +
        dcard * 0.3
    )

    print(f"{ticker} → News:{news:.2f} PTT:{ptt:.2f} Dcard:{dcard:.2f} => Final:{final_score:.2f}")

    return round(final_score, 2)


# ==========================================
# 6. 主執行（整合你原本 DB 更新）
# ==========================================
def run_sentiment_crawler(target_ticker=None):
    print("🚀 啟動多來源輿情分析系統...")

    try:
        df = pd.read_csv("股票清單_Cloud.csv", dtype={'股票代碼': str})
        df["股票代碼"] = df["股票代碼"].str.replace('="', '').str.replace('"', '')

        if target_ticker:
            tickers = [str(target_ticker)]
        else:
            tickers = df["股票代碼"].unique()[:50]

        engine = create_engine(MYSQL_CONN_STR)
        sentiment_map = {}

        for t in tickers:
            score = get_final_sentiment(t)
            sentiment_map[t] = score
            time.sleep(random.uniform(1, 2))

        print("📊 更新資料中...")

        # 更新 CSV
        for t, s in sentiment_map.items():
            idx = df[df['股票代碼'] == t].last_valid_index()
            if idx is not None:
                df.at[idx, '情緒分數'] = s

        df["股票代碼"] = df["股票代碼"].apply(lambda x: f'="{x}"')
        df.to_csv("股票清單_Cloud.csv", index=False, encoding='utf-8-sig')

        # 更新 MySQL (保留稍早修改的安全語法，避免 MySQL locked)
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
