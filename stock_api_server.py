# 程式名稱：ProQuant 正式版後端 API 伺服器 (MySQL 雲端版 + AI 預測)
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi import UploadFile, File
import shutil
from sqlalchemy import create_engine, text
import pandas as pd
import uvicorn
from datetime import datetime, date
import os
from datetime import datetime, date
import xgboost as xgb
import asyncio
import time
import yfinance as yf

current_dir = os.path.dirname(os.path.abspath(__file__))

# 引入連線字串與自訂模組
from config import MYSQL_CONN_STR, GEMINI_API_KEY
import google.generativeai as genai
import sentiment_crawler

# 建立全域資料庫連線池 (Connection Pool)
# pool_pre_ping=True 可確保每次連線前進行 PING 測試，防止連線閒置斷開導致 API 當機
engine = create_engine(MYSQL_CONN_STR, pool_pre_ping=True)

# 設定 Gemini API
if GEMINI_API_KEY and GEMINI_API_KEY != "在這裡填入您的金鑰":
    genai.configure(api_key=GEMINI_API_KEY)

app = FastAPI(title="ProQuant AI API Terminal", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 確保 uploads 資料夾存在並掛載
uploads_dir = os.path.join(current_dir, "uploads")
if not os.path.exists(uploads_dir):
    os.makedirs(uploads_dir)
app.mount("/uploads", StaticFiles(directory=uploads_dir), name="uploads")

# ==========================================
# 社群論壇資料結構
# ==========================================
class ReplyData(BaseModel):
    author: str
    content: str
    time: str

class Post(BaseModel):
    id: int
    user: str
    icon: str
    time: int
    sentiment: str
    tag: str
    content: str
    likes: int
    comments: int
    replies: Optional[List[dict]] = []

class LikeAction(BaseModel):
    action: str

# (JSON 邏輯已移除，改用 SQL 資料庫)


# ==========================================
# 背景定時排程區塊 (自動輿情爬蟲)
# ==========================================
async def periodic_sentiment_crawler():
    # 伺服器啟動後先等待 10 秒再開始第一次掃描，避免跟啟動流程衝突
    await asyncio.sleep(10)
    while True:
        print("\n🕒 [背景排程] 開始執行全自動輿情分析...")
        try:
            # 因為 requests.get 會阻塞事件迴圈，使用 asyncio.to_thread 放入背景執行序跑
            await asyncio.to_thread(sentiment_crawler.run_sentiment_crawler)
            print("✅ [背景排程] 輿情更新完成，等待下次執行")
        except Exception as e:
            print(f"❌ [背景排程] 發生嚴重錯誤: {e}")
            
        # 根據系統時間自動切換頻率：早上-下午 (08:00 ~ 18:00) 30分鐘，晚上 1小時一次
        current_hour = datetime.now().hour
        if 8 <= current_hour < 18:
            interval = 1800  # 白天 30 分鐘
            print(f"⏳ [背景排程] 目前為白天活躍時段，下次爬蟲執行將於 30 分鐘後")
        else:
            interval = 3600  # 晚上 1 小時
            print(f"⏳ [背景排程] 目前為夜間休盤時段，下次爬蟲執行將於 1 小時後")
            
        await asyncio.sleep(interval)

@app.on_event("startup")
async def start_background_tasks():
    print("🚀 API 伺服器啟動中... (暫時關閉背景定時爬蟲以防記憶體不足當機)")
    # 暫時註解掉自動執行，避免 Oracle 雲端 1GB RAM 被 CSV 灌爆當機
    # asyncio.create_task(periodic_sentiment_crawler())

# ==========================================
# AI 模型初始化區塊
# ==========================================
model = xgb.XGBClassifier()

try:
    model.load_model(os.path.join(current_dir, "model_universal.json"))
    print("🧠 成功載入 ProQuant 終極通用模型大腦！")
except Exception as e:
    print(f"⚠️ 模型載入失敗，請確認 model_universal.json 檔案是否存在。錯誤：{e}")

try:
    from autogluon.timeseries import TimeSeriesPredictor
    ag_predictor = TimeSeriesPredictor.load(os.path.join(current_dir, "autogluon_7days_model"))
    print("📈 成功載入 AutoGluon 7天走勢預測模型！")
except Exception as e:
    ag_predictor = None
    print(f"⚠️ AutoGluon 模型載入失敗，API 將暫停提供 7 天精確預測。錯誤: {e}")

# 必須與 Universal Trainer.py 訓練時完全一致的特徵清單
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

# ⚠️ 重要：中英對照翻譯字典 (請依據你 MySQL 實際的英文欄位名稱進行修改)
# 左邊是資料庫的英文欄位名，右邊是模型需要的中文特徵名
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
    "ma5_slope": "5日均線_斜率",
    "ma20_slope": "20日均線_斜率",
    "ma60_slope": "60日均線_斜率",
    "revenue_yoy": "營收YoY",
    "pe_ratio": "本益比",
    "pb_ratio": "股價淨值比",
    "volume": "成交量"
}

# ==========================================
# API 路由區塊
# ==========================================
@app.get("/")
def read_root():
    return {"status": "Online", "message": "ProQuant API Server is running safely on Oracle Cloud."}

class GeminiQuery(BaseModel):
    query: str
    ticker: Optional[str] = None

@app.post("/api/gemini/chat")
async def gemini_chat(data: GeminiQuery):
    if not GEMINI_API_KEY or GEMINI_API_KEY == "在這裡填入您的金鑰":
        raise HTTPException(status_code=500, detail="Gemini API Key 尚未設定，請至 config.py 修改")
    
    context_str = ""
    if data.ticker:
        ticker = data.ticker.strip()
        context_str = f"【ProQuant 系統實時偵測到個股代號：{ticker}】\n"
        
        # 1. 抓取基本面數據 (使用 yfinance)
        try:
            info = await asyncio.to_thread(fetch_yf_info, ticker)
            name = info.get("longName") or info.get("shortName") or ticker
            pe = info.get("trailingPE", "無資料")
            pb = info.get("priceToBook", "無資料")
            roe = f"{info.get('returnOnEquity', 0) * 100:.2f}%" if info.get("returnOnEquity") else "無資料"
            fcf = f"${info.get('freeCashflow', 0):,}" if info.get("freeCashflow") else "無資料"
            insider = info.get("heldPercentInsiders", 0) * 100 if info.get("heldPercentInsiders") else 0.0
            inst = info.get("heldPercentInstitutions", 0) * 100 if info.get("heldPercentInstitutions") else 0.0
            
            context_str += (
                f"- 個股名稱：{name}\n"
                f"- 基本面估值：本益比 (PE) = {pe}, 股價淨值比 (PB) = {pb}, 股東權益報酬率 (ROE) = {roe}\n"
                f"- 現金流與健康度：自由現金流 (FCF) = {fcf}\n"
                f"- 籌碼結構：內部人持股 = {insider:.2f}%, 機構持股 = {inst:.2f}%\n"
            )
        except Exception as e:
            context_str += f"- (基本面資料擷取失敗：{str(e)})\n"
            
        # 2. 抓取 XGBoost AI 預測數據
        try:
            if engine.dialect.name == 'mssql':
                query = f"SELECT TOP 2 * FROM StockPrice WHERE ticker = '{ticker}' ORDER BY trade_date DESC"
            else:
                query = f"SELECT * FROM StockPrice WHERE ticker = '{ticker}' ORDER BY trade_date DESC LIMIT 2"
            
            with engine.connect() as conn:
                df = pd.read_sql(text(query), conn)
            
            if not df.empty:
                df = df.sort_values("trade_date")
                if 'MA_5' in df.columns: df['5日均線_斜率'] = df['MA_5'].pct_change()
                df = df.tail(1).copy()
                
                # 計算 AI 勝率
                df = df.rename(columns=COLUMN_MAPPING)
                for col in UNIVERSAL_FEATURES:
                    if col not in df.columns: df[col] = 0.0
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
                
                X_pred = df[UNIVERSAL_FEATURES]
                probabilities = model.predict_proba(X_pred)[0]
                base_up_confidence = probabilities[1]
                
                sentiment = df['情緒分數'].values[0] if '情緒分數' in df.columns else 0.0
                sentiment = 0.0 if pd.isna(sentiment) or sentiment is None else float(sentiment)
                sentiment_adj = sentiment * 0.1
                adjusted_up_confidence = max(0.01, min(0.99, base_up_confidence + sentiment_adj))
                up_confidence = adjusted_up_confidence * 100
                prediction_result = "看漲 🚀" if up_confidence >= 50 else "看跌 📉"
                
                close_p = df['收盤價'].values[0] if '收盤價' in df.columns else "無資料"
                
                context_str += (
                    f"- 最新收盤價：{close_p}\n"
                    f"- ProQuant XGBoost AI 短線預測：{prediction_result} (上漲勝率: {up_confidence:.2f}%)\n"
                    f"- 輿情情緒指標：{sentiment:.2f} (影響調整分: {sentiment_adj*100:+.2f}%)\n"
                )
        except Exception as e:
            context_str += f"- (AI 預測與價格資料擷取失敗：{str(e)})\n"

    # 組合 System Prompt 餵給 Gemini
    system_prompt = (
        "你現在是 ProQuant 專業量化投資顧問，負責回答使用者關於台股投資的問題。\n"
        "如果下方有提供個股的實時數據，請**務必結合並引用**這些數據為使用者分析、給出理性、客觀的量化分析與建議。\n"
        "請用繁體中文（台灣習慣用語）回答，並使用適當的 markdown 格式（如粗體、條列式）讓排版清晰易讀。\n\n"
    )
    
    if context_str:
        system_prompt += f"[ProQuant 系統實時偵測個股數據]\n{context_str}\n\n"
        
    full_prompt = f"{system_prompt}使用者提出的問題或指令：\n{data.query}"
    
    try:
        gemini_model = genai.GenerativeModel('gemini-2.5-flash')
        response = gemini_model.generate_content(full_prompt)
        return {"status": "success", "response": response.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

LEADERBOARD_CACHE = {"data": [], "timestamp": 0}

@app.get("/api/leaderboard")
def get_leaderboard(limit: int = 5):
    """
    動態讀取歷史紀錄，計算 AI 上漲機率最高的前 N 檔股票 (快取+批次運算極速版)
    """
    import numpy as np
    import time
    
    now = time.time()
    # 如果快取未過期 (1小時)，直接秒回傳
    if now - LEADERBOARD_CACHE.get("timestamp", 0) < 3600 and LEADERBOARD_CACHE.get("data"):
        return {"top_tickers": LEADERBOARD_CACHE["data"][:limit]}
        
    try:
        csv_path = os.path.join(current_dir, "股票清單_Cloud.csv")
        df_raw = pd.read_csv(csv_path, dtype={'股票代碼': str})
        df_raw['股票代碼'] = df_raw['股票代碼'].str.replace('="', '', regex=False).str.replace('"', '', regex=False)
        
        # 1. 高效取得最後 2 天資料 (避免對全表做 groupby)
        if '交易日期' in df_raw.columns:
            df_raw = df_raw.dropna(subset=['交易日期'])
            latest_dates = sorted(df_raw['交易日期'].unique())[-2:]
            df_recent = df_raw[df_raw['交易日期'].isin(latest_dates)].copy()
        else:
            df_recent = df_raw.groupby("股票代碼").tail(2).copy()
            
        df_recent = df_recent.sort_values(["股票代碼", "交易日期"])
        
        # 2. 向量化批次計算斜率 (瞬間完成)
        for ma_col in ["5日均線", "20日均線", "60日均線"]:
            if ma_col in df_recent.columns:
                df_recent[f"{ma_col}_斜率"] = df_recent.groupby("股票代碼")[ma_col].pct_change()
                
        # 取最新一天
        last_row_df = df_recent.groupby("股票代碼").tail(1).copy()
        
        feat_names = model.get_booster().feature_names or UNIVERSAL_FEATURES
        
        for f in feat_names:
            if f not in last_row_df.columns: 
                last_row_df[f] = 0.0
                
        # 確保所有特徵都是 float，避免含有字串導致 XGBoost 崩潰
        for f in feat_names:
            last_row_df[f] = pd.to_numeric(last_row_df[f], errors='coerce').fillna(0.0)
            
        X = last_row_df[feat_names]
        
        # 3. XGBoost 矩陣批次預測 (免除 for 迴圈，一次預測 2000 檔只要 0.01 秒)
        probs = model.predict_proba(X)[:, 1]
        last_row_df["prob"] = probs
        
        # 輿情疊加
        if '情緒分數' in last_row_df.columns:
            last_row_df["prob"] += last_row_df['情緒分數'].fillna(0).astype(float) * 0.1
            last_row_df["prob"] = last_row_df["prob"].clip(0.01, 0.99)
            
        # 排序並存入快取 (存前 100 名)
        top_list = last_row_df.sort_values("prob", ascending=False)["股票代碼"].head(100).tolist()
        
        LEADERBOARD_CACHE["data"] = top_list
        LEADERBOARD_CACHE["timestamp"] = now
        
        return {"top_tickers": top_list[:limit]}
    except Exception as e:
        print(f"❌ Leaderboard 計算錯誤: {e}")
        # 如果出錯但有舊快取，加減回傳，避免前端當掉
        if LEADERBOARD_CACHE.get("data"):
             return {"top_tickers": LEADERBOARD_CACHE["data"][:limit]}
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/sentiment/run_crawler/{ticker}")
def trigger_sentiment_crawler(ticker: str):
    print(f"📡 收到啟動輿情爬蟲請求 ({ticker})...")
    try:
        # 直接呼叫我們新建的 crawler function，並傳入要單獨爬取的股票代碼
        result = sentiment_crawler.run_sentiment_crawler(ticker)
        if result["status"] == "success":
            return {"status": "success", "message": "輿情分析與資料更新完成"}
        else:
            raise HTTPException(status_code=500, detail=result["message"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/stock/{ticker}")
def get_stock_data(ticker: str, limit: int = 200):
    print(f"📡 收到全數據請求：股票代號 {ticker}")
    try:
        if engine.dialect.name == 'mssql':
            query = f"SELECT TOP {limit} * FROM StockPrice WHERE ticker = '{ticker}' ORDER BY trade_date DESC"
        else:
            query = f"SELECT * FROM StockPrice WHERE ticker = '{ticker}' ORDER BY trade_date DESC LIMIT {limit}"
        
        with engine.connect() as conn:
            df = pd.read_sql(text(query), conn)
        
        if df.empty:
            raise HTTPException(status_code=404, detail="資料庫中找不到該股票代碼")

        df = df.sort_values("trade_date")
        result = df.to_dict(orient='records')
        
        for row in result:
            for key, value in row.items():
                if isinstance(value, (datetime, date)):
                    row[key] = str(value)
            
        return {"ticker": ticker, "count": len(result), "data": result}

    except Exception as e:
        print(f"❌ 錯誤：{e}")
        raise HTTPException(status_code=500, detail=str(e))

STOCK_INFO_CACHE = {}
CACHE_TTL = 3600  # 快取 1 小時

INDUSTRY_TRANSLATION = {
    "Semiconductors": "半導體",
    "Consumer Electronics": "消費性電子",
    "Electronic Components": "電子零組件",
    "Computer Hardware": "電腦硬體",
    "Information Technology Services": "資訊服務",
    "Software - Infrastructure": "軟體服務",
    "Communication Equipment": "通訊設備",
    "Auto Parts": "汽車零組件",
    "Banks - Regional": "銀行業",
    "Capital Markets": "資本市場",
    "Insurance - Life": "人壽保險",
    "Chemicals": "化學工業",
    "Building Products": "建材",
    "Airlines": "航空業",
    "Marine Shipping": "航運業"
}

def fetch_yf_info(ticker: str):
    import yfinance as yf
    
    # 讓最新版的 yfinance 自己處理 session 與防封鎖 (它會使用 curl_cffi)
    # 嘗試上市 (.TW)
    tk = yf.Ticker(f"{ticker}.TW")
    info = tk.info
    
    # 如果找不到基本資訊，嘗試上櫃 (.TWO)
    if "longName" not in info and "shortName" not in info:
        tk = yf.Ticker(f"{ticker}.TWO")
        info = tk.info
    return info

@app.get("/api/stock_info/{ticker}")
async def get_stock_fundamental_info(ticker: str):
    print(f"📊 收到個股基本面請求：股票代號 {ticker}")
    now = time.time()
    
    # 檢查快取
    if ticker in STOCK_INFO_CACHE:
        cached_data, timestamp = STOCK_INFO_CACHE[ticker]
        if now - timestamp < CACHE_TTL:
            return {"status": "success", "data": cached_data}
            
    try:
        info = await asyncio.to_thread(fetch_yf_info, ticker)
        
        raw_industry = info.get("industry", "未分類")
        
        data = {
            # --- 基本資料 ---
            "longName":   info.get("longName") or info.get("shortName") or ticker,
            "industry":   INDUSTRY_TRANSLATION.get(raw_industry, raw_industry),
            "sector":     info.get("sector", ""),
            "employees":  info.get("fullTimeEmployees", 0),
            "marketCap":  info.get("marketCap", 0),
            "beta":       info.get("beta", 0),
            # --- 除權息 ---
            "dividendYield":  info.get("dividendYield", 0),
            "dividendRate":   info.get("dividendRate", 0),
            "exDividendDate": info.get("exDividendDate", 0),
            # --- 估值 ---
            "trailingPE":  info.get("trailingPE", 0),
            "forwardPE":   info.get("forwardPE", 0),
            "priceToBook": info.get("priceToBook", 0),
            "trailingEps": info.get("trailingEps", 0),
            "forwardEps":  info.get("forwardEps", 0),
            # --- 獲利能力 ---
            "grossMargins":     info.get("grossMargins", 0),
            "operatingMargins": info.get("operatingMargins", 0),
            "profitMargins":    info.get("profitMargins", 0),
            "returnOnEquity":   info.get("returnOnEquity", 0),
            "returnOnAssets":   info.get("returnOnAssets", 0),
            # --- 年度成長率 (YoY) ---
            "revenueGrowth":  info.get("revenueGrowth", 0),
            "earningsGrowth": info.get("earningsGrowth", 0),
            # --- 現金流 ---
            "freeCashflow":      info.get("freeCashflow", 0),
            "operatingCashflow": info.get("operatingCashflow", 0),
            # --- 財務健全 ---
            "debtToEquity": info.get("debtToEquity", 0),
            "currentRatio":  info.get("currentRatio", 0),
            "totalCash":     info.get("totalCash", 0),
            "totalDebt":     info.get("totalDebt", 0),
            # --- 52 週區間 ---
            "fiftyTwoWeekHigh":     info.get("fiftyTwoWeekHigh", 0),
            "fiftyTwoWeekLow":      info.get("fiftyTwoWeekLow", 0),
            "fiftyDayAverage":      info.get("fiftyDayAverage", 0),
            "twoHundredDayAverage": info.get("twoHundredDayAverage", 0),
            # --- 分析師評估 ---
            "targetHighPrice":         info.get("targetHighPrice", 0),
            "targetLowPrice":          info.get("targetLowPrice", 0),
            "targetMeanPrice":         info.get("targetMeanPrice", 0),
            "recommendationMean":      info.get("recommendationMean", 0),
            "numberOfAnalystOpinions": info.get("numberOfAnalystOpinions", 0),
            # --- 大戶籌碼 ---
            "heldPercentInstitutions": info.get("heldPercentInstitutions", 0),
            "heldPercentInsiders":     info.get("heldPercentInsiders", 0),
        }
        
        STOCK_INFO_CACHE[ticker] = (data, now)
        return {"status": "success", "data": data}
        
    except Exception as e:
        print(f"❌ Yahoo Finance 請求錯誤：{e}")
        if ticker in STOCK_INFO_CACHE:
             return {"status": "success", "data": STOCK_INFO_CACHE[ticker][0]}
        
        # 建立防呆預設資料，防止因為 yfinance 遭雲端 IP 擋掉而導致整個 API 報 500 錯誤
        default_data = {
            "longName": ticker,
            "industry": "暫時無法撈取 (雲端IP受限)",
            "sector": "",
            "employees": 0,
            "marketCap": 0,
            "beta": 0,
            "dividendYield": 0,
            "dividendRate": 0,
            "exDividendDate": 0,
            "trailingPE": 0,
            "forwardPE": 0,
            "priceToBook": 0,
            "trailingEps": 0,
            "forwardEps": 0,
            "grossMargins": 0,
            "operatingMargins": 0,
            "profitMargins": 0,
            "returnOnEquity": 0,
            "returnOnAssets": 0,
            "revenueGrowth": 0,
            "earningsGrowth": 0,
            "freeCashflow": 0,
            "operatingCashflow": 0,
            "debtToEquity": 0,
            "currentRatio": 0,
            "totalCash": 0,
            "totalDebt": 0,
            "fiftyTwoWeekHigh": 0,
            "fiftyTwoWeekLow": 0,
            "fiftyDayAverage": 0,
            "twoHundredDayAverage": 0,
            "targetHighPrice": 0,
            "targetLowPrice": 0,
            "targetMeanPrice": 0,
            "recommendationMean": 0,
            "numberOfAnalystOpinions": 0,
            "heldPercentInstitutions": 0,
            "heldPercentInsiders": 0
        }
        return {"status": "success", "data": default_data}

@app.get("/api/prediction/{ticker}")
def get_ai_prediction(ticker: str):
    print(f"🧠 收到 AI 分析請求：股票代號 {ticker}")
    try:
        # 抓取最近 2 天來做明日預測與斜率計算
        if engine.dialect.name == 'mssql':
            query = f"SELECT TOP 2 * FROM StockPrice WHERE ticker = '{ticker}' ORDER BY trade_date DESC"
        else:
            query = f"SELECT * FROM StockPrice WHERE ticker = '{ticker}' ORDER BY trade_date DESC LIMIT 2"
        
        with engine.connect() as conn:
            df = pd.read_sql(text(query), conn)

        if df.empty:
            raise HTTPException(status_code=404, detail="無資料可供預測")

        # 反轉排序讓舊的日期在前面，以便 pct_change 正確計算
        df = df.sort_values("trade_date")

        # 動態計算斜率
        if 'MA_5' in df.columns: df['5日均線_斜率'] = df['MA_5'].pct_change()
        if 'MA_20' in df.columns: df['20日均線_斜率'] = df['MA_20'].pct_change()
        if 'MA_60' in df.columns: df['60日均線_斜率'] = df['MA_60'].pct_change()

        # 取最新一天
        df = df.tail(1).copy()

        # 將英文欄位轉換為模型需要的中文特徵
        df = df.rename(columns=COLUMN_MAPPING)

        # 檢查並補齊缺失的特徵欄位 (防呆機制)
        for col in UNIVERSAL_FEATURES:
            if col not in df.columns:
                df[col] = 0.0 
        
        # 提取模型要的特徵資料並安全轉換為浮點數
        # 使用 pd.to_numeric 強制轉換，無法轉換的會變成 NaN，然後再填補為 0
        for col in UNIVERSAL_FEATURES:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
            
        X_pred = df[UNIVERSAL_FEATURES]

        # 進行 XGBoost 原始預測
        probabilities = model.predict_proba(X_pred)[0]
        base_up_confidence = probabilities[1]
        
        # --- 專家法則 (Heuristic Overlay) 輿情強制修正 ---
        sentiment = df['情緒分數'].values[0] if '情緒分數' in df.columns else 0.0
        sentiment = 0.0 if pd.isna(sentiment) or sentiment is None else float(sentiment)
        sentiment_adj = sentiment * 0.1
        adjusted_up_confidence = max(0.01, min(0.99, base_up_confidence + sentiment_adj))

        up_confidence = adjusted_up_confidence * 100  # 轉換為百分比

        prediction_result = "看漲 🚀" if up_confidence >= 50 else "看跌 📉"

        # --- 產生買賣點分析 (AI 原因) ---
        def parse_val(val):
            if pd.isna(val) or val is None: return 0.0
            try:
                if isinstance(val, str):
                    val = val.replace(',', '').replace(' ', '').strip()
                    if val == '' or val == '-': return 0.0
                return float(val)
            except: return 0.0

        reasons = []
        # 技術面
        ma5 = parse_val(df['5日均線_斜率'].values[0] if '5日均線_斜率' in df.columns else 0)
        ma20 = parse_val(df['20日均線_斜率'].values[0] if '20日均線_斜率' in df.columns else 0)
        if ma5 > 0.01 and ma20 > 0:
            reasons.append("均線多頭排列")
        elif ma5 < -0.01 and ma20 < 0:
            reasons.append("均線空頭排列")
        elif ma5 > 0:
            reasons.append("短期均線上揚")
        elif ma5 < 0:
            reasons.append("短期均線下彎")

        # 籌碼面
        foreign = parse_val(df['外資買賣超'].values[0] if '外資買賣超' in df.columns else 0)
        trust = parse_val(df['投信買賣超'].values[0] if '投信買賣超' in df.columns else 0)
        
        if foreign > 0 and trust > 0:
            reasons.append("土洋法人同步買超")
        elif foreign < 0 and trust < 0:
            reasons.append("土洋法人同步賣超")
        elif foreign > 0:
            reasons.append("外資買盤進駐")
        elif foreign < 0:
            reasons.append("外資賣超調節")
        elif trust > 0:
            reasons.append("投信作帳行情")

        # 輿情面
        if sentiment >= 0.8:
            reasons.append("重大市場利多")
        elif sentiment >= 0.3:
            reasons.append("輿情偏多發酵")
        elif sentiment <= -0.8:
            reasons.append("重大市場利空")
        elif sentiment <= -0.3:
            reasons.append("輿情偏空發酵")

        if not reasons:
             reasons.append("量縮震盪整理")

        final_reasons = "、".join(reasons[:3])

        return {
            "ticker": ticker, 
            "prediction": prediction_result, 
            "confidence": round(float(up_confidence), 2),
            "base_confidence": round(float(base_up_confidence * 100), 2),
            "sentiment_adjustment": round(float(sentiment_adj * 100), 2),
            "message": final_reasons
        }

    except Exception as e:
        print(f"❌ 預測錯誤：{e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/trend_7days/{ticker}")
def get_trend_7days(ticker: str):
    print(f"📊 收到 AutoGluon 七天預測請求：股票代號 {ticker}")
    if ag_predictor is None:
        raise HTTPException(status_code=503, detail="AutoGluon 模型尚未準備妥當")
        
    try:
        from autogluon.timeseries import TimeSeriesDataFrame
        
        # 取得該股票最近的 60 天歷史資料供 AutoGluon 做時序背景推演
        # 改從資料庫讀取，確保預測基準點與今日同步 (捨棄舊 CSV)
        if engine.dialect.name == 'mssql':
            query = f"SELECT TOP 60 * FROM StockPrice WHERE ticker = '{ticker}' ORDER BY trade_date DESC"
        else:
            query = f"SELECT * FROM StockPrice WHERE ticker = '{ticker}' ORDER BY trade_date DESC LIMIT 60"
            
        with engine.connect() as conn:
            df = pd.read_sql(text(query), conn)
            
        if df.empty:
            raise HTTPException(status_code=404, detail="查無此股票歷史紀錄")
            
        # 為了時間序列，必須排序讓舊的日期在上面
        df = df.sort_values("trade_date")
        
        # 將資料庫英文欄位轉為模型訓練時的中文特徵
        df = df.rename(columns={
            'ticker': '股票代碼',
            'trade_date': '交易日期',
            'close_price': '收盤價',
            'volume': '成交量'
        })
        
        if '外資買賣超' not in df.columns:
            df['外資買賣超'] = 0.0
            
        # 轉換為 AutoGluon 需要的指定欄位結構
        df['交易日期'] = pd.to_datetime(df['交易日期'])
            
        keep_cols = ['股票代碼', '交易日期', '收盤價', '成交量', '外資買賣超']
        keep_cols_exist = [c for c in keep_cols if c in df.columns]
        df = df[keep_cols_exist]
        
        ts_data = TimeSeriesDataFrame.from_data_frame(
            df,
            id_column="股票代碼",
            timestamp_column="交易日期"
        )
        ts_data = ts_data.convert_frequency(freq='B')
        
        # 自動推演未來 7 天
        predictions = ag_predictor.predict(ts_data)
        
        # xs 可以獨立把單一股票的 multi-index 抽出來變成一般 DataFrame
        pred_df = predictions.xs(ticker, level="item_id")
        
        # 輸出的 index 是預測的未來日期, columns 預設會有 mean, 以及信賴區間
        result = []
        for dt, row in pred_df.iterrows():
            result.append({
                "date": str(dt.date()),
                "mean": round(float(row.get("mean", row.iloc[0])), 2),
                "lower": round(float(row.get("0.1", row.get("0.05", float(row.get("mean", 100))*0.95))), 2),
                "upper": round(float(row.get("0.9", row.get("0.95", float(row.get("mean", 100))*1.05))), 2)
            })
            
        return {"ticker": ticker, "predictions": result}
        
    except Exception as e:
        print(f"❌ AutoGluon 預測遭遇錯誤：{e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/institutional")
def get_institutional_data_default(limit: int = 30):
    # 保留給 selection.html 大盤預設使用
    return get_institutional_data("2330", limit)

@app.get("/api/institutional/{ticker}")
def get_institutional_data(ticker: str, limit: int = 30):
    print(f"\n📡 [API] 收到三大法人數據請求 ({ticker})...")
    try:
        if engine.dialect.name == 'mssql':
            query = f"SELECT TOP {limit} * FROM StockPrice WHERE ticker = '{ticker}' ORDER BY trade_date DESC"
        else:
            query = f"SELECT * FROM StockPrice WHERE ticker = '{ticker}' ORDER BY trade_date DESC LIMIT {limit}"
            
        with engine.connect() as conn:
            df = pd.read_sql(text(query), conn)
        
        if df.empty:
            print("⚠️ [API] 資料庫中找不到 TSE 的資料")
            raise HTTPException(status_code=404, detail="資料庫中找不到法人資料")

        df = df.sort_values("trade_date")
        
        # 🧪 診斷功能：印出資料庫中實際的欄位名稱
        all_cols = list(df.columns)
        print(f"📊 [診斷] 資料庫實際欄位：{all_cols}")

        # 🌟 強化版模糊比對：只要包含關鍵字就抓取
        def find_col(keywords):
            for c in all_cols:
                if any(k in c for k in keywords):
                    return c
            return None

        f_col = find_col(['外資', 'foreign', 'Foreign'])
        t_col = find_col(['投信', 'trust', 'SITC', 'Trust'])
        d_col = find_col(['自營', 'dealer', 'Dealer'])
        
        print(f"🔍 [診斷] 比對結果 -> 外資:{f_col}, 投信:{t_col}, 自營:{d_col}")

        # 數字清洗函數
        def safe_float(val):
            if pd.isna(val) or val is None: return 0.0
            try:
                if isinstance(val, str):
                    val = val.replace(',', '').replace(' ', '').strip()
                    if val == '' or val == '-': return 0.0
                return float(val)
            except: return 0.0

        result = []
        for _, row in df.iterrows():
            f_val = safe_float(row[f_col]) if f_col else 0.0
            t_val = safe_float(row[t_col]) if t_col else 0.0
            d_val = safe_float(row[d_col]) if d_col else 0.0
            
            result.append({
                "date": str(row['trade_date'])[:10],
                "foreign": f_val,
                "trust": t_val,
                "dealer": d_val,
                "total": f_val + t_val + d_val
            })
            
        return {"count": len(result), "data": result}

    except Exception as e:
        print(f"❌ [API] 法人資料讀取錯誤：{e}")
        raise HTTPException(status_code=500, detail=str(e))

# ==========================================
# Gemini AI 深度解析 API
# ==========================================
@app.get("/api/gemini_analysis/{ticker}")
async def get_gemini_analysis(ticker: str):
    print(f"🤖 收到 Gemini 深度解析請求：股票代號 {ticker}")
    if not GEMINI_API_KEY or GEMINI_API_KEY == "在這裡填入您的金鑰":
        raise HTTPException(status_code=503, detail="尚未設定 Gemini API Key。請在 config.py 中填入您的金鑰。")
        
    try:
        # 從資料庫撈取最新一日的資料
        if engine.dialect.name == 'mssql':
            query = f"SELECT TOP 1 * FROM StockPrice WHERE ticker = '{ticker}' ORDER BY trade_date DESC"
        else:
            query = f"SELECT * FROM StockPrice WHERE ticker = '{ticker}' ORDER BY trade_date DESC LIMIT 1"
            
        with engine.connect() as conn:
            df = pd.read_sql(text(query), conn)
            
        if df.empty:
            raise HTTPException(status_code=404, detail="查無此股票資料")
            
        latest_data = df.iloc[0]
        
        # 準備送給 Gemini 的 Prompt
        prompt = f"""
        你是一位專業的股市分析師，請針對台灣股市的股票代碼 {ticker} (名稱: {latest_data.get('stock_name', '未知')}) 提供一份專業且人性化的深度解析報告。
        
        以下是該股票最新的盤後數據：
        - 交易日期：{latest_data.get('trade_date', '未知')}
        - 收盤價：{latest_data.get('close_price', '未知')}
        - 漲跌幅：{latest_data.get('Change_1D', 0) * 100:.2f}%
        - 成交量：{latest_data.get('volume', '未知')}
        - 5日均線：{latest_data.get('MA_5', '未知')}
        - 20日均線：{latest_data.get('MA_20', '未知')}
        - 外資買賣超：{latest_data.get('Foreign_Buy', 0)} 張
        - 投信買賣超：{latest_data.get('Trust_Buy', 0)} 張
        - 自營商買賣超：{latest_data.get('Dealer_Buy', 0)} 張
        - RSI (14)：{latest_data.get('RSI_14', '未知')}
        - MACD 柱狀體：{latest_data.get('MACD_Hist', '未知')}
        
        請根據以上數據，使用繁體中文，並以 Markdown 格式輸出報告。報告必須包含以下三個部分：
        1. **總評摘要** (簡短總結目前的趨勢是多頭、空頭還是盤整)
        2. **技術面解析** (分析價格與均線的關係、RSI是否過熱/超賣、MACD訊號等)
        3. **籌碼面解析** (分析三大法人的動向對後市的影響)
        
        語氣請專業且客觀，並給出一個適當的短線操作建議(例如：逢低佈局、觀望、注意追高風險等)。
        """
        
        # 呼叫 Gemini
        model = genai.GenerativeModel('gemini-1.5-flash') # 使用 1.5-flash 版本，速度快且便宜
        # 因為這可能會花費幾秒鐘，使用 asyncio 將其放到執行緒池
        response = await asyncio.to_thread(model.generate_content, prompt)
        
        return {"status": "success", "analysis": response.text}
        
    except Exception as e:
        print(f"❌ Gemini API 發生錯誤: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==========================================
# 社群論壇 API 區塊
# ==========================================
# ==========================================
# 社群論壇 API 區塊 (SQL 版)
# ==========================================
@app.get("/api/posts", response_model=List[Post])
def get_posts():
    try:
        user_col = "[user]" if "mssql" in MYSQL_CONN_STR.lower() else "user"
        query_posts = f"SELECT id, {user_col} as [user], icon, created_at, sentiment, tag, content, likes, comments FROM Posts ORDER BY created_at DESC"
        # MSSQL alias 語法稍微不同，修正為更通用的方式
        if "mysql" in MYSQL_CONN_STR.lower():
            query_posts = "SELECT id, user, icon, created_at, sentiment, tag, content, likes, comments FROM Posts ORDER BY created_at DESC"
        
        with engine.connect() as conn:
            df_posts = pd.read_sql(text(query_posts), conn)
            
            # 抓取所有回覆
            query_replies = "SELECT id, post_id, author, content, created_at FROM Replies ORDER BY created_at ASC"
            df_replies = pd.read_sql(text(query_replies), conn)

        result = []
        for _, row in df_posts.iterrows():
            post_id = row['id']
            # 過濾該貼文的回覆
            post_replies = df_replies[df_replies['post_id'] == post_id].to_dict(orient='records')
            
            # 格式化回覆中的時間
            for r in post_replies:
                if isinstance(r['created_at'], (datetime, date)):
                    r['time'] = r['created_at'].strftime("%Y-%m-%d %H:%M")
                else:
                    r['time'] = str(r['created_at'])

            result.append({
                "id": post_id,
                "user": row['user'],
                "icon": row['icon'],
                "time": int(row['created_at'].timestamp() * 1000) if hasattr(row['created_at'], 'timestamp') else 0,
                "sentiment": row['sentiment'],
                "tag": row['tag'],
                "content": row['content'],
                "likes": row['likes'],
                "comments": row['comments'],
                "replies": post_replies
            })
        return result
    except Exception as e:
        print(f"❌ 讀取貼文失敗: {e}")
        return []

@app.post("/api/posts")
def create_post(post: Post):
    try:
        user_col = "[user]" if "mssql" in MYSQL_CONN_STR.lower() else "user"
        with engine.begin() as conn:
            conn.execute(text(f"""
                INSERT INTO Posts ({user_col}, icon, created_at, sentiment, tag, content, likes, comments)
                VALUES (:user, :icon, :created_at, :sentiment, :tag, :content, 0, 0)
            """), {
                "user": post.user,
                "icon": post.icon,
                "created_at": datetime.now(),
                "sentiment": post.sentiment,
                "tag": post.tag,
                "content": post.content
            })
        return {"status": "success"}
    except Exception as e:
        print(f"❌ 建立貼文失敗: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/posts/{post_id}/like")
def toggle_like(post_id: int, action: LikeAction):
    try:
        with engine.begin() as conn:
            if action.action == "like":
                conn.execute(text("UPDATE Posts SET likes = likes + 1 WHERE id = :id1"), {"id1": post_id})
            elif action.action == "unlike":
                conn.execute(text("UPDATE Posts SET likes = CASE WHEN likes > 0 THEN likes - 1 ELSE 0 END WHERE id = :id2"), {"id2": post_id})
            
            # 取得最新按讚數
            res = conn.execute(text("SELECT likes FROM Posts WHERE id = :id3"), {"id3": post_id}).fetchone()
            if res:
                return {"status": "success", "likes": res[0]}
        raise HTTPException(status_code=404, detail="找不到該留言")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/posts/{post_id}")
def delete_post(post_id: int):
    try:
        with engine.begin() as conn:
            # 檢查並刪除貼文 (Replies 會因為 ON DELETE CASCADE 自動刪除)
            res = conn.execute(text("DELETE FROM Posts WHERE id = :id"), {"id": post_id})
            if res.rowcount == 0:
                raise HTTPException(status_code=404, detail="找不到該貼文")
        return {"status": "success", "message": "貼文已刪除"}
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ 刪除貼文失敗: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/posts/{post_id}/reply")
def add_reply(post_id: int, reply: ReplyData):
    try:
        with engine.begin() as conn:
            # 插入回覆 (交給資料庫自動產生時間，避免格式不相容)
            conn.execute(text("""
                INSERT INTO Replies (post_id, author, content)
                VALUES (:post_id, :author, :content)
            """), {
                "post_id": post_id,
                "author": reply.author,
                "content": reply.content
            })
            # 更新主貼文回覆數 (使用不同的參數名稱以防驅動程式報錯)
            conn.execute(text("UPDATE Posts SET comments = (SELECT COUNT(*) FROM Replies WHERE post_id = :id1) WHERE id = :id2"), {"id1": post_id, "id2": post_id})
            return {"status": "success", "message": "留言儲存成功"}
    except Exception as e:
        print(f"❌ 儲存回覆失敗: {e}")
        raise HTTPException(status_code=500, detail=f"儲存回覆失敗: {str(e)}")

@app.post("/api/upload")
async def upload_image(file: UploadFile = File(...)):
    try:
        # 生成唯一檔名
        import uuid
        file_ext = file.filename.split(".")[-1] if "." in file.filename else "jpg"
        file_name = f"{uuid.uuid4().hex}.{file_ext}"
        file_path = os.path.join(current_dir, "uploads", file_name)
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # 回傳圖片的公開 URL
        return {"status": "success", "url": f"/uploads/{file_name}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/trending_tags")
def get_trending_tags():
    try:
        # 從最近 3 天的貼文中抓取最常出現的股票代號 (4位數字)
        with engine.connect() as conn:
            query = "SELECT tag, content FROM Posts WHERE created_at >= DATE_SUB(NOW(), INTERVAL 3 DAY)"
            # 相容 MSSQL 語法
            if "mssql" in MYSQL_CONN_STR.lower():
                query = "SELECT tag, content FROM Posts WHERE created_at >= DATEADD(day, -3, GETDATE())"
            
            df = pd.read_sql(text(query), conn)
            
        import re
        ticker_counts = {}
        for _, row in df.iterrows():
            text_content = str(row['tag']) + " " + str(row['content'])
            # 尋找 4 位數字
            tickers = re.findall(r'(?<!\d)([0-9]{4})(?!\d)', text_content)
            for t in tickers:
                ticker_counts[t] = ticker_counts.get(t, 0) + 1
                
        # 排序並取前 4 名
        sorted_tickers = sorted(ticker_counts.items(), key=lambda x: x[1], reverse=True)
        top_tags = [t[0] for t in sorted_tickers[:4]]
        
        # 如果討論太少，給予預設值
        defaults = ["2330", "2454", "2603", "3008"]
        for d in defaults:
            if len(top_tags) < 4 and d not in top_tags:
                top_tags.append(d)
                
        return {"status": "success", "trending": top_tags[:4]}
    except Exception as e:
        print(f"❌ 取得熱門標籤失敗: {e}")
        return {"status": "success", "trending": ["2330", "2454", "2603", "3008"]}

if __name__ == "__main__":
    print("🚀 ProQuant 後端服務正在啟動... (支援高併發模式)")
    # 使用字串啟動以支援多進程 worker，預設開啟 4 個進程來處理萬人連線
    uvicorn.run("stock_api_server:app", host="0.0.0.0", port=8000, workers=4)