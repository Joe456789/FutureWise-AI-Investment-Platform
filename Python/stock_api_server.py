# 程式名稱：FutureWise 正式版後端 API 伺服器 (MySQL 雲端版 + AI 預測)
from fastapi import FastAPI, HTTPException, Depends, status
from pydantic import BaseModel
from passlib.context import CryptContext
import jwt
from fastapi.security import OAuth2PasswordBearer
from datetime import timedelta
from typing import List, Optional
from fastapi.middleware.cors import CORSMiddleware
from fastapi import UploadFile, File
import shutil
from sqlalchemy import create_engine, text
import pandas as pd
import uvicorn
from datetime import datetime, date
import os
import xgboost as xgb
import asyncio
import time
import yfinance as yf
from fastapi.staticfiles import StaticFiles

current_dir = os.path.dirname(os.path.abspath(__file__))

# 確保上傳資料夾存在，否則存檔會報錯
os.makedirs(os.path.join(current_dir, "uploads"), exist_ok=True)

# 引入連線字串與自訂模組
from config import MYSQL_CONN_STR, GEMINI_API_KEY, JWT_SECRET_KEY
import google.generativeai as genai
import sentiment_crawler

# 建立全域資料庫連線池 (Connection Pool)
# pool_pre_ping=True 可確保每次連線前進行 PING 測試，防止連線閒置斷開導致 API 當機
engine = create_engine(MYSQL_CONN_STR, pool_pre_ping=True)

# 設定 Gemini API
if GEMINI_API_KEY and GEMINI_API_KEY != "在這裡填入您的金鑰":
    genai.configure(api_key=GEMINI_API_KEY)

app = FastAPI(title="FutureWise AI API Terminal", version="3.0.0")

# 允許前端讀取上傳的圖片
app.mount("/uploads", StaticFiles(directory=os.path.join(current_dir, "uploads")), name="uploads")

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

# 網頁版：跟 App 共用同一個後端，掛在 /web 底下，避免跟根目錄的健康檢查、/api/* 路由打架
# 對應本機的 attachments/ 資料夾，上傳到伺服器時放進這裡指定的 web 資料夾
web_dir = os.path.join(current_dir, "web")
if not os.path.exists(web_dir):
    os.makedirs(web_dir)
app.mount("/web", StaticFiles(directory=web_dir, html=True), name="web")

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
    user_email: Optional[str] = None
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

class PostEditData(BaseModel):
    content: str

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
    print("🧠 成功載入 FutureWise 終極通用模型大腦！")
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
# 認證與登入系統 (JWT)
# ==========================================
if not JWT_SECRET_KEY:
    raise RuntimeError(
        "環境變數 JWT_SECRET_KEY 未設定。請在 .env 檔案加入一行 JWT_SECRET_KEY=一串隨機字串"
        "（例如用 python -c \"import secrets; print(secrets.token_hex(32))\" 產生）"
    )
SECRET_KEY = JWT_SECRET_KEY
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7  # 7 天

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")

def get_password_hash(password):
    return pwd_context.hash(password)

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
    except jwt.PyJWTError:
        raise credentials_exception
    return email

class UserRegister(BaseModel):
    email: str
    password: str
    name: Optional[str] = None
    phone: Optional[str] = None

class UserLogin(BaseModel):
    email: str
    password: str

@app.post("/api/auth/register")
def register_user(user: UserRegister):
    with engine.connect() as conn:
        res = conn.execute(text("SELECT id FROM Users WHERE email = :e"), {"e": user.email}).fetchone()
        if res:
            raise HTTPException(status_code=400, detail="Email already registered")

        hashed_pw = get_password_hash(user.password)
        conn.execute(
            text("INSERT INTO Users (email, password_hash, name, phone) VALUES (:e, :p, :n, :ph)"),
            {"e": user.email, "p": hashed_pw, "n": user.name, "ph": user.phone}
        )
        conn.commit()
    return {"status": "success", "message": "User registered successfully"}

@app.get("/api/user/me")
def get_my_profile(current_user: str = Depends(get_current_user)):
    with engine.connect() as conn:
        res = conn.execute(
            text("SELECT email, name, phone FROM Users WHERE email = :e"), {"e": current_user}
        ).fetchone()
        if not res:
            raise HTTPException(status_code=404, detail="User not found")
        return {"email": res[0], "name": res[1], "phone": res[2]}

class UserProfileUpdate(BaseModel):
    name: Optional[str] = None
    phone: Optional[str] = None

@app.put("/api/user/me")
def update_my_profile(payload: UserProfileUpdate, current_user: str = Depends(get_current_user)):
    with engine.connect() as conn:
        conn.execute(
            text("UPDATE Users SET name = :n, phone = :ph WHERE email = :e"),
            {"n": payload.name, "ph": payload.phone, "e": current_user}
        )
        conn.commit()
    return {"status": "success", "message": "Profile updated"}

class PasswordChange(BaseModel):
    old_password: str
    new_password: str

@app.put("/api/user/password")
def change_password(payload: PasswordChange, current_user: str = Depends(get_current_user)):
    with engine.connect() as conn:
        res = conn.execute(
            text("SELECT password_hash FROM Users WHERE email = :e"), {"e": current_user}
        ).fetchone()
        if not res or not verify_password(payload.old_password, res[0]):
            raise HTTPException(status_code=401, detail="舊密碼不正確")

        new_hash = get_password_hash(payload.new_password)
        conn.execute(
            text("UPDATE Users SET password_hash = :p WHERE email = :e"),
            {"p": new_hash, "e": current_user}
        )
        conn.commit()
    return {"status": "success", "message": "Password updated"}

# ==========================================
# 我的收藏
# ==========================================
@app.get("/api/favorites")
def get_favorites(current_user: str = Depends(get_current_user)):
    with engine.connect() as conn:
        rows = conn.execute(
            text("SELECT ticker, stock_name, created_at FROM Favorites WHERE user_email = :e ORDER BY created_at DESC"),
            {"e": current_user}
        ).fetchall()
        return [{"ticker": r[0], "stock_name": r[1], "created_at": r[2].isoformat() if r[2] else None} for r in rows]

class FavoriteAdd(BaseModel):
    stock_name: Optional[str] = None

@app.post("/api/favorites/{ticker}")
def add_favorite(ticker: str, payload: FavoriteAdd = FavoriteAdd(), current_user: str = Depends(get_current_user)):
    with engine.connect() as conn:
        conn.execute(
            text("""
                INSERT INTO Favorites (user_email, ticker, stock_name) VALUES (:e, :t, :n)
                ON DUPLICATE KEY UPDATE stock_name = VALUES(stock_name)
            """),
            {"e": current_user, "t": ticker, "n": payload.stock_name}
        )
        conn.commit()
    return {"status": "success", "message": f"{ticker} 已加入收藏"}

@app.delete("/api/favorites/{ticker}")
def remove_favorite(ticker: str, current_user: str = Depends(get_current_user)):
    with engine.connect() as conn:
        conn.execute(
            text("DELETE FROM Favorites WHERE user_email = :e AND ticker = :t"),
            {"e": current_user, "t": ticker}
        )
        conn.commit()
    return {"status": "success", "message": f"{ticker} 已移除收藏"}

# ==========================================
# 瀏覽歷史
# ==========================================
class HistoryAdd(BaseModel):
    stock_name: Optional[str] = None

@app.post("/api/history/{ticker}")
def add_history(ticker: str, payload: HistoryAdd = HistoryAdd(), current_user: str = Depends(get_current_user)):
    with engine.connect() as conn:
        conn.execute(
            text("INSERT INTO ViewHistory (user_email, ticker, stock_name) VALUES (:e, :t, :n)"),
            {"e": current_user, "t": ticker, "n": payload.stock_name}
        )
        conn.commit()
    return {"status": "success"}

@app.get("/api/history")
def get_history(current_user: str = Depends(get_current_user)):
    with engine.connect() as conn:
        rows = conn.execute(
            text("SELECT ticker, stock_name, viewed_at FROM ViewHistory WHERE user_email = :e ORDER BY viewed_at DESC LIMIT 100"),
            {"e": current_user}
        ).fetchall()
        return [{"ticker": r[0], "stock_name": r[1], "viewed_at": r[2].isoformat() if r[2] else None} for r in rows]

# ==========================================
# 通知偏好設定（App 內偏好開關，尚未串接推播）
# ==========================================
@app.get("/api/notifications/settings")
def get_notification_settings(current_user: str = Depends(get_current_user)):
    with engine.connect() as conn:
        res = conn.execute(
            text("SELECT ai_alert, community_reply, system_announce FROM NotificationSettings WHERE user_email = :e"),
            {"e": current_user}
        ).fetchone()
        if not res:
            # 尚未設定過，回傳預設值（全開）
            return {"ai_alert": True, "community_reply": True, "system_announce": True}
        return {"ai_alert": bool(res[0]), "community_reply": bool(res[1]), "system_announce": bool(res[2])}

class NotificationSettingsUpdate(BaseModel):
    ai_alert: bool = True
    community_reply: bool = True
    system_announce: bool = True

@app.put("/api/notifications/settings")
def update_notification_settings(payload: NotificationSettingsUpdate, current_user: str = Depends(get_current_user)):
    with engine.connect() as conn:
        conn.execute(
            text("""
                INSERT INTO NotificationSettings (user_email, ai_alert, community_reply, system_announce)
                VALUES (:e, :a, :c, :s)
                ON DUPLICATE KEY UPDATE ai_alert = VALUES(ai_alert), community_reply = VALUES(community_reply),
                                        system_announce = VALUES(system_announce)
            """),
            {"e": current_user, "a": payload.ai_alert, "c": payload.community_reply, "s": payload.system_announce}
        )
        conn.commit()
    return {"status": "success", "message": "Notification settings updated"}

# ==========================================
# 客服工單
# ==========================================
class SupportTicketCreate(BaseModel):
    subject: str
    message: str

@app.post("/api/support")
def create_support_ticket(payload: SupportTicketCreate, current_user: str = Depends(get_current_user)):
    with engine.connect() as conn:
        conn.execute(
            text("INSERT INTO SupportTickets (user_email, subject, message) VALUES (:e, :s, :m)"),
            {"e": current_user, "s": payload.subject, "m": payload.message}
        )
        conn.commit()
    return {"status": "success", "message": "已收到您的問題，我們會盡快回覆"}

# ==========================================
# 教學中心：閱讀進度
# ==========================================
@app.post("/api/education/read/{article_id}")
def mark_article_read(article_id: str, current_user: str = Depends(get_current_user)):
    with engine.connect() as conn:
        conn.execute(
            text("""
                INSERT INTO ArticleReadLog (user_email, article_id) VALUES (:e, :a)
                ON DUPLICATE KEY UPDATE read_at = read_at
            """),
            {"e": current_user, "a": article_id}
        )
        conn.commit()
    return {"status": "success"}

@app.get("/api/education/progress")
def get_reading_progress(current_user: str = Depends(get_current_user)):
    with engine.connect() as conn:
        rows = conn.execute(
            text("SELECT article_id FROM ArticleReadLog WHERE user_email = :e"), {"e": current_user}
        ).fetchall()
        return {"read_article_ids": [r[0] for r in rows]}

@app.post("/api/auth/login")
def login_user(user: UserLogin):
    with engine.connect() as conn:
        res = conn.execute(text("SELECT password_hash FROM Users WHERE email = :e"), {"e": user.email}).fetchone()
        if not res:
            raise HTTPException(status_code=401, detail="Invalid credentials")
        hashed_pw = res[0]
        if not verify_password(user.password, hashed_pw):
            raise HTTPException(status_code=401, detail="Invalid credentials")
            
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": user.email}, expires_delta=access_token_expires
        )
        return {"access_token": access_token, "token_type": "bearer"}

# ==========================================
# API 路由區塊
# ==========================================
@app.get("/")
def read_root():
    return {"status": "Online", "message": "FutureWise API Server is running safely on Oracle Cloud."}

class GeminiQuery(BaseModel):
    query: str
    ticker: Optional[str] = None

@app.post("/api/gemini/chat")
async def gemini_chat(data: GeminiQuery, current_user: str = Depends(get_current_user)):
    if not GEMINI_API_KEY or GEMINI_API_KEY == "在這裡填入您的金鑰":
        raise HTTPException(status_code=500, detail="Gemini API Key 尚未設定，請至 config.py 修改")
    
    context_str = ""
    if data.ticker:
        ticker = data.ticker.strip()
        context_str = f"【FutureWise 系統實時偵測到個股代號：{ticker}】\n"
        
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

            # --- 新增：抓取最新新聞標題放入提示詞 ---
            import yfinance as yf
            try:
                tk_news = yf.Ticker(f"{ticker}.TW")
                news_list = tk_news.news
                if not news_list:
                    tk_news = yf.Ticker(f"{ticker}.TWO")
                    news_list = tk_news.news
                    
                if news_list:
                    context_str += "- 近期最新新聞摘要：\n"
                    for n in news_list[:3]:  # 取最新3筆新聞
                        title = n.get("title", "")
                        publisher = n.get("publisher", "")
                        if title:
                            context_str += f"  * {title} (來源: {publisher})\n"
            except Exception as ne:
                pass # 若新聞抓取失敗則忽略
            # -----------------------------------
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
                    f"- FutureWise XGBoost AI 短線預測：{prediction_result} (上漲勝率: {up_confidence:.2f}%)\n"
                    f"- 輿情情緒指標：{sentiment:.2f} (影響調整分: {sentiment_adj*100:+.2f}%)\n"
                )
        except Exception as e:
            context_str += f"- (AI 預測與價格資料擷取失敗：{str(e)})\n"

    # 組合 System Prompt 餵給 Gemini
    # 設計重點：先直接回答問題本身，再用1~2個重點數據佐證，避免一般人看不懂的資料堆疊
    system_prompt = (
        "你是 FutureWise 的 AI 助理，要讓完全不懂投資的一般人也看得懂你的回答。\n"
        "回答規則：\n"
        "1. 第一句話一定要先直接回答使用者的問題本身。例如被問「會不會漲」，第一句就要明確講看漲、看跌、或持平，不要先鋪陳數據、不要顧左右而言他。\n"
        "2. 接著用 1~3 句白話文說明主要原因，只挑最相關的 1~2 個數據佐證（例如 AI 預測信心指數、近期漲跌趨勢），不要把下方提供的所有數據逐項列出來。\n"
        "3. 不要使用條列式或項目符號（不要用「-」開頭或數字列點），一律寫成完整的口語化句子，最重要的關鍵字可以用 **粗體** 標示。\n"
        "4. 全部使用繁體中文（台灣習慣用語）。\n"
        "5. 如果使用者的問題跟股票投資無關，禮貌說明你只回答投資相關問題即可，不用長篇大論。\n"
        "6. 結尾可視情況補一句簡短風險提醒（例如「僅供參考，不代表保證」），不用每次長篇強調。\n\n"
    )
    
    if context_str:
        system_prompt += f"[FutureWise 系統實時偵測個股數據]\n{context_str}\n\n"
        
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
def trigger_sentiment_crawler(ticker: str, current_user: str = Depends(get_current_user)):
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

@app.get("/api/coverage_stats")
def get_coverage_stats():
    try:
        query = """
            SELECT COUNT(DISTINCT ticker) AS total, MAX(trade_date) AS updated_at
            FROM StockPrice WHERE ticker NOT IN ('TSE', 'OTC')
        """
        with engine.connect() as conn:
            df = pd.read_sql(text(query), conn)
        row = df.iloc[0]
        return {
            "total": int(row['total']) if row['total'] is not None else 0,
            "updated_at": str(row['updated_at'])[:10] if row['updated_at'] is not None else None,
        }
    except Exception as e:
        print(f"❌ [API] 涵蓋範圍統計錯誤：{e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/search_stock")
def search_stock(q: str, limit: int = 8):
    q = (q or "").strip()
    if not q:
        return {"data": []}
    try:
        query = """
            SELECT DISTINCT ticker, stock_name FROM StockPrice
            WHERE trade_date = (SELECT MAX(trade_date) FROM StockPrice)
            AND (ticker LIKE :q OR stock_name LIKE :q)
            LIMIT :limit
        """
        with engine.connect() as conn:
            df = pd.read_sql(text(query), conn, params={"q": f"%{q}%", "limit": limit})
        result = [{"ticker": row['ticker'], "name": row['stock_name']} for _, row in df.iterrows()]
        return {"data": result}
    except Exception as e:
        print(f"❌ [API] 股票搜尋錯誤：{e}")
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
                elif isinstance(value, float) and (value != value or value in (float("inf"), float("-inf"))):
                    # NaN 或 Infinity 不是合法 JSON，通常出現在指數(TSE/OTC)這種沒有正常成交量的資料，指標算不出來
                    row[key] = None

        return {"ticker": ticker, "count": len(result), "data": result}

    except HTTPException:
        raise
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
def get_ai_prediction(ticker: str, current_user: str = Depends(get_current_user)):
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

TREND_7D_CACHE_FILE = os.path.join(current_dir, "trend_7days_cache.json")
TREND_7D_CACHE_TTL = 3600  # AutoGluon 推論一次要花好幾秒，同一股票同一小時內直接吃快取
# 用檔案存快取，而不是 Python 記憶體變數：因為伺服器是多 worker process 模式，
# 記憶體變數各個 process 互不相通，寫進檔案大家才能共用同一份快取。

def _load_trend_7d_cache():
    import json
    try:
        with open(TREND_7D_CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def _save_trend_7d_cache(cache):
    import json
    try:
        with open(TREND_7D_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(cache, f)
    except Exception as e:
        print(f"⚠️ 寫入 trend_7days 快取失敗：{e}")

@app.get("/api/trend_7days/{ticker}")
def get_trend_7days(ticker: str):
    cache = _load_trend_7d_cache()
    cached = cache.get(ticker)
    if cached and (time.time() - cached["ts"] < TREND_7D_CACHE_TTL):
        return cached["data"]

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
            
        response = {"ticker": ticker, "predictions": result}
        # 重新讀一次最新的快取檔再寫入，避免蓋掉其他 worker 這期間幫別支股票寫入的結果
        fresh_cache = _load_trend_7d_cache()
        fresh_cache[ticker] = {"data": response, "ts": time.time()}
        _save_trend_7d_cache(fresh_cache)
        return response

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
# 重大訊息（AI分析頁的新聞列表用）
# ==========================================
@app.get("/api/material_news/{ticker}")
def get_material_news(ticker: str, limit: int = 20):
    try:
        query = """
            SELECT ticker, company_name, announce_date, announce_time, subject, detail
            FROM MaterialNews
            WHERE ticker = :ticker
            ORDER BY announce_date DESC, announce_time DESC
            LIMIT :limit
        """
        with engine.connect() as conn:
            df = pd.read_sql(text(query), conn, params={"ticker": ticker, "limit": limit})

        result = []
        for _, row in df.iterrows():
            result.append({
                "date": str(row['announce_date'])[:10],
                "time": row['announce_time'],
                "subject": row['subject'],
                "detail": row['detail'],
            })
        return {"count": len(result), "data": result}
    except Exception as e:
        print(f"❌ [API] 重大訊息讀取錯誤：{e}")
        raise HTTPException(status_code=500, detail=str(e))

# ==========================================
# Gemini AI 深度解析 API
# ==========================================
@app.get("/api/gemini_analysis/{ticker}")
async def get_gemini_analysis(ticker: str, current_user: str = Depends(get_current_user)):
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
        query_posts = f"SELECT id, {user_col} as [user], user_email, icon, created_at, sentiment, tag, content, likes, comments FROM Posts ORDER BY created_at DESC"
        # MSSQL alias 語法稍微不同，修正為更通用的方式
        if "mysql" in MYSQL_CONN_STR.lower():
            query_posts = "SELECT id, user, user_email, icon, created_at, sentiment, tag, content, likes, comments FROM Posts ORDER BY created_at DESC"
        
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
                "user_email": row['user_email'] if pd.notna(row['user_email']) else None,
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
def create_post(post: Post, current_user: str = Depends(get_current_user)):
    try:
        user_col = "[user]" if "mssql" in MYSQL_CONN_STR.lower() else "user"
        with engine.begin() as conn:
            # 發文者身分一律以登入帳號查到的真實姓名為準，不信任前端送來的 user 欄位
            name_res = conn.execute(text("SELECT name FROM Users WHERE email = :e"), {"e": current_user}).fetchone()
            display_name = (name_res[0] if name_res and name_res[0] else current_user)

            conn.execute(text(f"""
                INSERT INTO Posts ({user_col}, user_email, icon, created_at, sentiment, tag, content, likes, comments)
                VALUES (:user, :user_email, :icon, :created_at, :sentiment, :tag, :content, 0, 0)
            """), {
                "user": display_name,
                "user_email": current_user,
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
def toggle_like(post_id: int, action: LikeAction, current_user: str = Depends(get_current_user)):
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
def delete_post(post_id: int, current_user: str = Depends(get_current_user)):
    try:
        with engine.begin() as conn:
            owner_res = conn.execute(text("SELECT user_email FROM Posts WHERE id = :id"), {"id": post_id}).fetchone()
            if not owner_res:
                raise HTTPException(status_code=404, detail="找不到該貼文")
            if owner_res[0] != current_user:
                raise HTTPException(status_code=403, detail="只能刪除自己發布的貼文")

            # 檢查並刪除貼文 (Replies 會因為 ON DELETE CASCADE 自動刪除)
            conn.execute(text("DELETE FROM Posts WHERE id = :id"), {"id": post_id})
        return {"status": "success", "message": "貼文已刪除"}
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ 刪除貼文失敗: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/api/posts/{post_id}")
def edit_post(post_id: int, data: PostEditData, current_user: str = Depends(get_current_user)):
    try:
        with engine.begin() as conn:
            owner_res = conn.execute(text("SELECT user_email FROM Posts WHERE id = :id"), {"id": post_id}).fetchone()
            if not owner_res:
                raise HTTPException(status_code=404, detail="找不到該貼文")
            if owner_res[0] != current_user:
                raise HTTPException(status_code=403, detail="只能編輯自己發布的貼文")

            conn.execute(text("UPDATE Posts SET content = :content WHERE id = :id"), {"content": data.content, "id": post_id})
        return {"status": "success", "message": "貼文已更新"}
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ 編輯貼文失敗: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/posts/{post_id}/reply")
def add_reply(post_id: int, reply: ReplyData, current_user: str = Depends(get_current_user)):
    try:
        with engine.begin() as conn:
            # 回覆者身分一律以登入帳號查到的真實姓名為準，不信任前端送來的 author 欄位
            name_res = conn.execute(text("SELECT name FROM Users WHERE email = :e"), {"e": current_user}).fetchone()
            display_name = (name_res[0] if name_res and name_res[0] else current_user)

            # 插入回覆 (交給資料庫自動產生時間，避免格式不相容)
            conn.execute(text("""
                INSERT INTO Replies (post_id, author, content)
                VALUES (:post_id, :author, :content)
            """), {
                "post_id": post_id,
                "author": display_name,
                "content": reply.content
            })
            # 更新主貼文回覆數 (使用不同的參數名稱以防驅動程式報錯)
            conn.execute(text("UPDATE Posts SET comments = (SELECT COUNT(*) FROM Replies WHERE post_id = :id1) WHERE id = :id2"), {"id1": post_id, "id2": post_id})
            return {"status": "success", "message": "留言儲存成功"}
    except Exception as e:
        print(f"❌ 儲存回覆失敗: {e}")
        raise HTTPException(status_code=500, detail=f"儲存回覆失敗: {str(e)}")

MAX_UPLOAD_SIZE = 5 * 1024 * 1024  # 5MB
ALLOWED_UPLOAD_TYPES = {"image/jpeg", "image/png", "image/gif", "image/webp"}

@app.post("/api/upload")
async def upload_image(file: UploadFile = File(...), current_user: str = Depends(get_current_user)):
    try:
        # 驗證檔案類型，避免有人上傳可執行檔或其他非圖片內容
        if file.content_type not in ALLOWED_UPLOAD_TYPES:
            raise HTTPException(status_code=400, detail="只允許上傳 JPG / PNG / GIF / WEBP 圖片")

        contents = await file.read()
        if len(contents) > MAX_UPLOAD_SIZE:
            raise HTTPException(status_code=400, detail="圖片大小不能超過 5MB")

        # 生成唯一檔名
        import uuid
        file_ext = file.filename.split(".")[-1] if "." in file.filename else "jpg"
        file_name = f"{uuid.uuid4().hex}.{file_ext}"
        file_path = os.path.join(current_dir, "uploads", file_name)

        with open(file_path, "wb") as buffer:
            buffer.write(contents)

        # 回傳圖片的公開 URL
        return {"status": "success", "url": f"/uploads/{file_name}"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ==========================================
# 社群檢舉與封鎖
# ==========================================
class ReportCreate(BaseModel):
    reason: str

@app.post("/api/posts/{post_id}/report")
def report_post(post_id: int, payload: ReportCreate, current_user: str = Depends(get_current_user)):
    with engine.connect() as conn:
        conn.execute(
            text("INSERT INTO Reports (reporter_email, target_type, target_id, reason) VALUES (:e, 'post', :id, :r)"),
            {"e": current_user, "id": post_id, "r": payload.reason}
        )
        conn.commit()
    return {"status": "success", "message": "已收到您的檢舉，我們會盡快處理"}

@app.post("/api/replies/{reply_id}/report")
def report_reply(reply_id: int, payload: ReportCreate, current_user: str = Depends(get_current_user)):
    with engine.connect() as conn:
        conn.execute(
            text("INSERT INTO Reports (reporter_email, target_type, target_id, reason) VALUES (:e, 'reply', :id, :r)"),
            {"e": current_user, "id": reply_id, "r": payload.reason}
        )
        conn.commit()
    return {"status": "success", "message": "已收到您的檢舉，我們會盡快處理"}

@app.get("/api/blocks")
def get_blocked_users(current_user: str = Depends(get_current_user)):
    with engine.connect() as conn:
        rows = conn.execute(
            text("SELECT blocked_email FROM Blocks WHERE blocker_email = :e"), {"e": current_user}
        ).fetchall()
        return {"blocked": [r[0] for r in rows]}

@app.post("/api/blocks/{target_email}")
def block_user(target_email: str, current_user: str = Depends(get_current_user)):
    if target_email == current_user:
        raise HTTPException(status_code=400, detail="不能封鎖自己")
    with engine.connect() as conn:
        conn.execute(
            text("""
                INSERT INTO Blocks (blocker_email, blocked_email) VALUES (:e, :t)
                ON DUPLICATE KEY UPDATE blocked_email = VALUES(blocked_email)
            """),
            {"e": current_user, "t": target_email}
        )
        conn.commit()
    return {"status": "success", "message": "已封鎖該使用者"}

@app.delete("/api/blocks/{target_email}")
def unblock_user(target_email: str, current_user: str = Depends(get_current_user)):
    with engine.connect() as conn:
        conn.execute(
            text("DELETE FROM Blocks WHERE blocker_email = :e AND blocked_email = :t"),
            {"e": current_user, "t": target_email}
        )
        conn.commit()
    return {"status": "success", "message": "已解除封鎖"}

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

# ==========================================
# 選股策略 API (讀取背景掃描快取)
# ==========================================
@app.get("/api/screener/{strategy_name}")
def get_screener_results(strategy_name: str):
    import json
    cache_path = os.path.join(current_dir, "strategy_cache.json")
    try:
        if not os.path.exists(cache_path):
            return {"status": "error", "message": "尚未建立掃描快取，請稍候或手動觸發背景掃描。"}
            
        with open(cache_path, "r", encoding="utf-8") as f:
            cache_data = json.load(f)
            
        strategies = cache_data.get("strategies", {})
        if strategy_name not in strategies:
            return {"status": "error", "message": f"未知的策略名稱: {strategy_name}"}
            
        return {
            "status": "success",
            "updated_at": cache_data.get("updated_at"),
            "data": strategies[strategy_name]
        }
    except Exception as e:
        return {"status": "error", "message": f"讀取快取失敗: {str(e)}"}

@app.get("/api/ai_top_picks")
def get_ai_top_picks():
    import json
    cache_path = os.path.join(current_dir, "ai_top_picks_cache.json")
    try:
        if not os.path.exists(cache_path):
            return {"status": "error", "message": "尚未建立AI精選快取，請稍候或確認排程是否已執行過 ai_top_picks_scanner.py"}

        with open(cache_path, "r", encoding="utf-8") as f:
            cache_data = json.load(f)

        return {
            "status": "success",
            "updated_at": cache_data.get("updated_at"),
            "scanned_count": cache_data.get("scanned_count"),
            "top_bullish": cache_data.get("top_bullish", []),
            "top_bearish": cache_data.get("top_bearish", [])
        }
    except Exception as e:
        return {"status": "error", "message": f"讀取快取失敗: {str(e)}"}

@app.get("/api/feature_importance")
def get_feature_importance():
    import json
    path = os.path.join(current_dir, "feature_importance.json")
    try:
        if not os.path.exists(path):
            return {"status": "error", "message": "尚未產生特徵重要性資料，請確認 Universal Trainer.py 是否已執行過"}

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return {
            "status": "success",
            "updated_at": data.get("updated_at"),
            "accuracy": data.get("accuracy"),
            "features": data.get("features", [])
        }
    except Exception as e:
        return {"status": "error", "message": f"讀取失敗: {str(e)}"}

@app.get("/api/prediction_accuracy")
def get_prediction_accuracy(days: int = 30):
    try:
        with engine.connect() as conn:
            row = conn.execute(text("""
                SELECT COUNT(*) AS total, SUM(is_correct) AS correct, MAX(predicted_at) AS latest_date
                FROM PredictionLog
                WHERE resolved = 1 AND predicted_at >= DATE_SUB(CURDATE(), INTERVAL :days DAY)
            """), {"days": days}).fetchone()

        total = row[0] or 0
        correct = row[1] or 0
        if total == 0:
            return {"status": "success", "sample_size": 0, "hit_rate": None, "days": days, "message": "尚無已驗證的預測資料"}

        return {
            "status": "success",
            "sample_size": total,
            "hit_rate": round(correct / total * 100, 2),
            "days": days,
            "latest_date": str(row[2]) if row[2] else None
        }
    except Exception as e:
        return {"status": "error", "message": f"讀取失敗: {str(e)}"}

# ==========================================
# 底部導覽列未讀紅點：比對前端傳來的「上次查看時間」，判斷有沒有新內容
# ==========================================
@app.get("/api/notifications/badge_status")
def get_badge_status(community_since: Optional[str] = None, replies_since: Optional[str] = None, current_user: str = Depends(get_current_user)):
    try:
        with engine.connect() as conn:
            has_new_posts = False
            if community_since:
                res = conn.execute(
                    text("SELECT COUNT(*) FROM Posts WHERE created_at > :since"),
                    {"since": community_since}
                ).fetchone()
                has_new_posts = (res[0] or 0) > 0

            has_new_replies = False
            if replies_since:
                res = conn.execute(
                    text("""
                        SELECT COUNT(*) FROM Replies r
                        JOIN Posts p ON r.post_id = p.id
                        WHERE p.user_email = :email AND r.created_at > :since
                    """),
                    {"email": current_user, "since": replies_since}
                ).fetchone()
                has_new_replies = (res[0] or 0) > 0

        return {
            "status": "success",
            "has_new_community_posts": has_new_posts,
            "has_new_replies_to_me": has_new_replies
        }
    except Exception as e:
        return {"status": "error", "message": f"讀取失敗: {str(e)}"}

if __name__ == "__main__":
    print("🚀 FutureWise 後端服務正在啟動... (支援高併發模式)")
    # 使用字串啟動以支援多進程 worker，預設開啟 4 個進程來處理萬人連線
    uvicorn.run("stock_api_server:app", host="0.0.0.0", port=8000, workers=4)