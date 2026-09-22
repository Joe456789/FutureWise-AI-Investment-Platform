# 程式名稱：FutureWise 正式版後端 API 伺服器 (MySQL 雲端版 + AI 預測)
import re
import json
from fastapi import FastAPI, HTTPException, Depends, status
from pydantic import BaseModel
from passlib.context import CryptContext
import jwt
from fastapi.security import OAuth2PasswordBearer
from datetime import timedelta
from typing import List, Optional, Literal
from fastapi.middleware.cors import CORSMiddleware
from fastapi import UploadFile, File
import shutil
from sqlalchemy import create_engine, text, bindparam
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
from config import (MYSQL_CONN_STR, GEMINI_API_KEY, JWT_SECRET_KEY,
                    SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASSWORD, SMTP_FROM, PUBLIC_BASE_URL, EMAIL_ENABLED,
                    EMAIL_DAILY_LIMIT)
import google.generativeai as genai
import sentiment_crawler
import random
import secrets
import hashlib
import smtplib
from email.mime.text import MIMEText
from email.header import Header
from fastapi import BackgroundTasks
from auth_schema import ensure_auth_schema

# 建立全域資料庫連線池 (Connection Pool)
# pool_pre_ping=True 可確保每次連線前進行 PING 測試，防止連線閒置斷開導致 API 當機
engine = create_engine(MYSQL_CONN_STR, pool_pre_ping=True)

# 設定 Gemini API
if GEMINI_API_KEY and GEMINI_API_KEY != "在這裡填入您的金鑰":
    genai.configure(api_key=GEMINI_API_KEY)

app = FastAPI(title="FutureWise AI API Terminal", version="3.0.0")

# 頻率限制：記錄存在資料庫(RateLimits表)，不是記憶體——API是uvicorn多worker，
# 各worker的記憶體互相獨立，記憶體版限制會被多個worker平均稀釋，等於形同虛設。
# 用 SELECT ... FOR UPDATE 鎖住同一個key的那一列，多個worker同時進來也不會同時通過。
def _rl_key(*parts: str) -> str:
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()

def enforce_rate_limit(user_email: str, action: str, cooldown_seconds: int):
    """同一個key(使用者/Email)對同一個動作，要間隔至少cooldown_seconds秒才能再觸發，否則回429"""
    key = _rl_key("cooldown", action, user_email)
    now = time.time()
    wait = None
    try:
        with engine.begin() as conn:
            row = conn.execute(text("SELECT last_call FROM RateLimits WHERE rl_key = :k FOR UPDATE"), {"k": key}).fetchone()
            if row and now - float(row[0]) < cooldown_seconds:
                wait = max(1, int(cooldown_seconds - (now - float(row[0]))))
            else:
                conn.execute(text("""
                    INSERT INTO RateLimits (rl_key, last_call, hits) VALUES (:k, :n, 1)
                    ON DUPLICATE KEY UPDATE last_call = :n, hits = hits + 1
                """), {"k": key, "n": now})
            if random.random() < 0.01:  # 順手清掉兩天前的舊記錄，避免表無限成長
                conn.execute(text("DELETE FROM RateLimits WHERE last_call < :old"), {"old": now - 2 * 86400})
    except Exception as e:
        print(f"⚠️ 頻率限制檢查失敗，本次放行：{e}")  # 資料庫本身出問題時整個API都會壞，這裡寧可放行不要多一個失敗點
        return
    if wait is not None:
        raise HTTPException(status_code=429, detail=f"操作過於頻繁，請等待約 {wait} 秒後再試一次")

def reserve_daily_quota(action: str, limit: int, message: str):
    """全站每日總量上限（例如每天最多寄N封信），超過回503。跟單一使用者的冷卻時間互補：
    冷卻時間擋不住『每次換一個Email』的濫用，總量上限才擋得住"""
    key = _rl_key("daily", action, date.today().isoformat())
    now = time.time()
    over = False
    try:
        with engine.begin() as conn:
            row = conn.execute(text("SELECT hits FROM RateLimits WHERE rl_key = :k FOR UPDATE"), {"k": key}).fetchone()
            if row and int(row[0]) >= limit:
                over = True
            else:
                conn.execute(text("""
                    INSERT INTO RateLimits (rl_key, last_call, hits) VALUES (:k, :n, 1)
                    ON DUPLICATE KEY UPDATE last_call = :n, hits = hits + 1
                """), {"k": key, "n": now})
    except Exception as e:
        print(f"⚠️ 每日總量檢查失敗，本次放行：{e}")
        return
    if over:
        raise HTTPException(status_code=503, detail=message)

# 允許前端讀取上傳的圖片
app.mount("/uploads", StaticFiles(directory=os.path.join(current_dir, "uploads")), name="uploads")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    # allow_credentials=True 搭配 allow_origins=["*"] 是已知的CORS誤設組合（瀏覽器實際上會把
    # Access-Control-Allow-Origin動態回填成請求方網域，等於允許任何網站帶憑證跨源呼叫）。
    # 這個API的登入機制是Bearer Token（Authorization header），不是cookie session，
    # 前端從來沒有用到 credentials:'include'，所以關掉這個選項不影響任何現有功能。
    allow_credentials=False,
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
    action: Literal["like", "unlike"]

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
    try:
        ensure_auth_schema(engine)
    except Exception as e:
        print(f"❌ Email驗證/同意系統資料表建立失敗，註冊/登入相關功能可能異常：{e}")
    if not EMAIL_ENABLED:
        print("⚠️ 未設定 SMTP：不會強制Email驗證，忘記密碼功能停用。請在 .env 設定 SMTP_HOST/SMTP_USER/SMTP_PASSWORD")
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
    to_encode.update({"exp": expire, "iat": int(time.time())})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

# 登入憑證撤銷：JWT本身無法作廢，所以在Users.token_valid_after記一個時間點，
# 簽發時間(iat)早於這個時間點的憑證一律視為失效。改密碼/重設密碼時把它設成「現在」，
# 就能讓所有舊裝置的登入立刻(最多15秒內)失效。沒有iat的舊憑證視為iat=0，只在密碼被改過之後才會失效。
_token_valid_after_cache = {}  # {email: (token_valid_after, 快取到期時間)}

def _get_token_valid_after(email: str):
    now = time.time()
    cached = _token_valid_after_cache.get(email)
    if cached and cached[1] > now:
        return cached[0]
    with engine.connect() as conn:
        row = conn.execute(text("SELECT token_valid_after FROM Users WHERE email = :e"), {"e": email}).fetchone()
    if row is None:
        return None  # 帳號已不存在
    value = int(row[0] or 0)
    _token_valid_after_cache[email] = (value, now + 15)
    return value

def revoke_tokens_for(email: str) -> int:
    """讓這個帳號目前所有已簽發的登入憑證失效，回傳新的有效起點"""
    now = int(time.time())
    with engine.begin() as conn:
        conn.execute(text("UPDATE Users SET token_valid_after = :t WHERE email = :e"), {"t": now, "e": email})
    _token_valid_after_cache.pop(email, None)
    return now

def get_current_user_base(token: str = Depends(oauth2_scheme)):
    """只驗證登入憑證本身(簽章、期限、是否被撤銷)，不檢查條款同意狀態。同意頁自己的API用這個"""
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

    valid_after = _get_token_valid_after(email)
    if valid_after is None or int(payload.get("iat", 0)) < valid_after:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="登入已失效，請重新登入",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return email

# 通過同意檢查的使用者快取5分鐘（只快取「通過」，沒通過的每次都重查，
# 這樣使用者在某個worker同意完之後，下一個請求不管落在哪個worker都會立刻通過）
_consent_ok_cache = {}

def get_current_user(email: str = Depends(get_current_user_base)):
    """一般API用：登入憑證有效，而且已經同意目前最新版本的所有必要條款，否則回403 CONSENT_REQUIRED"""
    now = time.time()
    expires = _consent_ok_cache.get(email)
    if expires and expires > now:
        return email
    try:
        with engine.connect() as conn:
            pending = pending_required_consents(conn, email)
    except Exception as e:
        print(f"⚠️ 檢查同意狀態失敗，本次放行：{e}")
        return email
    if pending:
        raise HTTPException(status_code=403, detail={
            "code": "CONSENT_REQUIRED",
            "message": "條款已更新，請先閱讀並同意後再繼續使用",
        })
    _consent_ok_cache[email] = now + 300
    return email

class ConsentChoice(BaseModel):
    code: str
    version: int
    granted: bool

class UserRegister(BaseModel):
    email: str
    password: str
    name: Optional[str] = None
    phone: Optional[str] = None
    consents: List[ConsentChoice] = []

class UserLogin(BaseModel):
    email: str
    password: str

# 只允許一般Email字元。原本的寫法 [^\s@]+ 連 < > " 都放行，別人的Email會被顯示在封鎖名單、檢舉頁，等於能塞HTML
EMAIL_PATTERN = re.compile(r"^[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}$")

def reject_markup(value, label: str):
    """這類欄位(姓名、電話、標籤、檢舉原因)會被顯示給其他使用者，不需要任何HTML，直接拒絕 < > 字元。
    前端輸出時也會跳脫，這裡是第二道防線，讓惡意內容一開始就存不進資料庫"""
    if value is not None and ("<" in value or ">" in value):
        raise HTTPException(status_code=400, detail=f"{label}不能包含 < 或 > 符號")

ICON_PATTERN = re.compile(r"^fa-[a-z0-9-]+$")
AVATAR_URL_PATTERN = re.compile(r"^(/uploads/[A-Za-z0-9_.\-]+|https?://[^\s\"'<>]+)$")

# ------------------------------------------
# Email 寄信 / 一次性驗證連結（註冊驗證、忘記密碼共用）
# ------------------------------------------
def _hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()

def send_email(to_email: str, subject: str, html_body: str):
    """寄信（由BackgroundTasks在背景執行，寄失敗只記log，不影響API回應）"""
    try:
        msg = MIMEText(html_body, "html", "utf-8")
        msg["Subject"] = Header(subject, "utf-8")
        msg["From"] = SMTP_FROM
        msg["To"] = to_email
        if SMTP_PORT == 465:
            server = smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT, timeout=15)
        else:
            server = smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=15)
            server.starttls()
        with server:
            server.login(SMTP_USER, SMTP_PASSWORD)
            server.sendmail(SMTP_FROM, [to_email], msg.as_string())
        print(f"📧 已寄出「{subject}」給 {to_email}")
    except Exception as e:
        print(f"❌ 寄信失敗（{to_email}）：{e}")

def create_email_token(conn, email: str, purpose: str, ttl_minutes: int) -> str:
    """產生一次性token；資料庫只存雜湊值，同一人同一用途重新申請時，舊的未使用token直接作廢"""
    token = secrets.token_urlsafe(32)
    conn.execute(text("DELETE FROM EmailTokens WHERE email = :e AND purpose = :p AND used_at IS NULL"),
                 {"e": email, "p": purpose})
    conn.execute(text("""
        INSERT INTO EmailTokens (email, purpose, token_hash, expires_at)
        VALUES (:e, :p, :h, :x)
    """), {"e": email, "p": purpose, "h": _hash_token(token), "x": datetime.utcnow() + timedelta(minutes=ttl_minutes)})
    return token

def consume_email_token(conn, token: str, purpose: str) -> Optional[str]:
    """驗證並用掉token，成功回傳對應的email；不存在、已用過、已過期一律回傳None"""
    row = conn.execute(text(
        "SELECT id, email, expires_at, used_at FROM EmailTokens WHERE token_hash = :h AND purpose = :p"
    ), {"h": _hash_token(token), "p": purpose}).fetchone()
    if not row or row[3] is not None or row[2] < datetime.utcnow():
        return None
    conn.execute(text("UPDATE EmailTokens SET used_at = :n WHERE id = :i"), {"n": datetime.utcnow(), "i": row[0]})
    return row[1]

def _email_layout(title: str, body_html: str) -> str:
    return (f'<div style="font-family:Segoe UI,Arial,sans-serif;max-width:480px;margin:0 auto;padding:24px;'
            f'background:#0f172a;color:#e2e8f0;border-radius:12px;">'
            f'<h2 style="color:#38bdf8;margin-top:0;">{title}</h2>{body_html}'
            f'<p style="font-size:12px;color:#64748b;margin-top:24px;">若這不是您本人的操作，請忽略這封信，不會有任何變更。</p></div>')

EMAIL_QUOTA_MESSAGE = "今日系統寄信量已達上限，請明天再試，或聯繫客服協助"

def queue_verification_email(background_tasks: BackgroundTasks, conn, email: str):
    reserve_daily_quota("email", EMAIL_DAILY_LIMIT, EMAIL_QUOTA_MESSAGE)
    token = create_email_token(conn, email, "verify", 60 * 24)
    link = f"{PUBLIC_BASE_URL}/web/verify_email.html?token={token}"
    html = _email_layout("驗證您的 Email", (
        f'<p>歡迎加入 FutureWise AI！請點下方按鈕完成 Email 驗證（連結 24 小時內有效）：</p>'
        f'<p><a href="{link}" style="display:inline-block;padding:10px 20px;background:#0ea5e9;color:#fff;'
        f'border-radius:8px;text-decoration:none;font-weight:bold;">驗證我的 Email</a></p>'
        f'<p style="font-size:12px;color:#94a3b8;word-break:break-all;">按鈕無法點擊時，請複製此網址貼到瀏覽器：<br>{link}</p>'))
    background_tasks.add_task(send_email, email, "【FutureWise AI】請驗證您的 Email", html)

def queue_reset_email(background_tasks: BackgroundTasks, conn, email: str):
    reserve_daily_quota("email", EMAIL_DAILY_LIMIT, EMAIL_QUOTA_MESSAGE)
    token = create_email_token(conn, email, "reset", 60)
    link = f"{PUBLIC_BASE_URL}/web/reset_password.html?token={token}"
    html = _email_layout("重設您的密碼", (
        f'<p>我們收到重設密碼的申請，請點下方按鈕設定新密碼（連結 1 小時內有效，只能使用一次）：</p>'
        f'<p><a href="{link}" style="display:inline-block;padding:10px 20px;background:#f59e0b;color:#fff;'
        f'border-radius:8px;text-decoration:none;font-weight:bold;">重設密碼</a></p>'
        f'<p style="font-size:12px;color:#94a3b8;word-break:break-all;">按鈕無法點擊時，請複製此網址貼到瀏覽器：<br>{link}</p>'))
    background_tasks.add_task(send_email, email, "【FutureWise AI】重設密碼", html)

# ------------------------------------------
# 動態知情同意：條款內容存在資料庫(ConsentItems)，可以升版；
# 使用者的同意/撤回只新增不修改(UserConsents)，保留完整稽核軌跡
# ------------------------------------------
def get_active_consent_items(conn) -> list:
    """每個條款(code)取目前啟用中的最新版本"""
    rows = conn.execute(text("""
        SELECT c.code, c.version, c.title, c.summary, c.body, c.required
        FROM ConsentItems c
        JOIN (SELECT code, MAX(version) AS v FROM ConsentItems WHERE is_active = 1 GROUP BY code) m
          ON m.code = c.code AND m.v = c.version
        WHERE c.is_active = 1
        ORDER BY c.required DESC, c.id ASC
    """)).fetchall()
    return [{"code": r[0], "version": r[1], "title": r[2], "summary": r[3], "body": r[4], "required": bool(r[5])}
            for r in rows]

def get_user_consent_map(conn, email: str) -> dict:
    """{code: (最後一次表態的版本, 是否同意)}"""
    rows = conn.execute(text("""
        SELECT uc.code, uc.version, uc.granted FROM UserConsents uc
        JOIN (SELECT code, MAX(id) AS mid FROM UserConsents WHERE user_email = :e GROUP BY code) m ON m.mid = uc.id
    """), {"e": email}).fetchall()
    return {r[0]: (r[1], bool(r[2])) for r in rows}

def pending_required_consents(conn, email: str) -> list:
    """還沒同意目前最新版本的「必要」條款（新版本上線後，舊使用者會出現在這裡）"""
    current = get_user_consent_map(conn, email)
    return [i for i in get_active_consent_items(conn)
            if i["required"] and current.get(i["code"]) != (i["version"], True)]

def record_consents(conn, email: str, items: list, choices: list, skip_unchanged: bool):
    """依使用者的勾選寫入同意紀錄。choices裡的版本必須是目前最新版，避免拿舊版本條款來同意"""
    by_code = {i["code"]: i for i in items}
    current = get_user_consent_map(conn, email) if skip_unchanged else {}
    for c in choices:
        item = by_code.get(c.code)
        if item is None:
            raise HTTPException(status_code=400, detail=f"未知的條款：{c.code}")
        if c.version != item["version"]:
            raise HTTPException(status_code=409, detail=f"「{item['title']}」已更新為新版本，請重新整理頁面後再確認")
        if item["required"] and not c.granted:
            raise HTTPException(status_code=400, detail=f"「{item['title']}」為必要條款，無法撤回；如不同意請停止使用並申請刪除帳號")
        if skip_unchanged and current.get(c.code) == (c.version, c.granted):
            continue
        conn.execute(text(
            "INSERT INTO UserConsents (user_email, code, version, granted) VALUES (:e, :c, :v, :g)"
        ), {"e": email, "c": c.code, "v": c.version, "g": 1 if c.granted else 0})

@app.get("/api/consent/items")
def list_consent_items():
    """註冊頁用：目前生效中的所有條款（含全文）"""
    with engine.connect() as conn:
        return {"items": get_active_consent_items(conn)}

@app.get("/api/consent/me")
def get_my_consents(current_user: str = Depends(get_current_user_base)):
    with engine.connect() as conn:
        items = get_active_consent_items(conn)
        current = get_user_consent_map(conn, current_user)
    result = []
    for i in items:
        state = current.get(i["code"])
        accepted = state == (i["version"], True)
        result.append({
            **i,
            "granted": accepted if state and state[0] == i["version"] else None,  # None=這個版本還沒表態
            "needs_action": i["required"] and not accepted,
        })
    return {"items": result, "consent_required": any(r["needs_action"] for r in result)}

class ConsentUpdate(BaseModel):
    consents: List[ConsentChoice]

@app.post("/api/consent/me")
def update_my_consents(payload: ConsentUpdate, current_user: str = Depends(get_current_user_base)):
    with engine.begin() as conn:
        items = get_active_consent_items(conn)
        record_consents(conn, current_user, items, payload.consents, skip_unchanged=True)
    _consent_ok_cache.pop(current_user, None)  # 撤回選填授權不影響必要條款，但重新同意後要馬上重新評估
    return {"status": "success"}

@app.post("/api/auth/register")
def register_user(user: UserRegister, background_tasks: BackgroundTasks):
    # 後端也要驗證一次，不能只靠前端擋——前端的檢查繞得過去（例如直接呼叫API），
    # 密碼長度門檻跟register.html前端的檢查保持一致
    if not EMAIL_PATTERN.match(user.email):
        raise HTTPException(status_code=400, detail="Email格式不正確")
    if len(user.password) < 6:
        raise HTTPException(status_code=400, detail="密碼長度至少需要6碼")
    reject_markup(user.name, "姓名")
    reject_markup(user.phone, "電話")

    with engine.begin() as conn:
        # 必要條款一定要在「目前最新版本」上勾選同意，後端強制檢查，不能只靠前端把按鈕鎖起來
        items = get_active_consent_items(conn)
        chosen = {c.code: c for c in user.consents}
        for i in items:
            c = chosen.get(i["code"])
            if i["required"] and not (c and c.granted and c.version == i["version"]):
                raise HTTPException(status_code=400, detail=f"請先閱讀並同意「{i['title']}」")

        res = conn.execute(text("SELECT id FROM Users WHERE email = :e"), {"e": user.email}).fetchone()
        if res:
            raise HTTPException(status_code=400, detail="Email already registered")

        hashed_pw = get_password_hash(user.password)
        conn.execute(
            text("INSERT INTO Users (email, password_hash, name, phone, email_verified) VALUES (:e, :p, :n, :ph, :v)"),
            {"e": user.email, "p": hashed_pw, "n": user.name, "ph": user.phone, "v": 0 if EMAIL_ENABLED else 1}
        )
        # 沒勾的選填項目也記錄成「不同意」，之後才分得出是「拒絕」還是「沒被問過」
        full_choices = [chosen.get(i["code"]) or ConsentChoice(code=i["code"], version=i["version"], granted=False)
                        for i in items]
        record_consents(conn, user.email, items, full_choices, skip_unchanged=False)

        if EMAIL_ENABLED:
            queue_verification_email(background_tasks, conn, user.email)

    return {
        "status": "success",
        "verification_required": EMAIL_ENABLED,
        "message": "註冊成功，請到信箱點擊驗證連結後再登入" if EMAIL_ENABLED else "註冊成功",
    }

class EmailOnly(BaseModel):
    email: str

class TokenOnly(BaseModel):
    token: str

class ResetPassword(BaseModel):
    token: str
    new_password: str

@app.post("/api/auth/verify_email")
def verify_email(payload: TokenOnly):
    with engine.begin() as conn:
        email = consume_email_token(conn, payload.token, "verify")
        if not email:
            raise HTTPException(status_code=400, detail="驗證連結無效或已過期，請回登入頁重新寄送驗證信")
        conn.execute(text("UPDATE Users SET email_verified = 1 WHERE email = :e"), {"e": email})
    return {"status": "success", "message": "Email 驗證完成，現在可以登入了"}

@app.post("/api/auth/resend_verification")
def resend_verification(payload: EmailOnly, background_tasks: BackgroundTasks):
    if not EMAIL_ENABLED:
        raise HTTPException(status_code=503, detail="系統尚未啟用寄信功能")
    enforce_rate_limit(payload.email.strip().lower(), "resend_verification", 60)
    with engine.begin() as conn:
        row = conn.execute(text("SELECT email_verified FROM Users WHERE email = :e"), {"e": payload.email}).fetchone()
        if row and not row[0]:
            queue_verification_email(background_tasks, conn, payload.email)
    # 不論帳號存在與否、是否已驗證，都回一樣的訊息，避免被拿來探測哪些Email註冊過
    return {"status": "success", "message": "如果這個Email已註冊且尚未驗證，驗證信已重新寄出，請查看信箱（也請檢查垃圾郵件）"}

@app.post("/api/auth/forgot_password")
def forgot_password(payload: EmailOnly, background_tasks: BackgroundTasks):
    if not EMAIL_ENABLED:
        raise HTTPException(status_code=503, detail="系統尚未啟用寄信功能，請聯繫管理員協助重設密碼")
    if not EMAIL_PATTERN.match(payload.email):
        raise HTTPException(status_code=400, detail="Email格式不正確")
    enforce_rate_limit(payload.email.strip().lower(), "forgot_password", 60)
    with engine.begin() as conn:
        row = conn.execute(text("SELECT id FROM Users WHERE email = :e"), {"e": payload.email}).fetchone()
        if row:
            queue_reset_email(background_tasks, conn, payload.email)
    return {"status": "success", "message": "如果這個Email已註冊，重設密碼信已寄出，請查看信箱（也請檢查垃圾郵件）"}

@app.post("/api/auth/reset_password")
def reset_password(payload: ResetPassword):
    if len(payload.new_password) < 6:
        raise HTTPException(status_code=400, detail="密碼長度至少需要6碼")
    with engine.begin() as conn:
        email = consume_email_token(conn, payload.token, "reset")
        if not email:
            raise HTTPException(status_code=400, detail="重設連結無效或已過期，請重新申請")
        # 能收到信、點到連結，也等於證明了這個Email是本人的，順便標記為已驗證
        conn.execute(text("UPDATE Users SET password_hash = :p, email_verified = 1 WHERE email = :e"),
                     {"p": get_password_hash(payload.new_password), "e": email})
    revoke_tokens_for(email)  # 重設密碼的情境通常是懷疑帳號被盜，所以所有已登入的裝置一併登出
    return {"status": "success", "message": "密碼已重設，請用新密碼登入"}

@app.get("/api/user/me")
def get_my_profile(current_user: str = Depends(get_current_user)):
    with engine.connect() as conn:
        res = conn.execute(
            text("SELECT email, name, phone, avatar_url FROM Users WHERE email = :e"), {"e": current_user}
        ).fetchone()
        if not res:
            raise HTTPException(status_code=404, detail="User not found")
        return {"email": res[0], "name": res[1], "phone": res[2], "avatar_url": res[3]}

class UserProfileUpdate(BaseModel):
    name: Optional[str] = None
    phone: Optional[str] = None
    avatar_url: Optional[str] = None

@app.put("/api/user/me")
def update_my_profile(payload: UserProfileUpdate, current_user: str = Depends(get_current_user)):
    # 只更新請求裡真的有帶到的欄位，避免像之前那樣沒帶的欄位被無條件覆蓋成NULL
    # （例如只想換大頭貼，結果連name/phone都被清空）
    updates = payload.dict(exclude_unset=True, exclude_none=True)
    if not updates:
        return {"status": "success", "message": "Profile updated"}
    reject_markup(updates.get("name"), "姓名")
    reject_markup(updates.get("phone"), "電話")
    if "avatar_url" in updates and not AVATAR_URL_PATTERN.match(updates["avatar_url"]):
        raise HTTPException(status_code=400, detail="大頭貼網址格式不正確")

    set_clause = ", ".join(f"{k} = :{k}" for k in updates)
    with engine.connect() as conn:
        conn.execute(
            text(f"UPDATE Users SET {set_clause} WHERE email = :e"),
            {**updates, "e": current_user}
        )
        conn.commit()
    return {"status": "success", "message": "Profile updated"}

class PasswordChange(BaseModel):
    old_password: str
    new_password: str

@app.put("/api/user/password")
def change_password(payload: PasswordChange, background_tasks: BackgroundTasks, current_user: str = Depends(get_current_user)):
    if len(payload.new_password) < 6:
        raise HTTPException(status_code=400, detail="新密碼長度至少需要6碼")
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

    # 改密碼後，所有其他裝置/舊的登入憑證立刻失效；目前這個裝置換發一張新的，前端要存起來，才不會被自己登出
    revoke_tokens_for(current_user)
    new_token = create_access_token(
        data={"sub": current_user}, expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )

    # 密碼被改掉時通知本人：帳號若被別人偷登入後改密碼，本人至少收得到提醒
    if EMAIL_ENABLED:
        html = _email_layout("您的密碼剛剛被變更", (
            f'<p>您的 FutureWise AI 帳號密碼已於 {datetime.now().strftime("%Y-%m-%d %H:%M")}（伺服器時間）變更。</p>'
            f'<p>如果是您本人操作，不需要做任何事。</p>'
            f'<p style="color:#fca5a5;">如果不是您本人，請立即到登入頁點「忘記密碼？」重新設定密碼，並聯繫客服。</p>'))
        background_tasks.add_task(send_email, current_user, "【FutureWise AI】您的密碼已被變更", html)
    return {"status": "success", "message": "Password updated", "access_token": new_token, "token_type": "bearer"}

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
# 自選股買賣訊號通知（由 watchlist_alert_scanner.py 排程寫入）
# ==========================================
@app.get("/api/watchlist_alerts")
def get_watchlist_alerts(limit: int = 50, current_user: str = Depends(get_current_user)):
    limit = max(1, min(limit, 200))
    with engine.connect() as conn:
        rows = conn.execute(
            text("""
                SELECT id, ticker, stock_name, signal_type, signal_detail, trade_date, created_at, is_read, payload
                FROM WatchlistAlerts WHERE user_email = :e
                ORDER BY is_read ASC, created_at DESC LIMIT :limit
            """),
            {"e": current_user, "limit": limit}
        ).fetchall()
        unread_count = conn.execute(
            text("SELECT COUNT(*) FROM WatchlistAlerts WHERE user_email = :e AND is_read = 0"),
            {"e": current_user}
        ).scalar()

    def parse_payload(raw):
        # payload 是排程寫入的卡片內容(JSON字串)；舊的通知沒有這欄，前端會退回顯示一行文字
        if not raw:
            return None
        try:
            return json.loads(raw)
        except (TypeError, ValueError):
            return None

    return {
        "unread_count": unread_count or 0,
        "alerts": [{
            "id": r[0], "ticker": r[1], "stock_name": r[2], "signal_type": r[3],
            "signal_detail": r[4], "trade_date": str(r[5]) if r[5] else None,
            # 資料庫存的是伺服器(UTC)的時間、沒有時區標記，加上Z前端才會換算成使用者當地時間，「幾小時前」才不會差8小時
            "created_at": (r[6].isoformat() + "Z") if r[6] else None, "is_read": bool(r[7]),
            "payload": parse_payload(r[8]),
        } for r in rows]
    }

@app.put("/api/watchlist_alerts/{alert_id}/read")
def mark_watchlist_alert_read(alert_id: int, current_user: str = Depends(get_current_user)):
    with engine.connect() as conn:
        conn.execute(
            text("UPDATE WatchlistAlerts SET is_read = 1 WHERE id = :id AND user_email = :e"),
            {"id": alert_id, "e": current_user}
        )
        conn.commit()
    return {"status": "success"}

@app.put("/api/watchlist_alerts/read_all")
def mark_all_watchlist_alerts_read(current_user: str = Depends(get_current_user)):
    with engine.connect() as conn:
        conn.execute(
            text("UPDATE WatchlistAlerts SET is_read = 1 WHERE user_email = :e AND is_read = 0"),
            {"e": current_user}
        )
        conn.commit()
    return {"status": "success"}

@app.get("/api/watchlist_alert_performance")
def get_watchlist_alert_performance(current_user: str = Depends(get_current_user)):
    """自選股買賣訊號成效追蹤：假設訊號觸發隔天開盤就照著做（買進訊號買進、賣出/偏空訊號視為賣出/不追），
    浮動報酬(用最新收盤價對比進場價)，依 signal_type 分別統計，由 watchlist_alert_scanner.py 每日排程寫入"""
    try:
        with engine.connect() as conn:
            rows = conn.execute(text("""
                SELECT signal_type, entry_price,
                       (SELECT close_price FROM StockPrice sp WHERE sp.ticker = w.ticker ORDER BY sp.trade_date DESC LIMIT 1) AS latest_close
                FROM WatchlistAlerts w
                WHERE user_email = :e AND entry_price IS NOT NULL
            """), {"e": current_user}).fetchall()
            pending_count = conn.execute(
                text("SELECT COUNT(*) FROM WatchlistAlerts WHERE user_email = :e AND entry_price IS NULL"),
                {"e": current_user}
            ).scalar()

        # 買進類訊號(ai_bullish/pattern_buy)：股價上漲才算猜對；
        # 偏空/賣出類訊號(ai_bearish/pattern_sell)：股價下跌才算猜對(意思是「提醒你賣掉/避開」有沒有猜中方向)
        BEARISH_TYPES = {"ai_bearish", "pattern_sell"}

        stats = {}
        for signal_type, entry_price, latest_close in rows:
            if entry_price is None or latest_close is None or float(entry_price) == 0:
                continue
            ret = (float(latest_close) - float(entry_price)) / float(entry_price) * 100
            is_bearish_signal = signal_type in BEARISH_TYPES
            s = stats.setdefault(signal_type, {"returns": [], "wins": 0, "count": 0})
            s["returns"].append(ret)
            s["count"] += 1
            correct = (ret < 0) if is_bearish_signal else (ret > 0)
            if correct:
                s["wins"] += 1

        result = {
            signal_type: {
                "count": s["count"],
                "avg_return": round(sum(s["returns"]) / s["count"], 2) if s["count"] else 0,
                "hit_rate": round(s["wins"] / s["count"] * 100, 2) if s["count"] else 0,
            }
            for signal_type, s in stats.items()
        }

        return {"status": "success", "stats": result, "pending_count": pending_count or 0}
    except Exception as e:
        return {"status": "error", "message": f"讀取成效失敗: {str(e)}"}

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
        res = conn.execute(text("SELECT password_hash, email_verified FROM Users WHERE email = :e"), {"e": user.email}).fetchone()
        if not res:
            raise HTTPException(status_code=401, detail="Invalid credentials")
        hashed_pw = res[0]
        if not verify_password(user.password, hashed_pw):
            raise HTTPException(status_code=401, detail="Invalid credentials")

        # 密碼確認正確之後才提示「尚未驗證」，避免被拿來探測某個Email有沒有註冊過
        if EMAIL_ENABLED and not res[1]:
            raise HTTPException(status_code=403, detail={
                "code": "EMAIL_NOT_VERIFIED",
                "message": "Email 尚未驗證，請先到信箱點擊驗證連結",
            })

        try:
            consent_required = bool(pending_required_consents(conn, user.email))
        except Exception as e:
            print(f"⚠️ 檢查同意狀態失敗，略過：{e}")
            consent_required = False

        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": user.email}, expires_delta=access_token_expires
        )
        return {"access_token": access_token, "token_type": "bearer", "consent_required": consent_required}

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

            # --- 抓取最新新聞標題放入提示詞：改用跟情緒分析(sentiment_crawler.py)同一套
            # Yahoo股市個股新聞頁爬蟲，比原本yfinance內建的.news屬性對中小型股的覆蓋率更可靠，
            # 兩邊用同一個資料來源，「有沒有新聞」的答案才會一致 ---
            from sentiment_crawler import fetch_yahoo_news
            try:
                news_titles = await asyncio.to_thread(fetch_yahoo_news, ticker)
                if news_titles:
                    context_str += "- 近期最新新聞摘要（來源：Yahoo股市）：\n"
                    for t in news_titles[:3]:  # 取最新3筆新聞
                        context_str += f"  * {t}\n"
            except Exception as ne:
                pass # 若新聞抓取失敗則忽略
            # -----------------------------------
        except Exception as e:
            context_str += f"- (基本面資料擷取失敗：{str(e)})\n"
            
        # 2. 抓取 XGBoost AI 預測數據
        try:
            if engine.dialect.name == 'mssql':
                query = "SELECT TOP 2 * FROM StockPrice WHERE ticker = :ticker ORDER BY trade_date DESC"
            else:
                query = "SELECT * FROM StockPrice WHERE ticker = :ticker ORDER BY trade_date DESC LIMIT 2"

            with engine.connect() as conn:
                df = pd.read_sql(text(query), conn, params={"ticker": ticker})
            
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

LEADERBOARD_CACHE_FILE = os.path.join(current_dir, "leaderboard_cache.json")
LEADERBOARD_CACHE_TTL = 3600
# 用檔案存快取，而不是 Python 記憶體變數：伺服器是多 worker process 模式，
# 記憶體變數各個 process 互不相通（同一套做法跟 TREND_7D_CACHE 一樣）。

def _load_leaderboard_cache():
    import json
    try:
        with open(LEADERBOARD_CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"data": [], "timestamp": 0}

def _save_leaderboard_cache(cache):
    import json
    try:
        with open(LEADERBOARD_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(cache, f)
    except Exception as e:
        print(f"⚠️ 寫入 leaderboard 快取失敗：{e}")

@app.get("/api/leaderboard")
def get_leaderboard(limit: int = 5):
    """
    動態讀取歷史紀錄，計算 AI 上漲機率最高的前 N 檔股票 (快取+批次運算極速版)
    """
    import numpy as np
    import time

    now = time.time()
    leaderboard_cache = _load_leaderboard_cache()
    # 如果快取未過期 (1小時)，直接秒回傳
    if now - leaderboard_cache.get("timestamp", 0) < LEADERBOARD_CACHE_TTL and leaderboard_cache.get("data"):
        return {"top_tickers": leaderboard_cache["data"][:limit]}

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

        _save_leaderboard_cache({"data": top_list, "timestamp": now})

        return {"top_tickers": top_list[:limit]}
    except Exception as e:
        print(f"❌ Leaderboard 計算錯誤: {e}")
        # 如果出錯但有舊快取，加減回傳，避免前端當掉
        if leaderboard_cache.get("data"):
             return {"top_tickers": leaderboard_cache["data"][:limit]}
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/sentiment/run_crawler/{ticker}")
def trigger_sentiment_crawler(ticker: str, current_user: str = Depends(get_current_user)):
    enforce_rate_limit(current_user, "sentiment_crawler", 30)
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
            query = "SELECT TOP (:limit) * FROM StockPrice WHERE ticker = :ticker ORDER BY trade_date DESC"
        else:
            query = "SELECT * FROM StockPrice WHERE ticker = :ticker ORDER BY trade_date DESC LIMIT :limit"

        with engine.connect() as conn:
            df = pd.read_sql(text(query), conn, params={"ticker": ticker, "limit": limit})
        
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

STOCK_INFO_CACHE_FILE = os.path.join(current_dir, "stock_info_cache.json")
CACHE_TTL = 3600  # 快取 1 小時
# 用檔案存快取，而不是 Python 記憶體變數：伺服器是多 worker process 模式，
# 記憶體變數各個 process 互不相通（同一套做法跟 TREND_7D_CACHE 一樣）。

def _load_stock_info_cache():
    import json
    try:
        with open(STOCK_INFO_CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def _save_stock_info_cache(cache):
    import json
    try:
        with open(STOCK_INFO_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(cache, f)
    except Exception as e:
        print(f"⚠️ 寫入 stock_info 快取失敗：{e}")

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
    stock_info_cache = _load_stock_info_cache()

    # 檢查快取
    cached = stock_info_cache.get(ticker)
    if cached and now - cached["ts"] < CACHE_TTL:
        return {"status": "success", "data": cached["data"]}

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
        
        fresh_cache = _load_stock_info_cache()
        fresh_cache[ticker] = {"data": data, "ts": now}
        _save_stock_info_cache(fresh_cache)
        return {"status": "success", "data": data}

    except Exception as e:
        print(f"❌ Yahoo Finance 請求錯誤：{e}")
        if ticker in stock_info_cache:
             return {"status": "success", "data": stock_info_cache[ticker]["data"]}
        
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
            query = "SELECT TOP 2 * FROM StockPrice WHERE ticker = :ticker ORDER BY trade_date DESC"
        else:
            query = "SELECT * FROM StockPrice WHERE ticker = :ticker ORDER BY trade_date DESC LIMIT 2"

        with engine.connect() as conn:
            df = pd.read_sql(text(query), conn, params={"ticker": ticker})

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

    except HTTPException:
        raise
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
def get_trend_7days(ticker: str, current_user: str = Depends(get_current_user)):
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
            query = "SELECT TOP 60 * FROM StockPrice WHERE ticker = :ticker ORDER BY trade_date DESC"
        else:
            query = "SELECT * FROM StockPrice WHERE ticker = :ticker ORDER BY trade_date DESC LIMIT 60"

        with engine.connect() as conn:
            df = pd.read_sql(text(query), conn, params={"ticker": ticker})
            
        if df.empty:
            raise HTTPException(status_code=404, detail="查無此股票歷史紀錄")
            
        # 為了時間序列，必須排序讓舊的日期在上面
        df = df.sort_values("trade_date")
        
        # 將資料庫英文欄位轉為模型訓練時的中文特徵（要跟 train_autogluon_7days.py 的欄位完全對齊，
        # 不然訓練好的模型在推論時看不到它認得的輔助欄位，會直接出錯）
        df = df.rename(columns={
            'ticker': '股票代碼',
            'trade_date': '交易日期',
            'close_price': '收盤價',
            'volume': '成交量',
            'Foreign_Buy': '外資買賣超',
            'Trust_Buy': '投信買賣超',
            'Dealer_Buy': '自營商買賣超',
            'Vol_Ratio': '量能比',
            'KD_K': 'K值',
            'KD_D': 'D值',
            'Bias_5': '5日乖離',
            'Change_1D': '漲跌幅_1日',
            'Market_Return': '大盤漲跌幅',
            'TWD_Exchange': '台幣匯率',
            'SOX_Return': '費半漲跌',
        })

        # 轉換為 AutoGluon 需要的指定欄位結構
        df['交易日期'] = pd.to_datetime(df['交易日期'])

        keep_cols = ['股票代碼', '交易日期', '收盤價', '成交量', '外資買賣超',
                     '投信買賣超', '自營商買賣超', '量能比', 'K值', 'D值', 'RSI_14',
                     '5日乖離', '漲跌幅_1日', '大盤漲跌幅', '台幣匯率', '費半漲跌']
        keep_cols_exist = [c for c in keep_cols if c in df.columns]
        df = df[keep_cols_exist]
        for col in keep_cols:
            if col not in df.columns:
                df[col] = 0.0
        
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

    except HTTPException:
        raise
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
            query = "SELECT TOP (:limit) * FROM StockPrice WHERE ticker = :ticker ORDER BY trade_date DESC"
        else:
            query = "SELECT * FROM StockPrice WHERE ticker = :ticker ORDER BY trade_date DESC LIMIT :limit"

        with engine.connect() as conn:
            df = pd.read_sql(text(query), conn, params={"ticker": ticker, "limit": limit})
        
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

    except HTTPException:
        raise
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
# 社群論壇 API 區塊
# ==========================================
# ==========================================
# 社群論壇 API 區塊 (SQL 版)
# ==========================================
@app.get("/api/posts", response_model=List[Post])
def get_posts(limit: int = 50, offset: int = 0, sort: str = "newest"):
    """社群貼文列表（公開）。分頁：limit(1~100，預設50)、offset；sort=newest(最新) 或 popular(最多讚)。
    回覆只撈「這一頁貼文」的，不再一次撈全部貼文跟全部回覆"""
    return _query_posts(limit, offset, sort)

@app.get("/api/my_posts", response_model=List[Post])
def get_my_posts(limit: int = 100, offset: int = 0, current_user: str = Depends(get_current_user)):
    """只回傳登入者自己的貼文（「我的貼文」頁用）。不能沿用公開列表再由前端篩選，
    公開列表分頁後只有最新幾十筆，比較舊的自己的貼文會看不到"""
    return _query_posts(limit, offset, "newest", owner=current_user)

def _query_posts(limit: int, offset: int, sort: str, owner: Optional[str] = None):
    limit = max(1, min(limit, 100))
    offset = max(0, offset)
    # 排序方式只從白名單挑，不直接把使用者傳入的字串放進SQL
    order_by = "likes DESC, created_at DESC, id DESC" if sort == "popular" else "created_at DESC, id DESC"
    try:
        is_mssql = "mssql" in MYSQL_CONN_STR.lower()
        user_col = "[user]" if is_mssql else "user"
        page_clause = "OFFSET :offset ROWS FETCH NEXT :limit ROWS ONLY" if is_mssql else "LIMIT :limit OFFSET :offset"
        where_clause = "WHERE user_email = :owner" if owner else ""
        query_params = {"limit": limit, "offset": offset}
        if owner:
            query_params["owner"] = owner
        query_posts = (f"SELECT id, {user_col} AS [user], user_email, icon, created_at, sentiment, tag, content, likes, comments "
                       f"FROM Posts {where_clause} ORDER BY {order_by} {page_clause}")
        if not is_mssql:
            query_posts = (f"SELECT id, user, user_email, icon, created_at, sentiment, tag, content, likes, comments "
                           f"FROM Posts {where_clause} ORDER BY {order_by} {page_clause}")

        with engine.connect() as conn:
            df_posts = pd.read_sql(text(query_posts), conn, params=query_params)
            if df_posts.empty:
                return []
            reply_query = text(
                "SELECT id, post_id, author, content, created_at FROM Replies WHERE post_id IN :ids ORDER BY created_at ASC"
            ).bindparams(bindparam("ids", expanding=True))
            df_replies = pd.read_sql(reply_query, conn, params={"ids": [int(i) for i in df_posts["id"].tolist()]})

        result = []
        for _, row in df_posts.iterrows():
            post_id = row['id']
            post_replies = df_replies[df_replies['post_id'] == post_id].to_dict(orient='records')

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
    reject_markup(post.tag, "標籤")
    post.icon = post.icon if ICON_PATTERN.match(post.icon or "") else "fa-user"
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
                # INSERT IGNORE 靠 PostLikes 的 PRIMARY KEY(user_email, post_id) 擋掉重複按讚，
                # 只有這次真的新增了一筆紀錄(rowcount>0)才加計數，避免同一人狂打這支API無限刷讚數
                result = conn.execute(
                    text("INSERT IGNORE INTO PostLikes (user_email, post_id) VALUES (:e, :id)"),
                    {"e": current_user, "id": post_id}
                )
                if result.rowcount > 0:
                    conn.execute(text("UPDATE Posts SET likes = likes + 1 WHERE id = :id1"), {"id1": post_id})
            else:  # unlike
                result = conn.execute(
                    text("DELETE FROM PostLikes WHERE user_email = :e AND post_id = :id"),
                    {"e": current_user, "id": post_id}
                )
                if result.rowcount > 0:
                    conn.execute(text("UPDATE Posts SET likes = CASE WHEN likes > 0 THEN likes - 1 ELSE 0 END WHERE id = :id2"), {"id2": post_id})

            # 取得最新按讚數
            res = conn.execute(text("SELECT likes FROM Posts WHERE id = :id3"), {"id3": post_id}).fetchone()
            if res:
                return {"status": "success", "likes": res[0]}
        raise HTTPException(status_code=404, detail="找不到該留言")
    except HTTPException:
        raise
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
UPLOAD_CONTENT_TYPE_EXT = {"image/jpeg": "jpg", "image/png": "png", "image/gif": "gif", "image/webp": "webp"}

@app.post("/api/upload")
async def upload_image(file: UploadFile = File(...), current_user: str = Depends(get_current_user)):
    try:
        # 驗證檔案類型，避免有人上傳可執行檔或其他非圖片內容
        if file.content_type not in ALLOWED_UPLOAD_TYPES:
            raise HTTPException(status_code=400, detail="只允許上傳 JPG / PNG / GIF / WEBP 圖片")

        contents = await file.read()
        if len(contents) > MAX_UPLOAD_SIZE:
            raise HTTPException(status_code=400, detail="圖片大小不能超過 5MB")

        # 生成唯一檔名：副檔名完全由已驗證過的 content_type 決定，不採用使用者送來的原始檔名，
        # 避免有人用 "a.png/../../../evil" 這種檔名做路徑穿越、或偽裝成圖片實際存成 .html/.svg 造成儲存型XSS
        import uuid
        file_ext = UPLOAD_CONTENT_TYPE_EXT[file.content_type]
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

def check_report_reason(reason: str):
    # 檢舉原因會顯示在管理員的檢舉頁，所以不能夾帶HTML，也限制長度
    reject_markup(reason, "檢舉原因")
    if len(reason) > 500:
        raise HTTPException(status_code=400, detail="檢舉原因最多500字")

@app.post("/api/posts/{post_id}/report")
def report_post(post_id: int, payload: ReportCreate, current_user: str = Depends(get_current_user)):
    check_report_reason(payload.reason)
    with engine.connect() as conn:
        conn.execute(
            text("INSERT INTO Reports (reporter_email, target_type, target_id, reason) VALUES (:e, 'post', :id, :r)"),
            {"e": current_user, "id": post_id, "r": payload.reason}
        )
        conn.commit()
    return {"status": "success", "message": "已收到您的檢舉，我們會盡快處理"}

@app.post("/api/replies/{reply_id}/report")
def report_reply(reply_id: int, payload: ReportCreate, current_user: str = Depends(get_current_user)):
    check_report_reason(payload.reason)
    with engine.connect() as conn:
        conn.execute(
            text("INSERT INTO Reports (reporter_email, target_type, target_id, reason) VALUES (:e, 'reply', :id, :r)"),
            {"e": current_user, "id": reply_id, "r": payload.reason}
        )
        conn.commit()
    return {"status": "success", "message": "已收到您的檢舉，我們會盡快處理"}

async def get_current_admin(current_user: str = Depends(get_current_user)):
    with engine.connect() as conn:
        res = conn.execute(
            text("SELECT is_admin FROM Users WHERE email = :e"), {"e": current_user}
        ).fetchone()
    if not res or not res[0]:
        raise HTTPException(status_code=403, detail="僅限管理員存取")
    return current_user

@app.get("/api/admin/reports")
def get_reports(current_user: str = Depends(get_current_admin)):
    with engine.connect() as conn:
        reports = conn.execute(
            text("SELECT id, reporter_email, target_type, target_id, reason, status, created_at FROM Reports ORDER BY created_at DESC")
        ).fetchall()

        result = []
        for r in reports:
            report_id, reporter_email, target_type, target_id, reason, status_val, created_at = r
            target_content, target_author = None, None
            if target_type == 'post':
                row = conn.execute(text("SELECT user, content FROM Posts WHERE id = :id"), {"id": target_id}).fetchone()
            else:
                row = conn.execute(text("SELECT author, content FROM Replies WHERE id = :id"), {"id": target_id}).fetchone()
            if row:
                target_author, target_content = row[0], row[1]

            result.append({
                "id": report_id, "reporter_email": reporter_email, "target_type": target_type,
                "target_id": target_id, "reason": reason, "status": status_val,
                "created_at": str(created_at), "target_author": target_author,
                "target_content": target_content
            })
        return {"status": "success", "reports": result}

class ReportStatusUpdate(BaseModel):
    status: str  # 'open' 或 'resolved'

@app.put("/api/admin/reports/{report_id}")
def update_report_status(report_id: int, payload: ReportStatusUpdate, current_user: str = Depends(get_current_admin)):
    with engine.connect() as conn:
        conn.execute(
            text("UPDATE Reports SET status = :s WHERE id = :id"),
            {"s": payload.status, "id": report_id}
        )
        conn.commit()
    return {"status": "success", "message": "已更新檢舉狀態"}

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
            "top_bullish_momentum": cache_data.get("top_bullish_momentum", []),
            "top_bullish_emerging": cache_data.get("top_bullish_emerging", []),
            "top_bearish": cache_data.get("top_bearish", [])
        }
    except Exception as e:
        return {"status": "error", "message": f"讀取快取失敗: {str(e)}"}

AI_PICK_FEE_RATE = 0.6  # 百分比：概估來回手續費0.1425%*2 + 賣出證交稅0.3%，扣掉後才是比較貼近實際的報酬，寫論文時分開呈現才嚴謹

@app.get("/api/ai_pick_performance")
def get_ai_pick_performance():
    """AI精選模擬交易的長期績效追蹤，浮動報酬(用最新收盤價對比進場價)，
    分動能延續(momentum)/潛力發掘(potential)/看跌追蹤(bearish，模擬放空)三類分別統計，
    每類都同時給「無成本理想版」(avg_return)跟「概估手續費/滑價後」(avg_net_return)兩種數字，
    並附上每一筆的個股明細，不只是分類後的整體平均。由 ai_top_picks_scanner.py 每日排程寫入"""
    try:
        with engine.connect() as conn:
            rows = conn.execute(text("""
                SELECT t.ticker, t.stock_name, t.signal_type, t.entry_price, t.entry_date,
                       (SELECT close_price FROM StockPrice sp WHERE sp.ticker = t.ticker ORDER BY sp.trade_date DESC LIMIT 1) AS latest_close
                FROM AiPickTrades t
                WHERE entry_price IS NOT NULL
            """)).fetchall()
            pending_count = conn.execute(text("SELECT COUNT(*) FROM AiPickTrades WHERE entry_price IS NULL")).scalar()

        stats = {}
        trades = []
        for ticker, name, signal_type, entry_price, entry_date, latest_close in rows:
            if entry_price is None or latest_close is None or float(entry_price) == 0:
                continue
            entry_price = float(entry_price)
            latest_close = float(latest_close)
            # bearish(看跌追蹤)是模擬放空，股價跌越多報酬越高，方向跟做多的兩類相反
            raw_return = (latest_close - entry_price) / entry_price * 100
            gross_return = -raw_return if signal_type == "bearish" else raw_return
            net_return = gross_return - AI_PICK_FEE_RATE

            s = stats.setdefault(signal_type, {"gross": [], "net": [], "wins": 0, "count": 0})
            s["gross"].append(gross_return)
            s["net"].append(net_return)
            s["count"] += 1
            if gross_return > 0:
                s["wins"] += 1

            trades.append({
                "ticker": ticker, "name": name, "signal_type": signal_type,
                "entry_date": str(entry_date) if entry_date else None,
                "entry_price": entry_price, "latest_close": latest_close,
                "gross_return": round(gross_return, 2), "net_return": round(net_return, 2),
            })

        result = {
            signal_type: {
                "count": s["count"],
                "avg_return": round(sum(s["gross"]) / s["count"], 2) if s["count"] else 0,
                "avg_net_return": round(sum(s["net"]) / s["count"], 2) if s["count"] else 0,
                "win_rate": round(s["wins"] / s["count"] * 100, 2) if s["count"] else 0,
            }
            for signal_type, s in stats.items()
        }

        trades.sort(key=lambda t: t["gross_return"], reverse=True)

        return {
            "status": "success", "stats": result, "pending_count": pending_count or 0,
            "trades": trades, "fee_rate": AI_PICK_FEE_RATE,
        }
    except Exception as e:
        return {"status": "error", "message": f"讀取績效失敗: {str(e)}"}

@app.get("/api/ai_pick_performance_history")
def get_ai_pick_performance_history(days: int = 30):
    """AI精選模擬績效隨時間的變化(近N天平均報酬率)，由 ai_top_picks_scanner.py 每日排程寫入快照，
    給前端畫趨勢圖用，不然只看累積總平均看不出績效是在變好還變差"""
    days = max(1, min(days, 180))
    try:
        with engine.connect() as conn:
            rows = conn.execute(text("""
                SELECT snapshot_date, signal_type, avg_return, win_rate, trade_count
                FROM AiPickPerformanceHistory
                WHERE snapshot_date >= DATE_SUB(CURDATE(), INTERVAL :days DAY)
                ORDER BY snapshot_date ASC
            """), {"days": days}).fetchall()

        history = {}
        for snapshot_date, signal_type, avg_return, win_rate, trade_count in rows:
            history.setdefault(signal_type, []).append({
                "date": str(snapshot_date),
                "avg_return": float(avg_return) if avg_return is not None else None,
                "win_rate": float(win_rate) if win_rate is not None else None,
                "trade_count": trade_count,
            })

        return {"status": "success", "history": history}
    except Exception as e:
        return {"status": "error", "message": f"讀取趨勢失敗: {str(e)}"}

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
            "metrics": data.get("metrics"),
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

        # 除了整體命中率(準確率)，再由混淆矩陣算「精確率/召回率」：
        #   精確率 = 預測會漲的裡面真的漲了多少(TP/(TP+FP))；召回率 = 真的漲的裡面被抓到多少(TP/(TP+FN))
        # 預測方向以信心度>=50為「漲」；actual_direction 是下一個交易日收盤相對預測當天收盤（持平算未上漲）
        with engine.connect() as conn:
            cm = conn.execute(text("""
                SELECT
                    COALESCE(SUM(predicted_direction = 'up'   AND actual_direction = 'up'),   0) AS tp,
                    COALESCE(SUM(predicted_direction = 'up'   AND actual_direction = 'down'), 0) AS fp,
                    COALESCE(SUM(predicted_direction = 'down' AND actual_direction = 'up'),   0) AS fn,
                    COALESCE(SUM(predicted_direction = 'down' AND actual_direction = 'down'), 0) AS tn,
                    COALESCE(SUM(predicted_direction = 'up' AND confidence >= 65), 0)                          AS hi_n,
                    COALESCE(SUM(predicted_direction = 'up' AND confidence >= 65 AND actual_direction = 'up'), 0) AS hi_tp
                FROM PredictionLog
                WHERE resolved = 1 AND predicted_at >= DATE_SUB(CURDATE(), INTERVAL :days DAY)
            """), {"days": days}).fetchone()
        tp, fp, fn, tn, hi_n, hi_tp = (int(x) for x in cm)
        ratio = lambda a, b: round(a / b * 100, 2) if b else None
        up_rate = ratio(tp + fn, total)

        return {
            "status": "success",
            "sample_size": total,
            "hit_rate": round(correct / total * 100, 2),
            "days": days,
            "latest_date": str(row[2]) if row[2] else None,
            "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
            "precision_up": ratio(tp, tp + fp),
            "recall_up": ratio(tp, tp + fn),
            "precision_down": ratio(tn, tn + fn),
            "recall_down": ratio(tn, tn + fp),
            "up_rate": up_rate,
            "majority_baseline": max(up_rate, 100 - up_rate) if up_rate is not None else None,
            "high_confidence_up": {"threshold": 65, "sample_size": hi_n, "precision": ratio(hi_tp, hi_n)},
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