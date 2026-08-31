import platform
import urllib.parse
import socket
import os

# 嘗試讀取「跟 config.py 同一個資料夾」的 .env 檔案（如果有安裝 python-dotenv）。
# 用絕對路徑指定，而不是讓 load_dotenv() 自己找目前工作目錄——
# 因為 crontab 執行 python 腳本時，工作目錄不一定是這個資料夾，用絕對路徑才能穩定讀到。
# 沒裝 dotenv 或沒有 .env 檔案都不影響運作，之後改用系統環境變數也一樣有效。
try:
    from dotenv import load_dotenv
    _env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    load_dotenv(dotenv_path=_env_path)
except ImportError:
    pass

# 1. 自動偵測作業系統
OS_TYPE = platform.system()  # 會回傳 'Windows' 或 'Linux'

# ==========================================
# 情況 A：在本機 Windows (SSMS / MSSQL)
# ==========================================
if OS_TYPE == 'Windows':
    DB_TYPE = "MSSQL"
    HOSTNAME = socket.gethostname()
    DATABASE_NAME = "StockDB1"

    params = urllib.parse.quote_plus(
        f"DRIVER={{ODBC Driver 17 for SQL Server}};"
        f"SERVER={HOSTNAME};"
        f"DATABASE={DATABASE_NAME};"
        f"Trusted_Connection=yes;"
        f"TrustServerCertificate=yes;"
    )
    DB_CONN_STR = f"mssql+pyodbc:///?odbc_connect={params}"
    MYSQL_CONN_STR = DB_CONN_STR  # 統一變數名稱給 API 伺服器用

# ==========================================
# 情況 B：在 Oracle Linux 伺服器 (MySQL)
# ==========================================
else:
    DB_TYPE = "MYSQL"
    DB_USER = os.environ.get("DB_USER", "root")

    # 密碼一律從環境變數讀取，程式碼裡不再寫死。
    # 伺服器上要在 /home/ubuntu/stock_python/Python/.env 建立這個檔案（不要進 git）：
    #   DB_PASSWORD=你的密碼
    #   GEMINI_API_KEY=你的金鑰
    #   JWT_SECRET_KEY=一串隨機字串
    DB_PASSWORD = os.environ.get("DB_PASSWORD")
    if not DB_PASSWORD:
        raise RuntimeError(
            "環境變數 DB_PASSWORD 未設定。請在 config.py 同資料夾建立 .env 檔案，"
            "內容加入一行 DB_PASSWORD=你的資料庫密碼"
        )

    DB_HOST = os.environ.get("DB_HOST", "127.0.0.1")
    DB_PORT = os.environ.get("DB_PORT", "3306")
    DB_NAME = os.environ.get("DB_NAME", "StockDB1")

    # ★ 關鍵修改 1：用 quote_plus 保護密碼，這樣密碼有特殊符號或中文都不會報 latin-1 錯誤
    safe_password = urllib.parse.quote_plus(DB_PASSWORD)

    # ★ 關鍵修改 2：加上 ?charset=utf8mb4，讓台積電等中文名稱可以順利丟給同學的前端
    DB_CONN_STR = f"mysql+pymysql://{DB_USER}:{safe_password}@{DB_HOST}:{DB_PORT}/{DB_NAME}?charset=utf8mb4"
    MYSQL_CONN_STR = DB_CONN_STR  # 統一變數名稱給 API 伺服器用

# ==========================================
# Gemini API 設定
# ==========================================
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

# ==========================================
# JWT 簽章密鑰（stock_api_server.py 的登入系統用）
# ==========================================
JWT_SECRET_KEY = os.environ.get("JWT_SECRET_KEY", "")
