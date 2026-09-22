# 程式名稱：auth_schema.py
# 功能：建立/補齊「Email驗證」「忘記密碼」「動態知情同意」需要的資料表與欄位，並寫入預設同意條款。
#      可重複執行（idempotent）：stock_api_server.py 啟動時會自動呼叫一次，也可以手動執行：
#          python auth_schema.py
# 注意：預設同意條款文字是依照本系統實際蒐集的資料寫的草稿版本，正式上線前建議自行審閱、
#      必要時請法務或專業人士確認（尤其是個資法告知事項）。要修改條款：新增一筆同 code、version 更大的
#      ConsentItems，舊使用者下次登入就會被要求重新同意新版本，不要直接改舊版本的文字。

from sqlalchemy import create_engine, text

CONSENT_SEED = [
    {
        "code": "terms", "version": 1, "required": 1,
        "title": "服務條款",
        "summary": "使用本平台需遵守的基本規則",
        "body": (
            "一、服務內容：FutureWise AI 提供股票資料查詢、AI 模型預測、選股、社群討論等功能，僅供學習與研究參考。\n"
            "二、帳號責任：您應提供正確的 Email，並妥善保管密碼；因密碼外洩造成的損失由您自行負擔。\n"
            "三、社群守則：不得發布違法、侵權、詐騙、人身攻擊或惡意刷版內容；平台得視情況刪除內容、停權，並依檢舉機制處理。\n"
            "四、服務變更：平台得隨時調整、暫停或終止部分功能，重大變更會於系統公告。\n"
            "五、責任限制：資料可能延遲或有誤，平台不保證資料之即時性、正確性與完整性。"
        ),
    },
    {
        "code": "privacy", "version": 1, "required": 1,
        "title": "個人資料蒐集、處理及利用告知",
        "summary": "我們蒐集哪些資料、為什麼蒐集、怎麼使用、您有哪些權利",
        "body": (
            "依《個人資料保護法》第8條規定告知：\n"
            "一、蒐集機關：FutureWise AI 專題團隊。\n"
            "二、蒐集目的：會員管理與身分驗證、提供個人化功能（收藏、歷史紀錄、通知）、社群討論與檢舉處理、客服聯繫、系統安全與改善。\n"
            "三、蒐集之資料類別：Email、密碼（僅儲存不可還原的雜湊值）、姓名、聯絡電話、大頭貼、收藏股票、瀏覽歷史、教學閱讀紀錄、"
            "社群貼文／回覆／按讚、檢舉與封鎖紀錄、通知設定、客服工單、同意紀錄。\n"
            "四、利用期間：自您註冊起至您申請刪除帳號為止；法令另有規定者從其規定。\n"
            "五、利用地區與對象：資料儲存於雲端主機（Oracle Cloud）；僅供本平台內部使用，除法令要求外不會出售或提供給第三方行銷。\n"
            "六、第三方處理：當您使用 AI 聊天／分析功能時，您輸入的問題內容與股票代號會傳送至 Google Gemini 服務以產生回覆；"
            "寄送驗證信、重設密碼信會透過電子郵件服務商寄出。\n"
            "七、您的權利：得依個資法向我們請求查詢或閱覽、製給複製本、補充或更正、停止蒐集處理利用、刪除您的個人資料；"
            "可透過「幫助與客服」提出。\n"
            "八、不提供的影響：Email 與密碼為建立帳號之必要資料，若不提供將無法註冊；姓名、電話為選填之聯絡資料。"
        ),
    },
    {
        "code": "ai_disclaimer", "version": 1, "required": 1,
        "title": "AI 預測與投資風險聲明",
        "summary": "AI 預測不是投資建議，投資有風險",
        "body": (
            "一、本平台所有 AI 預測、信心分數、選股結果、買賣訊號與模擬績效，皆由統計模型自動產生，僅供學習研究參考，"
            "不構成任何投資建議、買賣邀約或獲利保證。\n"
            "二、模型有其限制，過去表現不代表未來結果；模擬績效為假設情境，未必能在實際交易中達成。\n"
            "三、投資決策應由您自行判斷並承擔風險，因使用本平台資訊而產生的任何損失，平台不負賠償責任。"
        ),
    },
    {
        "code": "research_data", "version": 1, "required": 0,
        "title": "匿名化統計與模型改進授權（選填）",
        "summary": "同意以去識別化方式使用您的操作資料，協助改善功能與學術研究",
        "body": (
            "若您勾選同意，我們可能將您的使用資料（例如收藏的股票代號、功能使用次數）"
            "在「去除 Email、姓名、電話等可識別個人資訊」之後，用於統計分析、改善模型與功能設計，以及專題／學術研究成果的彙整呈現。\n"
            "不同意不影響您使用任何功能。您可隨時在「隱私與資料授權」頁面撤回，撤回後不再納入之後的統計。"
        ),
    },
    {
        "code": "marketing_email", "version": 1, "required": 0,
        "title": "功能更新與公告 Email 通知（選填）",
        "summary": "同意我們以 Email 寄送功能更新、系統公告",
        "body": (
            "若您勾選同意，我們可能以您註冊的 Email 寄送新功能介紹、重大系統公告或維護通知。\n"
            "與帳號安全相關的信件（Email 驗證、重設密碼）不受此選項影響。不同意不影響您使用任何功能，可隨時撤回。"
        ),
    },
]


def _column_exists(conn, table, column):
    row = conn.execute(text(
        "SELECT COUNT(*) FROM information_schema.COLUMNS "
        "WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = :t AND COLUMN_NAME = :c"
    ), {"t": table, "c": column}).scalar()
    return bool(row)


def _add_column(conn, table, column, ddl):
    """新增欄位；回傳這次是不是真的由自己新增的。
    伺服器有多個 worker 同時啟動，可能兩個都判斷「還沒有」，後到的那個會撞到 Duplicate column，視為已存在即可"""
    if _column_exists(conn, table, column):
        return False
    try:
        conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}"))
        return True
    except Exception as e:
        if "Duplicate column" in str(e) or "1060" in str(e):
            return False
        raise


def ensure_alert_payload_column(engine):
    """WatchlistAlerts 新增 payload 欄位（通知卡片要顯示的結構化內容，存JSON字串）。
    API 伺服器啟動時與 watchlist_alert_scanner.py 執行前都會呼叫，兩邊誰先跑都不會缺欄位；表還不存在就略過"""
    if engine.dialect.name != "mysql":
        return
    with engine.begin() as conn:
        exists = conn.execute(text(
            "SELECT COUNT(*) FROM information_schema.TABLES WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = 'WatchlistAlerts'"
        )).scalar()
        if exists and _add_column(conn, "WatchlistAlerts", "payload", "MEDIUMTEXT NULL"):
            print("✅ [auth_schema] WatchlistAlerts.payload 已新增")


def ensure_auth_schema(engine):
    if engine.dialect.name != "mysql":
        print("ℹ️ [auth_schema] 非 MySQL 資料庫，略過建表（Email驗證/同意系統僅支援伺服器上的 MySQL）")
        return

    with engine.begin() as conn:
        # 新增 email_verified；第一次新增時把「既有帳號」視為已驗證，避免現有使用者被擋在門外
        if _add_column(conn, "Users", "email_verified", "TINYINT(1) NOT NULL DEFAULT 0"):
            conn.execute(text("UPDATE Users SET email_verified = 1"))
            print("✅ [auth_schema] Users.email_verified 已新增，既有帳號全部視為已驗證")

        # 登入憑證撤銷用：簽發時間(iat)早於這個時間點的憑證視為失效，改密碼/重設密碼時更新
        if _add_column(conn, "Users", "token_valid_after", "BIGINT NOT NULL DEFAULT 0"):
            print("✅ [auth_schema] Users.token_valid_after 已新增")

        # 頻率限制/每日總量：存資料庫而不是記憶體，多個 uvicorn worker 才會共用同一份記錄
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS RateLimits (
                rl_key CHAR(64) NOT NULL PRIMARY KEY,
                last_call DOUBLE NOT NULL,
                hits INT NOT NULL DEFAULT 0,
                INDEX idx_last_call (last_call)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """))

        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS EmailTokens (
                id INT AUTO_INCREMENT PRIMARY KEY,
                email VARCHAR(255) NOT NULL,
                purpose VARCHAR(20) NOT NULL,
                token_hash CHAR(64) NOT NULL,
                expires_at DATETIME NOT NULL,
                used_at DATETIME DEFAULT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE KEY uniq_token_hash (token_hash),
                INDEX idx_email_purpose (email, purpose)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """))

        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS ConsentItems (
                id INT AUTO_INCREMENT PRIMARY KEY,
                code VARCHAR(50) NOT NULL,
                version INT NOT NULL,
                title VARCHAR(100) NOT NULL,
                summary VARCHAR(255),
                body MEDIUMTEXT NOT NULL,
                required TINYINT(1) NOT NULL DEFAULT 0,
                is_active TINYINT(1) NOT NULL DEFAULT 1,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE KEY uniq_code_version (code, version)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """))

        # 只新增不修改：每次同意/撤回都是新增一筆，目前狀態=該使用者該條款最新的一筆，保留完整稽核軌跡
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS UserConsents (
                id INT AUTO_INCREMENT PRIMARY KEY,
                user_email VARCHAR(255) NOT NULL,
                code VARCHAR(50) NOT NULL,
                version INT NOT NULL,
                granted TINYINT(1) NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                INDEX idx_user_code (user_email, code, id)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """))

        for item in CONSENT_SEED:
            conn.execute(text("""
                INSERT IGNORE INTO ConsentItems (code, version, title, summary, body, required)
                VALUES (:code, :version, :title, :summary, :body, :required)
            """), item)
            conn.execute(text("""
                UPDATE ConsentItems SET title = :title, summary = :summary, body = :body
                WHERE code = :code AND version = :version AND body LIKE '%ProQuant%'
            """), item)

    ensure_alert_payload_column(engine)
    print("✅ [auth_schema] Email驗證 / 忘記密碼 / 同意系統資料表已就緒")


if __name__ == "__main__":
    from config import MYSQL_CONN_STR
    ensure_auth_schema(create_engine(MYSQL_CONN_STR))
