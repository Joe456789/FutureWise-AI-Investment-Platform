-- 一次性遷移：重大訊息公告（AI分析頁的新聞列表用）
-- 在 Oracle 主機的 MySQL 上執行一次即可（只有 1 段 CREATE TABLE，直接執行即可）

CREATE TABLE IF NOT EXISTS MaterialNews (
    id              INT AUTO_INCREMENT PRIMARY KEY,
    ticker          VARCHAR(20)   NOT NULL,
    company_name    VARCHAR(100),
    announce_date   DATE          NOT NULL,   -- 發言日期
    announce_time   VARCHAR(10),              -- 發言時間 (HHMMSS)
    subject         VARCHAR(500)  NOT NULL,   -- 主旨（新聞列表顯示用）
    detail          TEXT,                     -- 完整說明內容
    created_at      DATETIME DEFAULT CURRENT_TIMESTAMP,
    UNIQUE KEY uniq_news (ticker, announce_date, subject(191)),
    INDEX idx_ticker_date (ticker, announce_date)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
