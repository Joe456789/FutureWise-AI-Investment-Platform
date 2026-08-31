-- 一次性遷移：AI 預測命中率追蹤（回測儀表板用）
-- 在 Oracle 主機的 MySQL 上執行一次即可（只有 1 段 CREATE TABLE，直接執行即可）

CREATE TABLE IF NOT EXISTS PredictionLog (
    id                  INT AUTO_INCREMENT PRIMARY KEY,
    ticker              VARCHAR(20)   NOT NULL,
    predicted_at        DATE          NOT NULL,   -- 預測當天所用的最新交易日資料
    close_at_predict    DECIMAL(18,4) NOT NULL,    -- 預測當下的收盤價
    predicted_direction VARCHAR(10)   NOT NULL,    -- 'up' 或 'down'
    confidence          DECIMAL(6,2)  NOT NULL,
    resolved            TINYINT(1)    DEFAULT 0,
    actual_direction    VARCHAR(10)   NULL,
    is_correct          TINYINT(1)    NULL,
    resolved_at         DATETIME      NULL,
    created_at          DATETIME      DEFAULT CURRENT_TIMESTAMP,
    UNIQUE KEY uniq_ticker_date (ticker, predicted_at),
    INDEX idx_resolved (resolved),
    INDEX idx_predicted_at (predicted_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
