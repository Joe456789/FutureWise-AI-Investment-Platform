-- 自選股買賣訊號通知記錄表
-- signal_type: ai_bullish(AI轉強烈偏多) / ai_bearish(AI轉強烈偏空) / pattern_buy(技術面買進訊號) / pattern_sell(技術面賣出/警示訊號)
CREATE TABLE IF NOT EXISTS WatchlistAlerts (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_email VARCHAR(255) NOT NULL,
    ticker VARCHAR(20) NOT NULL,
    stock_name VARCHAR(100),
    signal_type VARCHAR(20) NOT NULL,
    signal_detail VARCHAR(255) NOT NULL,
    trade_date DATE NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    is_read TINYINT(1) DEFAULT 0,
    UNIQUE KEY uniq_alert (user_email, ticker, signal_type, trade_date),
    INDEX idx_user_read (user_email, is_read)
);
