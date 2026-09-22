-- 模擬「AI精選出爐隔天開盤買入」的紙上交易記錄，用來追蹤AI精選長期實際績效
-- signal_type: potential(潛力發掘) / momentum(動能延續)
-- entry_price 在挑出的隔天開盤才會補上（挑出當下還沒有隔天的開盤價）
CREATE TABLE IF NOT EXISTS AiPickTrades (
    id INT AUTO_INCREMENT PRIMARY KEY,
    ticker VARCHAR(20) NOT NULL,
    stock_name VARCHAR(100),
    signal_type VARCHAR(20) NOT NULL,
    pick_date DATE NOT NULL,
    entry_date DATE DEFAULT NULL,
    entry_price DECIMAL(10,2) DEFAULT NULL,
    confidence DECIMAL(5,2),
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    UNIQUE KEY uniq_pick (ticker, pick_date, signal_type),
    INDEX idx_signal_entry (signal_type, entry_price)
);
