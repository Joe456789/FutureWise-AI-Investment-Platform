-- AI精選模擬績效的每日快照，用來畫「近30天報酬率變化」趨勢圖，
-- 不然/api/ai_pick_performance只有累積至今的總平均，看不出績效隨時間的變化
CREATE TABLE IF NOT EXISTS AiPickPerformanceHistory (
    id INT AUTO_INCREMENT PRIMARY KEY,
    snapshot_date DATE NOT NULL,
    signal_type VARCHAR(20) NOT NULL,
    avg_return DECIMAL(6,2),
    win_rate DECIMAL(5,2),
    trade_count INT,
    UNIQUE KEY uniq_snapshot (snapshot_date, signal_type)
);
