-- 自選股買賣訊號成效追蹤：比照 AiPickTrades 的做法，記錄訊號觸發後隔天開盤價，
-- 用來追蹤「如果照訊號進場」的浮動報酬，不只是單純通知而已
ALTER TABLE WatchlistAlerts
    ADD COLUMN entry_date DATE DEFAULT NULL,
    ADD COLUMN entry_price DECIMAL(10,2) DEFAULT NULL;
