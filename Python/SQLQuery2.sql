/*
  ProQuant AI 交易系統 - 終極資料庫初始化腳本
  目的：確保 SQL 欄位與 Python 爬蟲 (V5版) 100% 契合，解決 Market_Close 遺漏問題
  修正：將檔案標籤修正為 SQL，避免系統誤以 LaTeX 編譯
*/

-- 1. 建立資料庫 (如果不存在)
IF NOT EXISTS (SELECT * FROM sys.databases WHERE name = 'StockDB1')
    CREATE DATABASE StockDB1;
GO

USE StockDB1;
GO

-- 2. 刪除舊表 (清空地基，確保欄位重新排列)
IF OBJECT_ID('StockPrice', 'U') IS NOT NULL
    DROP TABLE StockPrice;
GO

-- 3. 建立 StockPrice 資料表 (包含所有量化特徵)
CREATE TABLE StockPrice (
    -- [識別碼]
    id INT IDENTITY(1,1) PRIMARY KEY,
    ticker NVARCHAR(20) NOT NULL,          -- 股票代碼
    stock_name NVARCHAR(50),               -- 股票名稱
    trade_date DATE NOT NULL,              -- 交易日期
    
    -- [基礎價格]
    open_price DECIMAL(18, 4),
    high_price DECIMAL(18, 4),
    low_price DECIMAL(18, 4),
    close_price DECIMAL(18, 4),
    volume BIGINT,

    -- [1. 技術指標 - 均線系統]
    MA_5 DECIMAL(18, 4), MA_10 DECIMAL(18, 4), MA_20 DECIMAL(18, 4),
    MA_60 DECIMAL(18, 4), MA_120 DECIMAL(18, 4), MA_240 DECIMAL(18, 4),

    -- [2. 技術指標 - 動能與震盪]
    RSI_6 DECIMAL(18, 4), RSI_14 DECIMAL(18, 4),
    MACD_DIF DECIMAL(18, 4), MACD_Signal DECIMAL(18, 4), MACD_Hist DECIMAL(18, 4),
    KD_K DECIMAL(18, 4), KD_D DECIMAL(18, 4),
    BB_Upper DECIMAL(18, 4), BB_Lower DECIMAL(18, 4),
    ATR DECIMAL(18, 4),

    -- [3. 成交量與乖離特徵]
    Vol_MA_5 BIGINT, Vol_MA_20 BIGINT,
    Vol_Ratio DECIMAL(18, 4),
    Bias_5 DECIMAL(18, 4), Bias_20 DECIMAL(18, 4),
    
    -- [4. 預測工程特徵]
    Change_1D DECIMAL(18, 6), Change_3D DECIMAL(18, 6), Change_5D DECIMAL(18, 6),
    Gap DECIMAL(18, 4),

    -- [5. 基本面特徵]
    Revenue_YoY DECIMAL(18, 4),
    EPS DECIMAL(18, 4),
    PE_Ratio DECIMAL(18, 4),
    PB_Ratio DECIMAL(18, 4),

    -- [6. 全球大盤連動指標 (解決您報錯的關鍵)]
    Market_Close DECIMAL(18, 4),  -- 台股大盤收盤價
    Market_Return DECIMAL(18, 6), -- 台股大盤漲跌幅
    SOX_Close DECIMAL(18, 4),     -- 費城半導體收盤
    SOX_Return DECIMAL(18, 6),    -- 費半昨日漲跌幅
    TWD_Exchange DECIMAL(18, 4),  -- 台幣兌美金匯率

    -- [7. 籌碼面與另類數據]
    Foreign_Buy BIGINT DEFAULT 0,
    Trust_Buy BIGINT DEFAULT 0,
    Dealer_Buy BIGINT DEFAULT 0,
    Sentiment_Score DECIMAL(18, 6) DEFAULT 0,

    -- 建立唯一限制，防止同一代碼在同一日期重複存入
    CONSTRAINT UQ_Stock_Date UNIQUE(ticker, trade_date)
);
GO

-- 建立索引以提升 Python 讀取速度
CREATE INDEX IDX_Ticker_Date ON StockPrice (ticker, trade_date);
GO

PRINT '✅ 終極對齊版 StockPrice 表格已重新建立。';

-- 驗證最終表格結構
SELECT TOP 10 * FROM StockPrice;

SELECT 
    COLUMN_NAME AS [欄位名稱], 
    DATA_TYPE AS [資料型態], 
    NUMERIC_PRECISION AS [整數長度], 
    NUMERIC_SCALE AS [小數點位數]
FROM INFORMATION_SCHEMA.COLUMNS 
WHERE TABLE_NAME = 'StockPrice'
AND COLUMN_NAME IN ('Market_Close', 'SOX_Close', 'TWD_Exchange', 'open_price', 'close_price');

-- 1. 檢查 2/5 當天總共有幾支股票成功存入 (正常應該要有 1000 支左右)
SELECT COUNT(*) AS [2月5日總筆數] 
FROM StockPrice 
WHERE trade_date = '2026-02-26';

-- 2. 檢查特定的權值股 (如 2330 台積電) 在 2/5 的詳細資料
-- 順便檢查「費半收盤」跟「台幣匯率」是不是 0.00
SELECT *
FROM StockPrice 
WHERE trade_date = '2026-07-26' AND ticker = '3595';

-- 3. 看看最近 5 天資料庫存了哪些日期 (確認資料是否有斷層)
SELECT DISTINCT TOP 5 trade_date 
FROM StockPrice 
ORDER BY trade_date DESC;

USE StockDB1;
GO

SELECT TOP 10 
    ticker AS [代碼], 
    trade_date AS [日期], 
    close_price AS [收盤價], 
    Foreign_Buy AS [外資買賣超], 
    Trust_Buy AS [投信買賣超], 
    Dealer_Buy AS [自營商買賣超]
FROM StockPrice 
WHERE ticker = '2330'
ORDER BY trade_date DESC;


SELECT TOP 100 *
FROM StockPrice
WHERE ticker = '0050'
ORDER BY trade_date DESC;