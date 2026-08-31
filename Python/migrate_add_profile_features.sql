-- 一次性遷移：個人頁面 5 大功能（收藏／設定／通知／歷史／客服）
-- 在 Oracle 主機的 MySQL 上執行一次即可
--
-- 注意：這個檔案有 4 段 CREATE TABLE。若在 Navicat 貼上後按單次「執行」出現
-- 1064 語法錯誤，通常是 Navicat 把整段當成「一句」送給 MySQL，沒有照 ; 拆開。
-- 請改用「檔案 → 執行 SQL 文件 (Run SQL File)」開啟這個 .sql 檔執行，
-- 或是一次只反白其中一段 CREATE TABLE 執行，避免一次貼全部文字進查詢框執行。

-- 1. 我的收藏
CREATE TABLE IF NOT EXISTS Favorites (
    id         INT AUTO_INCREMENT PRIMARY KEY,
    user_email VARCHAR(100) NOT NULL,
    ticker     VARCHAR(20)  NOT NULL,
    stock_name VARCHAR(50),
    created_at DATETIME     DEFAULT CURRENT_TIMESTAMP,
    UNIQUE KEY uniq_user_ticker (user_email, ticker)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 2. 瀏覽歷史（查過的個股預測）
CREATE TABLE IF NOT EXISTS ViewHistory (
    id         INT AUTO_INCREMENT PRIMARY KEY,
    user_email VARCHAR(100) NOT NULL,
    ticker     VARCHAR(20)  NOT NULL,
    stock_name VARCHAR(50),
    viewed_at  DATETIME     DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_user_time (user_email, viewed_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 3. 通知偏好設定（目前僅為 App 內偏好開關，尚未串接真正推播）
CREATE TABLE IF NOT EXISTS NotificationSettings (
    user_email       VARCHAR(100) PRIMARY KEY,
    ai_alert         TINYINT(1) DEFAULT 1,
    community_reply  TINYINT(1) DEFAULT 1,
    system_announce  TINYINT(1) DEFAULT 1,
    updated_at       DATETIME   DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 4. 客服工單
CREATE TABLE IF NOT EXISTS SupportTickets (
    id         INT AUTO_INCREMENT PRIMARY KEY,
    user_email VARCHAR(100) NOT NULL,
    subject    VARCHAR(200) NOT NULL,
    message    TEXT         NOT NULL,
    status     VARCHAR(20)  DEFAULT 'open',
    created_at DATETIME     DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
