-- 一次性遷移：社群檢舉與封鎖機制
-- 在 Oracle 主機的 MySQL 上執行一次即可
--
-- 注意：這個檔案有 2 段 CREATE TABLE，請用 Navicat/DBeaver 的「執行 SQL 文件 / Execute SQL Script」
-- 整份執行，不要用單句執行按鈕貼上整段文字執行，否則會遇到 1064 語法錯誤。

CREATE TABLE IF NOT EXISTS Reports (
    id            INT AUTO_INCREMENT PRIMARY KEY,
    reporter_email VARCHAR(100) NOT NULL,
    target_type   VARCHAR(10)  NOT NULL,   -- 'post' 或 'reply'
    target_id     INT          NOT NULL,
    reason        VARCHAR(200) NOT NULL,
    status        VARCHAR(20)  DEFAULT 'open',
    created_at    DATETIME     DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS Blocks (
    id            INT AUTO_INCREMENT PRIMARY KEY,
    blocker_email VARCHAR(100) NOT NULL,
    blocked_email VARCHAR(100) NOT NULL,
    created_at    DATETIME     DEFAULT CURRENT_TIMESTAMP,
    UNIQUE KEY uniq_block (blocker_email, blocked_email)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
