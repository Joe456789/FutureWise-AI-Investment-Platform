-- 一次性遷移：讓教學頁的「AI 專屬閱讀路徑」可以記錄真實的個人閱讀進度
-- 在 Oracle 主機的 MySQL 上執行一次即可（只有 1 段 CREATE TABLE，直接執行即可）

CREATE TABLE IF NOT EXISTS ArticleReadLog (
    id         INT AUTO_INCREMENT PRIMARY KEY,
    user_email VARCHAR(100) NOT NULL,
    article_id VARCHAR(20)  NOT NULL,
    read_at    DATETIME     DEFAULT CURRENT_TIMESTAMP,
    UNIQUE KEY uniq_user_article (user_email, article_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
