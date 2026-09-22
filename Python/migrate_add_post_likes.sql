-- 記錄「誰對哪篇貼文按過讚」，避免同一人重複呼叫 /api/posts/{id}/like 無限刷讚數/砍到0
CREATE TABLE IF NOT EXISTS PostLikes (
    user_email VARCHAR(255) NOT NULL,
    post_id INT NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (user_email, post_id)
);
