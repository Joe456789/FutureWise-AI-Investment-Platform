-- 新增使用者大頭貼欄位：儲存 /api/upload 回傳的圖片URL(例如 /uploads/xxxx.jpg)
ALTER TABLE Users ADD COLUMN avatar_url VARCHAR(255) DEFAULT NULL;
