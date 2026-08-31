-- 一次性清理：刪除遷移前的舊測試貼文（這些貼文因為沒有 user_email，會被新版後端擋掉無法刪除）
-- 這會清空 Posts 和 Replies 兩張表，且無法復原，請確認裡面真的都是測試資料再執行
-- 執行前記得先跑過 migrate_add_post_ownership.sql

SET FOREIGN_KEY_CHECKS = 0;
TRUNCATE TABLE Replies;
TRUNCATE TABLE Posts;
SET FOREIGN_KEY_CHECKS = 1;
