-- 一次性遷移：讓 Posts 記錄真正的發文者 email（用來判斷刪文權限）
-- 在 Oracle 主機的 MySQL 上執行一次即可
--
-- 注意：這個檔案只有 1 段 ALTER TABLE，直接執行即可，不會遇到之前
-- 多個 CREATE TABLE 被 Navicat 當成一句送出的問題。

ALTER TABLE Posts
    ADD COLUMN user_email VARCHAR(100) NULL AFTER user;

-- 補充說明：遷移前既有的舊貼文 user_email 會是 NULL，
-- 後端會擋掉這些舊貼文的刪除請求（找不到擁有者，視為無法刪除），
-- 這是預期行為，不影響新貼文正常運作。
