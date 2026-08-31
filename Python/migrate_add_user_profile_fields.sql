-- 一次性遷移：讓 Users 表可以存姓名與電話
-- 在 Oracle 主機的 MySQL 上執行一次即可

ALTER TABLE Users
    ADD COLUMN name  VARCHAR(50)  NULL AFTER email,
    ADD COLUMN phone VARCHAR(20)  NULL AFTER name;
