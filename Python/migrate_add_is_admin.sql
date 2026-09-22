-- 新增管理員權限欄位，並把你自己的帳號設成管理員
ALTER TABLE Users ADD COLUMN is_admin BOOLEAN DEFAULT FALSE;
UPDATE Users SET is_admin = TRUE WHERE email = 'joe20050727@gmail.com';
