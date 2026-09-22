-- 索引檢視（待辦清單第6項）：Posts表目前只有PRIMARY KEY，/api/posts每次都是
-- `SELECT ... FROM Posts ORDER BY created_at DESC`（沒有WHERE條件），
-- 隨社群貼文數增加，加這個索引能讓ORDER BY直接走索引排序，不用每次全表排序。
-- 這個索引不是急迫問題（目前貼文量遠小於StockPrice），可以找空檔跑就好。
CREATE INDEX idx_posts_created_at ON Posts (created_at);
