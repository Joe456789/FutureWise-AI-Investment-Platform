# 程式名稱：setup_community_db.py
# 功能：初始化社群資料表並遷移 JSON 資料至 MySQL/MSSQL (相容版)

import os
import json
from sqlalchemy import create_engine, text
from datetime import datetime
from config import MYSQL_CONN_STR

# 1. 連線至資料庫
print(f"Connecting to DB: {MYSQL_CONN_STR.split('@')[-1]}")
engine = create_engine(MYSQL_CONN_STR)

def get_create_sql(db_type):
    if db_type == "mysql":
        return [
            """
            CREATE TABLE IF NOT EXISTS Posts (
                id         INT AUTO_INCREMENT PRIMARY KEY,
                user       VARCHAR(100) NOT NULL,
                icon       VARCHAR(50)  DEFAULT 'User',
                created_at DATETIME     DEFAULT CURRENT_TIMESTAMP,
                sentiment  VARCHAR(20)  DEFAULT 'Neutral',
                tag        VARCHAR(50)  DEFAULT 'Observation',
                content    TEXT         NOT NULL,
                likes      INT          DEFAULT 0,
                comments   INT          DEFAULT 0
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
            """,
            """
            CREATE TABLE IF NOT EXISTS Replies (
                id         INT AUTO_INCREMENT PRIMARY KEY,
                post_id    INT          NOT NULL,
                author     VARCHAR(100) NOT NULL,
                content    TEXT         NOT NULL,
                created_at DATETIME     DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (post_id) REFERENCES Posts(id) ON DELETE CASCADE
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
            """
        ]
    else:
        # MSSQL 語法
        return [
            """
            IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='Posts' and xtype='U')
            CREATE TABLE Posts (
                id         INT IDENTITY(1,1) PRIMARY KEY,
                [user]     NVARCHAR(100) NOT NULL,
                icon       NVARCHAR(50)  DEFAULT 'User',
                created_at DATETIME     DEFAULT GETDATE(),
                sentiment  NVARCHAR(20)  DEFAULT 'Neutral',
                tag        NVARCHAR(50)  DEFAULT 'Observation',
                content    NVARCHAR(MAX) NOT NULL,
                likes      INT          DEFAULT 0,
                comments   INT          DEFAULT 0
            );
            """,
            """
            IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='Replies' and xtype='U')
            CREATE TABLE Replies (
                id         INT IDENTITY(1,1) PRIMARY KEY,
                post_id    INT          NOT NULL,
                author     NVARCHAR(100) NOT NULL,
                content    NVARCHAR(MAX) NOT NULL,
                created_at DATETIME     DEFAULT GETDATE(),
                FOREIGN KEY (post_id) REFERENCES Posts(id) ON DELETE CASCADE
            );
            """
        ]

def setup_db():
    try:
        db_type = "mysql" if "mysql" in MYSQL_CONN_STR.lower() else "mssql"
        print(f"Detected DB Type: {db_type}")

        current_dir = os.path.dirname(os.path.abspath(__file__))
        POSTS_DB_FILE = os.path.join(current_dir, "posts_db.json")

        with engine.begin() as conn:
            print("Running: Cleaning and Creating tables...")
            # 只有在遷移模式才刪除舊表確保欄位更新
            if os.path.exists(POSTS_DB_FILE):
                if db_type == "mysql":
                    conn.execute(text("SET FOREIGN_KEY_CHECKS = 0;"))
                    conn.execute(text("DROP TABLE IF EXISTS Replies;"))
                    conn.execute(text("DROP TABLE IF EXISTS Posts;"))
                    conn.execute(text("SET FOREIGN_KEY_CHECKS = 1;"))
                else:
                    try: conn.execute(text("DROP TABLE Replies;"))
                    except: pass
                    try: conn.execute(text("DROP TABLE Posts;"))
                    except: pass

            sqls = get_create_sql(db_type)
            for sql in sqls:
                conn.execute(text(sql))
            print("OK: Tables ready.")

        # 3. 遷移資料
        current_dir = os.path.dirname(os.path.abspath(__file__))
        POSTS_DB_FILE = os.path.join(current_dir, "posts_db.json")
        
        if os.path.exists(POSTS_DB_FILE):
            print(f"Data: Found legacy file {POSTS_DB_FILE}, starting migration...")
            with open(POSTS_DB_FILE, "r", encoding="utf-8") as f:
                old_posts = json.load(f)
            
            with engine.begin() as conn:
                if db_type == "mysql":
                    conn.execute(text("SET FOREIGN_KEY_CHECKS = 0;"))
                    conn.execute(text("TRUNCATE TABLE Replies;"))
                    conn.execute(text("TRUNCATE TABLE Posts;"))
                    conn.execute(text("SET FOREIGN_KEY_CHECKS = 1;"))
                else:
                    conn.execute(text("DELETE FROM Replies;"))
                    conn.execute(text("DELETE FROM Posts;"))

                for p in old_posts:
                    ts = p.get('time', int(datetime.now().timestamp() * 1000))
                    # 處理毫秒 -> 秒
                    if ts > 10**11: 
                        ts = ts / 1000
                    dt = datetime.fromtimestamp(ts)
                    
                    user_col = "[user]" if db_type == "mssql" else "user"
                    
                    res = conn.execute(text(f"""
                        INSERT INTO Posts ({user_col}, icon, created_at, sentiment, tag, content, likes, comments)
                        VALUES (:user, :icon, :created_at, :sentiment, :tag, :content, :likes, :comments)
                    """), {
                        "user": p.get('user', 'User'),
                        "icon": p.get('icon', 'User'),
                        "created_at": dt,
                        "sentiment": p.get('sentiment', 'Neutral'),
                        "tag": p.get('tag', 'Observation'),
                        "content": p.get('content', ''),
                        "likes": p.get('likes', 0),
                        "comments": p.get('comments', 0)
                    })
                    
                    if db_type == "mysql":
                        new_id = res.lastrowid
                    else:
                        new_id_res = conn.execute(text("SELECT @@IDENTITY"))
                        new_id = new_id_res.scalar()
                    
                    replies = p.get('replies', [])
                    for r in replies:
                        conn.execute(text("""
                            INSERT INTO Replies (post_id, author, content)
                            VALUES (:post_id, :author, :content)
                        """), {
                            "post_id": new_id,
                            "author": r.get('author', 'Anonymous'),
                            "content": r.get('content', '')
                        })
                
                print(f"Success: Migrated {len(old_posts)} posts.")
        else:
            print("Info: No legacy file found.")

    except Exception as e:
        print(f"Error occurred during setup.")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    setup_db()
