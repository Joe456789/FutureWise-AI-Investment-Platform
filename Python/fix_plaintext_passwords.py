# 一次性檢查/修復工具：找出 Users 表裡「password_hash」欄位不是 bcrypt 雜湊格式的帳號
# （例如當初用 Navicat 手動 insert 測試帳號、直接填了明文密碼），並把它們正確雜湊化。
#
# 用法：在你這台 Windows 電腦上執行（跟 Navicat 一樣直接連 Oracle 上的 MySQL）
#   pip install pymysql passlib bcrypt
#   python fix_plaintext_passwords.py
#
# 這支工具只會「讀取」再詢問確認，不會自動悄悄修改資料。

import re
import pymysql
from passlib.context import CryptContext

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
BCRYPT_PATTERN = re.compile(r"^\$2[aby]\$.{56}$")

# 這支請直接在 Oracle 主機上執行（SSH 進去跑），所以用 127.0.0.1，
# 跟 config.py 裡 Linux 分支的 DB_HOST 一致，不受 3306 對外防火牆規則影響。
# 密碼一律從環境變數讀取，不要寫死在程式碼裡
import os

DB_HOST = "127.0.0.1"
DB_PORT = 3306
DB_USER = "root"
DB_PASSWORD = os.environ.get("DB_PASSWORD", "")
DB_NAME = "StockDB1"

if not DB_PASSWORD:
    raise RuntimeError("請先設定環境變數 DB_PASSWORD，例如：DB_PASSWORD=你的密碼 python fix_plaintext_passwords.py")


def main():
    conn = pymysql.connect(
        host=DB_HOST, port=DB_PORT, user=DB_USER,
        password=DB_PASSWORD, database=DB_NAME, charset="utf8mb4"
    )
    cur = conn.cursor()
    cur.execute("SELECT id, email, password_hash FROM Users")
    rows = cur.fetchall()

    bad_rows = [r for r in rows if not (r[2] and BCRYPT_PATTERN.match(r[2]))]

    print(f"共 {len(rows)} 位使用者，其中 {len(bad_rows)} 位的密碼欄位不是 bcrypt 雜湊格式：")
    for uid, email, ph in bad_rows:
        preview = (ph[:10] + "...") if ph else "(空值)"
        print(f"  id={uid}  email={email}  目前存的值={preview}")

    if not bad_rows:
        print("全部都是正確的 bcrypt 雜湊，不需要處理。")
        conn.close()
        return

    print("\n注意：")
    print("- 如果上面列出的值『看起來就是明文密碼』，把它雜湊化之後，該使用者仍可用原密碼登入，不受影響。")
    print("- 如果值是『其他來源產生的雜湊』（不是明文），雜湊化後密碼會失效，該使用者要改用「忘記密碼」重設。")
    confirm = input("\n是否要把上面這些帳號的值直接雜湊化？輸入 yes 繼續，其他任意鍵取消：")

    if confirm.strip().lower() != "yes":
        print("已取消，未做任何變更。")
        conn.close()
        return

    for uid, email, ph in bad_rows:
        new_hash = pwd_context.hash(ph)
        cur.execute("UPDATE Users SET password_hash = %s WHERE id = %s", (new_hash, uid))
    conn.commit()
    print(f"已完成，共更新 {len(bad_rows)} 筆。")
    conn.close()


if __name__ == "__main__":
    main()
