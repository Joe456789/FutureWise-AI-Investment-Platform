# 程式名稱：test_email.py
# 功能：獨立測試 SMTP 寄信設定，直接印出真正的失敗原因（不經過 API，也不會印出密碼）。
# 用法（在伺服器上，跟 config.py 同一個資料夾）：
#     python3 test_email.py 你的收件信箱@gmail.com

import sys
import smtplib
from email.mime.text import MIMEText
from email.header import Header

from config import SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASSWORD, SMTP_FROM, EMAIL_ENABLED


def main():
    if len(sys.argv) < 2:
        print("用法：python3 test_email.py 收件信箱")
        return
    to = sys.argv[1]

    print("--- 目前讀到的設定（不顯示密碼內容）---")
    print(f"SMTP_HOST     = {SMTP_HOST or '(空的)'}")
    print(f"SMTP_PORT     = {SMTP_PORT}")
    print(f"SMTP_USER     = {SMTP_USER or '(空的)'}")
    print(f"SMTP_PASSWORD = {'已設定，長度 %d 碼' % len(SMTP_PASSWORD) if SMTP_PASSWORD else '(空的)'}")
    print(f"SMTP_FROM     = {SMTP_FROM or '(空的)'}")
    if SMTP_PASSWORD and len(SMTP_PASSWORD) != 16:
        print("⚠️ Gmail 應用程式密碼應該是 16 碼（4組字連在一起，中間不能有空白或引號），長度不對很可能就是問題所在")
    if SMTP_USER and not SMTP_USER.isascii():
        print("❌ SMTP_USER 含有中文字，看起來還是範例文字，請換成你真正的 Gmail 地址（例如 abc@gmail.com）")
        return
    if SMTP_PASSWORD and (not SMTP_PASSWORD.isascii() or " " in SMTP_PASSWORD or "\"" in SMTP_PASSWORD or "'" in SMTP_PASSWORD):
        print("❌ SMTP_PASSWORD 含有空白、引號或非英數字元，請改成 16 個字母連在一起")
        return
    if not EMAIL_ENABLED:
        print("❌ SMTP 設定不完整（HOST/USER/PASSWORD 有空的），系統不會寄信。請檢查 .env")
        return

    msg = MIMEText("這是 FutureWise AI 的寄信測試信，收到代表 SMTP 設定正確。", "plain", "utf-8")
    msg["Subject"] = Header("【FutureWise AI】寄信測試", "utf-8")
    msg["From"] = SMTP_FROM
    msg["To"] = to

    try:
        if SMTP_PORT == 465:
            server = smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT, timeout=15)
        else:
            server = smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=15)
            server.starttls()
        with server:
            server.login(SMTP_USER, SMTP_PASSWORD)
            refused = server.sendmail(SMTP_FROM, [to], msg.as_string())
        print(f"✅ 已交給 SMTP 伺服器寄出（拒收清單：{refused or '無'}）。請到 {to} 查看，也檢查垃圾郵件匣")
    except smtplib.SMTPAuthenticationError as e:
        print(f"❌ 登入 Gmail 失敗（{e.smtp_code}）：帳號或應用程式密碼不對。")
        print("   → 確認 SMTP_USER 是產生應用程式密碼的那個 Gmail，且密碼是『應用程式密碼』(16碼)，不是登入密碼；")
        print("   → 密碼若曾重新產生，舊的會立刻失效。")
    except (smtplib.SMTPConnectError, smtplib.SMTPServerDisconnected, TimeoutError, OSError) as e:
        print(f"❌ 連不上 {SMTP_HOST}:{SMTP_PORT}：{type(e).__name__}: {e}")
        print("   → 可能是伺服器防火牆/Oracle 安全清單擋住對外的 587 埠，可改試 SMTP_PORT=465")
    except smtplib.SMTPException as e:
        print(f"❌ SMTP 錯誤：{type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
