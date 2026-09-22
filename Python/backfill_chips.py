# 程式名稱：backfill_chips.py
# 功能：自動補齊 MySQL/MSSQL 中缺失的三大法人資料 (0 值補正)
# 修正：改用多股指標 + 支援手動指定日期範圍補齊

import requests
import pandas as pd
import time
from sqlalchemy import create_engine, text
from datetime import datetime, timedelta
from config import MYSQL_CONN_STR
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ==========================================
# 設定區
# FULL_SCAN = True  → 第一次使用，掃描資料庫所有歷史資料
# FULL_SCAN = False → 之後日常使用，只掃最近 90 天
# FORCE_START/END   → 手動強制補齊特定日期區間
# ==========================================
FULL_SCAN   = False   # ← 第一次執行請設 True，之後改 False
FORCE_START = "2026-02-25"   # 例如 "2026-04-01"
FORCE_END   = "2026-04-22"   # 例如 "2026-05-10"

# TWSE 公開 API 最早可查詢日期（2012-05-02 = 民國 101年 05月02日）
TWSE_API_MIN_DATE = datetime(2012, 5, 2)

# 1. 初始化連線
print(f"Connecting to DB: {MYSQL_CONN_STR.split('@')[-1]}")
engine = create_engine(MYSQL_CONN_STR)

# 用多檔股票交叉比對，只要其中「大多數」法人資料為 0 就認定為缺失日
INDICATOR_TICKERS = ['2330', '2454', '2317', '2382', '2412']

def get_missing_dates_smart(full_scan=False):
    """
    改良版偵測：同時掃描多檔指標股，
    只要有 3 檔以上的法人資料全為 0，就認定該日為缺失。
    full_scan=True 時掃全部歷史，False 只掃近 90 天。
    """
    ticker_list = "', '".join(INDICATOR_TICKERS)
    
    if full_scan:
        print("   模式：全量掃描（資料庫所有歷史資料）...")
        query = f"""
        SELECT trade_date, ticker
        FROM StockPrice 
        WHERE ticker IN ('{ticker_list}')
          AND Foreign_Buy = 0 
          AND Trust_Buy = 0 
          AND Dealer_Buy = 0
        ORDER BY trade_date ASC
        """
        with engine.connect() as conn:
            df = pd.read_sql(text(query), conn)
    else:
        start_date = (datetime.now() - timedelta(days=90)).date()
        print(f"   模式：近 90 天掃描（{start_date} 起）...")
        query = f"""
        SELECT trade_date, ticker
        FROM StockPrice 
        WHERE ticker IN ('{ticker_list}')
          AND Foreign_Buy = 0 
          AND Trust_Buy = 0 
          AND Dealer_Buy = 0
          AND trade_date >= :start_date
        ORDER BY trade_date DESC
        """
        with engine.connect() as conn:
            df = pd.read_sql(text(query), conn, params={"start_date": start_date})
    
    if df.empty:
        return []

    # 計算每一天有幾檔指標股的法人資料為 0
    date_counts = df.groupby('trade_date')['ticker'].count()
    
    # 超過 3 檔都為 0 → 認定為缺失日
    missing_raw = date_counts[date_counts >= 3].index.tolist()
    
    missing_dates = []
    skipped_old = 0
    for d in missing_raw:
        dt = pd.to_datetime(d)
        if dt < TWSE_API_MIN_DATE:
            skipped_old += 1
            continue  # TWSE API 最早只能查到 2012-05-02，更早的資料永遠取不到
        if dt.weekday() < 5:  # 過濾假日
            missing_dates.append(dt)
    
    if skipped_old > 0:
        print(f"   ℹ️  已略過 {skipped_old} 個 2012-05-02 以前的舊日期（TWSE API 無法查詢）")
    
    return sorted(missing_dates)

def get_dates_from_range(start_str, end_str):
    """手動指定日期區間，回傳所有交易日"""
    start = datetime.strptime(start_str, "%Y-%m-%d")
    end   = datetime.strptime(end_str,   "%Y-%m-%d")
    dates = []
    cur = start
    while cur <= end:
        if cur.weekday() < 5:
            dates.append(cur)
        cur += timedelta(days=1)
    return dates

def fetch_day_chips(dt):
    """抓取特定日期的全市場法人資料"""
    all_chips = []
    dt_key = dt.strftime("%Y-%m-%d")
    dt_twse = dt.strftime("%Y%m%d")
    tw_yr = dt.year - 1911
    dt_tpex = f"{tw_yr}/{dt.strftime('%m/%d')}"
    
    session = requests.Session()
    session.headers.update({'User-Agent': 'Mozilla/5.0'})

    # --- 1. 上市公司 (TWSE) ---
    try:
        url = f"https://www.twse.com.tw/rwd/zh/fund/T86?date={dt_twse}&selectType=ALLBUT0999&response=json"
        res = session.get(url, timeout=15, verify=False).json()
        if res.get('stat') == 'OK' and 'data' in res:
            fields = res['fields']
            c_idx = fields.index('證券代號')
            f_idx = next(i for i, f in enumerate(fields) if ('外資' in f or '外陸資' in f) and '買賣超' in f)
            t_idx = next(i for i, f in enumerate(fields) if '投信' in f and '買賣超' in f)
            d_idx = next(i for i, f in enumerate(fields) if f == '自營商買賣超股數')
            for row in res['data']:
                all_chips.append({
                    'ticker': row[c_idx].strip(), 
                    'foreign': float(row[f_idx].replace(',', '')) / 1000, 
                    'trust':   float(row[t_idx].replace(',', '')) / 1000, 
                    'dealer':  float(row[d_idx].replace(',', '')) / 1000
                })
            print(f"  [TWSE] {dt_key} 取得 {len(res['data'])} 筆")
        else:
            print(f"  [TWSE] {dt_key} 無資料 (可能為假日或休市): {res.get('stat')}")
    except Exception as e:
        print(f"  [TWSE] {dt_key} 失敗: {e}")

    time.sleep(2)

    # --- 2. 上櫃公司 (TPEX) ---
    try:
        url = f"https://www.tpex.org.tw/web/stock/3insti/daily_trade/3itrade_hedge_result.php?l=zh-tw&o=json&se=EW&t=D&d={dt_tpex}"
        res = session.get(url, timeout=15, verify=False).json()
        tpex_rows = res.get('tables', [{}])[0].get('data', [])
        if tpex_rows:
            for row in tpex_rows:
                all_chips.append({
                    'ticker':  str(row[0]).strip(),
                    'foreign': float(str(row[10]).replace(',', '')) / 1000,
                    'trust':   float(str(row[13]).replace(',', '')) / 1000,
                    'dealer':  (float(str(row[16]).replace(',', '')) + float(str(row[19]).replace(',', ''))) / 1000
                })
            print(f"  [TPEX] {dt_key} 取得 {len(tpex_rows)} 筆")
        else:
            print(f"  [TPEX] {dt_key} 無資料")
    except Exception as e:
        print(f"  [TPEX] {dt_key} 失敗: {e}")

    return all_chips

def update_db(dt, chips):
    if not chips:
        return
    dt_str = dt.strftime("%Y-%m-%d")
    
    try:
        with engine.begin() as conn:
            updated = 0
            for item in chips:
                result = conn.execute(text("""
                    UPDATE StockPrice 
                    SET Foreign_Buy = :f, Trust_Buy = :t, Dealer_Buy = :d
                    WHERE ticker = :ticker AND trade_date = :dt
                """), {
                    "f":      item['foreign'],
                    "t":      item['trust'],
                    "d":      item['dealer'],
                    "ticker": item['ticker'],
                    "dt":     dt_str
                })
                updated += result.rowcount
            print(f"  ✅ {dt_str} 成功更新 {updated} 筆記錄")
    except Exception as e:
        print(f"  ❌ Update error: {e}")

def main():
    # 判斷執行模式
    if FORCE_START and FORCE_END:
        print(f"🔧 強制補齊模式：{FORCE_START} ～ {FORCE_END}")
        target_dates = get_dates_from_range(FORCE_START, FORCE_END)
        print(f"   共 {len(target_dates)} 個交易日待處理")
    else:
        if FULL_SCAN:
            print("🔍 Step 1: 全量掃描模式 - 掃描資料庫所有歷史資料（首次執行建議使用）...")
        else:
            print("🔍 Step 1: 日常掃描模式 - 掃描最近 90 天缺失的法人資料...")

        target_dates = get_missing_dates_smart(full_scan=FULL_SCAN)
        
        if not target_dates:
            if FULL_SCAN:
                print("✅ Done: 資料庫所有歷史資料中皆未發現明顯缺失。")
                print("\n💡 下次執行請將程式頂部的 FULL_SCAN 改為 False，改用日常掃描模式即可。")
            else:
                print("✅ Done: 近 90 天內無明顯缺失的法人資料。")
                print("\n💡 如果您確定有特定日期資料為 0，請在程式頂部設定：")
                print('   FORCE_START = "2026-04-01"')
                print('   FORCE_END   = "2026-05-10"')
            return

        print(f"⚠️  找到 {len(target_dates)} 個缺失日期：")
        for d in target_dates:
            print(f"   - {d.strftime('%Y-%m-%d')}")

    print(f"\n🚀 開始補齊 {len(target_dates)} 個日期的法人資料...")
    for i, dt in enumerate(target_dates):
        print(f"\n[進度 {i+1}/{len(target_dates)}] 📥 Processing {dt.strftime('%Y-%m-%d')}...")
        chips = fetch_day_chips(dt)
        if chips:
            update_db(dt, chips)
        else:
            print(f"  ⚠️  Warning: 無法從交易所取得 {dt.strftime('%Y-%m-%d')} 的資料（可能是假日或休市）")
        
        time.sleep(3)

    print("\n🎉 Mission Accomplished: 所有日期處理完成！")
    if FULL_SCAN:
        print("💡 下次執行請將程式頂部的 FULL_SCAN 改為 False。")

if __name__ == "__main__":
    main()
