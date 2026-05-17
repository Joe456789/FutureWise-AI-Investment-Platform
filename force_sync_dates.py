# 程式名稱：force_sync_dates.py
# 功能：強力補抓指定區間的全台股資料 (價格 + 指標 + 籌碼) 並同步至資料庫

import yfinance as yf
import pandas as pd
import ta
import time
import requests
import os
import random
import numpy as np
from sqlalchemy import create_engine, text
from datetime import datetime, timedelta
from config import MYSQL_CONN_STR
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# --- 配置區 ---
START_DATE = "2026-04-01"  # 起始日期
END_DATE = "2026-05-08"    # 結束日期
BATCH_SIZE = 20            # 每 20 支股票寫入一次資料庫

# 欄位對應 (與 cloud_crawler_TWSE.py 一致)
DB_COLUMN_MAP = {
    "股票代碼": "ticker", "股票名稱": "stock_name", "交易日期": "trade_date",
    "開盤價": "open_price", "最高價": "high_price", "最低價": "low_price", "收盤價": "close_price", "成交量": "volume",
    "5日均線": "MA_5", "10日均線": "MA_10", "20日均線": "MA_20", 
    "60日均線": "MA_60", "120日均線": "MA_120", "240日均線": "MA_240",
    "RSI_6": "RSI_6", "RSI_14": "RSI_14", "MACD_快線": "MACD_DIF", "MACD_慢線": "MACD_Signal", "MACD_柱狀": "MACD_Hist",
    "K值": "KD_K", "D值": "KD_D", "ATR": "ATR", "量能比": "Vol_Ratio", "5日乖離": "Bias_5", "20日乖離": "Bias_20",
    "漲跌幅_1日": "Change_1D", "漲跌幅_3日": "Change_3D", "漲跌幅_5日": "Change_5D", "跳空缺口": "Gap",
    "外資買賣超": "Foreign_Buy", "投信買賣超": "Trust_Buy", "自營商買賣超": "Dealer_Buy"
}

engine = create_engine(MYSQL_CONN_STR)

def get_trading_dates(start, end):
    # 利用大盤資料取得真正的交易日清單
    tk = yf.Ticker("^TWII")
    df = tk.history(start=start, end=end)
    return df.index

def fetch_all_market_chips(trading_dates):
    all_chips = []
    session = requests.Session()
    session.headers.update({'User-Agent': 'Mozilla/5.0'})
    
    for d in trading_dates:
        dt_key = d.strftime("%Y-%m-%d")
        dt_twse = d.strftime("%Y%m%d")
        tw_yr = d.year - 1911
        dt_tpex = f"{tw_yr}/{d.strftime('%m/%d')}"
        
        print(f"   Downloading Chips for {dt_key}...")
        
        # TWSE
        try:
            url = f"https://www.twse.com.tw/rwd/zh/fund/T86?date={dt_twse}&selectType=ALLBUT0999&response=json"
            res = session.get(url, timeout=10, verify=False).json()
            if res.get('stat') == 'OK' and 'data' in res:
                fields = res['fields']
                c_idx = fields.index('證券代號')
                f_idx = next(i for i, f in enumerate(fields) if ('外資' in f or '外陸資' in f) and '買賣超' in f)
                t_idx = next(i for i, f in enumerate(fields) if '投信' in f and '買賣超' in f)
                d_idx = next(i for i, f in enumerate(fields) if f == '自營商買賣超股數')
                for row in res['data']:
                    all_chips.append({
                        'date_str': dt_key, 'ticker': row[c_idx].strip(), 
                        'f': float(row[f_idx].replace(',', '')) / 1000, 
                        't': float(row[t_idx].replace(',', '')) / 1000, 
                        'd': float(row[d_idx].replace(',', '')) / 1000
                    })
        except: pass

        # TPEX
        try:
            url = f"https://www.tpex.org.tw/web/stock/3insti/daily_trade/3itrade_hedge_result.php?l=zh-tw&o=json&se=EW&t=D&d={dt_tpex}"
            res = session.get(url, timeout=10, verify=False).json()
            if 'aaData' in res:
                for row in res['aaData']:
                    all_chips.append({
                        'date_str': dt_key, 'ticker': str(row[0]).strip(), 
                        'f': float(str(row[10]).replace(',', '')) / 1000, 
                        't': float(str(row[13]).replace(',', '')) / 1000, 
                        'd': (float(str(row[16]).replace(',', '')) + float(str(row[19]).replace(',', ''))) / 1000
                    })
        except: pass
        time.sleep(2)

    if all_chips:
        df = pd.DataFrame(all_chips)
        return df.groupby(['date_str', 'ticker']).sum()
    return pd.DataFrame()

def calculate_indicators(df):
    if len(df) < 5: return df
    c = df["收盤價"]
    for d in [5, 10, 20, 60, 120, 240]: df[f"{d}日均線"] = c.rolling(d).mean()
    df["RSI_6"] = ta.momentum.rsi(c, window=6)
    df["RSI_14"] = ta.momentum.rsi(c, window=14)
    macd = ta.trend.MACD(c)
    df["MACD_快線"] = macd.macd(); df["MACD_慢線"] = macd.macd_signal(); df["MACD_柱狀"] = macd.macd_diff()
    stoch = ta.momentum.StochasticOscillator(high=df["最高價"], low=df["最低價"], close=c, window=9)
    df["K值"] = stoch.stoch(); df["D值"] = stoch.stoch_signal()
    df["ATR"] = ta.volatility.average_true_range(df["最高價"], df["最低價"], c, window=14)
    df["量能比"] = (df["成交量"] / df["成交量"].rolling(5).mean()).fillna(0)
    df["5日乖離"] = ((c - df["5日均線"]) / df["5日均線"]) * 100
    df["20日乖離"] = ((c - df["20日均線"]) / df["20日均線"]) * 100
    df["漲跌幅_1日"] = c.pct_change(1)
    df["漲跌幅_3日"] = c.pct_change(3)
    df["漲跌幅_5日"] = c.pct_change(5)
    df["跳空缺口"] = df["開盤價"] - c.shift(1)
    return df

def save_to_db(batch_df):
    if batch_df.empty: return
    try:
        with engine.begin() as conn:
            for t_code in batch_df['ticker'].unique():
                dates = batch_df[batch_df['ticker'] == t_code]['trade_date'].tolist()
                date_strs = ",".join([f"'{d}'" for d in dates])
                conn.execute(text(f"DELETE FROM StockPrice WHERE ticker = '{t_code}' AND trade_date IN ({date_strs})"))
            batch_df.to_sql("StockPrice", conn, if_exists="append", index=False)
    except Exception as e:
        print(f"Database Error: {e}")

def main():
    print(f"Force Sync Mode: {START_DATE} to {END_DATE}")
    trading_dates = get_trading_dates(START_DATE, END_DATE)
    print(f"Trading Days Found: {len(trading_dates)}")
    
    print("Fetching Global Institutional Data...")
    chips_master = fetch_all_market_chips(trading_dates)
    
    # 取得股票名單
    from cloud_crawler_TWSE import get_tw_stock_map
    stock_map = get_tw_stock_map()
    tickers = sorted(stock_map.keys())
    
    batch_list = []
    for i, code in enumerate(tickers):
        print(f"[{i+1}/{len(tickers)}] Syncing {code}...", end="\r")
        try:
            tk = yf.Ticker(f"{code}.TW" if code.isdigit() else f"{code}")
            df = tk.history(start=(datetime.strptime(START_DATE, "%Y-%m-%d") - timedelta(days=300)).strftime("%Y-%m-%d"), end=END_DATE)
            if df.empty: 
                tk = yf.Ticker(f"{code}.TWO")
                df = tk.history(start=(datetime.strptime(START_DATE, "%Y-%m-%d") - timedelta(days=300)).strftime("%Y-%m-%d"), end=END_DATE)
            if df.empty: continue
            
            df = df.reset_index().rename(columns={"Date":"交易日期","Open":"開盤價","High":"最高價","Low":"最低價","Close":"收盤價","Volume":"成交量"})
            df["交易日期"] = pd.to_datetime(df["交易日期"]).dt.date
            df["股票代碼"] = code
            df["股票名稱"] = stock_map[code]
            
            df = calculate_indicators(df)
            
            # 只保留目標區間
            target_start = datetime.strptime(START_DATE, "%Y-%m-%d").date()
            df = df[df["交易日期"] >= target_start]
            
            # 合併籌碼
            df["date_str"] = df["交易日期"].apply(lambda x: x.strftime("%Y-%m-%d"))
            df = df.merge(chips_master.reset_index(), left_on=["date_str", "股票代碼"], right_on=["date_str", "ticker"], how="left")
            df["外資買賣超"] = df["f"].fillna(0); df["投信買賣超"] = df["t"].fillna(0); df["自營商買賣超"] = df["d"].fillna(0)
            
            # 清理並準備寫入
            final_cols = [c for c in DB_COLUMN_MAP.keys() if c in df.columns]
            df_to_save = df[final_cols].rename(columns=DB_COLUMN_MAP)
            df_to_save = df_to_save.fillna(0).replace([np.inf, -np.inf], 0)
            
            batch_list.append(df_to_save)
            
            if len(batch_list) >= BATCH_SIZE:
                save_to_db(pd.concat(batch_list))
                batch_list = []
                
            time.sleep(random.uniform(1, 2))
        except: pass

    if batch_list:
        save_to_db(pd.concat(batch_list))
    
    print("\nForce Sync Completed Successfully!")

if __name__ == "__main__":
    main()
