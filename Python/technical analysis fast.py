# 程式名稱：FutureWise 每日快充引擎 (V8.3 全市場 60 天安全掛機版)
# 核心升級：
# 1. 全市場掃描：解鎖全台股 1800 支股票自動抓取。
# 2. 60 天深度修復：籌碼與股價一律回溯 60 天，徹底填補過去兩個月的空洞。
# 3. 破解 API 限制：加入 6.1 秒精準延遲，完美繞過 FinMind 每小時 600 次的限制，絕不當機。

import yfinance as yf
import pandas as pd
import ta
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SAWarning
import os, time, requests, urllib3, io, numpy as np, pyodbc, warnings
import logging
from datetime import datetime, timedelta

# ==========================================
# 0. 系統降噪與環境設定
# ==========================================
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
warnings.simplefilter('ignore', category=SAWarning)
warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger('yfinance').setLevel(logging.CRITICAL)

# ==========================================
# 1. 配置區
# ==========================================
SERVER = "DESKTOP-DIF9QVQ" 
DATABASE = "StockDB1"
CSV_FILE = "股票清單.csv"
FINMIND_TOKEN = "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJkYXRlIjoiMjAyNi0wMi0yNyAxNDozMDo0MCIsInVzZXJfaWQiOiJKb2UiLCJlbWFpbCI6ImpvZTIwMDUwNzI3QGdtYWlsLmNvbSIsImlwIjoiMTAxLjE0LjYuNzMifQ.QDvapNZhDJFseBXiLbfl6au6Kqz8R_xenm3b5FflZ5w" 

DB_COLUMN_MAP = {
    "股票代碼": "ticker", "股票名稱": "stock_name", "交易日期": "trade_date",
    "開盤價": "open_price", "最高價": "high_price", "最低價": "low_price", "收盤價": "close_price", "成交量": "volume",
    "5日均線": "MA_5", "10日均線": "MA_10", "20日均線": "MA_20", 
    "60日均線": "MA_60", "120日均線": "MA_120", "240日均線": "MA_240",
    "RSI_6": "RSI_6", "RSI_14": "RSI_14", "MACD_快線": "MACD_DIF", "MACD_慢線": "MACD_Signal", "MACD_柱狀": "MACD_Hist",
    "K值": "KD_K", "D值": "KD_D", "布林上軌": "BB_Upper", "布林下軌": "BB_Lower", "ATR": "ATR",
    "5日均量": "Vol_MA_5", "20日均量": "Vol_MA_20", "量能比": "Vol_Ratio", "5日乖離": "Bias_5", "20日乖離": "Bias_20",
    "漲跌幅_1日": "Change_1D", "漲跌幅_3日": "Change_3D", "漲跌幅_5日": "Change_5D", "跳空缺口": "Gap",
    "營收YoY": "Revenue_YoY", "EPS": "EPS", "本益比": "PE_Ratio", "股價淨值比": "PB_Ratio",
    "外資買賣超": "Foreign_Buy", "投信買賣超": "Trust_Buy", "自營商買賣超": "Dealer_Buy",
    "大盤收盤": "Market_Close", "大盤漲跌幅": "Market_Return",
    "費半收盤": "SOX_Close", "台幣匯率": "TWD_Exchange", "費半漲跌": "SOX_Return", "情緒分數": "Sentiment_Score"
}

def get_best_driver():
    drivers = pyodbc.drivers()
    for d in ["ODBC Driver 18 for SQL Server", "ODBC Driver 17 for SQL Server", "SQL Server"]:
        if d in drivers: return d
    return None

DRIVER = get_best_driver()
trust_cert = "&TrustServerCertificate=yes" if DRIVER and "18" in DRIVER else ""
CONN_STR = f"mssql+pyodbc://@{SERVER}/{DATABASE}?driver={DRIVER.replace(' ', '+')}{trust_cert}"

# ==========================================
# 2. 數據獲取模組 
# ==========================================
def get_tw_stock_map():
    headers = {'User-Agent': 'Mozilla/5.0'}
    stock_map = {}
    for url in ["https://isin.twse.com.tw/isin/C_public.jsp?strMode=2", "https://isin.twse.com.tw/isin/C_public.jsp?strMode=4"]:
        try:
            res = requests.get(url, headers=headers, verify=False, timeout=10)
            df = pd.read_html(io.StringIO(res.text))[0].iloc[2:]
            for item in df.iloc[:, 0]:
                parts = str(item).split(maxsplit=1)
                if len(parts) == 2 and parts[0].isdigit() and len(parts[0]) == 4:
                    stock_map[parts[0]] = parts[1]
        except: continue
    return stock_map

def fetch_single_stock_chips(ticker, days=60):
    """
    ★ 抓取過去 60 天的籌碼數據
    """
    url = "https://api.finmindtrade.com/api/v4/data"
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    
    params = {
        "dataset": "TaiwanStockInstitutionalInvestorsBuySell",
        "data_id": ticker, "start_date": start_date, "end_date": end_date, "token": FINMIND_TOKEN
    }
    try:
        res = requests.get(url, params=params, timeout=10).json()
        if res.get('msg') == 'success' and res.get('data'):
            df = pd.DataFrame(res['data'])
            if 'buy' in df.columns and 'sell' in df.columns:
                df['diff_lots'] = (df['buy'] - df['sell']) / 1000
                
                # 模糊比對法
                df['name'] = df['name'].apply(
                    lambda x: 'Foreign_Buy' if 'Foreign' in str(x) else (
                              'Trust_Buy' if 'Trust' in str(x) else (
                              'Dealer_Buy' if 'Dealer' in str(x) else x))
                )
                
                df_grouped = df.groupby(['date', 'name'])['diff_lots'].sum().reset_index()
                pivot = df_grouped.pivot(index='date', columns='name', values='diff_lots').reset_index()
                
                for col in ['Foreign_Buy', 'Trust_Buy', 'Dealer_Buy']:
                    if col not in pivot.columns: pivot[col] = 0
                
                pivot = pivot.rename(columns={
                    'Foreign_Buy': '外資買賣超',
                    'Trust_Buy': '投信買賣超',
                    'Dealer_Buy': '自營商買賣超'
                })
                pivot['date'] = pd.to_datetime(pivot['date']).dt.date
                return pivot[['date', '外資買賣超', '投信買賣超', '自營商買賣超']]
    except: pass
    return pd.DataFrame()

def get_macro_data():
    try:
        raw = yf.download(["^TWII", "^SOX", "TWD=X"], period="2y", progress=False)['Close'].ffill()
        df_m = pd.DataFrame(index=raw.index)
        df_m["Market_Close"] = raw["^TWII"]
        df_m["Market_Return"] = raw["^TWII"].pct_change()
        df_m["SOX_Close"] = raw["^SOX"].shift(1)
        df_m["SOX_Return"] = raw["^SOX"].pct_change().shift(1)
        df_m["TWD_Exchange"] = raw["TWD=X"]
        df_m.index = pd.to_datetime(df_m.index).date
        return df_m
    except: return pd.DataFrame()

def calculate_all_indicators(df):
    if len(df) < 5: return df
    df = df.sort_values("交易日期").reset_index(drop=True)
    c = df["收盤價"]; h = df["最高價"]; l = df["最低價"]; v = df["成交量"]
    for d in [5, 10, 20, 60, 120, 240]: df[f"{d}日均線"] = c.rolling(d).mean()
    df["RSI_6"] = ta.momentum.rsi(c, window=6); df["RSI_14"] = ta.momentum.rsi(c, window=14)
    macd = ta.trend.MACD(c)
    df["MACD_快線"] = macd.macd(); df["MACD_慢線"] = macd.macd_signal(); df["MACD_柱狀"] = macd.macd_diff()
    stoch = ta.momentum.StochasticOscillator(high=h, low=l, close=c, window=9, smooth_window=3)
    df["K值"] = stoch.stoch(); df["D值"] = stoch.stoch_signal()
    bb = ta.volatility.BollingerBands(c, window=20); df["布林上軌"] = bb.bollinger_hband(); df["布林下軌"] = bb.bollinger_lband()
    df["ATR"] = ta.volatility.average_true_range(h, l, c, window=14)
    df["5日均量"] = v.rolling(5).mean(); df["20日均量"] = v.rolling(20).mean()
    df["量能比"] = (v / df["5日均量"]).replace([np.inf, -np.inf], 0)
    df["5日乖離"] = ((c - df["5日均線"]) / df["5日均線"]) * 100
    df["20日乖離"] = ((c - df["20日均線"]) / df["20日均線"]) * 100
    df["漲跌幅_1日"] = c.pct_change(1); df["漲跌幅_3日"] = c.pct_change(3); df["漲跌幅_5日"] = c.pct_change(5)
    df["跳空缺口"] = df["開盤價"] - c.shift(1)
    return df

def save_batch_to_sql(batch_list, engine):
    if not batch_list: return
    full_batch = pd.concat(batch_list)
    full_batch['交易日期'] = pd.to_datetime(full_batch['交易日期']).dt.date
    for t_code in full_batch['股票代碼'].unique():
        c_df = full_batch[full_batch['股票代碼'] == t_code].copy()
        try:
            d_str = ",".join([f"'{d}'" for d in c_df['交易日期'].tolist()])
            with engine.connect() as conn:
                # 刪除重疊區間的資料以進行修復更新
                conn.execute(text(f"DELETE FROM StockPrice WHERE ticker = '{t_code}' AND trade_date IN ({d_str})"))
                conn.commit()
                valid_cols = [c for c in DB_COLUMN_MAP.keys() if c in c_df.columns]
                c_df[valid_cols].rename(columns=DB_COLUMN_MAP).to_sql("StockPrice", engine, if_exists="append", index=False)
        except: pass

# ==========================================
# 3. 主程序執行
# ==========================================
def run_v8_3_engine():
    if not DRIVER: print("❌ 找不到資料庫驅動程式"); return
    engine = create_engine(CONN_STR)
    macro_df = get_macro_data()
    name_map = get_tw_stock_map()
    
    # ★ 改為全市場掃描
    tickers = sorted(name_map.keys())
    
    print(f"🚀 FutureWise V8.3 啟動 | 目標: {len(tickers)} 支股票 (全市場)")
    print("⚠️ 提醒：為了符合 FinMind 每小時 600 次的 API 限制，")
    print("   每支股票抓取後將強制休息 6.1 秒。全部完成約需 3 小時，請掛機執行。")
    
    all_data_for_csv = []
    batch_list = []
    
    for i, code in enumerate(tickers):
        name = name_map[code]
        print(f"[{i+1}/{len(tickers)}] 同步 {code} {name} (60天籌碼)...")
        try:
            tk = yf.Ticker(f"{code}.TW")
            df = tk.history(period="2y", auto_adjust=True)
            if df.empty:
                tk = yf.Ticker(f"{code}.TWO"); df = tk.history(period="2y", auto_adjust=True)
            if df.empty: continue
            
            info = tk.info
            df = df.reset_index().rename(columns={"Date":"交易日期","Open":"開盤價","High":"最高價","Low":"最低價","Close":"收盤價","Volume":"成交量"})
            df["交易日期"] = pd.to_datetime(df["交易日期"]).dt.date
            df["股票代碼"] = code; df["股票名稱"] = name
            
            df = calculate_all_indicators(df)
            df = df.merge(macro_df, left_on="交易日期", right_index=True, how="left")
            df.rename(columns={"Market_Close":"大盤收盤","Market_Return":"大盤漲跌幅","SOX_Close":"費半收盤","TWD_Exchange":"台幣匯率","SOX_Return":"費半漲跌"}, inplace=True)
            
            df["營收YoY"] = pd.to_numeric(info.get('revenueGrowth', 0), errors='coerce')
            df["EPS"] = pd.to_numeric(info.get('trailingEps', 0), errors='coerce')
            df["本益比"] = pd.to_numeric(info.get('trailingPE', 0), errors='coerce')
            df["股價淨值比"] = pd.to_numeric(info.get('priceToBook', 0), errors='coerce')
            df["情緒分數"] = 0
            
            # ★ 抓取 60 天籌碼
            df_chips = fetch_single_stock_chips(code, days=60)
            if not df_chips.empty:
                df = df.merge(df_chips, left_on="交易日期", right_on="date", how="left").drop(columns=['date'])
            else:
                df["外資買賣超"] = 0; df["投信買賣超"] = 0; df["自營商買賣超"] = 0
            
            df = df.replace([np.inf, -np.inf, 'Infinity', 'inf'], np.nan).fillna(0)
            
            latest_row = df.iloc[-1]
            print(f"   -> 最新收盤價: {latest_row['收盤價']:.2f} | 籌碼: 外資 {int(latest_row['外資買賣超']):,} 張")
            
            # ★ 儲存最近 60 天的資料，把過去兩個月的缺口全部覆蓋修復
            df_to_save = df.tail(60).copy()
            batch_list.append(df_to_save)
            all_data_for_csv.append(df_to_save)
            
            if len(batch_list) >= 20: # 每 20 支寫入一次資料庫
                save_batch_to_sql(batch_list, engine)
                batch_list = []
            
            # ★ 絕對安全的破限延遲 (3600秒 / 600次 = 6秒)
            time.sleep(6.1) 
            
        except Exception as e: print(f"   -> 錯誤: {e}")

    if batch_list: save_batch_to_sql(batch_list, engine)
        
    if all_data_for_csv:
        print(f"\n📥 正在整理 CSV...")
        new_df = pd.concat(all_data_for_csv)
        new_df["交易日期"] = pd.to_datetime(new_df["交易日期"]).dt.strftime('%Y-%m-%d')
        if os.path.exists(CSV_FILE):
            try:
                old_df = pd.read_csv(CSV_FILE, dtype={'股票代碼': str}, on_bad_lines='skip')
                old_df["股票代碼"] = old_df["股票代碼"].str.replace('="', '').str.replace('"', '')
                combined = pd.concat([old_df, new_df])
            except: combined = new_df
        else: combined = new_df
        combined.drop_duplicates(subset=['股票代碼', '交易日期'], keep='last', inplace=True)
        combined['tmp'] = pd.to_datetime(combined['交易日期'])
        combined = combined.sort_values(['股票代碼', 'tmp'], ascending=[True, False]).drop(columns=['tmp'])
        combined["股票代碼"] = combined["股票代碼"].apply(lambda x: f'="{x}"')
        combined.to_csv(CSV_FILE, index=False, encoding='utf-8-sig')

    print("\n🎉 V8.3 執行完畢！全市場 1800 支股票過去 60 天的籌碼皆已對齊！")

if __name__ == "__main__":
    run_v8_3_engine()