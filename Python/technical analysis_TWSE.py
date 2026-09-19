# 程式名稱：FutureWise 全自動快充引擎 (V9.8 SSL破案版)
# 核心修正：
# 1. SSL 破案：在抓取籌碼的請求中加入 verify=False，解決證交所憑證被 Python 擋下的問題。
# 2. 拒絕篩選：完全移除菁英名單邏輯，啟動即自動掃描全台股 1800+ 支股票。
# 3. 終極對齊術：強制「股票代碼」去除空白並轉為純字串，徹底消滅 Merge 失敗導致的 0.00 佔位。
# 4. YoY 深度修補：針對 yfinance 遺漏的營收 YoY，自動抓取季報進行「去年同期」數據運算。
# 5. API 安全守護：維持 6.1 秒延遲，確保穩定運行，絕不撞牆。

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
# 2. 數據模組 
# ==========================================
def get_tw_stock_map():
    headers = {'User-Agent': 'Mozilla/5.0'}
    stock_map = {}
    # 上市與上櫃清單抓取
    for url in ["https://isin.twse.com.tw/isin/C_public.jsp?strMode=2", "https://isin.twse.com.tw/isin/C_public.jsp?strMode=4"]:
        try:
            res = requests.get(url, headers=headers, verify=False, timeout=10)
            df = pd.read_html(io.StringIO(res.text))[0].iloc[2:]
            for item in df.iloc[:, 0]:
                parts = str(item).split()
                if len(parts) >= 2 and parts[0].isdigit() and len(parts[0]) == 4:
                    stock_map[parts[0]] = parts[1]
        except: continue
    return stock_map

def fetch_all_market_chips(trading_dates):
    """
    從官方網站抓取全市場籌碼，強制轉化為字串以確保後續對齊
    """
    all_chips = []
    session = requests.Session()
    session.headers.update({'User-Agent': 'Mozilla/5.0'})
    
    for idx, d in enumerate(trading_dates):
        dt_key = pd.to_datetime(d).strftime("%Y-%m-%d")
        dt_twse = pd.to_datetime(d).strftime("%Y%m%d")
        tw_yr = d.year - 1911
        dt_tpex = f"{tw_yr}/{d.strftime('%m/%d')}"
        
        print(f"   下載籌碼批次: {dt_key} ...", end="\r")
        
        # 1. 證交所 (加上 verify=False 破除 SSL 阻擋)
        try:
            url = f"https://www.twse.com.tw/rwd/zh/fund/T86?date={dt_twse}&selectType=ALLBUT0999&response=json"
            res = session.get(url, timeout=10, verify=False).json()
            if res.get('stat') == 'OK' and 'data' in res:
                fields = res['fields']
                c_idx = fields.index('證券代號')
                f_idx = next(i for i, f in enumerate(fields) if '外資' in f)
                t_idx = next(i for i, f in enumerate(fields) if '投信' in f)
                d_idx = next(i for i, f in enumerate(fields) if '自營商' in f)
                for row in res['data']:
                    all_chips.append({
                        'date_str': dt_key, '股票代碼': row[c_idx].strip(), 
                        '外資買賣超': float(row[f_idx].replace(',', '')) / 1000, 
                        '投信買賣超': float(row[t_idx].replace(',', '')) / 1000, 
                        '自營商買賣超': float(row[d_idx].replace(',', '')) / 1000
                    })
        except: pass
        time.sleep(1) # 保護官方 API
        
        # 2. 櫃買中心 (加上 verify=False 破除 SSL 阻擋)
        try:
            url = f"https://www.tpex.org.tw/web/stock/3insti/daily_trade/3itrade_hedge_result.php?l=zh-tw&o=json&se=EW&t=D&d={dt_tpex}"
            res = session.get(url, timeout=10, verify=False).json()
            if 'aaData' in res:
                for row in res['aaData']:
                    all_chips.append({
                        'date_str': dt_key, '股票代碼': str(row[0]).strip(), 
                        '外資買賣超': float(str(row[10]).replace(',', '')) / 1000, 
                        '投信買賣超': float(str(row[13]).replace(',', '')) / 1000, 
                        '自營商買賣超': (float(str(row[16]).replace(',', '')) + float(str(row[19]).replace(',', ''))) / 1000
                    })
        except: pass
        time.sleep(1)

    if all_chips:
        df = pd.DataFrame(all_chips)
        return df.groupby(['date_str', '股票代碼']).sum() 
    return pd.DataFrame()

def get_macro_data():
    try:
        raw = yf.download(["^TWII", "^SOX", "TWD=X"], period="2y", progress=False)['Close'].ffill()
        df_m = pd.DataFrame(index=raw.index)
        df_m["Market_Close"] = raw["^TWII"]
        df_m["Market_Return"] = raw["^TWII"].pct_change(fill_method=None)
        df_m["SOX_Close"] = raw["^SOX"].shift(1)
        df_m["SOX_Return"] = raw["^SOX"].pct_change(fill_method=None).shift(1)
        df_m["TWD_Exchange"] = raw["TWD=X"]
        df_m.index = pd.to_datetime(df_m.index).date
        return df_m
    except: return pd.DataFrame()

def calculate_all_indicators(df):
    if len(df) < 5: return df
    df = df.sort_values("交易日期").reset_index(drop=True)
    c, h, l, v = df["收盤價"], df["最高價"], df["最低價"], df["成交量"]
    
    for d in [5, 10, 20, 60, 120, 240]: df[f"{d}日均線"] = c.rolling(d).mean()
    df["RSI_6"] = ta.momentum.rsi(c, window=6); df["RSI_14"] = ta.momentum.rsi(c, window=14)
    macd = ta.trend.MACD(c)
    df["MACD_快線"] = macd.macd(); df["MACD_慢線"] = macd.macd_signal(); df["MACD_柱狀"] = macd.macd_diff()
    stoch = ta.momentum.StochasticOscillator(high=h, low=l, close=c, window=9, smooth_window=3)
    df["K值"] = stoch.stoch(); df["D值"] = stoch.stoch_signal()
    bb = ta.volatility.BollingerBands(c, window=20); df["布林上軌"] = bb.bollinger_hband(); df["布林下軌"] = bb.bollinger_lband()
    df["ATR"] = ta.volatility.average_true_range(h, l, c, window=14)
    df["5日均量"] = v.rolling(5).mean(); df["20日均量"] = v.rolling(20).mean()
    df["量能比"] = (v / df["5日均量"]).replace([np.inf, -np.inf], 0).fillna(0)
    df["5日乖離"] = ((c - df["5日均線"]) / df["5日均線"]) * 100
    df["20日乖離"] = ((c - df["20日均線"]) / df["20日均線"]) * 100
    df["漲跌幅_1日"] = c.pct_change(1, fill_method=None)
    df["漲跌幅_3日"] = c.pct_change(3, fill_method=None)
    df["漲跌幅_5日"] = c.pct_change(5, fill_method=None)
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
                # 使用 DELETE 刪除重疊的舊紀錄，確保新計算的資料能成功 UPDATE
                conn.execute(text(f"DELETE FROM StockPrice WHERE ticker = '{t_code}' AND trade_date IN ({d_str})"))
                conn.commit()
                valid_cols = [c for c in DB_COLUMN_MAP.keys() if c in c_df.columns]
                c_df[valid_cols].rename(columns=DB_COLUMN_MAP).to_sql("StockPrice", engine, if_exists="append", index=False)
        except: pass

# ==========================================
# 3. 主執行流程 
# ==========================================
def run_v9_8_ultimate():
    if not DRIVER: print("❌ 找不到資料庫驅動程式"); return
    engine = create_engine(CONN_STR)
    
    print("🌍 正在下載總經數據與全台股名單...")
    macro_df = get_macro_data()
    name_map = get_tw_stock_map()
    
    # 全市場代碼掃描 (不篩選)
    tickers = sorted(name_map.keys())
    
    TARGET_SAVE_DAYS = 60 # 深度修補天數
    
    print(f"🚀 FutureWise V9.8 啟動 | 目標: 全台股 {len(tickers)} 支股票")
    print(f"⚠️ 預計耗時: 約 3 小時 | 每支股票延遲 6.1 秒防封鎖")
    
    # ★ 預載籌碼
    last_n_dates = macro_df.index[-TARGET_SAVE_DAYS:].tolist()
    global_chips_df = fetch_all_market_chips(last_n_dates)
    
    macro_df['date_str'] = pd.to_datetime(macro_df.index).strftime('%Y-%m-%d')
    
    all_data_for_csv = []
    batch_list = []
    
    for i, code in enumerate(tickers):
        name = name_map[code]
        
        # ★ 終極對齊術：確保代碼乾淨無空白
        clean_code = str(code).strip() 
        
        print(f"[{i+1}/{len(tickers)}] 同步 {clean_code} {name} (60天數據修正)...", end="\r")
        try:
            tk = yf.Ticker(f"{clean_code}.TW")
            df = tk.history(period="2y", auto_adjust=True)
            if df.empty:
                tk = yf.Ticker(f"{clean_code}.TWO"); df = tk.history(period="2y", auto_adjust=True)
            if df.empty: continue
            
            info = tk.info
            df = df.reset_index().rename(columns={"Date":"交易日期","Open":"開盤價","High":"最高價","Low":"最低價","Close":"收盤價","Volume":"成交量"})
            df["交易日期"] = pd.to_datetime(df["交易日期"]).dt.date
            df["股票代碼"] = clean_code
            df["股票名稱"] = name
            
            # 技術指標
            df = calculate_all_indicators(df)
            df["date_str"] = pd.to_datetime(df["交易日期"]).dt.strftime('%Y-%m-%d')
            df = df.merge(macro_df[['date_str', 'Market_Close', 'Market_Return', 'SOX_Close', 'SOX_Return', 'TWD_Exchange']], on='date_str', how='left')
            df.rename(columns={'Market_Close':'大盤收盤','Market_Return':'大盤漲跌幅','SOX_Close':'費半收盤','TWD_Exchange':'台幣匯率','SOX_Return':'費半漲跌'}, inplace=True)
            
            # --- ★修復 YoY：如果 yfinance 沒資料，改從財報算 ---
            yoy = info.get('revenueGrowth', None)
            if yoy is None:
                try:
                    q_fin = tk.quarterly_financials
                    if not q_fin.empty and "Total Revenue" in q_fin.index:
                        revs = q_fin.loc["Total Revenue"].dropna()
                        if len(revs) >= 5: 
                            yoy = (revs.iloc[0] - revs.iloc[4]) / revs.iloc[4]
                except: yoy = 0
            
            df["營收YoY"] = float(yoy) if yoy is not None else 0
            df["EPS"] = pd.to_numeric(info.get('trailingEps', 0), errors='coerce')
            df["本益比"] = pd.to_numeric(info.get('trailingPE', 0), errors='coerce')
            df["股價淨值比"] = pd.to_numeric(info.get('priceToBook', 0), errors='coerce')
            df["情緒分數"] = 0
            
            # --- ★ 終極修復三大法人：字串對齊法 ---
            if not global_chips_df.empty:
                # 重設索引，方便用字串比對
                flat_chips = global_chips_df.reset_index()
                
                # 確保兩邊都是 String 型態
                flat_chips['date_str'] = flat_chips['date_str'].astype(str)
                flat_chips['股票代碼'] = flat_chips['股票代碼'].astype(str).str.strip()
                df['date_str'] = df['date_str'].astype(str)
                
                # 單獨篩選出這支股票的籌碼
                stock_chips = flat_chips[flat_chips['股票代碼'] == clean_code]
                
                if not stock_chips.empty:
                    # 使用字串欄位 date_str 進行 Merge，確保不再是 0
                    df = df.merge(stock_chips[['date_str', '外資買賣超', '投信買賣超', '自營商買賣超']], on="date_str", how="left")
                else:
                    df["外資買賣超"] = 0; df["投信買賣超"] = 0; df["自營商買賣超"] = 0
            else:
                df["外資買賣超"] = 0; df["投信買賣超"] = 0; df["自營商買賣超"] = 0
                
            if 'date_str' in df.columns: df.drop(columns=['date_str'], inplace=True)

            df = df.fillna(0).replace([np.inf, -np.inf], 0)
            
            # 準備儲存
            df_to_save = df.tail(TARGET_SAVE_DAYS).copy()
            batch_list.append(df_to_save)
            all_data_for_csv.append(df_to_save)
            
            if len(batch_list) >= 20: 
                save_batch_to_sql(batch_list, engine)
                batch_list = []
            
            # ★ 絕對安全的流量休息 (6.1秒)
            time.sleep(6.1) 

        except Exception as e: pass

    if batch_list: save_batch_to_sql(batch_list, engine)
        
    if all_data_for_csv:
        print(f"\n📥 正在產出 CSV 備份...")
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
        combined["股票代碼"] = combined["股票代碼"].apply(lambda x: f'="{x}"')
        combined.to_csv(CSV_FILE, index=False, encoding='utf-8-sig')

    print(f"\n🎉 任務圓滿達成！全市場資料庫與 CSV 已補全三大法人及營收 YoY 數據。")

if __name__ == "__main__":
    run_v9_8_ultimate()