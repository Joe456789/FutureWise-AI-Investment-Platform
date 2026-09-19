# 程式名稱：FutureWise 全自動快充引擎 (V10.0 雲端終極版)
# 核心修正：
# 1. 雲端對接：全面改用 MySQL 連線，對接 config.py 總開關。
# 2. yfinance 維度防呆：加入 MultiIndex 降維處理，防止 1-dimensional 報錯。
# 3. MySQL 寫入優化：優化 DELETE 舊資料與批次寫入的語法。

import yfinance as yf
import pandas as pd
import ta
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SAWarning
import os, time, requests, urllib3, io, numpy as np, warnings
import random
import logging
from datetime import datetime, timedelta

# ==========================================
# 0. 系統降噪與環境設定
# ==========================================
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
warnings.simplefilter('ignore', category=SAWarning)
warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger('yfinance').setLevel(logging.CRITICAL)

# 1. 引入雲端 MySQL 連線字串 (從 config.py)
from config import MYSQL_CONN_STR

# ==========================================
# 1. 配置區
# ==========================================
CSV_FILE = "股票清單_Cloud.csv"

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

# ==========================================
# 2. 數據模組 
# ==========================================
def get_tw_stock_map():
    headers = {'User-Agent': 'Mozilla/5.0'}
    stock_map = {}
    
    # 手動加入大盤與櫃買指數
    stock_map['TSE'] = '加權指數'
    stock_map['OTC'] = '櫃買指數'
    
    for url in ["https://isin.twse.com.tw/isin/C_public.jsp?strMode=2", "https://isin.twse.com.tw/isin/C_public.jsp?strMode=4"]:
        try:
            res = requests.get(url, headers=headers, verify=False, timeout=10)
            df = pd.read_html(io.StringIO(res.text))[0].iloc[2:]
            for item in df.iloc[:, 0]:
                parts = str(item).split()
                if len(parts) >= 2:
                    ticker = parts[0].strip()
                    name = parts[1].strip()
                    
                    # 🔥 終極智能過濾邏輯 🔥
                    is_valid = False
                    
                    if len(ticker) == 4 and ticker.isdigit():
                        # 1. 普通股 (例如: 2330)
                        is_valid = True
                    elif ticker.startswith('00') and 4 <= len(ticker) <= 6:
                        # 2. 所有 ETF (例如: 0050, 00919)
                        is_valid = True
                    elif len(ticker) == 5 and ticker[:4].isdigit() and ticker[4].isalpha():
                        # 3. 特別股 (例如: 2881A)
                        is_valid = True
                        
                    if is_valid:
                        stock_map[ticker] = name
        except Exception as e: 
            pass
            
    return stock_map

def fetch_all_market_chips(trading_dates):
    all_chips = []
    session = requests.Session()
    session.headers.update({'User-Agent': 'Mozilla/5.0'})
    
    for idx, d in enumerate(trading_dates):
        dt_key = pd.to_datetime(d).strftime("%Y-%m-%d")
        dt_twse = pd.to_datetime(d).strftime("%Y%m%d")
        tw_yr = d.year - 1911
        dt_tpex = f"{tw_yr}/{d.strftime('%m/%d')}"
        
        print(f"   下載籌碼批次: {dt_key} ...", end="\r")
        
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
                        'date_str': dt_key, '股票代碼': row[c_idx].strip(), 
                        '外資買賣超': float(row[f_idx].replace(',', '')) / 1000, 
                        '投信買賣超': float(row[t_idx].replace(',', '')) / 1000, 
                        '自營商買賣超': float(row[d_idx].replace(',', '')) / 1000
                    })
        except: pass
        time.sleep(1)
        
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
        
        # 防呆：處理 yfinance 可能回傳的 MultiIndex
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
            
        df_m = pd.DataFrame(index=raw.index)
        df_m["Market_Close"] = raw["^TWII"].squeeze()
        df_m["Market_Return"] = df_m["Market_Close"].pct_change(fill_method=None)
        df_m["SOX_Close"] = raw["^SOX"].squeeze().shift(1)
        df_m["SOX_Return"] = df_m["SOX_Close"].pct_change(fill_method=None).shift(1)
        df_m["TWD_Exchange"] = raw["TWD=X"].squeeze()
        df_m.index = pd.to_datetime(df_m.index).date
        return df_m
    except: return pd.DataFrame()

def calculate_all_indicators(df):
    if len(df) < 5: return df
    df = df.sort_values("交易日期").reset_index(drop=True)
    
    # 強制轉為 1-dimensional Series，防止 ta 報錯
    c = df["收盤價"].squeeze()
    h = df["最高價"].squeeze()
    l = df["最低價"].squeeze()
    v = df["成交量"].squeeze()
    
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
    df["跳空缺口"] = df["開盤價"].squeeze() - c.shift(1)
    return df

def save_batch_to_sql(batch_list, engine):
    if not batch_list: return
    full_batch = pd.concat(batch_list)
    full_batch['交易日期'] = pd.to_datetime(full_batch['交易日期']).dt.date
    
    valid_cols = [c for c in DB_COLUMN_MAP.keys() if c in full_batch.columns]
    final_df = full_batch[valid_cols].rename(columns=DB_COLUMN_MAP)
    
    try:
        # 使用 engine.begin() 會在結束時自動 Commit，出錯自動 Rollback
        with engine.begin() as conn: 
            for t_code in final_df['ticker'].unique():
                c_df = final_df[final_df['ticker'] == t_code]
                d_str = ",".join([f"'{d}'" for d in c_df['trade_date']])
                # 執行刪除
                conn.execute(text(f"DELETE FROM StockPrice WHERE ticker = '{t_code}' AND trade_date IN ({d_str})"))
            
            # ✅ 關鍵修正：將 engine 改為 conn
            final_df.to_sql("StockPrice", conn, if_exists="append", index=False)
            print(f"  ✅ 成功批次寫入 {len(final_df)} 筆資料", end="\r")
    except Exception as e:
        print(f"❌ 批次寫入失敗: {e}")

# ==========================================
# 3. 主執行流程 
# ==========================================
def run_v10_cloud():
    print("🌍 正在連線至 MySQL 雲端資料庫...")
    engine = create_engine(MYSQL_CONN_STR)
    
    print("🌍 正在下載總經數據與全台股名單...")
    macro_df = get_macro_data()
    name_map = get_tw_stock_map()

    name_map['TSE'] = '加權指數'
    name_map['OTC'] = '櫃買指數'
    
    tickers = sorted(name_map.keys())
    TARGET_SAVE_DAYS = 60 
    
    print(f"🚀 FutureWise 雲端版啟動 | 目標: 全台股 {len(tickers)} 支股票")
    
    last_n_dates = macro_df.index[-TARGET_SAVE_DAYS:].tolist()
    global_chips_df = fetch_all_market_chips(last_n_dates)
    
    macro_df['date_str'] = pd.to_datetime(macro_df.index).strftime('%Y-%m-%d')
    
    all_data_for_csv = []
    batch_list = []
    
    for i, code in enumerate(tickers):
        name = name_map[code]
        clean_code = str(code).strip() 
        
        print(f"[{i+1}/{len(tickers)}] 同步 {clean_code} {name} ...", end="\r")
        try:
            if clean_code == 'TSE':
                tk = yf.Ticker("^TWII")
            elif clean_code == 'OTC':
                tk = yf.Ticker("^TWOII")
            else:
                tk = yf.Ticker(f"{clean_code}.TW")
                
            df = tk.history(period="2y", auto_adjust=True)
            if df.empty and clean_code not in ['TSE', 'OTC']:
                tk = yf.Ticker(f"{clean_code}.TWO")
                df = tk.history(period="2y", auto_adjust=True)
                
            if df.empty: continue
            
            # 防呆：處理 MultiIndex
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
                
            info = tk.info
            df = df.reset_index().rename(columns={"Date":"交易日期","Open":"開盤價","High":"最高價","Low":"最低價","Close":"收盤價","Volume":"成交量"})
            df["交易日期"] = pd.to_datetime(df["交易日期"]).dt.date
            df["股票代碼"] = clean_code
            df["股票名稱"] = name
            
            df = calculate_all_indicators(df)
            df["date_str"] = pd.to_datetime(df["交易日期"]).dt.strftime('%Y-%m-%d')
            df = df.merge(macro_df[['date_str', 'Market_Close', 'Market_Return', 'SOX_Close', 'SOX_Return', 'TWD_Exchange']], on='date_str', how='left')
            df.rename(columns={'Market_Close':'大盤收盤','Market_Return':'大盤漲跌幅','SOX_Close':'費半收盤','TWD_Exchange':'台幣匯率','SOX_Return':'費半漲跌'}, inplace=True)
            
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
            
            if not global_chips_df.empty:
                flat_chips = global_chips_df.reset_index()
                flat_chips['date_str'] = flat_chips['date_str'].astype(str)
                flat_chips['股票代碼'] = flat_chips['股票代碼'].astype(str).str.strip()
                df['date_str'] = df['date_str'].astype(str)
                
                stock_chips = flat_chips[flat_chips['股票代碼'] == clean_code]
                
                if not stock_chips.empty:
                    df = df.merge(stock_chips[['date_str', '外資買賣超', '投信買賣超', '自營商買賣超']], on="date_str", how="left")
                else:
                    df["外資買賣超"] = 0; df["投信買賣超"] = 0; df["自營商買賣超"] = 0
            else:
                df["外資買賣超"] = 0; df["投信買賣超"] = 0; df["自營商買賣超"] = 0
                
            if 'date_str' in df.columns: df.drop(columns=['date_str'], inplace=True)

            # 🚀 修正：基本面資料 (EPS, 本益比等) 屬於長期有效，適合前向填充 (ffill)
            # 但「每日籌碼與交易量」(外資買賣超) 每日獨立，不能 ffill，否則會捏造假數據！
            cols_to_ffill = ['營收YoY', 'EPS', '本益比', '股價淨值比']
            for c in cols_to_ffill:
                if c in df.columns:
                    df[c] = df[c].replace([np.inf, -np.inf, ''], np.nan).ffill()

            # 剩下的空值（包含籌碼與缺漏的交易日）乖乖補 0，保持數據真實性
            df = df.fillna(0).replace([np.inf, -np.inf], 0)
            
            df_to_save = df.tail(TARGET_SAVE_DAYS).copy()
            batch_list.append(df_to_save)
            all_data_for_csv.append(df_to_save)
            
            # 每累積 20 支股票寫入一次資料庫
            if len(batch_list) >= 20: 
                save_batch_to_sql(batch_list, engine)
                batch_list = []
            
            sleep_time = random.uniform(3, 5)
            time.sleep(sleep_time)

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
        
    print(f"\n🎉 任務圓滿達成！全市場資料庫已更新完畢。")

if __name__ == "__main__":
    run_v10_cloud()