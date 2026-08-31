# 程式名稱：ProQuant 雙機同步工具 (自動切換 雲端MySQL / 本機SSMS 版)
import pandas as pd
from sqlalchemy import create_engine, text, exc
import os
import numpy as np

# 1. ★ 關鍵修改：引入能自動判斷環境的總開關
from config import DB_CONN_STR, DB_TYPE

# 2. 配置區
CSV_FILE = "股票清單.csv" 

def sync_csv_to_sql():
    if not os.path.exists(CSV_FILE):
        print(f"❌ 錯誤：找不到 {CSV_FILE}，請確認檔案已經放在同一個資料夾。")
        return

    try:
        # 3. ★ 關鍵修改：使用 DB_CONN_STR，並印出目前環境
        print(f"🌍 系統偵測到目前環境為：[{DB_TYPE}]，準備連線...")
        engine = create_engine(DB_CONN_STR, isolation_level="AUTOCOMMIT") 
        
        print(f"📂 正在載入 CSV 數據...")
        df = pd.read_csv(CSV_FILE, dtype={'股票代碼': str})
        
        # 清理格式 (日期強制轉為字串 YYYY-MM-DD)
        df["股票代碼"] = df["股票代碼"].astype(str).str.replace('="', '', regex=False).str.replace('"', '', regex=False)
        df["交易日期"] = pd.to_datetime(df["交易日期"]).dt.strftime('%Y-%m-%d')
        
        column_map = {
            "股票代碼": "ticker", "股票名稱": "stock_name", "交易日期": "trade_date",
            "開盤價": "open_price", "最高價": "high_price", "最低價": "low_price", "收盤價": "close_price", "成交量": "volume",
            "5日均線": "MA_5", "10日均線": "MA_10", "20日均線": "MA_20", "60日均線": "MA_60", "120日均線": "MA_120", "240日均線": "MA_240",
            "RSI_6": "RSI_6", "RSI_14": "RSI_14", "MACD_快線": "MACD_DIF", "MACD_慢線": "MACD_Signal", "MACD_柱狀": "MACD_Hist",
            "K值": "KD_K", "D值": "KD_D", "布林上軌": "BB_Upper", "布林下軌": "BB_Lower", "ATR": "ATR",
            "5日均量": "Vol_MA_5", "20日均量": "Vol_MA_20", "量能比": "Vol_Ratio", "5日乖離": "Bias_5", "20日乖離": "Bias_20",
            "漲跌幅_1日": "Change_1D", "漲跌幅_3日": "Change_3D", "漲跌幅_5日": "Change_5D", "跳空缺口": "Gap",
            "營收YoY": "Revenue_YoY", "EPS": "EPS", "本益比": "PE_Ratio", "股價淨值比": "PB_Ratio",
            "外資買賣超": "Foreign_Buy", "投信買賣超": "Trust_Buy", "自營商買賣超": "Dealer_Buy",
            "大盤收盤": "Market_Close", "大盤漲跌幅": "Market_Return",
            "費半收盤": "SOX_Close", "台幣匯率": "TWD_Exchange", "費半漲跌": "SOX_Return", "情緒分數": "Sentiment_Score"
        }

        tickers = df["股票代碼"].unique()
        print(f"🔄 開始同步 {len(tickers)} 支股票到 {DB_TYPE}...")

        total_added = 0
        skip_count = 0

        for i, t in enumerate(tickers):
            if (i + 1) % 100 == 0: print(f"   進度: {i+1}/{len(tickers)}...")
            
            c_df = df[df["股票代碼"] == t].copy()
            
            with engine.connect() as conn:
                try:
                    # 查詢該股在 DB 的日期
                    check_sql = text(f"SELECT trade_date FROM StockPrice WHERE ticker = '{t}'")
                    exist_data = pd.read_sql(check_sql, conn)
                    
                    if not exist_data.empty:
                        exist_set = set(pd.to_datetime(exist_data.iloc[:, 0]).dt.strftime('%Y-%m-%d'))
                        c_df = c_df[~c_df['交易日期'].isin(exist_set)]
                    
                    if not c_df.empty:
                        df_to_db = c_df.rename(columns=column_map)
                        df_to_db = df_to_db.replace([np.inf, -np.inf], np.nan).fillna(0)
                        
                        valid_cols = [c for c in df_to_db.columns if c in column_map.values()]
                        
                        df_to_db[valid_cols].to_sql("StockPrice", engine, if_exists="append", index=False)
                        total_added += len(c_df)
                    else:
                        skip_count += 1

                except exc.IntegrityError:
                    print(f"\n⚠️ {t} 偵測到重複主鍵 (已跳過)")
                    continue 
                except Exception as inner_e:
                    print(f"\n❌ {t} 發生錯誤: {inner_e}")
                    continue

        print(f"\n✨ 同步圓滿完成！")
        print(f"📈 成功寫入: {total_added} 筆 | 😴 已是最新: {skip_count} 支")

    except Exception as e:
        print(f"💥 發生嚴重錯誤: {e}")

if __name__ == "__main__":
    sync_csv_to_sql()