# 程式名稱：SQL 轉 CSV 重建工具 (雙棲自動切換版)
# 功能：當 CSV 損壞時，直接從資料庫匯出所有資料，確保欄位 100% 正確
# 目的：重建「股票清單.csv」供 AI 訓練或備份使用

import pandas as pd
from sqlalchemy import create_engine
import os

# 1. ★ 關鍵修改：引入能自動判斷環境的總開關
from config import DB_CONN_STR, DB_TYPE

# 2. 配置區
CSV_FILE = "股票清單.csv"

# 反向映射表：將資料庫英文欄位轉回 CSV 中文標題
column_map_rev = {
    "ticker": "股票代碼", "stock_name": "股票名稱", "trade_date": "交易日期",
    "open_price": "開盤價", "high_price": "最高價", "low_price": "最低價", "close_price": "收盤價", "volume": "成交量",
    "MA_5": "5日均線", "MA_10": "10日均線", "MA_20": "20日均線", "MA_60": "60日均線", "MA_120": "120日均線", "MA_240": "240日均線",
    "RSI_6": "RSI_6", "RSI_14": "RSI_14", "MACD_DIF": "MACD_快線", "MACD_Signal": "MACD_慢線", "MACD_Hist": "MACD_柱狀",
    "KD_K": "K值", "KD_D": "D值", "BB_Upper": "布林上軌", "BB_Lower": "布林下軌", "ATR": "ATR",
    "Vol_MA_5": "5日均量", "Vol_MA_20": "20日均量", "Vol_Ratio": "量能比", "Bias_5": "5日乖離", "Bias_20": "20日乖離",
    "Change_1D": "漲跌幅_1日", "Change_3D": "漲跌幅_3日", "Change_5D": "漲跌幅_5日", "Gap": "跳空缺口",
    "Revenue_YoY": "營收YoY", "EPS": "EPS", "PE_Ratio": "本益比", "PB_Ratio": "股價淨值比",
    "Foreign_Buy": "外資買賣超", "Trust_Buy": "投信買賣超", "Dealer_Buy": "自營商買賣超",
    "Market_Close": "大盤收盤", "Market_Return": "大盤漲跌幅",
    "SOX_Close": "費半收盤", "TWD_Exchange": "台幣匯率", "SOX_Return": "費半漲跌", "Sentiment_Score": "情緒分數"
}

def rebuild_csv():
    try:
        print(f"🌍 系統偵測到目前環境為：[{DB_TYPE}]，準備連線...")
        engine = create_engine(DB_CONN_STR)
        
        # 1. 從 SQL 讀取所有資料
        # 注意：這裡的 SQL 語法標準通用，不管 MySQL 還是 MSSQL 都能看懂
        query = "SELECT * FROM StockPrice ORDER BY ticker, trade_date DESC"
        print("📥 正在從資料庫撈取全部海量資料，請稍候...")
        df = pd.read_sql(query, engine)
        
        if df.empty:
            print("📭 資料庫目前是空的，無法匯出 CSV。")
            return

        # 如果資料表裡有多餘的 id 欄位，自動過濾掉
        if 'id' in df.columns:
            df = df.drop(columns=['id'])

        # 2. 轉換欄位名稱回中文
        print("📝 正在轉換標題並格式化...")
        df = df.rename(columns=column_map_rev)
        
        # 3. 處理日期格式 (確保 CSV 讀取時不會出錯)
        df["交易日期"] = pd.to_datetime(df["交易日期"]).dt.strftime('%Y-%m-%d')

        # 4. 處理 Excel 防呆格式 (="2330")，讓股票代碼的 0 不會被吃掉
        df["股票代碼"] = df["股票代碼"].apply(lambda x: f'="{x}"')

        # 5. 寫入新的 CSV
        print(f"💾 正在寫入全新的 {CSV_FILE} (共 {len(df)} 筆資料)...")
        df.to_csv(CSV_FILE, index=False, encoding='utf-8-sig')
        
        print(f"✅ 重建成功！已經產出最新鮮的 {CSV_FILE}。")

    except Exception as e:
        print(f"❌ 重建失敗：{e}")

if __name__ == "__main__":
    rebuild_csv()