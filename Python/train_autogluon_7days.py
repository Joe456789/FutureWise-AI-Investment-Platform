import pandas as pd
import warnings
import os
from sqlalchemy import create_engine, text
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
from config import MYSQL_CONN_STR

# 忽略不必要的警告，保持終端機乾淨
warnings.filterwarnings('ignore')

def train_autogluon():
    # 改成直接讀 MySQL，不再依賴 cloud_crawler_TWSE.py 額外匯出的 CSV 備份檔
    # （原本讀 CSV 曾發生 CSV 沒寫成功、訓練資料卡在舊日期的問題，MySQL 才是每天真正在更新的資料源）
    print("[Step 1] 正在從 MySQL 讀取近 2 年歷史資料...")
    try:
        engine = create_engine(MYSQL_CONN_STR)
        query = """
            SELECT ticker AS 股票代碼, trade_date AS 交易日期, close_price AS 收盤價,
                   volume AS 成交量, Foreign_Buy AS 外資買賣超
            FROM StockPrice
            WHERE trade_date >= (SELECT DATE_SUB(MAX(trade_date), INTERVAL 2 YEAR) FROM StockPrice)
        """
        with engine.connect() as conn:
            df = pd.read_sql(text(query), conn)
    except Exception as e:
        print(f"Error 無法讀取資料: {e}")
        return

    if df.empty:
        print("Error 查無資料，中止訓練")
        return

    # 處理時間欄位、過濾缺漏收盤價的資料列
    df['交易日期'] = pd.to_datetime(df['交易日期'])
    df = df.dropna(subset=['收盤價'])

    print("[Step 1] 資料讀取與清洗完成！")
    print(df.head())

    print("\n[Step 2] 轉換為 AutoGluon 專用的 TimeSeriesDataFrame...")
    
    # AutoGluon 轉換
    ts_data = TimeSeriesDataFrame.from_data_frame(
        df,
        id_column="股票代碼",
        timestamp_column="交易日期"
    )

    # AutoGluon 目前版本會自動判斷時間頻率與填補 missing，若出現跳號使用 convert_frequency
    ts_data = ts_data.convert_frequency(freq='B')
    
    print("[Step 2] 轉換成功！開始構建預測網路...")

    print("\n[Step 3] AutoGluon 模型訓練開始 (時間限制設定為 3 分鐘)")
    # 預測未來 7 天的收盤價
    predictor = TimeSeriesPredictor(
        prediction_length=7,
        path="autogluon_7days_model", # 儲存至此資料夾
        target="收盤價",
        eval_metric="MAPE",           # 使用平均絕對百分比誤差，越低越好
        freq="B"                      # 台灣股市為 Business Days
    )

    # 正式上雲端後改用 high_quality：fast_training 對股價這種高噪音資料太容易選到
    # 「幾乎拉平複製最後一天數值」的簡單統計模型，7天預測線看起來都差不多平
    predictor.fit(
        ts_data,
        presets="high_quality",
        time_limit=3600, # 訓練時間上限：3600秒（1小時），每週六排程執行，時間充裕不趕
        random_seed=42
    )

    print("\n[Step 4] 恭喜！AutoGluon 七天趨勢模型已生成完成！")
    print("模型檔案已被儲存至：'autogluon_7days_model' 資料夾中。")
    print("您現在可以將這個資料夾連同 API 伺服器一起上傳到雲端部署了！")

if __name__ == "__main__":
    train_autogluon()
