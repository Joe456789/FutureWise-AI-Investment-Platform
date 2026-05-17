import pandas as pd
import warnings
import os
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor

# 忽略不必要的警告，保持終端機乾淨
warnings.filterwarnings('ignore')

def train_autogluon():
    print("[Step 1] 正在讀取歷史資料 (股票清單_Cloud.csv)...")
    try:
        # 自動抓取本腳本所在的資料夾路徑，避免發生找不到檔案的錯誤
        current_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(current_dir, "股票清單_Cloud.csv")
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error 無法讀取資料: {e}")
        return

    # 1. 清理股票代碼 (去除 Excel 強制字串符號 '="0050"')
    if '股票代碼' in df.columns:
        df['股票代碼'] = df['股票代碼'].astype(str).str.replace('="', '').str.replace('"', '')
    else:
        print("Error CSV 中找不到 '股票代碼' 欄位")
        return

    # 2. 處理時間欄位
    df['交易日期'] = pd.to_datetime(df['交易日期'])
    
    # 3. 過濾掉包含太多 NaN 的行數，或是極端值 (保持資料穩定)
    df = df.dropna(subset=['收盤價'])
    
    # 只需要保留時序預測最關鍵的欄位，減少記憶體負擔
    # 外資買賣超與成交量可以作為輔助特徵 (Covariates)
    keep_cols = ['股票代碼', '交易日期', '收盤價', '成交量', '外資買賣超']
    keep_cols_exist = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols_exist]
    
    # 如果資料量過於龐大（例如幾萬行），可考慮只取最近 3 年的值來預測未來 7 天，加快速度
    recent_date_threshold = df['交易日期'].max() - pd.DateOffset(years=2)
    df = df[df['交易日期'] >= recent_date_threshold]

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

    # presets="fast_training" 是為求快速的測試組合，正式上雲端時可以改為 "high_quality"
    predictor.fit(
        ts_data,
        presets="fast_training",
        time_limit=3600, # 180秒內盡可能訓練出最好的模型
        random_seed=42
    )

    print("\n[Step 4] 恭喜！AutoGluon 七天趨勢模型已生成完成！")
    print("模型檔案已被儲存至：'autogluon_7days_model' 資料夾中。")
    print("您現在可以將這個資料夾連同 API 伺服器一起上傳到雲端部署了！")

if __name__ == "__main__":
    train_autogluon()
