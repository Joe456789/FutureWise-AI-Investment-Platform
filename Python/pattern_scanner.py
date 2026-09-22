import os
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from scipy.signal import argrelextrema
import json
from datetime import datetime

# ==========================================
# 資料庫連線
# ==========================================
from config import MYSQL_CONN_STR
engine = create_engine(MYSQL_CONN_STR)
current_dir = os.path.dirname(os.path.abspath(__file__))
CACHE_FILE = os.path.join(current_dir, "strategy_cache.json")

# ==========================================
# 1. 均線多頭排列 (Bullish MA)
# ==========================================
def scan_bullish_ma(df):
    """
    條件：收盤價 > MA_5 > MA_10 > MA_20 > MA_60
    且 MA_5 趨勢向上
    """
    try:
        latest = df.iloc[-1]
        
        # 確保所有需要的均線都有資料
        req_cols = ['close_price', 'MA_5', 'MA_10', 'MA_20', 'MA_60']
        if not all(col in df.columns for col in req_cols):
            return False, ""
            
        for col in req_cols:
            if pd.isna(latest[col]) or latest[col] == 0:
                return False, ""
                
        # 多頭排列條件
        c1 = latest['close_price'] > latest['MA_5']
        c2 = latest['MA_5'] > latest['MA_10']
        c3 = latest['MA_10'] > latest['MA_20']
        c4 = latest['MA_20'] > latest['MA_60']
        
        # MA5 向上
        prev_ma5 = df.iloc[-2]['MA_5']
        c5 = latest['MA_5'] > prev_ma5
        
        if c1 and c2 and c3 and c4 and c5:
            return True, f"收盤價({latest['close_price']:.2f})站穩四線之上，均線呈現完美多頭排列。"
        return False, ""
    except Exception:
        return False, ""

# ==========================================
# 2. W底 (W-Bottom)
# ==========================================
def scan_w_bottom(df, order=5):
    """
    使用 scipy argrelextrema 尋找局部極小值(波谷)與極大值(波峰)
    W底特徵：近60天內，有兩個相近的波谷，且中間夾著一個波峰。
    當前價格突破中間波峰(頸線)。
    """
    try:
        if len(df) < 30: return False, ""
        
        prices = df['close_price'].values
        # 尋找波谷 (Local Minima)
        local_min_idx = argrelextrema(prices, np.less, order=order)[0]
        # 尋找波峰 (Local Maxima)
        local_max_idx = argrelextrema(prices, np.greater, order=order)[0]
        
        if len(local_min_idx) >= 2 and len(local_max_idx) >= 1:
            # 取最近的兩個波谷
            trough2_idx = local_min_idx[-1]
            trough1_idx = local_min_idx[-2]
            
            # 確保兩個波谷之間有一個波峰
            peaks_between = [p for p in local_max_idx if trough1_idx < p < trough2_idx]
            if not peaks_between:
                return False, ""
                
            neckline_idx = peaks_between[-1]
            trough1_price = prices[trough1_idx]
            trough2_price = prices[trough2_idx]
            neckline_price = prices[neckline_idx]
            current_price = prices[-1]
            
            # 條件 1: 兩個波谷價格相近 (相差不超過 3%)
            diff_pct = abs(trough1_price - trough2_price) / max(trough1_price, trough2_price)
            if diff_pct > 0.03:
                return False, ""
                
            # 條件 2: 第二個波谷發生在最近 15 天內
            if (len(prices) - 1 - trough2_idx) > 15:
                return False, ""
                
            # 條件 3: 目前價格剛突破頸線 (或正在突破邊緣 1% 內)
            if current_price >= neckline_price * 0.99:
                return True, f"形成W底型態！雙底支撐約在 {trough1_price:.2f}，並成功突破頸線 {neckline_price:.2f}。"
                
        return False, ""
    except Exception:
        return False, ""

# ==========================================
# 3. 黃金交叉 (Golden Cross)
# ==========================================
def scan_golden_cross(df):
    """
    條件：MA_5 由下往上穿越 MA_20（今天 MA_5 > MA_20，昨天 MA_5 <= MA_20）
    且兩條均線都向上，避免盤整區間的假交叉
    """
    try:
        if len(df) < 3:
            return False, ""
        latest = df.iloc[-1]
        prev = df.iloc[-2]

        req_cols = ['MA_5', 'MA_20']
        for row in (latest, prev):
            for col in req_cols:
                if col not in row or pd.isna(row[col]) or row[col] == 0:
                    return False, ""

        crossed_up = prev['MA_5'] <= prev['MA_20'] and latest['MA_5'] > latest['MA_20']
        ma5_rising = latest['MA_5'] > prev['MA_5']

        if crossed_up and ma5_rising:
            return True, f"5日均線({latest['MA_5']:.2f})今日向上穿越20日均線({latest['MA_20']:.2f})，形成黃金交叉。"
        return False, ""
    except Exception:
        return False, ""


# ==========================================
# 4. 布林通道突破 (Bollinger Breakout)
# ==========================================
def scan_bollinger_breakout(df):
    """
    條件：通道先前壓縮（近5日帶寬明顯小於近20日平均帶寬），
    今日帶量價格突破上軌，屬於「開布林」起漲訊號
    """
    try:
        if len(df) < 25:
            return False, ""
        req_cols = ['close_price', 'BB_Upper', 'BB_Lower', 'volume']
        if not all(col in df.columns for col in req_cols):
            return False, ""

        recent = df.tail(25).copy()
        if recent[req_cols].isna().any().any():
            return False, ""

        recent['bandwidth'] = (recent['BB_Upper'] - recent['BB_Lower']) / recent['close_price']
        avg_bandwidth_20 = recent['bandwidth'].iloc[:20].mean()
        recent_bandwidth_5 = recent['bandwidth'].iloc[-6:-1].mean()

        latest = recent.iloc[-1]
        prev_close = recent.iloc[-2]['close_price']
        avg_vol_5 = recent['volume'].iloc[-6:-1].mean()

        was_squeezed = recent_bandwidth_5 < avg_bandwidth_20 * 0.7
        breaks_upper = latest['close_price'] > latest['BB_Upper'] and prev_close <= recent.iloc[-2]['BB_Upper']
        volume_confirmed = avg_vol_5 > 0 and latest['volume'] > avg_vol_5 * 1.2

        if was_squeezed and breaks_upper and volume_confirmed:
            return True, f"布林通道壓縮後帶量突破上軌({latest['BB_Upper']:.2f})，收盤價({latest['close_price']:.2f})，為開口起漲訊號。"
        return False, ""
    except Exception:
        return False, ""


# ==========================================
# 5. 杯柄型態 (Cup and Handle)
# ==========================================
def scan_cup_handle(df, order=5):
    """
    杯柄型態特徵：
    1. 左杯唇：過去 20-60 天的高點
    2. 杯底：中間的平滑低谷 (跌幅約 15-30%)
    3. 右杯唇：反彈回接近左杯唇的高點
    4. 握柄：近 5-10 天的小幅回檔 (跌幅小於 10%)
    5. 突破：目前價格突破右杯唇
    """
    try:
        if len(df) < 40: return False, ""
        
        prices = df['close_price'].values
        local_max_idx = argrelextrema(prices, np.greater, order=order)[0]
        local_min_idx = argrelextrema(prices, np.less, order=order)[0]
        
        if len(local_max_idx) >= 2 and len(local_min_idx) >= 1:
            right_lip_idx = local_max_idx[-1]
            left_lip_idx = local_max_idx[-2]
            
            # 確保中間有低谷
            troughs_between = [t for t in local_min_idx if left_lip_idx < t < right_lip_idx]
            if not troughs_between: return False, ""
            bottom_idx = troughs_between[np.argmin([prices[t] for t in troughs_between])]
            
            left_lip_price = prices[left_lip_idx]
            right_lip_price = prices[right_lip_idx]
            bottom_price = prices[bottom_idx]
            current_price = prices[-1]
            
            # 條件1: 左右杯唇價格相近 (相差 < 5%)
            if abs(left_lip_price - right_lip_price) / max(left_lip_price, right_lip_price) > 0.05:
                return False, ""
                
            # 條件2: 杯底深度足夠 (大於 10%)
            cup_depth = (left_lip_price - bottom_price) / left_lip_price
            if cup_depth < 0.10:
                return False, ""
                
            # 條件3: 握柄回檔幅度不大 (小於 10%)，發生在右杯唇之後
            min_after_right = np.min(prices[right_lip_idx:])
            handle_depth = (right_lip_price - min_after_right) / right_lip_price
            if handle_depth > 0.10:
                return False, ""
                
            # 條件4: 目前正在突破或剛突破
            if current_price >= right_lip_price * 0.98:
                return True, f"完美杯柄型態！杯底約在 {bottom_price:.2f}，目前正準備突破杯緣壓力 {right_lip_price:.2f}。"
                
        return False, ""
    except Exception:
        return False, ""


# ==========================================
# 6. MACD 翻紅 (MACD Histogram 由負轉正)
# ==========================================
def scan_macd_flip(df):
    """今日 MACD 柱狀由負轉正，是波段起漲最佳上車點之一（教學頁「MACD 中長波段神器」同一套邏輯）"""
    try:
        if len(df) < 3:
            return False, ""
        latest = df.iloc[-1]
        prev = df.iloc[-2]

        if 'MACD_Hist' not in df.columns:
            return False, ""
        if pd.isna(latest['MACD_Hist']) or pd.isna(prev['MACD_Hist']):
            return False, ""

        if prev['MACD_Hist'] <= 0 and latest['MACD_Hist'] > 0:
            return True, f"MACD 柱狀今日由負轉正({latest['MACD_Hist']:.3f})，動能翻多訊號出現。"
        return False, ""
    except Exception:
        return False, ""


# ==========================================
# 7. KD 低檔黃金交叉
# ==========================================
def scan_kd_golden_cross(df):
    """K值由下往上穿越D值，且交叉發生在相對低檔區（D < 30），避免高檔鈍化區的雜訊交叉"""
    try:
        if len(df) < 3:
            return False, ""
        latest = df.iloc[-1]
        prev = df.iloc[-2]

        req_cols = ['KD_K', 'KD_D']
        for row in (latest, prev):
            for col in req_cols:
                if col not in row or pd.isna(row[col]):
                    return False, ""

        crossed_up = prev['KD_K'] <= prev['KD_D'] and latest['KD_K'] > latest['KD_D']
        in_low_zone = latest['KD_D'] < 30

        if crossed_up and in_low_zone:
            return True, f"KD 指標於低檔區出現黃金交叉（K:{latest['KD_K']:.1f} > D:{latest['KD_D']:.1f}），短線跌深反彈訊號。"
        return False, ""
    except Exception:
        return False, ""


# ==========================================
# 8. 量價齊揚
# ==========================================
def scan_volume_breakout(df):
    """今日收紅上漲，且成交量明顯放大（> 近5日均量的1.8倍），代表追價意願強，量增價漲最健康"""
    try:
        if len(df) < 8:
            return False, ""
        req_cols = ['close_price', 'volume']
        if not all(col in df.columns for col in req_cols):
            return False, ""

        latest = df.iloc[-1]
        prev_close = df.iloc[-2]['close_price']
        recent_vol = df['volume'].iloc[-6:-1]

        if recent_vol.isna().any() or pd.isna(latest['volume']) or pd.isna(latest['close_price']):
            return False, ""

        avg_vol_5 = recent_vol.mean()
        price_up = latest['close_price'] > prev_close
        volume_surge = avg_vol_5 > 0 and latest['volume'] > avg_vol_5 * 1.8

        if price_up and volume_surge:
            pct = (latest['close_price'] - prev_close) / prev_close * 100
            return True, f"今日上漲{pct:.2f}%，成交量放大至近5日均量的{latest['volume']/avg_vol_5:.1f}倍，量價齊揚。"
        return False, ""
    except Exception:
        return False, ""


# ==========================================
# 9. M頭型態 (雙重頂，示警用)
# ==========================================
def scan_m_top(df, order=5):
    """
    W底的鏡像：近期兩個相近波峰、中間夾一個波谷，目前價格跌破波谷頸線。
    這是警示型態（提醒留意風險），不是買進訊號。
    """
    try:
        if len(df) < 30:
            return False, ""

        prices = df['close_price'].values
        local_max_idx = argrelextrema(prices, np.greater, order=order)[0]
        local_min_idx = argrelextrema(prices, np.less, order=order)[0]

        if len(local_max_idx) >= 2 and len(local_min_idx) >= 1:
            peak2_idx = local_max_idx[-1]
            peak1_idx = local_max_idx[-2]

            troughs_between = [t for t in local_min_idx if peak1_idx < t < peak2_idx]
            if not troughs_between:
                return False, ""

            neckline_idx = troughs_between[-1]
            peak1_price = prices[peak1_idx]
            peak2_price = prices[peak2_idx]
            neckline_price = prices[neckline_idx]
            current_price = prices[-1]

            diff_pct = abs(peak1_price - peak2_price) / max(peak1_price, peak2_price)
            if diff_pct > 0.03:
                return False, ""

            if (len(prices) - 1 - peak2_idx) > 15:
                return False, ""

            if current_price <= neckline_price * 1.01:
                return True, f"形成M頭型態！雙頭壓力約在 {peak1_price:.2f}，已跌破頸線 {neckline_price:.2f}，留意回檔風險。"

        return False, ""
    except Exception:
        return False, ""


# ==========================================
# 主程式：全市場掃描與快取寫入
# ==========================================
def run_scanner():
    print("🚀 開始執行全市場型態掃描 (Pattern Scanner)...")
    
    # 排除大盤指數，以及 00 開頭的 ETF/槓桿/反向商品（例如 00647L、00674R）——
    # 這是個股技術型態選股功能，混進去會讓槓桿/反向 ETF 洗版排擠掉真正的個股
    # 也不再 LIMIT 200，避免因為沒有 ORDER BY，撈到的剛好都是代碼較小的一批 ETF
    try:
        with engine.connect() as conn:
            query_tickers = """
                SELECT DISTINCT ticker FROM StockPrice
                WHERE ticker NOT IN ('TSE', 'OTC') AND ticker NOT LIKE '00%'
            """
            if engine.dialect.name == 'mssql':
                query_tickers = """
                    SELECT DISTINCT ticker FROM StockPrice
                    WHERE ticker NOT IN ('TSE', 'OTC') AND ticker NOT LIKE '00%'
                """

            tickers_df = pd.read_sql(text(query_tickers), conn)
            tickers = tickers_df['ticker'].tolist()
            print(f"📋 本次掃描個股數量：{len(tickers)} 檔（已排除大盤指數與 00 開頭 ETF/槓反商品）")

            # 初始化結果字典
            results = {
                "bullish_ma": [],
                "w_bottom": [],
                "golden_cross": [],
                "bollinger_breakout": [],
                "cup_handle": [],
                "macd_flip": [],
                "kd_golden_cross": [],
                "volume_breakout": [],
                "m_top": []
            }
            
            # 針對每一檔進行掃描
            for ticker in tickers:
                query_data = "SELECT * FROM StockPrice WHERE ticker = :ticker ORDER BY trade_date DESC LIMIT 60"
                if engine.dialect.name == 'mssql':
                    query_data = "SELECT TOP 60 * FROM StockPrice WHERE ticker = :ticker ORDER BY trade_date DESC"

                df = pd.read_sql(text(query_data), conn, params={"ticker": ticker})
                if len(df) < 30: continue
                
                # 反轉讓日期舊到新
                df = df.sort_values("trade_date").reset_index(drop=True)
                
                latest_close = float(df.iloc[-1]['close_price'])
                stock_name = df.iloc[-1].get('stock_name', ticker)
                
                # 執行掃描
                is_bullish, msg_bullish = scan_bullish_ma(df)
                if is_bullish:
                    results["bullish_ma"].append({"ticker": ticker, "name": stock_name, "price": latest_close, "reason": msg_bullish})
                    
                is_w, msg_w = scan_w_bottom(df)
                if is_w:
                    results["w_bottom"].append({"ticker": ticker, "name": stock_name, "price": latest_close, "reason": msg_w})

                is_golden, msg_golden = scan_golden_cross(df)
                if is_golden:
                    results["golden_cross"].append({"ticker": ticker, "name": stock_name, "price": latest_close, "reason": msg_golden})

                is_boll, msg_boll = scan_bollinger_breakout(df)
                if is_boll:
                    results["bollinger_breakout"].append({"ticker": ticker, "name": stock_name, "price": latest_close, "reason": msg_boll})

                is_cup, msg_cup = scan_cup_handle(df)
                if is_cup:
                    results["cup_handle"].append({"ticker": ticker, "name": stock_name, "price": latest_close, "reason": msg_cup})

                is_macd, msg_macd = scan_macd_flip(df)
                if is_macd:
                    results["macd_flip"].append({"ticker": ticker, "name": stock_name, "price": latest_close, "reason": msg_macd})

                is_kd, msg_kd = scan_kd_golden_cross(df)
                if is_kd:
                    results["kd_golden_cross"].append({"ticker": ticker, "name": stock_name, "price": latest_close, "reason": msg_kd})

                is_vol, msg_vol = scan_volume_breakout(df)
                if is_vol:
                    results["volume_breakout"].append({"ticker": ticker, "name": stock_name, "price": latest_close, "reason": msg_vol})

                is_mtop, msg_mtop = scan_m_top(df)
                if is_mtop:
                    results["m_top"].append({"ticker": ticker, "name": stock_name, "price": latest_close, "reason": msg_mtop})

            # 將結果寫入 Cache JSON
            cache_data = {
                "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "strategies": results
            }

            with open(CACHE_FILE, "w", encoding="utf-8") as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=4)

            print(f"✅ 掃描完成！共找到 多頭排列:{len(results['bullish_ma'])} W底:{len(results['w_bottom'])} "
                  f"黃金交叉:{len(results['golden_cross'])} 布林突破:{len(results['bollinger_breakout'])} 杯柄:{len(results['cup_handle'])} "
                  f"MACD翻紅:{len(results['macd_flip'])} KD黃金交叉:{len(results['kd_golden_cross'])} "
                  f"量價齊揚:{len(results['volume_breakout'])} M頭:{len(results['m_top'])} 檔。")
            print(f"📁 結果已快取至: {CACHE_FILE}")

    except Exception as e:
        print(f"❌ 掃描過程發生錯誤: {e}")

if __name__ == "__main__":
    run_scanner()
