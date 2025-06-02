import pandas as pd
import numpy as np
import random
import sqlite3

def calculate_technical_indicators(file_path):
    df = pd.read_csv(file_path)
    
    df['Date'] = pd.to_datetime(df['Date'])
    df['Close'] = df['Close/Last'].str.replace('$', '').astype(float).round(3)
    df['Open'] = df['Open'].str.replace('$', '').astype(float).round(3)
    df['High'] = df['High'].str.replace('$', '').astype(float).round(3)
    df['Low'] = df['Low'].str.replace('$', '').astype(float).round(3)
    
    df = df.sort_values('Date')
    
    window_5 = 5
    window_10 = 10
    window_20 = 20
    
    df['SMA_5'] = df['Close'].rolling(window=window_5).mean().round(3)
    df['SMA_10'] = df['Close'].rolling(window=window_10).mean().round(3)
    df['SMA_20'] = df['Close'].rolling(window=window_20).mean().round(3)
    
    df['EMA_5'] = df['Close'].ewm(span=window_5, adjust=False).mean().round(3)
    df['EMA_10'] = df['Close'].ewm(span=window_10, adjust=False).mean().round(3)
    df['EMA_20'] = df['Close'].ewm(span=window_20, adjust=False).mean().round(3)
    
    delta = df['Close'].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ema_up = up.ewm(com=13, adjust=False).mean()
    ema_down = down.ewm(com=13, adjust=False).mean()
    rs = ema_up / ema_down
    df['RSI'] = (100 - (100 / (1 + rs))).round(3)
    
    df['Daily_Return'] = (df['Close'].pct_change() * 100).round(3)
    
    df['MACD'] = (df['Close'].ewm(span=12, adjust=False).mean() - df['Close'].ewm(span=26, adjust=False).mean()).round(3)
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean().round(3)
    df['MACD_Histogram'] = (df['MACD'] - df['MACD_Signal']).round(3)
    
    df['Bollinger_Middle'] = df['SMA_20']
    df['Bollinger_Std'] = df['Close'].rolling(window=window_20).std().round(3)
    df['Bollinger_Upper'] = (df['Bollinger_Middle'] + 2 * df['Bollinger_Std']).round(3)
    df['Bollinger_Lower'] = (df['Bollinger_Middle'] - 2 * df['Bollinger_Std']).round(3)
    
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    df['ATR'] = true_range.rolling(14).mean().round(3)
    
    df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum().round(3)
    
    sentiments = []
    for _ in range(len(df)):
        if random.random() < 0.7:
            sentiments.append(round(random.uniform(-0.3, 0.3), 3))
        else:
            sentiments.append(round(random.uniform(-1, 1), 3))
    
    df['Sentiment'] = sentiments
    
    return df

if __name__ == "__main__":
    file_path = "AAPL.csv"
    df_with_indicators = calculate_technical_indicators(file_path)
    
    print(df_with_indicators.tail())
    
    conn = sqlite3.connect('stock_indicators.db')
    df_with_indicators.to_sql('apple_indicators', conn, if_exists='replace', index=False)
    conn.close()
    
    print("Technical analysis complete. Results saved to stock_indicators.db")