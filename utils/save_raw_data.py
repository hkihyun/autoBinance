import os
from dotenv import load_dotenv
load_dotenv()

api_key = os.getenv("BINANCE_API_KEY")
secret = os.getenv("BINANCE_SECRET_KEY")

import ccxt
import pandas as pd


def _ohlcv2refined_df(ohlcv):
    '''
    refined_df:
    df = pd.DataFrame({
        'open': [26000, 26050, 26080],
        'high': [26100, 26120, 26150],
        'low': [25950, 26000, 26070],
        'close': [26050, 26080, 26130],
        'volume': [12.5, 8.2, 5.9],
        'timestamp': [
            '2024-09-16 00:00:00',
            '2024-09-16 00:01:00',
            '2024-09-16 00:02:00'
        ]
    })
    '''
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

    # timestamp를 datetime 형식으로 변환
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

    return df

def _save_refined_df(df):
    # 저장 경로 설정
    os.makedirs('data', exist_ok=True)
    start = df['timestamp'].iloc[0].strftime('%Y-%m-%d')
    end   = df['timestamp'].iloc[-1].strftime('%Y-%m-%d')
    filename = f"BTCUSDT_{start}_to_{end}.csv"
    save_path = os.path.join('data/raw_data', filename)

    # 파일 존재 여부 확인
    if os.path.exists(save_path):
        print(f"이미 같은 이름의 파일이 존재합니다: {save_path}")
    else:
        print(f"새 파일로 저장합니다: {save_path}")

    # CSV로 저장
    df.to_csv(save_path, index=False)
    return df


def btc_1min_chadle(day = 7):
    # Binance 선물 거래소 초기화
    exchange = ccxt.binance({
        'options': {
            'defaultType': 'future',  # 선물 거래소로 설정
        }
    })
    # 차트 데이터 저장하기
    ohlcv = exchange.fetch_ohlcv('BTC/USDT', timeframe='1m', limit=1440 * day)
    refined_df = _ohlcv2refined_df(ohlcv)
    return _save_refined_df(refined_df)


def btc_1day_candle(month = 12):
    # Binance 선물 거래소 초기화
    exchange = ccxt.binance({
        'options': {
            'defaultType': 'future',  # 선물 거래소로 설정
        }
    })
    # 차트 데이터 가져오기
    ohlcv = exchange.fetch_ohlcv('BTC/USDT', timeframe='1d', limit=30 * month)
    refined_df = _ohlcv2refined_df(ohlcv)
    return _save_refined_df(refined_df)

if __name__=='__main__':
    print(btc_1min_chadle(900))