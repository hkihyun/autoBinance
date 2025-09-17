import os
from dotenv import load_dotenv
load_dotenv()

api_key = os.getenv("BINANCE_API_KEY")
secret = os.getenv("BINANCE_SECRET_KEY")

import ccxt
import pandas as pd
import time


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


def btc_1min_candle(day):
    # Binance 선물 거래소 초기화
    exchange = ccxt.binance({
        'options': {
            'defaultType': 'future',  # 선물 거래소로 설정
        }
    })

    symbol = 'BTC/USDT'
    timeframe = '1m'
    total_mins = 1440 * day  # 가져올 총 일 수
    limit = 1500              # API 한 번 호출 시 최대 개수
    since = exchange.parse8601('2017-08-17T00:00:00Z')  # BTC/USDT 선물 시작일 근처
    # since = exchange.parse8601('2022-02-24T00:00:00Z')  # BTC/USDT 선물 시작일 근처

    all_ohlcv = []

    # limit 단위로 나눠서 호출
    while len(all_ohlcv) < total_mins:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit)
        if not ohlcv:
            break

        all_ohlcv.extend(ohlcv)
        
        # 다음 요청을 위해 since 값을 마지막 캔들 이후로 갱신
        since = ohlcv[-1][0] + 60 * 1000  # 마지막 캔들 다음날 (ms 단위)
        
        # API rate limit 보호
        time.sleep(exchange.rateLimit / 1000)

        # 진행상황 표시
        print(f"----------------      {round(len(all_ohlcv)/total_mins*100)}%...      ----------------")

    # 필요한 개수만큼만 자르기
    all_ohlcv = all_ohlcv[-total_mins:]

    # 차트 데이터 저장하기
    refined_df = _ohlcv2refined_df(all_ohlcv)
    return _save_refined_df(refined_df)
def btc_1day_candle(month=12):
    # Binance 선물 거래소 초기화
    exchange = ccxt.binance({
        'options': {
            'defaultType': 'future',  # 선물 거래소로 설정
        }
    })

    symbol = 'BTC/USDT'
    timeframe = '1d'
    total_days = 30 * month  # 가져올 총 일 수
    limit = 1500              # API 한 번 호출 시 최대 개수
    since = exchange.parse8601('2017-08-17T00:00:00Z')  # BTC/USDT 선물 시작일 근처

    all_ohlcv = []

    # limit 단위로 나눠서 호출
    while len(all_ohlcv) < total_days:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit)
        if not ohlcv:
            break

        all_ohlcv.extend(ohlcv)
        
        # 다음 요청을 위해 since 값을 마지막 캔들 이후로 갱신
        since = ohlcv[-1][0] + 24 * 60 * 60 * 1000  # 마지막 캔들 다음날 (ms 단위)
        
        # API rate limit 보호
        time.sleep(exchange.rateLimit / 1000)
        print(f"----------------      {round(len(all_ohlcv)/total_days*100)}%...      ----------------")

    # 필요한 개수만큼만 자르기
    all_ohlcv = all_ohlcv[-total_days:]

    refined_df = _ohlcv2refined_df(all_ohlcv)
    return _save_refined_df(refined_df)


def btc_1min_training_data():
    # Binance 선물 거래소 초기화
    exchange = ccxt.binance({
        'options': {
            'defaultType': 'future',  # 선물 거래소로 설정
        }
    })

    symbol = 'BTC/USDT'
    timeframe = '1m'
    total_mins = 1440 * 1800  # 가져올 총 일 수
    limit = 1500              # API 한 번 호출 시 최대 개수
    since = exchange.parse8601('2017-08-17T00:00:00Z')  # BTC/USDT 선물 시작일 근처
    # since = exchange.parse8601('2022-02-24T00:00:00Z')  # BTC/USDT 선물 시작일 근처

    all_ohlcv = []

    # limit 단위로 나눠서 호출
    while len(all_ohlcv) < total_mins:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit)
        if not ohlcv:
            break

        all_ohlcv.extend(ohlcv)
        
        # 다음 요청을 위해 since 값을 마지막 캔들 이후로 갱신
        since = ohlcv[-1][0] + 60 * 1000  # 마지막 캔들 다음날 (ms 단위)
        
        # API rate limit 보호
        time.sleep(exchange.rateLimit / 1000)

        # 진행상황 표시
        print(f"----------------      {round(len(all_ohlcv)/total_mins*100)}%...      ----------------")

    # 필요한 개수만큼만 자르기
    all_ohlcv = all_ohlcv[-total_mins:]

    # 차트 데이터 저장하기
    refined_df = _ohlcv2refined_df(all_ohlcv)
    return _save_refined_df(refined_df)
def btc_1min_testing_data():
    # Binance 선물 거래소 초기화
    exchange = ccxt.binance({
        'options': {
            'defaultType': 'future',  # 선물 거래소로 설정
        }
    })

    symbol = 'BTC/USDT'
    timeframe = '1m'
    total_mins = 1440 * 300  # 가져올 총 일 수
    limit = 1500              # API 한 번 호출 시 최대 개수
    since = exchange.parse8601('2024-08-13T00:00:00Z')  # BTC/USDT 선물 시작일 근처

    all_ohlcv = []

    # limit 단위로 나눠서 호출
    while len(all_ohlcv) < total_mins:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit)
        if not ohlcv:
            break

        all_ohlcv.extend(ohlcv)
        
        # 다음 요청을 위해 since 값을 마지막 캔들 이후로 갱신
        since = ohlcv[-1][0] + 60 * 1000  # 마지막 캔들 다음날 (ms 단위)
        
        # API rate limit 보호
        time.sleep(exchange.rateLimit / 1000)

        # 진행상황 표시
        print(f"----------------      {round(len(all_ohlcv)/total_mins*100)}%...      ----------------")

    # 필요한 개수만큼만 자르기
    all_ohlcv = all_ohlcv[-total_mins:]

    # 차트 데이터 저장하기
    refined_df = _ohlcv2refined_df(all_ohlcv)
    return _save_refined_df(refined_df)


if __name__=='__main__':
    btc_1min_testing_data()