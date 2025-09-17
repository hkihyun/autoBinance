import pandas as pd
import ta
import numpy as np

'''
# 예시 데이터
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
def search_cross_ema(df):
    # 5EMA, 20EMA 계산
    df['ema_5'] = ta.trend.ema_indicator(df['close'], window=5)
    df['ema_20'] = ta.trend.ema_indicator(df['close'], window=20)

    # 크로스 발생 조건
    # 5EMA가 20EMA를 상향 돌파 → (이전에는 5 < 20) and (현재는 5 >= 20)
    # 5EMA가 20EMA를 하향 돌파 → (이전에는 5 > 20) and (현재는 5 <= 20)
    cross_indices = []
    for i in range(1, len(df)):
        prev_diff = df.loc[i-1, 'ema_5'] - df.loc[i-1, 'ema_20']
        curr_diff = df.loc[i, 'ema_5'] - df.loc[i, 'ema_20']
        if prev_diff < 0 and curr_diff >= 0:   # 골든크로스
            cross_indices.append(i)
        elif prev_diff > 0 and curr_diff <= 0: # 데드크로스
            cross_indices.append(i)
    return cross_indices

def raw_data_to_ema_precess_data(df):
    # ema 지점 추출
    cross_indices = search_cross_ema(df)
    # 각 크로스 지점 기준으로 전후 30개 row 추출
    result_segments = []
    for idx in cross_indices:
        start = idx - 39
        after_start = idx + 1
        after_end = idx + 6 
        if start < 0 or  after_end > len(df):
            continue

        # 이전 40개 구간
        prev_segment = df.iloc[start:idx+1].copy()
        base_price_prev = prev_segment.iloc[0]['open']
        prev_segment['open'] = (prev_segment['open'] / base_price_prev - 1) * 100  # 퍼센트(%)
        prev_segment['close'] = (prev_segment['close'] / base_price_prev - 1) * 100  # 퍼센트(%)
        prev_segment['high'] = (prev_segment['high'] / base_price_prev - 1) * 100  # 퍼센트(%)
        prev_segment['low'] = (prev_segment['low'] / base_price_prev - 1) * 100  # 퍼센트(%)
        prev_segment.drop(columns=['timestamp'], inplace=True) # timestamp column 제거
        prev_segment.drop(columns=['ema_5'], inplace=True) # ema_5 column 제거
        prev_segment.drop(columns=['ema_20'], inplace=True) # ema_20 column 제거

        # 이후 5개 구간
        after_segment = df.iloc[after_start:after_end].copy()
        base_price_after = after_segment.iloc[0]['open']
        after_segment['open'] = (after_segment['open'] / base_price_after - 1) * 100  # 퍼센트(%)
        after_segment['close'] = (after_segment['close'] / base_price_after - 1) * 100  # 퍼센트(%)
        after_segment['high'] = (after_segment['high'] / base_price_after - 1) * 100  # 퍼센트(%)
        after_segment['low'] = (after_segment['low'] / base_price_after - 1) * 100  # 퍼센트(%)
        end = after_segment['open'].to_frame().T

        # 두 구간을 붙이다
        # segment = pd.concat([prev_segment, end])
        # result_segments.append(segment)

        # 두 구간을 numpy 배열로 변환
        prev_np = prev_segment.to_numpy()
        end_np = end.to_numpy()

        # 수직으로 붙이기 (40행 + 1행 = 41행)
        segment_np = np.vstack([prev_np, end_np])

        result_segments.append(segment_np)


    final_np = np.stack(result_segments)
    final_np = np.round(final_np, 5)
    print(final_np.shape)
    # 결과 확인
    print(final_np)
    return final_np
