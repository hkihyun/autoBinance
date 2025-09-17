'''
startegy 에서 인덱스를 추출. 
이 함수는 추출한 인덱스와 df를 이용해 
1. df에서 필요한 부분만을 추출
2. 추출한 부분의 구조를 바꾸고 shape(41, 5, _)
2. df를 np.array로 바꿔준다
'''

import numpy as np

def make_array_structure(indexs, df):
    # 각 크로스 지점 기준으로 전후 30개 row 추출
    result_segments = []
    for idx in indexs:
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

        # 이후 5개 구간
        after_segment = df.iloc[after_start:after_end].copy()
        base_price_after = after_segment.iloc[0]['open']
        after_segment['open'] = (after_segment['open'] / base_price_after - 1) * 100  # 퍼센트(%)
        after_segment['close'] = (after_segment['close'] / base_price_after - 1) * 100  # 퍼센트(%)
        after_segment['high'] = (after_segment['high'] / base_price_after - 1) * 100  # 퍼센트(%)
        after_segment['low'] = (after_segment['low'] / base_price_after - 1) * 100  # 퍼센트(%)
        end = after_segment['open'].to_frame().T

        # 두 구간을 numpy 배열로 변환
        prev_np = prev_segment.to_numpy()
        end_np = end.to_numpy()

        # 수직으로 붙이기 (40행 + 1행 = 41행)
        segment_np = np.vstack([prev_np, end_np])
        result_segments.append(segment_np)

    final_np = np.stack(result_segments)
    final_np = np.array(final_np, dtype=np.float32)
    final_np = np.round(final_np, 5)

    # 결과 확인
    print(final_np.shape)
    print(final_np)
    
    return final_np