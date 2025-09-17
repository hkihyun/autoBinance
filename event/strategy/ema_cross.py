

import ta

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

class EmaCrossStrategy():
    def detect(self, df):
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
        df.drop(columns=['ema_5'], inplace=True)
        df.drop(columns=['ema_20'], inplace=True)
        print(len(cross_indices))
        return cross_indices
