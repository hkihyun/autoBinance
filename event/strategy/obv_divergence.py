import ta

class ObvDivergenceStrategy():
    def detect(self, df, lookback=5):
        # OBV 계산
        df['obv'] = ta.volume.OnBalanceVolumeIndicator(
            close=df['close'], volume=df['volume']
        ).on_balance_volume()

        divergence_indices = []

        # 최근 구간 가격/OBV 고점저점 비교
        for i in range(lookback, len(df)):
            price_prev = df['close'].iloc[i-lookback]
            price_curr = df['close'].iloc[i]
            obv_prev = df['obv'].iloc[i-lookback]
            obv_curr = df['obv'].iloc[i]

            # 가격은 상승, OBV는 하락 → 매도 시그널
            if price_curr > price_prev and obv_curr < obv_prev:
                divergence_indices.append(i)

            # # 가격은 하락, OBV는 상승 → 매수 시그널
            # if price_curr < price_prev and obv_curr > obv_prev:
            #     divergence_indices.append(i)

        df.drop(columns=['obv'], inplace=True)
        print(len(divergence_indices))
        return divergence_indices