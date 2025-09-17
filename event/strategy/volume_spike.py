class VolumeSpikeStrategy():
    def detect(self, df, lookback=10, spike_ratio=3.0):
        """
        lookback: 평균 거래량을 구할 과거 봉 수 (기본 10)
        spike_ratio: 평균 대비 몇 배 이상일 때 스파이크로 볼지 (기본 3배)
        """
        signal_indices = []

        for i in range(lookback, len(df)):
            avg_vol = df['volume'].iloc[i-lookback:i].mean()
            curr_vol = df['volume'].iloc[i]

            if avg_vol == 0:
                continue

            # 거래량이 평균의 spike_ratio배 이상일 때 신호 발생
            if curr_vol >= avg_vol * spike_ratio:
                signal_indices.append(i)

        print(len(signal_indices))
        return signal_indices
