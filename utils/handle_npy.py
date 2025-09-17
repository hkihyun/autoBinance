import pandas as pd
import numpy as np

'''
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

class HandleNpy():
    def __init__(self):
        self.npy_file = None
        self.df_file = None

    def df2npy(self, df : np.array):
        # timestamp 컬럼 제거하고 값만 추출
        values_only = df.drop(columns=['timestamp']).to_numpy()
        self.npy_file = values_only
        return
    
    def save_np(self, np):
        self.npy_file = np
        return
    
    def save_npy(self, file_name: str):
        np.save(f"data/raw_data/{file_name}.npy", self.npy_file)
        print("data/raw_data/에 저장되었습니다")
        