import pandas as pd

def load_csv_to_df(filepath: str) -> pd.DataFrame:
    # CSV 파일 읽기
    df = pd.read_csv(filepath)
    return df


'''
# 사용 예시
df = load_csv_to_df('data/BTCUSDT_1m_2024-09-09_to_2024-09-16.csv')
print(df.head())
'''