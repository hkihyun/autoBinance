import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from utils.load_raw_data import load_csv_to_df
from event.event_indicates_to_data import make_array_structure
from utils.save_events import save_event_npy


# ---------------------------
# Strategy Interface
# ---------------------------
class EventStrategy(ABC):
    @abstractmethod
    def detect(self, data: np.ndarray) -> np.ndarray:
        """
        이벤트 탐지 로직을 구현하는 메서드
        data: (N, F)
        return: (E, L, F)
        """
        pass


# ---------------------------
# Context (탐지기 본체)
# ---------------------------
class DetectEvent:
    def __init__(self, strategy: EventStrategy):
        self.strategy = strategy

    def load_raw_data(self, path: str) -> pd.DataFrame:
        return load_csv_to_df(path)

    def find_events(self, data: pd.DataFrame) -> np.ndarray:
        indexs = self.strategy.detect(data)
        events = make_array_structure(indexs=indexs, df=data)
        return events

    def save_events(self, events: np.ndarray, path: str) -> None:
        save_event_npy(events, path)

'''
# ---------------------------
# 실행 예시
# ---------------------------
if __name__ == "__main__":
    # 전략 선택해서 주입
    strategy = EmaCrossStrategy()
    detector = DetectEvent(strategy)

    raw = detector.load_raw_data("data/raw_data/minute_data.npy")
    events = detector.find_events(raw)
    detector.save_events(events, "data/precessed_data/events.npy")

'''