# 원하는 전략 클라스
from event.strategy.ema_cross import EmaCrossStrategy
from event.interface import DetectEvent

# event 찾는 전략. 이 클래스만 교체하면 된다
strategy = EmaCrossStrategy()

# interface *이 코드는 고정*
detector = DetectEvent(strategy)

# 이 위치에서 raw data 가져오기
raw = detector.load_raw_data("data/raw_data/minute_data.npy")

# raw data 안에서 event 탐지
events = detector.find_events(raw)

# 이 위치에 이 이름으로 저장
detector.save_events(events, "data/precessed_data/events.npy")