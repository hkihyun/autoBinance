import numpy as np




def save_event_npy(event_np: np.array, path: str):
    np.save(path, event_np)
    print(f"{path}에 저장 완료")
        