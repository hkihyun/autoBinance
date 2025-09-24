import argparse
import json
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from model import MODEL_REGISTRY
from train import SequenceDataset, compute_metrics   # train.py에서 정의한 함수 재사용
import matplotlib.pyplot as plt



# -------------------
# 1. 모델 예측 기반 성능 평가
# -------------------
def backtest_model(model, loader, config, device):
    model.eval() # 평가 모드
    # 모델 예측값, 실제 결과
    all_preds, all_targets = [], []

    with torch.no_grad():
        for Xb, yb in loader:
            Xb, yb = Xb.to(device), yb.to(device)
            preds = model(Xb)

            all_preds.append(preds.cpu())
            all_targets.append(yb.cpu())

    all_preds = torch.cat(all_preds) # (N, output_dim)
    all_targets = torch.cat(all_targets) # (N, target_dim)

    # 평가 기준
    mae, rmse, sign_acc = compute_metrics(all_targets, all_preds)

    print(
        f"[Backtest Metrics] "
        f"MAE: {mae:.6f}, RMSE: {rmse:.6f}, SignAcc: {sign_acc:.6f}"
    )

    return all_preds, all_targets


# -------------------
# 2. 거래 시뮬레이션 함수
# -------------------
def trading_backtest(preds, targets, buy_threshold, stop_loss, take_profit, holding_period):
    if isinstance(preds, torch.Tensor):
        preds = preds.numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.numpy()

    equity = 1.0   # 초기 자본 (100%)
    trades = []
    equity_curve = [equity]   # 자본 추이 저장

    i = 0
    n = len(preds) # 데이터 길이
    while i < n:
        # 매수 조건
        if preds[i].mean().item() > buy_threshold:   # preds도 다차원일 수 있으니 평균
            idx = i
            pred = preds[i].mean().item()  # 평균 예측값
            exit_type = None
            exit_ret = 0.0

            # 청산 조건 체크
            ret = targets[i].mean().item()   # ✅ 평균 수익률 사용

            if ret < stop_loss:
                equity = equity * (ret * 10 - 9)
                exit_type = "STOP_LOSS"
                exit_ret = ret
                print(f'----------{ret}----------')

            elif ret >= take_profit:
                equity = equity * (ret * 10 - 9)
                exit_type = "TAKE_PROFIT"
                exit_ret = ret

            # 기간 종료로 청산
            else:
                equity = equity * (ret * 10 - 9) # 10배 레버리지 가정, 소수점 5자리
                exit_type = "TIME_EXIT"
                exit_ret = ret

            trades.append((
                exit_type,
                idx,
                float(pred),
                float(exit_ret),
                float(equity)
            ))

            # 실시간 거래 로그 출력
            print(f"[거래 발생] {exit_type} | 거래 인덱스={idx}"
                  f"예측={pred:.4f}, 실제={exit_ret:.4f}, "
                  f"수익률={exit_ret:.4f}, 자본={equity:.4f}")

            equity_curve.append(equity)   #  자본 업데이트 기록

            i += 1  # 다음 인덱스로 이동

        else:
            i += 1

    return equity, trades, equity_curve




# 에쿼티 커브 시각화 함수
def plot_equity_curve(equity_curve):
    plt.figure(figsize=(10,5))
    plt.plot(equity_curve, label="Equity Curve")
    plt.xlabel("Trade Number")
    plt.ylabel("Equity (Capital)")
    plt.title("Backtest Equity Curve")
    plt.legend()
    plt.grid(True)
    plt.savefig("equity_curve.png", dpi=150)   # 저장
    plt.show()


# -------------------
# 3. 메인 실행
# -------------------   
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    args = parser.parse_args()

    # config 로드
    with open(args.config, "r") as f:
        config = json.load(f)

    model_name = config["model_name"]
    ModelClass = MODEL_REGISTRY[model_name]

    # 데이터셋 로드
    if config["preprocessed_data_path"].endswith(".npy"):
        arr = np.load(config["preprocessed_data_path"], allow_pickle=True) # shape: (_, 41, 5)
        X, y =  arr[:, :-1, :], arr[:, -1, :]  
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
        # print("X shape: ", X.shape)
        # print("y shape: ", y.shape)
    else:
        X, y = torch.load(config["preprocessed_data_path"])

    if model_name in ["tfmodel"]:
        dataset = SequenceDataset(X, y, config["seq_len"])
    else:
        dataset = TensorDataset(X, y)

    loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=False)

    # 모델 로드
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ModelClass(**config).to(device)
    model.load_state_dict(torch.load(config["model_path"], map_location=device))

    # 1) 기본 성능 평가
    preds, targets = backtest_model(model, loader, config, device)

    # 2) 거래 시뮬레이션 (config)
    final_equity, trade_log, equity_curve = trading_backtest(
        preds, targets,
        buy_threshold=config.get("buy_threshold"),
        stop_loss=config.get("stop_loss"),
        take_profit=config.get("take_profit"),
        holding_period=config.get("holding_period")
    )

    print(f"[Trading Backtest] 최종 누적 수익률: {(final_equity-1)*100:.2f}%")
    print(f"총 거래 횟수: {len(trade_log)}")

    # 에쿼티 커브 출력
    plot_equity_curve(equity_curve)

    # 결과 저장
    np.save("backtest_preds.npy", preds.numpy())
    np.save("backtest_targets.npy", targets.numpy())


if __name__ == "__main__":
    main()
