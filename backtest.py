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

    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)

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
def trading_backtest(preds, targets,
                     buy_threshold, stop_loss, take_profit, holding_period):
    if isinstance(preds, torch.Tensor):
        preds = preds.numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.numpy()

    # 추가된 부분: preds와 targets가 다차원일 경우 첫 번째 피처 선택
    if preds.ndim > 1:
        preds = preds[:, 0]
    if targets.ndim > 1:
        targets = targets[:, 0]

    n = len(preds)
    equity = 1.0   # 초기 자본 (100%)
    trades = []
    equity_curve = [equity]   # 자본 추이 저장

    i = 0
    while i < n:
        # 매수 조건
        if preds[i] > buy_threshold:
            entry_idx = i
            entry_pred = preds[i]
            exit_idx = None
            exit_type = None
            exit_ret = 0.0

            for j in range(i+1, min(i+holding_period, n)):
                ret = targets[j]

                if ret < stop_loss:
                    equity *= (1 + ret)
                    exit_idx = j
                    exit_type = "STOP_LOSS"
                    exit_ret = ret
                    break

                if ret > take_profit:
                    equity *= (1 + ret)
                    exit_idx = j
                    exit_type = "TAKE_PROFIT"
                    exit_ret = ret
                    break

            if exit_idx is None:
                exit_idx = min(i+holding_period-1, n-1)
                ret = targets[exit_idx]
                equity *= (1 + ret)
                exit_type = "TIME_EXIT"
                exit_ret = ret

            trades.append((
                exit_type,
                entry_idx,
                exit_idx,
                float(entry_pred),
                float(exit_ret),
                float(exit_ret * 100),
                float(equity)
            ))

            # 실시간 거래 로그 출력
            print(f"[거래 발생] {exit_type} | 진입={entry_idx}, 청산={exit_idx}, "
                  f"예측={entry_pred:.4f}, 실제={exit_ret:.4f}, "
                  f"수익률={exit_ret*100:.2f}%, 자본={equity:.4f}")

            equity_curve.append(equity)   #  자본 업데이트 기록

            i = exit_idx + 1
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
        arr = np.load(config["preprocessed_data_path"], allow_pickle=True)
        X, y = arr[0], arr[1]
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
    else:
        X, y = torch.load(config["preprocessed_data_path"])

    if model_name in ["tfmodel"]:
        dataset = SequenceDataset(X, y, config["seq_len"])
    else:
        dataset = TensorDataset(X, y)

    loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=False)

    # 모델 로드
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ModelClass(**config).to(device)
    model.load_state_dict(torch.load(config["model_path"], map_location=device))

    # 1) 기본 성능 평가
    preds, targets = backtest_model(model, loader, config, device)

    # 2) 거래 시뮬레이션 (config)
    final_equity, trade_log, equity_curve = trading_backtest(
        preds, targets,
        buy_threshold=config.get("buy_threshold", 0.005),
        stop_loss=config.get("stop_loss", -0.01),
        take_profit=config.get("take_profit", 0.02),
        holding_period=config.get("holding_period", 5)
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
