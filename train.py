import argparse
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, TensorDataset
import mlflow
import mlflow.pytorch
import torch.optim as optim
import numpy as np
from model import MODEL_REGISTRY
from pytorch_forecasting import TimeSeriesDataSet
import pandas as pd

# -------------------
# 지표 계산 함수
# -------------------
def compute_metrics(y_true, y_pred):
    """
    y_true, y_pred: torch.Tensor [batch, output_size]
    """
    mae = torch.mean(torch.abs(y_true - y_pred)).item()
    rmse = torch.sqrt(torch.mean((y_true - y_pred) ** 2)).item()
    sign_acc = (torch.sign(y_true) == torch.sign(y_pred)).float().mean().item()
    return mae, rmse, sign_acc


# -------------------
# 학습 함수
# -------------------
def train_model(model, train_loader, val_loader, config, device):
    # Loss / Optimizer
    loss_class = getattr(nn, config.get("loss", "MSELoss"))
    criterion = loss_class()

    opt_class = getattr(optim, config.get("optimizer", "AdamW"))
    optimizer = opt_class(model.parameters(), lr=config.get("learning_rate", 1e-3))

    # MLflow experiment
    mlflow.set_experiment(f"{config['model_name']}_prediction")

    with mlflow.start_run():
        mlflow.log_params(config)

        for epoch in range(config["epochs"]):
            # ---- Training ----
            model.train()
            train_loss, train_mae, train_rmse, train_sign = 0, 0, 0, 0
            for Xb, yb in train_loader:
                Xb, yb = Xb.to(device), yb.to(device)

                optimizer.zero_grad()
                preds = model(Xb)
                loss = criterion(preds, yb)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * Xb.size(0)
                mae, rmse, sign_acc = compute_metrics(yb, preds)
                train_mae += mae * Xb.size(0)
                train_rmse += rmse * Xb.size(0)
                train_sign += sign_acc * Xb.size(0)

            train_loss /= len(train_loader.dataset)
            train_mae /= len(train_loader.dataset)
            train_rmse /= len(train_loader.dataset)
            train_sign /= len(train_loader.dataset)

            # ---- Validation ----
            model.eval()
            val_loss, val_mae, val_rmse, val_sign = 0, 0, 0, 0
            with torch.no_grad():
                for Xb, yb in val_loader:
                    Xb, yb = Xb.to(device), yb.to(device)
                    preds = model(Xb)
                    loss = criterion(preds, yb)

                    val_loss += loss.item() * Xb.size(0)
                    mae, rmse, sign_acc = compute_metrics(yb, preds)
                    val_mae += mae * Xb.size(0)
                    val_rmse += rmse * Xb.size(0)
                    val_sign += sign_acc * Xb.size(0)

            val_loss /= len(val_loader.dataset)
            val_mae /= len(val_loader.dataset)
            val_rmse /= len(val_loader.dataset)
            val_sign /= len(val_loader.dataset)

            # ---- MLflow 로깅 ----
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            mlflow.log_metric("train_mae", train_mae, step=epoch)
            mlflow.log_metric("val_mae", val_mae, step=epoch)
            mlflow.log_metric("train_rmse", train_rmse, step=epoch)
            mlflow.log_metric("val_rmse", val_rmse, step=epoch)
            mlflow.log_metric("train_sign_acc", train_sign, step=epoch)
            mlflow.log_metric("val_sign_acc", val_sign, step=epoch)

            print(
                f"Epoch {epoch+1}/{config['epochs']} | "
                f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f} | "
                f"Train MAE: {train_mae:.6f}, Val MAE: {val_mae:.6f} | "
                f"Train RMSE: {train_rmse:.6f}, Val RMSE: {val_rmse:.6f} | "
                f"Train SignAcc: {train_sign:.6f}, Val SignAcc: {val_sign:.6f}"
            )

        torch.save(model.state_dict(), f"model/{config['model_name']}_final.pth")

        # 최종 모델 저장
        mlflow.pytorch.log_model(model, f"model_{config['model_name']}")





def main():
    # 1. argparse 설정
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    args = parser.parse_args()

    # -------------------
    # 2. config 로드
    # -------------------
    with open(args.config, "r") as f:
        config = json.load(f)

    model_name = config["model_name"]   
    ModelClass = MODEL_REGISTRY[model_name]

    # -------------------
    # 3. 데이터셋 로드 
    # -------------------

    if config["preprocessed_data_path"].endswith(".npy"):
        arr = np.load(config["preprocessed_data_path"], allow_pickle=True)
        X, y = arr[0], arr[1]

        # numpy → torch 변환
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
    else:
        X, y = torch.load(config["preprocessed_data_path"])


    dataset = TensorDataset(X, y)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config["batch_size"])

    # -------------------
    # 4. 모델 생성
    # -------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ModelClass(**config).to(device)

    train_model(model, train_loader, val_loader, config, device)






if __name__ == "__main__":
    main()