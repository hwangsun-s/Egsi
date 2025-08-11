import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np

# 설정
CSV_PATH    = 'hand_sign_data.csv'
MODEL_PATH  = 'hand_sign_binary.pth'
BATCH_SIZE  = 32
EPOCHS      = 7
LR          = 1e-3
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
RANDOM_SEED = 42

# MLP 모델 정의
class MLPClassifier(nn.Module):
    def __init__(self, input_dim, n_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128,  64),       nn.ReLU(),
            nn.Linear(64,   n_classes)
        )
    def forward(self, x):
        return self.net(x)

# 학습 함수
def train_model():
    # CSV 읽기 (헤더 없음)
    df = pd.read_csv(CSV_PATH, header=None)

    # 마지막 열이 label
    df = df[df.iloc[:, -1].isin([1, 2])].copy()
    df.iloc[:, -1] = df.iloc[:, -1].map({1: 0, 2: 1})  # 라벨 1→0, 2→1

    # 특징(X), 라벨(y) 분리
    X = df.iloc[:, :-1].values.astype(np.float32)
    y = df.iloc[:, -1].values.astype(np.int64)

    input_dim = X.shape[1]
    print(f"[INFO] 입력 특징 수: {input_dim}, 샘플 수: {len(X)}")

    # 데이터 셋 분할
    rng = np.random.RandomState(RANDOM_SEED)
    perm = rng.permutation(len(X))
    split = int(0.8 * len(X))
    train_idx, val_idx = perm[:split], perm[split:]

    X_train, y_train = X[train_idx], y[train_idx]
    X_val,   y_val   = X[val_idx],   y[val_idx]

    # 텐서 변환
    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_ds   = TensorDataset(torch.from_numpy(X_val),   torch.from_numpy(y_val))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE)

    # 모델 준비
    model = MLPClassifier(input_dim=input_dim, n_classes=2).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_acc = 0.0
    print("[INFO] 학습 시작...")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for bx, by in train_loader:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            logits = model(bx)
            loss = criterion(logits, by)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * bx.size(0)

        avg_loss = total_loss / len(train_loader.dataset)

        # 검증 정확도 계산
        model.eval()
        correct = 0
        with torch.no_grad():
            for vx, vy in val_loader:
                vx, vy = vx.to(DEVICE), vy.to(DEVICE)
                preds = model(vx).argmax(dim=1)
                correct += (preds == vy).sum().item()
        val_acc = correct / len(val_loader.dataset)

        print(f"Epoch {epoch}/{EPOCHS} | Loss: {avg_loss:.4f} | Val Acc: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), MODEL_PATH)
            print(" → Best 모델 저장됨.")

    print(f"[INFO] 학습 완료. 최고 검증 정확도: {best_acc:.4f}")

if __name__ == '__main__':
    train_model()


