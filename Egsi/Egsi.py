import cv2
import torch
import numpy as np
import pandas as pd
import mediapipe as mp
import os
import time
import pyttsx3
from torch import nn

# 기본 설정
CSV_PATH       = 'hand_sign_data.csv'
MODEL_PATH     = 'hand_sign_binary.pth'
DEVICE         = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CONF_THRESHOLD = 0.9  # 🔥 0.900 이상일 때만 결과 인정
SPEAK_INTERVAL = 3.0  # 🔊 동일 신호가 3초 지속될 경우만 음성 출력

# 음성 합성 초기화
engine = pyttsx3.init()
engine.setProperty('rate', 150)

class MLPClassifier(nn.Module):
    def __init__(self, input_dim, n_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64),        nn.ReLU(),
            nn.Linear(64, n_classes)
        )
    def forward(self, x):
        return self.net(x)

def preprocess_landmarks(landmarks):
    feats = []
    for lm in landmarks:
        feats.extend([lm.x, lm.y, lm.z])
    return np.array(feats, dtype=np.float32)

def run_inference():
    if not os.path.exists(MODEL_PATH):
        print("[ERROR] 모델 파일이 없습니다. 먼저 학습을 진행하세요.")
        return

    df = pd.read_csv(CSV_PATH, header=None)
    input_dim = df.shape[1] - 1

    model = MLPClassifier(input_dim=input_dim, n_classes=2).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    print("[INFO] 모델 로드 완료. Jetson IR 실시간 추론 시작")

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] 카메라 열기 실패")
        return

    # 👇 전체화면 창 설정
    window_name = "IR Hand Sign Inference (Jetson)"
    cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    label_map = {0: 'I am sick', 1: 'Help me please'}
    last_pred = None
    start_time = None
    last_announced = None

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # IR 대비 강화
        if frame.ndim == 2 or (frame.ndim == 3 and frame.shape[2] == 1):
            gray = frame[:, :] if frame.ndim == 2 else frame[:, :, 0]
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        enhanced = clahe.apply(gray)
        frame = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)
        disp = cv2.resize(frame.copy(), (640, 480))

        text = 'No sign detected'
        current_time = time.time()

        if res.multi_hand_landmarks:
            lm_list = res.multi_hand_landmarks[0].landmark
            if len(lm_list) == 21:
                feats = preprocess_landmarks(lm_list)
                if not np.any(np.isnan(feats)) and feats.shape[0] == input_dim:
                    x = torch.from_numpy(feats).unsqueeze(0).to(DEVICE)
                    with torch.no_grad():
                        logits = model(x)
                        probs = torch.softmax(logits, dim=1)[0]
                        conf, pred = probs.max(dim=0)
                        pred = pred.item()
                        conf = conf.item()

                        print(f"[DEBUG] Predicted={pred} | Confidence={conf:.3f}")

                        if conf >= CONF_THRESHOLD:
                            text = label_map[pred]

                            if last_pred == pred:
                                if start_time and (current_time - start_time) >= SPEAK_INTERVAL:
                                    if last_announced != pred:
                                        engine.say(label_map[pred])
                                        engine.runAndWait()
                                        last_announced = pred
                            else:
                                last_pred = pred
                                start_time = current_time
                        else:
                            text = "No sign detected"
                            last_pred = None
                            start_time = None
                            last_announced = None
                else:
                    print(f"[WARN] feature shape mismatch: {feats.shape}")

            mp.solutions.drawing_utils.draw_landmarks(
                disp, res.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)
        else:
            print("[WARN] 손 인식 실패")
            last_pred = None
            start_time = None
            last_announced = None

        cv2.putText(disp, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.imshow(window_name, disp)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    run_inference()


