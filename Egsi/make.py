import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import time
import os

# 설정
CSV_PATH    = 'hand_sign_data.csv'  # 저장 파일
LABEL       = 2                    # <-- 여기를 직접 바꿔서 사용 (0: 없음, 1: 아프다, 2: 도와주세요)
MAX_SAMPLES = 400                   # 최대 저장 수
CAPTURE_FPS = 3                     # 초당 저장 수

# MediaPipe Hands 초기화
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
draw = mp.solutions.drawing_utils

# 시작 안내
print("==== 좌표 기반 데이터 자동 수집 ====")
print(f"레이블: {LABEL} (0:없음, 1:아프다, 2:도와주세요)")
print("3초 후 시작...")
time.sleep(3)

# 카메라 열기
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[ERROR] 카메라 열기 실패")
    exit()

# 타이밍 설정
interval = 1.0 / CAPTURE_FPS
last_time = time.time()
data = []

while len(data) < MAX_SAMPLES:
    ret, frame = cap.read()
    if not ret:
        break

    # IR 카메라 흑백 → BGR 변환
    if frame.ndim == 2 or (frame.ndim == 3 and frame.shape[2] == 1):
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = hands.process(rgb)
    disp = frame.copy()

    if res.multi_hand_landmarks:
        lm_list = res.multi_hand_landmarks[0].landmark
        if len(lm_list) == 21:
            draw.draw_landmarks(disp, res.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)

            # 주기적 저장
            current_time = time.time()
            if current_time - last_time >= interval:
                last_time = current_time
                feats = []
                for lm in lm_list:
                    feats.extend([lm.x, lm.y, lm.z])
                feats.append(LABEL)
                data.append(feats)
                print(f"[INFO] 저장됨: {len(data)}/{MAX_SAMPLES}")

    cv2.putText(disp, f"Label: {LABEL} | Captured: {len(data)}/{MAX_SAMPLES}",
                (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
    cv2.imshow("Auto Data Capture", disp)

    if cv2.waitKey(1) & 0xFF == 27:
        print("ESC 눌림. 강제 종료")
        break

cap.release()
cv2.destroyAllWindows()

# CSV 저장
if data:
    df = pd.DataFrame(data)
    if os.path.exists(CSV_PATH):
        df.to_csv(CSV_PATH, mode='a', index=False, header=False)
    else:
        df.to_csv(CSV_PATH, index=False, header=False)
    print(f"[완료] 총 {len(data)}개 샘플 저장됨: {CSV_PATH}")
else:
    print("[경고] 저장된 데이터 없음")
