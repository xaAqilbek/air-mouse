import math
import os
import time

import cv2
import mediapipe as mp
import pyautogui
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "hand_landmarker.task")
MARGIN = 0.15
SMOOTHING = 5
PINCH_RATIO = 0.3
ENABLE_CLICK = True

pyautogui.PAUSE = 0
screen_w, screen_h = pyautogui.size()

options = vision.HandLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=vision.RunningMode.VIDEO,
    num_hands=1,                             # track only one hand
    min_hand_detection_confidence=0.6,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)
landmarker = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)
prev_x, prev_y = pyautogui.position()
was_pinching = False
start_time = time.monotonic()

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Could not read from camera")
            break

        frame = cv2.flip(frame, 1)                       # mirror
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)     # MediaPipe хочет RGB, а не BGR
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        timestamp_ms = int((time.monotonic() - start_time) * 1000)
        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        # Рисуем рамку "рабочей зоны" (внутри неё рука управляет всем экраном)
        x1, y1 = int(MARGIN * w), int(MARGIN * h)
        x2, y2 = int((1 - MARGIN) * w), int((1 - MARGIN) * h)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)

        if result.hand_landmarks:
            hand = result.hand_landmarks[0]      # 21 points from one hand
            wrist = hand[0]                      # wrist
            thumb_tip = hand[4]                  # кончик большого пальца
            index_tip = hand[8]                  # кончик указательного
            middle_base = hand[9]                # основание среднего пальца
            middle_tip = hand[12]                # кончик среднего

            # 1. Двигаем курсор указательным пальцем
            # x и y у точек от 0 до 1. Растягиваем рабочую зону на весь экран.
            nx = (index_tip.x - MARGIN) / (1 - 2 * MARGIN)
            ny = (index_tip.y - MARGIN) / (1 - 2 * MARGIN)
            nx = min(max(nx, 0), 1)              # не даём выйти за 0..1
            ny = min(max(ny, 0), 1)
            target_x = nx * (screen_w - 1)
            target_y = ny * (screen_h - 1)

            # сглаживание: каждый кадр проходим только часть пути до цели
            cur_x = prev_x + (target_x - prev_x) / SMOOTHING
            cur_y = prev_y + (target_y - prev_y) / SMOOTHING
            pyautogui.moveTo(cur_x, cur_y)
            prev_x, prev_y = cur_x, cur_y

            # 2. Клик: большой + средний палец вместе
            hand_size = math.hypot(wrist.x - middle_base.x, wrist.y - middle_base.y)
            pinch_dist = math.hypot(thumb_tip.x - middle_tip.x, thumb_tip.y - middle_tip.y)
            is_pinching = pinch_dist < PINCH_RATIO * hand_size

            if ENABLE_CLICK and is_pinching and not was_pinching:
                pyautogui.click()                # клик только в момент сведения
            was_pinching = is_pinching

            # Показываем на кадре, что видит программа
            color = (0, 0, 255) if is_pinching else (0, 255, 0)
            cv2.circle(frame, (int(index_tip.x * w), int(index_tip.y * h)), 12, color, -1)

        cv2.imshow("Air Mouse", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except pyautogui.FailSafeException:
    print("Stopped: cursor reached the screen corner (fail-safe).")

finally:
    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()

