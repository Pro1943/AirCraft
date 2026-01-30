import cv2
import mediapipe as mp
import math
import time
import numpy as np

model_path = 'hand_landmarker.task'
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_tracking_confidence=0.6
)

blocks = []
last_place_time = 0
hand2_was_open = False
BLOCK_SIZE = 50
SCAN_LINE_Y = 0

def distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def is_hand_open(landmarks):
    return landmarks[8].y < landmarks[6].y and \
           landmarks[12].y < landmarks[10].y and \
           landmarks[16].y < landmarks[14].y

def draw_cool_block(img, x, y, size):
    bx, by = x - size // 2, y - size // 2 
    cv2.rectangle(img, (bx-2, by-2), (bx+size+2, by+size+2), (255, 0, 255), 2)
    overlay = img.copy()
    cv2.rectangle(overlay, (bx, by), (bx+size, by+size), (100, 0, 100), -1)
    cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
    cv2.rectangle(img, (bx+5, by+5), (bx+size-5, by+size-5), (255, 255, 0), 1)

def get_nearest_block(x, y):
    if not blocks: return None
    return min(blocks, key=lambda b: distance((x, y), b))

cap = cv2.VideoCapture(0)

with HandLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        success, img = cap.read()
        if not success: break
        
        img = cv2.flip(img, 1)
        h, w, _ = img.shape

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        result = landmarker.detect_for_video(mp_image, int(time.time() * 1000))

        cursor_pos = None
        
        if result.hand_landmarks:
            for idx, hand_lms in enumerate(result.hand_landmarks):
                lm8 = (int(hand_lms[8].x * w), int(hand_lms[8].y * h)) # Index
                lm4 = (int(hand_lms[4].x * w), int(hand_lms[4].y * h)) # Thumb
                lm12 = (int(hand_lms[12].x * w), int(hand_lms[12].y * h)) # Middle

                if idx == 0:
                    cursor_pos = lm8
                    # Draw Reticle
                    cv2.drawMarker(img, cursor_pos, (0, 255, 255), cv2.MARKER_CROSS, 20, 2)
                    
                    # Removal Logic (Fist on Hand 1)
                    if distance(lm8, lm4) < 40 and distance(lm8, lm12) < 40:
                        nearest = get_nearest_block(lm8[0], lm8[1])
                        if nearest and distance(lm8, nearest) < 70:
                            blocks.remove(nearest)

                elif idx == 1: # HAND 2: GESTURE TRIGGER
                    currently_open = is_hand_open(hand_lms)
                    
                    if currently_open:
                        hand2_was_open = True
                        cv2.putText(img, "SYSTEM READY", (w-200, h-30), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (0, 255, 0), 1)
                    elif hand2_was_open and not currently_open:
                        if cursor_pos and (time.time() - last_place_time > 0.3):
                            blocks.append(cursor_pos)
                            last_place_time = time.time()
                            hand2_was_open = False
                    
                    for lm in hand_lms:
                        cv2.circle(img, (int(lm.x*w), int(lm.y*h)), 2, (255, 255, 0), -1)

        for b in blocks:
            draw_cool_block(img, b[0], b[1], BLOCK_SIZE)
        cv2.putText(img, f"AIR CRAFT  | BLOCKS: {len(blocks)}", (20, 40), 
                    cv2.FONT_HERSHEY_TRIPLEX, 0.8, (0, 255, 255), 1)
        cv2.imshow("Air Craft Neon", img)
        if cv2.waitKey(1) & 0xFF == 27: break

cap.release()
cv2.destroyAllWindows()
