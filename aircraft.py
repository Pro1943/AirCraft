import cv2
import mediapipe as mp
import math
import time
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

model_path = 'hand_landmarker.task'
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2, # Enabled 2 hands
    min_hand_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# --- State for Hand 2 (Gesture Detection) ---
hand2_was_open = False
blocks = []
last_place_time = 0
PLACE_DELAY = 0.3

def distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def is_hand_open(landmarks):
    # Simple check: Are finger tips (8, 12, 16, 20) higher than their bases (6, 10, 14, 18)?
    # In image coordinates, smaller Y means "higher"
    count = 0
    if landmarks[8].y < landmarks[6].y: count += 1
    if landmarks[12].y < landmarks[10].y: count += 1
    if landmarks[16].y < landmarks[14].y: count += 1
    if landmarks[20].y < landmarks[18].y: count += 1
    return count >= 3 # True if at least 3 fingers are extended

def draw_block(img, x, y, size=40):
    cv2.rectangle(img, (x, y), (x+size, y+size), (200, 200, 200), -1)
    cv2.rectangle(img, (x, y), (x+size, y+size), (0, 0, 0), 2)

def get_nearest_block(x, y):
    if not blocks: return None
    return min(blocks, key=lambda b: distance((x, y), (b[0], b[1])))

cap = cv2.VideoCapture(0)
with HandLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        success, img = cap.read()
        if not success: break

        img = cv2.flip(img, 1)
        h, w, _ = img.shape
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        
        frame_timestamp_ms = int(time.time() * 1000)
        result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)

        trigger_place = False
        cursor_pos = (0, 0)

        if result.hand_landmarks:
            # Process Hand 1 (The Cursor / Remover)
            h1 = result.hand_landmarks[0]
            cursor_pos = (int(h1[8].x * w), int(h1[8].y * h))
            it = cursor_pos
            tt = (int(h1[4].x * w), int(h1[4].y * h))
            mt = (int(h1[12].x * w), int(h1[12].y * h))

            # Fist logic to remove blocks
            if distance(it, mt) < 35 and distance(it, tt) < 35:
                nearest = get_nearest_block(it[0], it[1])
                if nearest and distance(it, nearest) < 60:
                    blocks.remove(nearest)

            # Process Hand 2 (The Trigger)
            if len(result.hand_landmarks) > 1:
                h2 = result.hand_landmarks[1]
                currently_open = is_hand_open(h2)

                # Detection of Open -> Closed sequence
                if currently_open:
                    hand2_was_open = True
                elif hand2_was_open and not currently_open:
                    # Just closed the hand!
                    trigger_place = True
                    hand2_was_open = False
                
                # Visual feedback for Hand 2
                color = (0, 255, 0) if currently_open else (0, 0, 255)
                cv2.putText(img, "H2 READY" if hand2_was_open else "H2 OPEN FIST", (w-200, 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Draw both hands
            for hand_lms in result.hand_landmarks:
                for lm in hand_lms:
                    cv2.circle(img, (int(lm.x * w), int(lm.y * h)), 3, (255, 255, 255), -1)

            # Execute Placement if Hand 2 triggered
            if trigger_place:
                if time.time() - last_place_time > PLACE_DELAY:
                    blocks.append(cursor_pos)
                    last_place_time = time.time()

        for bx, by in blocks:
            draw_block(img, bx, by)

        cv2.circle(img, cursor_pos, 10, (0, 255, 255), 2) # Cursor highlight
        cv2.putText(img, "Air Craft v0.3: H1 Cursor, H2 Clench to Place", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.imshow("Air Craft", img)

        if cv2.waitKey(1) & 0xFF == 27: break

cap.release()
cv2.destroyAllWindows()
