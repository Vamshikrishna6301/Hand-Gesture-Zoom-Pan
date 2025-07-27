import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)

# Load image and setup
original_image = cv2.imread("image.jpg")
if original_image is None:
    raise ValueError("Image not found. Make sure 'image.jpg' is in the same folder.")

image_center = (original_image.shape[1] // 2, original_image.shape[0] // 2)
scale = 1.0
min_scale, max_scale = 0.5, 3.0
prev_zoom_distance = None

# For pan
offset_x, offset_y = 0, 0
prev_left_hand_pos = None

def extract_hand_label(results, index):
    return results.multi_handedness[index].classification[0].label

def zoom_image(img, scale, offset_x, offset_y):
    h, w = img.shape[:2]
    new_w, new_h = int(w * scale), int(h * scale)

    resized = cv2.resize(img, (new_w, new_h))

    # Center crop to 640x480
    x_start = max(0, (new_w // 2 - 320) + offset_x)
    y_start = max(0, (new_h // 2 - 240) + offset_y)

    x_end = x_start + 640
    y_end = y_start + 480

    cropped = np.zeros((480, 640, 3), dtype=np.uint8)
    cropped_y, cropped_x = min(resized.shape[0] - y_start, 480), min(resized.shape[1] - x_start, 640)
    cropped[:cropped_y, :cropped_x] = resized[y_start:y_start + cropped_y, x_start:x_start + cropped_x]

    return cropped

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks and results.multi_handedness:
        for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
            label = extract_hand_label(results, i)
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract coordinates
            lm = hand_landmarks.landmark

            if label == "Right":
                # Zoom control using distance between thumb tip and index tip
                thumb = lm[mp_hands.HandLandmark.THUMB_TIP]
                index = lm[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                thumb_pos = np.array([thumb.x * frame.shape[1], thumb.y * frame.shape[0]])
                index_pos = np.array([index.x * frame.shape[1], index.y * frame.shape[0]])
                distance = np.linalg.norm(index_pos - thumb_pos)

                if prev_zoom_distance is not None:
                    diff = distance - prev_zoom_distance
                    scale += diff * 0.01  # Sensitivity
                    scale = np.clip(scale, min_scale, max_scale)
                prev_zoom_distance = distance

            elif label == "Left":
                # Pan control using index finger position
                index_finger = lm[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                pos = np.array([index_finger.x * frame.shape[1], index_finger.y * frame.shape[0]])

                if prev_left_hand_pos is not None:
                    delta = pos - prev_left_hand_pos
                    offset_x += int(delta[0])
                    offset_y += int(delta[1])
                prev_left_hand_pos = pos
    else:
        prev_zoom_distance = None
        prev_left_hand_pos = None

    zoomed_img = zoom_image(original_image, scale, offset_x, offset_y)

    combined = np.hstack((frame, zoomed_img))
    cv2.imshow("Hand Gesture Control", combined)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC key
        break

cap.release()
cv2.destroyAllWindows()
