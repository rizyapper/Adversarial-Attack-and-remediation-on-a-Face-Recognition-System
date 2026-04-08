# ar_overlay_demo.py
# Benign demo: overlays patch.png on eyes region in real time.
# Requires: pip install opencv-python mediapipe numpy

import cv2
import mediapipe as mp
import numpy as np

# Load overlay image (PNG with alpha)
overlay_path = "patch.png"   # any PNG you want to overlay for demo
overlay = cv2.imread(overlay_path, cv2.IMREAD_UNCHANGED)  # BGRA

# MediaPipe face mesh
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(max_num_faces=1, refine_landmarks=True,
                            min_detection_confidence=0.6, min_tracking_confidence=0.6)

def alpha_blend(bg, fg, x, y):
    """Alpha blend fg (with alpha) onto bg at position (x,y)."""
    h, w = fg.shape[:2]
    if x < 0 or y < 0 or x + w > bg.shape[1] or y + h > bg.shape[0]:
        # Clip if overlay goes out of bounds
        x0 = max(0, -x); y0 = max(0, -y)
        x1 = min(w, bg.shape[1] - x); y1 = min(h, bg.shape[0] - y)
        fg = fg[y0:y1, x0:x1]
        h, w = fg.shape[:2]
        x = max(x, 0); y = max(y, 0)
    if fg.shape[2] == 4:
        alpha = fg[:, :, 3] / 255.0
        for c in range(3):
            bg[y:y+h, x:x+w, c] = (alpha * fg[:, :, c] + (1-alpha) * bg[y:y+h, x:x+w, c])
    else:
        bg[y:y+h, x:x+w] = fg
    return bg

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = face_mesh.process(img_rgb)

    if res.multi_face_landmarks:
        lm = res.multi_face_landmarks[0].landmark
        # Use eyes positions (MediaPipe landmark indices for eye corners)
        # left_eye_outer = 33 ; right_eye_outer = 263
        left = lm[33]; right = lm[263]
        h, w = frame.shape[:2]
        # compute center between eyes in pixel coordinates
        cx = int((left.x + right.x) / 2.0 * w)
        cy = int((left.y + right.y) / 2.0 * h)

        # compute distance between eyes to scale overlay
        eye_dx = (right.x - left.x) * w
        eye_dy = (right.y - left.y) * h
        eye_dist = int(np.hypot(eye_dx, eye_dy))

        # desired overlay width proportional to eye distance
        overlay_w = int(eye_dist * 2.2)   # tweak multiplier for visual fit
        overlay_h = int(overlay_w * overlay.shape[0] / overlay.shape[1])

        # resize and place overlay centered at (cx, cy - some offset)
        resized = cv2.resize(overlay, (overlay_w, overlay_h), interpolation=cv2.INTER_AREA)
        x = int(cx - overlay_w/2)
        y = int(cy - overlay_h/2 - overlay_h*0.15)  # a little upward offset

        # alpha-blend overlay onto frame
        frame = alpha_blend(frame, resized, x, y)

    cv2.imshow("AR overlay demo", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
