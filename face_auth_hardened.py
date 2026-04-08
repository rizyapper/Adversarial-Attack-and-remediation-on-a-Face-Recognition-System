# face_auth_hardened.py
import cv2
import numpy as np
import pickle
import time
import io
import random
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import torch


DEVICE = torch.device('cpu')   
EMBED_THRESHOLD = 0.85        
CONSECUTIVE_MATCHES = 3       
LIVENESS_TIMEOUT = 4.0        
HEAD_TURN_DELTA = 0.03        
JPEG_QUALITY = 80             
LOGFILE = "attack_log.csv"
DBFILE = "face_db.pkl"


mtcnn = MTCNN(image_size=160, margin=0, min_face_size=20, device=DEVICE)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(DEVICE)


try:
    with open(DBFILE, "rb") as f:
        face_db = pickle.load(f)
except Exception:
    face_db = {}  

def save_db():
    with open(DBFILE, "wb") as f:
        pickle.dump(face_db, f)


def pil_jpeg_roundtrip(pil_img, quality=JPEG_QUALITY):
    buf = io.BytesIO()
    pil_img.save(buf, format='JPEG', quality=quality)
    buf.seek(0)
    return Image.open(buf).convert('RGB')

def get_embedding_from_pil(pil_img):

    pil_img = pil_jpeg_roundtrip(pil_img, quality=JPEG_QUALITY)
    face = mtcnn(pil_img)  
    if face is not None:
        face = face.unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            emb = resnet(face).cpu().numpy()
        return emb  # shape (1,512)
    return None

def cosine_sim(a, b):
    
    a = a / (np.linalg.norm(a) + 1e-10)
    b = b / (np.linalg.norm(b) + 1e-10)
    return float(np.dot(a, b))

def log_attempt(method, score, result, name_attempted=""):

    ts = time.time()
    with open(LOGFILE, "a") as f:
        f.write(f"{ts},{method},{score:.6f},{result},{name_attempted}\n")


def compute_normalized_nose_offset(landmarks, box):
 
    left_eye = landmarks[0]
    right_eye = landmarks[1]
    nose = landmarks[2]
    eye_center_x = (left_eye[0] + right_eye[0]) / 2.0
    nose_x = nose[0]
    face_width = max(1.0, box[2] - box[0])
    return (nose_x - eye_center_x) / face_width  
def run_head_turn_challenge(cap, initial_box, initial_landmarks, direction, timeout=LIVENESS_TIMEOUT):
    
    start_time = time.time()
    baseline_offset = compute_normalized_nose_offset(initial_landmarks, initial_box)
    
    expected_sign = -1 if direction == 'left' else 1

    while time.time() - start_time < timeout:
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes, probs, points = mtcnn.detect(rgb, landmarks=True)
        if boxes is None or len(boxes) == 0:
            
            cv2.putText(frame, f"Perform action: turn {direction.upper()}", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
            cv2.imshow("Liveness Challenge", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        
        i_best = 0
        best_overlap = 0
        for i, b in enumerate(boxes):
            cx = (b[0] + b[2]) / 2.0; cy = (b[1] + b[3]) / 2.0
            ibx = (initial_box[0] + initial_box[2]) / 2.0; iby = (initial_box[1] + initial_box[3]) / 2.0
            dist = np.hypot(cx-ibx, cy-iby)
            if i == 0 or dist < best_overlap:
                best_overlap = dist
                i_best = i

        new_box = boxes[i_best]
        new_points = points[i_best]  
        new_offset = compute_normalized_nose_offset(new_points, new_box)
        delta = new_offset - baseline_offset
        
        cv2.putText(frame, f"Perform action: turn {direction.upper()}", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2)
        cv2.putText(frame, f"Delta: {delta:.4f}", (10,60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        # draw the chosen box
        x1,y1,x2,y2 = [int(v) for v in new_box]
        cv2.rectangle(frame, (x1,y1), (x2,y2), (255,0,0), 2)
        cv2.imshow("Liveness Challenge", frame)
        if expected_sign * delta > HEAD_TURN_DELTA:
            # Passed
            return True
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    return False

# -----------------------
# Main loop: enroll (e) and authenticate (a)
# -----------------------
cap = cv2.VideoCapture(0)
print("Press 'e' to enroll a new face, 'a' to authenticate (with liveness challenge), 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    boxes, probs = mtcnn.detect(rgb)  # landmarks not needed for live overlay; later call detect with landmarks
    # draw boxes
    if boxes is not None:
        for box in boxes:
            x1,y1,x2,y2 = [int(v) for v in box]
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
    cv2.imshow("Face Auth (Hardened)", frame)
    key = cv2.waitKey(1) & 0xFF

    # ---------------- Enroll ----------------
    if key == ord('e') and boxes is not None:
        name = input("Enter name for enrollment: ").strip()
        # take the current frame for enrollment
        pil = Image.fromarray(rgb)
        emb = get_embedding_from_pil(pil)
        if emb is not None:
            face_db[name] = emb  # one embedding per person for simplicity; can store list/centroid
            save_db()
            print(f"[✔] Enrolled {name}")
            log_attempt("enroll", float(0.0), "enrolled", name)
        else:
            print("[✘] No face detected for enrollment.")
            log_attempt("enroll", 0.0, "fail", name)

    # ---------------- Authenticate (hardened) ----------------
    if key == ord('a'):
        # first detect with landmarks to get initial box & landmarks for challenge baseline
        boxes_lm, probs_lm, points = mtcnn.detect(rgb, landmarks=True)
        if boxes_lm is None or len(boxes_lm) == 0:
            print("[✘] No face found for authentication.")
            continue
        # choose largest face (or first)
        idx = 0
        init_box = boxes_lm[idx]
        init_lm = points[idx]  # shape (5,2)
        # pick random challenge direction
        direction = random.choice(['left', 'right'])
        print(f"[~] Liveness challenge: Please turn your head {direction.upper()} within {LIVENESS_TIMEOUT} seconds.")
        passed = run_head_turn_challenge(cap, init_box, init_lm, direction, timeout=LIVENESS_TIMEOUT)
        cv2.destroyWindow("Liveness Challenge")
        if not passed:
            print("[✘] Liveness check failed (no sufficient head movement).")
            log_attempt("liveness", 0.0, "fail")
            continue
        # Liveness passed: now capture frames and require consecutive matches
        consecutive = 0
        matched_name = None
        matched_score = None
        start_time = time.time()
        max_wait = 5.0  # maximum time to collect consecutive frames
        while time.time() - start_time < max_wait and consecutive < CONSECUTIVE_MATCHES:
            ret2, frame2 = cap.read()
            if not ret2:
                break
            rgb2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
            pil2 = Image.fromarray(rgb2)
            emb_probe = get_embedding_from_pil(pil2)
            if emb_probe is None:
                consecutive = 0
                continue
            # compare against DB
            best_score = -1.0
            best_name = "Unknown"
            for name, db_emb in face_db.items():
                score = cosine_sim(db_emb.flatten(), emb_probe.flatten())
                if score > best_score:
                    best_score = score
                    best_name = name
            # check threshold
            if best_score >= EMBED_THRESHOLD:
                consecutive += 1
                matched_name = best_name
                matched_score = best_score
                print(f"[~] Match attempt {consecutive}/{CONSECUTIVE_MATCHES}: {best_name} ({best_score:.3f})")
            else:
                consecutive = 0
                print(f"[~] No match in this frame (best {best_score:.3f})")
            # small pause so we don't flood
            cv2.waitKey(150)

        if consecutive >= CONSECUTIVE_MATCHES:
            print(f"[✔] Authentication SUCCESS: {matched_name} (score={matched_score:.3f})")
            log_attempt("authenticate", matched_score, "success", matched_name)
        else:
            print("[✘] Authentication FAILED (consistency requirement not met).")
            if matched_score is None:
                log_attempt("authenticate", 0.0, "fail", "Unknown")
            else:
                log_attempt("authenticate", matched_score, "fail", matched_name)

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
