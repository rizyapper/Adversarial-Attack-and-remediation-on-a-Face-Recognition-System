import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import time
import random
import os

# ------------------ CONFIG ------------------
TOTAL_CHALLENGES = 5
CHALLENGE_TIMEOUT = 8  # seconds per challenge
EMBED_THRESHOLD = 1.0
DEVICE = "cpu"

CHALLENGE_POOL = ["blink", "open_mouth", "look_up", "look_down"]
EMBED_DIR = "embeddings"
os.makedirs(EMBED_DIR, exist_ok=True)

# ------------------ FACE MODELS ------------------
mtcnn = MTCNN(keep_all=False, device=DEVICE)
facenet = InceptionResnetV1(pretrained="vggface2").eval().to(DEVICE)
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True,
                                  min_detection_confidence=0.6, min_tracking_confidence=0.6)

# ------------------ EMBEDDINGS ------------------
def load_embeddings():
    embeds = {}
    for f in os.listdir(EMBED_DIR):
        if f.endswith(".npy"):
            embeds[f[:-4]] = np.load(os.path.join(EMBED_DIR, f))
    return embeds

def save_embedding(name, emb):
    np.save(os.path.join(EMBED_DIR, f"{name}.npy"), emb)

# ------------------ HELPERS ------------------
def dist(p1, p2):
    return ((p1.x - p2.x)**2 + (p1.y - p2.y)**2)**0.5

def eye_aspect_ratio(lm, ids):
    p = [lm[i] for i in ids]
    vertical = dist(p[1], p[5]) + dist(p[2], p[4])
    horizontal = dist(p[0], p[3]) + 1e-8
    return vertical / (2.0 * horizontal)

def mouth_open_ratio(lm):
    return dist(lm[13], lm[14]) / (dist(lm[78], lm[308]) + 1e-8)

def head_pose_vertical(lm):
    nose = lm[1]; eye = lm[159]; chin = lm[152]
    ratio = dist(nose, eye) / (dist(nose, chin) + 1e-8)
    if ratio < 0.38: return "down"
    if ratio > 0.48: return "up"
    return "center"

def get_face_embedding(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    face_tensor = mtcnn(pil)
    if face_tensor is None:
        return None
    with torch.no_grad():
        emb = facenet(face_tensor.unsqueeze(0).to(DEVICE)).cpu().numpy()
    return emb.flatten()

def recognize_face(frame, registered_embs):
    emb = get_face_embedding(frame)
    if emb is None:
        return None
    best_name = None
    best_dist = float("inf")
    for name, stored in registered_embs.items():
        if stored.shape[0] != emb.shape[0]:
            continue
        d = np.linalg.norm(emb - stored)
        if d < best_dist:
            best_dist = d
            best_name = name
    if best_dist < EMBED_THRESHOLD:
        return best_name
    return None

# ------------------ STREAMLIT UI ------------------
st.title("Secure Face Authentication + L1 Demo")

frame_window = st.image([])

cap = cv2.VideoCapture(0)
registered_embs = load_embeddings()

if not cap.isOpened():
    st.error("Cannot open webcam.")
else:
    # Dropdown menu for action
    action = st.selectbox("Select Action", ["Enroll", "Authenticate"])
    name_input = st.text_input("Enter User Name (for Enrollment only)")

    if action == "Enroll":
        st.info("Show your face to the camera for enrollment")
        while True:
            ret, frame = cap.read()
            if not ret: continue
            frame_window.image(frame, channels="BGR")
            if st.button("Capture & Enroll"):
                emb = get_face_embedding(frame)
                if emb is not None:
                    if name_input:
                        save_embedding(name_input, emb)
                        registered_embs = load_embeddings()
                        st.success(f"✅ {name_input} enrolled successfully!")
                        break
                    else:
                        st.warning("Enter a valid name")
                else:
                    st.warning("No face detected. Try again.")
                time.sleep(0.5)

    elif action == "Authenticate":
        st.info("Perform L1 challenges and authentication")
        completed = 0
        for i in range(TOTAL_CHALLENGES):
            challenge = random.choice(CHALLENGE_POOL)
            st.write(f"Challenge {i+1}: {challenge}")
            action_done = False
            start_time = time.time()
            while time.time() - start_time < CHALLENGE_TIMEOUT:
                ret, frame = cap.read()
                if not ret: continue
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = face_mesh.process(rgb)
                if res.multi_face_landmarks:
                    lm = res.multi_face_landmarks[0].landmark
                    EAR_L = eye_aspect_ratio(lm,[33,160,158,133,153,144])
                    EAR_R = eye_aspect_ratio(lm,[263,387,385,362,380,373])
                    blink = (EAR_L<0.19 and EAR_R<0.19)
                    mouth_open = (mouth_open_ratio(lm)>0.35)
                    vert = head_pose_vertical(lm)
                    if (challenge=="blink" and blink) or \
                       (challenge=="open_mouth" and mouth_open) or \
                       (challenge=="look_up" and vert=="up") or \
                       (challenge=="look_down" and vert=="down"):
                        action_done = True
                        completed += 1
                        st.success(f"✅ Challenge {i+1} completed!")
                        time.sleep(0.5)
                        break
                frame_window.image(frame, channels="BGR")
            if not action_done:
                st.warning(f"❌ Challenge {i+1} failed!")

        st.info(f"L1 Results: {completed}/{TOTAL_CHALLENGES} challenges passed")

        # Recognition after L1
        auth_name = None
        for _ in range(15):
            ret, frame = cap.read()
            if not ret: continue
            auth_name = recognize_face(frame, registered_embs)
            frame_window.image(frame, channels="BGR")
            if auth_name:
                break
        if auth_name:
            st.success(f"Authentication successful! Recognized: {auth_name}")
        else:
            st.error("Authentication failed. No match.")

cap.release()
