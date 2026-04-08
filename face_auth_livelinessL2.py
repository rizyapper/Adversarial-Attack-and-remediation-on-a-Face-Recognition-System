import cv2
import mediapipe as mp
import time
import random
import numpy as np
import os
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch


device = 'cpu'


if not os.path.exists("embeddings"):
    os.makedirs("embeddings")


mtcnn = MTCNN(keep_all=False, device=device)
facenet = InceptionResnetV1(pretrained='vggface2').eval().to(device)



def load_embeddings():
    embeds = {}
    for f in os.listdir("embeddings"):
        if f.endswith(".npy"):
            embeds[f[:-4]] = np.load(os.path.join("embeddings", f))
    return embeds



def recognize_face(face_tensor, registered_embeds, threshold=0.9):
    if not registered_embeds:
        print("⚠ No user enrolled yet. Please enroll first.")
        return None

    with torch.no_grad():
        emb = facenet(face_tensor.unsqueeze(0).to(device)).cpu().numpy()

    for name, saved_emb in registered_embeds.items():
        dist = np.linalg.norm(emb - saved_emb)
        if dist < threshold:
            return name

    return None



def enroll_user():
    name = input("Enter your name for enrollment: ").strip()
    if not name:
        print("Invalid name.")
        return

    cap = cv2.VideoCapture(0)
    print("Align your face. Capturing in 3 seconds...")
    time.sleep(3)

    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame.")
        cap.release()
        return

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face, prob = mtcnn(rgb, return_prob=True)

    if face is None or prob < 0.90:
        print("Face not detected clearly. Try again.")
        cap.release()
        return

    with torch.no_grad():
        emb = facenet(face.unsqueeze(0).to(device)).cpu().numpy()

    np.save(f"embeddings/{name}.npy", emb)
    print(f"Enrollment successful for {name}!")

    cap.release()



mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

def dist(p1, p2):
    return ((p1.x - p2.x)**2 + (p1.y - p2.y)**2) ** 0.5

def eye_aspect_ratio(lm, ids):
    p = [lm[i] for i in ids]
    vertical = dist(p[1], p[5]) + dist(p[2], p[4])
    horizontal = dist(p[0], p[3])
    return vertical / (2.0 * horizontal)

def mouth_open_ratio(lm):
    return dist(lm[13], lm[14]) / dist(lm[78], lm[308])

def head_pose_vertical(lm):
    nose = lm[1]
    eye = lm[159]
    chin = lm[152]
    ratio = dist(nose, eye) / dist(nose, chin)
    if ratio < 0.38:
        return "down"
    elif ratio > 0.48:
        return "up"
    return "center"

def random_challenge():
    return random.choice(["blink", "open_mouth", "look_up", "look_down"])



def authenticate_user():
    registered = load_embeddings()

    challenge = random_challenge()
    completed = 0
    TOTAL = 5
    action_done = False

    print("\nStarting Liveness Check...")
    print("Challenge 1:", challenge)

    cap = cv2.VideoCapture(0)


    while True:
        if completed >= TOTAL:
            print("Liveness passed! Proceeding to face recognition...")
            break

        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = face_mesh.process(rgb)

        if res.multi_face_landmarks:
            lm = res.multi_face_landmarks[0].landmark

            EAR_L = eye_aspect_ratio(lm, [33,160,158,133,153,144])
            EAR_R = eye_aspect_ratio(lm, [263,387,385,362,380,373])
            blink = EAR_L < 0.19 and EAR_R < 0.19

            mouth_open = mouth_open_ratio(lm) > 0.35

            vert = head_pose_vertical(lm)

            if challenge == "blink" and blink:
                action_done = True
            elif challenge == "open_mouth" and mouth_open:
                action_done = True
            elif challenge == "look_up" and vert == "up":
                action_done = True
            elif challenge == "look_down" and vert == "down":
                action_done = True

            if action_done:
                completed += 1
                print(f"Completed {completed}/{TOTAL}")
                cv2.putText(frame, "Success!", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                time.sleep(1)
                if completed < TOTAL:
                    challenge = random_challenge()
                    print(f"Challenge {completed+1}: {challenge}")
                action_done = False

        cv2.putText(frame, f"Do: {challenge}", (20,460),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

        cv2.imshow("Liveness Check", frame)
        if cv2.waitKey(1) == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            return


    print("Recognizing face...")
    time.sleep(1)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face, prob = mtcnn(rgb, return_prob=True)

        if face is not None and prob > 0.90:
            user = recognize_face(face, registered)
            if user:
                print(f"\nAUTH SUCCESS → Welcome, {user}!\n")
            else:
                print("\nAUTH FAILED → Face not recognized.\n")
            break

        cv2.imshow("Authentication", frame)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()



while True:
    print("\n------------------------")
    print("AI SECURITY PROJECT MENU")
    print("------------------------")
    print("E - Enroll New Face")
    print("A - Authenticate (Liveness + Face Recognition)")
    print("Q - Quit")
    print("------------------------")

    choice = input("Choose an option: ").lower().strip()

    if choice == 'e':
        enroll_user()
    elif choice == 'a':
        authenticate_user()
    elif choice == 'q':
        print("Exiting...")
        break
    else:
        print("Invalid choice. Try again.")
