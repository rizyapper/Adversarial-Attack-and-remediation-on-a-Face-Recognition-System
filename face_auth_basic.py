
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import cv2
import pickle
import numpy as np
import torch

device=torch.device("cpu")
mtcnn= MTCNN(image_size=160,margin=0, min_face_size=20,device=device)
resnet= InceptionResnetV1(pretrained='vggface2').eval().to(device)

try:
    with open("face_db.pkl","rb") as f:
        face_db=pickle.load(f)
except FileNotFoundError:
    face_db={}

def get_embedding(img):
    face=mtcnn(img)
    
    if face is not None:
        face = face.unsqueeze(0).to(device)
        face = face.to(torch.float32) 
        emb = resnet(face).detach().cpu().numpy()
        return emb
    return None
cap=cv2.VideoCapture(0)
print("Press 'e' to enroll a new face, 'a' to authenticate, 'q' to quit.")
while True:
    ret, frame=cap.read()

    if not ret:
        print("Failed to grab frame")
        break
    rgb_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    boxes,_=mtcnn.detect(rgb_frame)
    if boxes is not None:
        for box in boxes:
            
            x1, y1, x2, y2 = [int(v) for v in box]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.imshow("Face Auth",frame)
    key=cv2.waitKey(1) & 0xFF
    if key==ord('e'):
        name=input("Enter name: ")
        img=Image.fromarray(rgb_frame)
        emb=get_embedding(img)
        if emb is not None:
            face_db[name]=emb
            with open("face_db.pkl","wb") as f:
                pickle.dump(face_db,f)
            print(f"Enrolled {name}")
        else:
            print("No face detected. Try again.")
    elif key==ord('a'):
        img=Image.fromarray(rgb_frame)
        emb=get_embedding(img)
        if emb is not None:
            min_dist=float('inf')
            identity=None
            for name, db_emb in face_db.items():
                dist=np.linalg.norm(emb-db_emb)
                if dist<min_dist:
                    min_dist=dist
                    identity=name
            if min_dist<0.8:
                print(f"Authenticated as {identity} (distance: {min_dist:.2f})")
            else:
                print("Authentication failed")
        else:
            print("No face detected. Try again.")
    elif key==ord('q'):
        break
cap.release()
cv2.destroyAllWindows()