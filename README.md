# Adversarial-Attack-and-remediation-on-a-Face-Recognition-System
Face Recognition + Liveness Detection + rPPG Heartbeat Verification + Secure Encrypted Embeddings + Red-Team Adversarial Attack Analysis
This repository contains a complete end-to-end face authentication system designed for cybersecurity research, adversarial ML demonstrations, and hackathon projects.
It includes:
✔ Face recognition (FaceNet)
✔ Liveness detection (blink, mouth, head pose challenges)
✔ rPPG heartbeat detection (CHROM method)
✔ Secure encrypted embedding storage
✔ Red-team adversarial patch attack
✔ Documentation of vulnerabilities + mitigations
This project demonstrates both sides of AI security:
building a secure biometric system and breaking it using adversarial ML techniques


Features:
Secure Face Enrollment
Uses MTCNN for face alignment
Uses FaceNet (InceptionResnetV1) to generate 512-dim embeddings
Embeddings are AES-encrypted (Fernet)
Stored in secure directory: secure_embeddings/

Authentication includes 2 layers:
a) Liveness Level-1
Random challenges:
Blink
Open mouth
Look up
Look down
This blocks:
Printed photos
Flat videos
Basic deepfake playback

b)rPPG Heartbeat Detection (CHROM Method)
Extracts subtle skin-color changes at ~1–2 Hz from live video frames.
Used to detect:
Screens
Deepfake renderings
Replay attacks
Masks
Deepfakes cannot replicate real blood flow, making this an effective security layer.


Secure Embedding Storage:
.npy files are never stored in plaintext
Encrypted using Fernet (AES-128 CBC with HMAC)
Keys stored separately in secure_store/key.key
This prevents:
Embedding theft
Replay attacks
Model inversion attacks

Red-Team: Adversarial Patch Attack
The repository includes:
Attack code to generate adversarial patches
AR overlay to place patch on the live face
Evaluation pipeline
Write-up on bypass attempts + failures
This demonstrates how:
ML systems can be fooled
Adversarial perturbations work
Patches sometimes fail due to liveness + rPPG layers

Red-Team: Adversarial Patch Attack
The repository includes:
Attack code to generate adversarial patches
AR overlay to place patch on the live face
Evaluation pipeline
Write-up on bypass attempts + failures
This demonstrates how:
ML systems can be fooled
Adversarial perturbations work
Patches sometimes fail due to liveness + rPPG layers
