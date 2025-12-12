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


How It Works (High-Level Architecture)
1.Face Enrollment
User presses E to enroll
System detects & aligns face using MTCNN
FaceNet creates a normalized 512-dim embedding
Embedding → AES encryption → saved to disk
2.Authentication Process
Step 1 — Liveness Challenge
System randomly asks:
Blink
Open mouth
Look up/down
User must pass 5 randomized challenges.
Step 2 — rPPG Verification
Extracts heartbeat signal from:
Forehead region
Converts RGB → chrominance signals
Performs temporal filtering
Checks pulse range (0.8–2.0 Hz)
If no consistent heartbeat → DENY.
Step 3 — Face Match
Builds embedding from current frame
Decrypts stored embedding
Compares vector distance
If < threshold: authenticated
Else: denied
