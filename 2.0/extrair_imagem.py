import cv2
import os
import torch
import clip
import numpy as np
from PIL import Image
from ultralytics import YOLO
from collections import deque

# ================= CONFIG =================
BASE_DIR = "base"
VIDEO_PATH = "video.mp4"
OUTPUT_DIR = "output"

FRAME_SKIP = 5
SIMILARITY_THRESHOLD = 0.85
ANTI_DUP_BASE_THRESHOLD = 0.95
ANTI_DUP_RESULT_THRESHOLD = 0.93

TEMPORAL_WINDOW = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ==========================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------- Models ----------
yolo = YOLO("yolov8n.pt")
clip_model, preprocess = clip.load("ViT-B/32", device=DEVICE)

# ---------- Utils ----------
def get_embedding(pil_img):
    img = preprocess(pil_img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        emb = clip_model.encode_image(img)
    emb /= emb.norm(dim=-1, keepdim=True)
    return emb.cpu().numpy()[0]

def cosine(a, b):
    return float(np.dot(a, b))

# ---------- Load base embeddings ----------
base_embeddings = []
for file in os.listdir(BASE_DIR):
    img = Image.open(os.path.join(BASE_DIR, file)).convert("RGB")
    base_embeddings.append(get_embedding(img))

base_embeddings = np.array(base_embeddings)
base_mean_embedding = base_embeddings.mean(axis=0)
base_mean_embedding /= np.linalg.norm(base_mean_embedding)

# ---------- Video ----------
cap = cv2.VideoCapture(VIDEO_PATH)
frame_id = 0
saved_embeddings = []
similarity_window = deque(maxlen=TEMPORAL_WINDOW)
saved_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_id += 1
    if frame_id % FRAME_SKIP != 0:
        continue

    detections = yolo(frame, verbose=False)[0]

    for box in detections.boxes:
        if int(box.cls[0]) != 0:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        pil_crop = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        emb = get_embedding(pil_crop)

        sim = cosine(base_mean_embedding, emb)
        similarity_window.append(sim)

        if len(similarity_window) < TEMPORAL_WINDOW:
            continue

        avg_sim = sum(similarity_window) / TEMPORAL_WINDOW
        if avg_sim < SIMILARITY_THRESHOLD:
            continue

        # ---- Anti-duplicate vs base ----
        if max(cosine(emb, b) for b in base_embeddings) > ANTI_DUP_BASE_THRESHOLD:
            continue

        # ---- Anti-duplicate vs results ----
        if saved_embeddings:
            if max(cosine(emb, s) for s in saved_embeddings[-5:]) > ANTI_DUP_RESULT_THRESHOLD:
                continue

        # ---- Save ----
        filename = f"{OUTPUT_DIR}/match_{frame_id}_{saved_count}.jpg"
        cv2.imwrite(filename, crop)
        saved_embeddings.append(emb)
        saved_count += 1

        print(f"[SAVE] Frame {frame_id} | Similaridade média: {avg_sim:.3f}")

cap.release()
print(f"Finalizado. {saved_count} imagens salvas.")
