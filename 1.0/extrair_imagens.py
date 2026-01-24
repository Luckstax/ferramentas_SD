import cv2
import os
import torch
import clip
import numpy as np
from PIL import Image
from ultralytics import YOLO

# ================= CONFIG =================
BASE_IMAGE_PATH = "base.jpg"
VIDEO_PATH = "video.mp4"
OUTPUT_DIR = "output"
SIMILARITY_THRESHOLD = 0.85
FRAME_SKIP = 5  # processa 1 a cada N frames
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ==========================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---- Load YOLO (detecção de pessoas)
yolo = YOLO("yolov8n.pt")

# ---- Load CLIP
clip_model, preprocess = clip.load("ViT-B/32", device=DEVICE)

def get_embedding(pil_image):
    image = preprocess(pil_image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        embedding = clip_model.encode_image(image)
    embedding /= embedding.norm(dim=-1, keepdim=True)
    return embedding.cpu().numpy()[0]

# ---- Embedding da imagem base
base_img = Image.open(BASE_IMAGE_PATH).convert("RGB")
base_embedding = get_embedding(base_img)

# ---- Abrir vídeo
cap = cv2.VideoCapture(VIDEO_PATH)
frame_idx = 0
saved_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_idx += 1
    if frame_idx % FRAME_SKIP != 0:
        continue

    results = yolo(frame, verbose=False)[0]

    for box in results.boxes:
        cls = int(box.cls[0])
        if cls != 0:  # 0 = pessoa
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        person_crop = frame[y1:y2, x1:x2]

        if person_crop.size == 0:
            continue

        pil_crop = Image.fromarray(cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB))
        emb = get_embedding(pil_crop)

        similarity = np.dot(base_embedding, emb)

        if similarity >= SIMILARITY_THRESHOLD:
            filename = f"{OUTPUT_DIR}/match_{frame_idx}_{saved_count}.jpg"
            cv2.imwrite(filename, person_crop)
            saved_count += 1
            print(f"[MATCH] Frame {frame_idx} | Similarity: {similarity:.3f}")

cap.release()
print(f"Finalizado. {saved_count} imagens salvas.")
