import os
import cv2
import torch
import imagehash
from PIL import Image
import os
import clip

def load_clip():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval()
    return model, preprocess


def ensure_dirs():
    os.makedirs("base", exist_ok=True)
    os.makedirs("output", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("input", exist_ok=True)



def load_base_embeddings(base_dir, model, preprocess):
    embeddings = []
    hashes = []

    for file in os.listdir(base_dir):
        if not file.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
            continue

        path = os.path.join(base_dir, file)

        # OpenCV -> NumPy (BGR)
        img = cv2.imread(path)
        if img is None:
            continue

        # BGR -> RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # NumPy -> PIL
        img_pil = Image.fromarray(img_rgb)

        # Hash para deduplicação
        img_hash = imagehash.phash(img_pil)
        hashes.append(img_hash)

        # Preprocess CLIP
        img_tensor = preprocess(img_pil).unsqueeze(0)

        with torch.no_grad():
            emb = model.encode_image(img_tensor)
            emb = emb / emb.norm(dim=-1, keepdim=True)

        embeddings.append(emb)

    return embeddings, hashes


def save_result(frame, crop, frame_idx, det, output_dir="output"):
    os.makedirs(output_dir, exist_ok=True)

    x1, y1, x2, y2 = det["bbox"]

    filename = f"frame_{frame_idx}_{x1}_{y1}_{x2}_{y2}.jpg"
    path = os.path.join(output_dir, filename)

    cv2.imwrite(path, crop)

    return {
        "frame": frame_idx,
        "file": filename,
        "bbox": det["bbox"],
    }
