import cv2
import os
import json

from src.detector import detect_people
from src.embedder import embed_image
from src.similarity import is_similar
from src.dedup import is_duplicate
from src.utils import (
    ensure_dirs,
    load_clip,
    load_base_embeddings,
    save_result
)

VIDEO_PATH = "input/video.mp4"
BASE_DIR = "base"
OUTPUT_DIR = "output"
FRAME_STRIDE = 10


def main():
    ensure_dirs()

    clip_model, preprocess = load_clip()
    base_embeddings, base_hashes = load_base_embeddings(
        BASE_DIR, clip_model, preprocess
    )

    cap = cv2.VideoCapture(VIDEO_PATH)
    frame_idx = 0
    results = []

    while cap.isOpened():
        print("frame", frame_idx)
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % FRAME_STRIDE != 0:
            frame_idx += 1
            continue

        detections = detect_people(frame)
        print("detections:", len(detections))

        for det in detections:
            crop = det["crop"]
            emb = embed_image(crop, clip_model, preprocess)

            if not is_similar(emb, base_embeddings):
                continue

            if is_duplicate(crop, base_hashes):
                continue

            meta = save_result(frame, crop, frame_idx, det)
            results.append(meta)

        frame_idx += 1

    with open(os.path.join(OUTPUT_DIR, "metadata.json"), "w") as f:
        json.dump(results, f, indent=2)

    cap.release()


if __name__ == "__main__":
    print("programa iniciado")
    main()
