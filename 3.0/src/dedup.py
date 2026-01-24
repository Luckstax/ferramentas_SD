import imagehash
from PIL import Image
import cv2


def is_duplicate(image, base_hashes, threshold=5):
    pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    h = imagehash.phash(pil)

    for bh in base_hashes:
        if abs(h - bh) <= threshold:
            return True

    base_hashes.append(h)
    return False
