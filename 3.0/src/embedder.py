import torch
import cv2
from PIL import Image

def embed_image(image, model, preprocess):
    # OpenCV (BGR) → RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # numpy → PIL
    image = Image.fromarray(image)

    image = preprocess(image).unsqueeze(0)

    with torch.no_grad():
        emb = model.encode_image(image)

    return emb / emb.norm(dim=-1, keepdim=True)
