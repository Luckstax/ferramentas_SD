import os
import numpy as np
import csv
import cv2  # Necessário para o filtro de nitidez
from pathlib import Path
from PIL import Image
from huggingface_hub import hf_hub_download
import onnxruntime as ort

# --- CONFIGURAÇÕES DO MODELO ---
MODEL_REPO = "SmilingWolf/wd-v1-4-convnext-tagger-v2"
MODEL_FILE = "model.onnx"
LABEL_FILE = "selected_tags.csv"

def calcular_nitidez(img_path):
    """Calcula se a imagem está borrada usando Variância do Laplaciano."""
    img = cv2.imread(str(img_path))
    cinza = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(cinza, cv2.CV_64F).var()

def carregar_modelo():
    print("⏳ Baixando/Carregando modelo WD14 (IA de Visão)...")
    model_path = hf_hub_download(MODEL_REPO, MODEL_FILE)
    label_path = hf_hub_download(MODEL_REPO, LABEL_FILE)
    session = ort.InferenceSession(model_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    with open(label_path, 'r') as f:
        reader = csv.reader(f)
        next(reader)
        tags = [row[1] for row in reader]
    return session, tags

def processar_imagem(img_path, session):
    img = Image.open(img_path).convert("RGB")
    img = img.resize((448, 448), Image.Resampling.BICUBIC)
    img_array = np.array(img).astype(np.float32)
    img_array = img_array[:, :, ::-1] # RGB para BGR
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def gerar_tags():
    # --- INPUTS DO USUÁRIO ---
    caminho_pasta = input("📂 Pasta das imagens: ").strip('"')
    trigger_tag = input("🏷️ Tag de Gatilho (ex: viperbug_miraculous): ").strip()
    extra_style = input("🎨 Estilo Fixo (ex: illustration, 3d render): ").strip()
    
    threshold = 0.35  # Confiança da IA
    min_nitidez = 100 # Abaixo disso a imagem é considerada borrada
    
    pasta = Path(caminho_pasta)
    imagens = [f for f in pasta.iterdir() if f.suffix.lower() in [".jpg", ".png", ".webp", ".jpeg"]]

    if not imagens:
        print("❌ Nenhuma imagem encontrada.")
        return

    session, lista_tags = carregar_modelo()
    input_name = session.get_inputs()[0].name

    print(f"🚀 Analisando {len(imagens)} imagens...")

    for img_path in imagens:
        # 1. Checa Nitidez
        score_nitidez = calcular_nitidez(img_path)
        if score_nitidez < min_nitidez:
            print(f"⚠️ Pulando {img_path.name} (Muito borrada: {score_nitidez:.2f})")
            continue

        # 2. IA gera as tags
        img_input = processar_imagem(img_path, session)
        preds = session.run(None, {input_name: img_input})[0][0]

        tags_encontradas = []
        for i, score in enumerate(preds):
            if score >= threshold and i >= 4:
                tags_encontradas.append(lista_tags[i].replace("_", " "))

        # 3. Monta o Prompt (Ordem: Gatilho -> Características IA -> Estilo)
        prompt_final = f"{trigger_tag}, {', '.join(tags_encontradas)}, {extra_style}"

        # 4. Salva o .txt
        txt_path = img_path.with_suffix(".txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(prompt_final)

    print(f"✅ Concluído! Seus arquivos .txt estão prontos.")

if __name__ == "__main__":
    gerar_tags()
