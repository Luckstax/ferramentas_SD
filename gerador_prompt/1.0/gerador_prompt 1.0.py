import os
import numpy as np
import csv
from pathlib import Path
from PIL import Image
from huggingface_hub import hf_hub_download
import onnxruntime as ort

# --- CONFIGURAÇÕES DO MODELO ---
MODEL_REPO = "SmilingWolf/wd-v1-4-convnext-tagger-v2"
MODEL_FILE = "model.onnx"
LABEL_FILE = "selected_tags.csv"

def carregar_modelo():
    print("⏳ Baixando/Carregando modelo WD14 (aprox. 100MB)...")
    model_path = hf_hub_download(MODEL_REPO, MODEL_FILE)
    label_path = hf_hub_download(MODEL_REPO, LABEL_FILE)
    
    # Inicia a sessão da IA
    session = ort.InferenceSession(model_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    
    # Carrega os nomes das tags
    with open(label_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        tags = [row[1] for row in reader]
    
    return session, tags

def processar_imagem(img_path, session):
    # Redimensiona para 448x448 (padrão do WD14)
    img = Image.open(img_path).convert("RGB")
    img = img.resize((448, 448), Image.Resampling.BICUBIC)
    
    # Converte para o formato que a IA entende
    img_array = np.array(img).astype(np.float32)
    img_array = img_array[:, :, ::-1] # RGB para BGR
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

def gerar_tags():
    caminho_pasta = input("📂 Digite o caminho da pasta de imagens: ").strip('"')
    tag_fixa = input("🏷️ Digite a tag principal (ex: illustrator_style): ").strip()
    threshold = 0.30 # Sensibilidade (0.1 a 0.9)

    pasta = Path(caminho_pasta)
    imagens = [f for f in pasta.iterdir() if f.suffix.lower() in [".jpg", ".png", ".webp", ".jpeg"]]

    if not imagens:
        print("❌ Nenhuma imagem encontrada.")
        return

    session, lista_tags = carregar_modelo()
    input_name = session.get_inputs()[0].name

    print(f"🚀 Processando {len(imagens)} imagens...")

    for img_path in imagens:
        img_input = processar_imagem(img_path, session)
        preds = session.run(None, {input_name: img_input})[0][0]

        # Filtra tags com confiança acima do threshold
        tags_encontradas = []
        for i, score in enumerate(preds):
            if score >= threshold and i >= 4: # Pula tags de categoria geral
                tags_encontradas.append(lista_tags[i].replace("_", " "))

        # Monta o texto final: Tag Fixa + Tags da IA
        prompt_final = f"{tag_fixa}, " + ", ".join(tags_encontradas)

        # Salva o .txt
        txt_path = img_path.with_suffix(".txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(prompt_final)

    print(f"✅ Concluído! Tags criadas para {len(imagens)} imagens.")

if __name__ == "__main__":
    gerar_tags()
