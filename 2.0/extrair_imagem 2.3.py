import cv2
import os
import torch
import clip
import numpy as np
import customtkinter as ctk
from tkinter import filedialog, messagebox
from PIL import Image
from ultralytics import YOLO
from collections import deque
from pathlib import Path
import threading

# Configurações de UI
ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")

class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Extrator PRO Miraculous | YOLOv8 + CLIP + Revisão")
        self.geometry("700x650")

        # Variáveis
        self.base_path = ctk.StringVar()
        self.video_path = ctk.StringVar()
        self.sim_threshold = ctk.DoubleVar(value=0.85)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.setup_ui()

    def setup_ui(self):
        self.grid_columnconfigure(0, weight=1)

        self.label_title = ctk.CTkLabel(self, text="Extração Adaptativa de Personagens", font=ctk.CTkFont(size=22, weight="bold"))
        self.label_title.pack(pady=20)

        self.frame_main = ctk.CTkFrame(self)
        self.frame_main.pack(padx=20, pady=10, fill="both", expand=True)

        # Seletores
        self.add_path_selector("Pasta de Referência (Imagens da Personagem):", self.base_path, is_dir=True)
        self.add_path_selector("Vídeo do Episódio:", self.video_path, is_dir=False)

        # Configurações
        self.label_sim = ctk.CTkLabel(self.frame_main, text="Sensibilidade do CLIP (Similaridade):")
        self.label_sim.pack(pady=(15, 0))
        
        self.slider = ctk.CTkSlider(self.frame_main, from_=0.5, to=1.0, variable=self.sim_threshold, command=self.update_label)
        self.slider.pack(pady=5, padx=20, fill="x")
        
        self.label_val = ctk.CTkLabel(self.frame_main, text="0.85")
        self.label_val.pack()

        # Progresso
        self.progress_bar = ctk.CTkProgressBar(self, width=500)
        self.progress_bar.set(0)
        self.progress_bar.pack(pady=20)

        self.label_status = ctk.CTkLabel(self, text="Aguardando início...")
        self.label_status.pack()

        self.btn_start = ctk.CTkButton(self, text="EXECUTAR PIPELINE COMPLETO", command=self.start_worker, height=50, font=ctk.CTkFont(weight="bold"))
        self.btn_start.pack(pady=20)

    def add_path_selector(self, label_text, var, is_dir):
        label = ctk.CTkLabel(self.frame_main, text=label_text)
        label.pack(anchor="w", padx=20, pady=(10, 0))
        
        frame = ctk.CTkFrame(self.frame_main, fg_color="transparent")
        frame.pack(fill="x", padx=15)
        
        entry = ctk.CTkEntry(frame, textvariable=var)
        entry.pack(side="left", fill="x", expand=True, padx=5, pady=5)
        
        btn = ctk.CTkButton(frame, text="Abrir", width=70, 
                            command=lambda: var.set(filedialog.askdirectory() if is_dir else filedialog.askopenfilename()))
        btn.pack(side="right", padx=5)

    def update_label(self, val):
        self.label_val.configure(text=f"{val:.2f}")

    def get_embedding(self, pil_img, model, preprocess):
        img = preprocess(pil_img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            emb = model.encode_image(img)
        emb /= emb.norm(dim=-1, keepdim=True)
        return emb.cpu().numpy()[0]

    def start_worker(self):
        if not self.base_path.get() or not self.video_path.get():
            messagebox.showwarning("Aviso", "Selecione todos os arquivos!")
            return
        self.btn_start.configure(state="disabled")
        threading.Thread(target=self.main_process, daemon=True).start()

    def main_process(self):
        try:
            output_dir = "dataset_extraido"
            os.makedirs(output_dir, exist_ok=True)

            # 1. Carregamento de Modelos
            self.label_status.configure(text="Iniciando IAs (YOLOv8 + CLIP)...")
            yolo = YOLO("yolov8m.pt") # Medium é o equilíbrio perfeito
            clip_model, preprocess = clip.load("ViT-L/14", device=self.device)

            # 2. Processar Referências
            self.label_status.configure(text="Analisando suas imagens de referência...")
            base_embs = []
            ref_path = Path(self.base_path.get())
            for f in ref_path.glob("*.*"):
                if f.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                    img = Image.open(f).convert("RGB")
                    base_embs.append(self.get_embedding(img, clip_model, preprocess))
            
            ref_embedding = np.mean(base_embs, axis=0)
            ref_embedding /= np.linalg.norm(ref_embedding)

            # 3. Processar Vídeo
            cap = cv2.VideoCapture(self.video_path.get())
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            skip = max(1, int(fps * 0.3)) # Analisa 3 frames por segundo de vídeo
            
            frame_idx, saved_count = 0, 0
            saved_embs = []

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break

                if frame_idx % skip == 0:
                    results = yolo(frame, verbose=False)[0]
                    for box in results.boxes:
                        if int(box.cls[0]) != 0: continue # Foca em pessoas

                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        crop = frame[y1:y2, x1:x2]
                        if crop.size == 0: continue

                        # CLIP Similarity
                        pil_crop = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                        emb = self.get_embedding(pil_crop, clip_model, preprocess)
                        sim = float(np.dot(ref_embedding, emb))

                        if sim >= self.sim_threshold.get():
                            # Anti-duplicata rápido
                            is_dup = False
                            if saved_embs:
                                if max(float(np.dot(emb, s)) for s in saved_embs[-10:]) > 0.96:
                                    is_dup = True
                            
                            if not is_dup:
                                # Salva quadrado para o LoRA
                                h, w = crop.shape[:2]
                                dim = min(h, w)
                                final_crop = crop[0:dim, 0:dim] # Simplificado para o exemplo
                                
                                out_path = f"{output_dir}/raw_{frame_idx}.jpg"
                                cv2.imwrite(out_path, final_crop)
                                saved_embs.append(emb)
                                saved_count += 1

                frame_idx += 1
                if frame_idx % 20 == 0:
                    self.progress_bar.set(frame_idx / total_frames)
                    self.label_status.configure(text=f"Analisando: {frame_idx}/{total_frames} frames")

            cap.release()

            # 4. REVISÃO AUTOMÁTICA
            self.label_status.configure(text="Iniciando Revisão Final (Removendo erros)...")
            self.revisar_pasta(output_dir, ref_embedding, clip_model, preprocess)

            self.label_status.configure(text=f"Sucesso! Dataset finalizado.")
            messagebox.showinfo("Fim", "Processamento e Revisão concluídos!")

        except Exception as e:
            messagebox.showerror("Erro", str(e))
        finally:
            self.btn_start.configure(state="normal")

    def revisar_pasta(self, pasta, ref_emb, model, preprocess):
        """Varre a pasta gerada e deleta outliers ou clones."""
        caminho = Path(pasta)
        arquivos = list(caminho.glob("*.jpg"))
        embeddings = []
        validos = []

        # Remove imagens que não batem com a referência na segunda passada
        for arq in arquivos:
            img = Image.open(arq).convert("RGB")
            emb = self.get_embedding(img, model, preprocess)
            sim = float(np.dot(ref_emb, emb))
            
            if sim < self.sim_threshold.get():
                arq.unlink() # Deleta se estiver abaixo do threshold
            else:
                embeddings.append(emb)
                validos.append(arq)

        # Deleta imagens quase idênticas (mesma pose em tempos diferentes)
        for i in range(len(embeddings)):
            if not validos[i].exists(): continue
            for j in range(i + 1, len(embeddings)):
                if not validos[j].exists(): continue
                
                dist = float(np.dot(embeddings[i], embeddings[j]))
                if dist > 0.98: # 98% de similaridade entre capturas
                    validos[j].unlink()

if __name__ == "__main__":
    app = App()
    app.mainloop()
