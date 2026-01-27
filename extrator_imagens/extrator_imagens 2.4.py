import cv2
import os
import torch
import clip
import numpy as np
import customtkinter as ctk
from tkinter import filedialog, messagebox
from PIL import Image
from ultralytics import YOLO
from pathlib import Path
import threading
import time

# Configurações de UI
ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")

class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Extrator PRO Miraculous | YOLOv8 + CLIP + Controle de Blur")
        self.geometry("800x850")

        # Variáveis de Caminho
        self.base_path = ctk.StringVar()
        self.video_path = ctk.StringVar()
        
        # Variáveis de Controle (Solicitadas)
        self.frame_skip = ctk.IntVar(value=10)
        self.sim_threshold = ctk.DoubleVar(value=0.85)
        self.anti_dup_base = ctk.DoubleVar(value=0.96)
        self.anti_dup_result = ctk.DoubleVar(value=0.98)
        self.temporal_window = ctk.IntVar(value=10)
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.setup_ui()

    def add_slider(self, parent, label_text, var, from_, to, resolution=1):
        label = ctk.CTkLabel(parent, text=label_text)
        label.pack(pady=(10, 0))
        
        val_label = ctk.CTkLabel(parent, text=str(var.get()), font=("Arial", 10, "bold"))
        
        slider = ctk.CTkSlider(parent, from_=from_, to=to, variable=var, 
                               command=lambda v: val_label.configure(text=f"{float(v):.2f}" if resolution < 1 else f"{int(v)}"))
        slider.pack(pady=5, padx=20, fill="x")
        val_label.pack()

    def setup_ui(self):
        self.scroll_frame = ctk.CTkScrollableFrame(self)
        self.scroll_frame.pack(padx=20, pady=20, fill="both", expand=True)

        ctk.CTkLabel(self.scroll_frame, text="Configurações de Extração", font=("Arial", 18, "bold")).pack(pady=10)

        # Seletores de Arquivo
        self.add_path_selector("Pasta de Referência:", self.base_path, True)
        self.add_path_selector("Vídeo do Episódio:", self.video_path, False)

        # Sliders de Controle
        self.add_slider(self.scroll_frame, "FRAME SKIP (Pular N frames)", self.frame_skip, 1, 60)
        self.add_slider(self.scroll_frame, "SIMILARITY THRESHOLD (Confiança da Personagem)", self.sim_threshold, 0.5, 1.0, 0.01)
        self.add_slider(self.scroll_frame, "ANTI_DUP BASE (Durante captura)", self.anti_dup_base, 0.80, 0.99, 0.01)
        self.add_slider(self.scroll_frame, "ANTI_DUP RESULT (Revisão final)", self.anti_dup_result, 0.80, 0.99, 0.01)
        self.add_slider(self.scroll_frame, "TEMPORAL WINDOW (Janela de comparação de duplicatas)", self.temporal_window, 1, 50)

        # Barra de Progresso e Status
        self.progress_bar = ctk.CTkProgressBar(self)
        self.progress_bar.set(0)
        self.progress_bar.pack(pady=10, padx=20, fill="x")

        self.label_status = ctk.CTkLabel(self, text="Aguardando...")
        self.label_status.pack()

        self.btn_start = ctk.CTkButton(self, text="EXECUTAR PIPELINE", command=self.start_worker, height=50, fg_color="green")
        self.btn_start.pack(pady=20)

    def add_path_selector(self, text, var, is_dir):
        frame = ctk.CTkFrame(self.scroll_frame)
        frame.pack(fill="x", padx=10, pady=5)
        ctk.CTkLabel(frame, text=text).pack(side="left", padx=5)
        ctk.CTkEntry(frame, textvariable=var).pack(side="left", fill="x", expand=True, padx=5)
        ctk.CTkButton(frame, text="...", width=40, command=lambda: var.set(filedialog.askdirectory() if is_dir else filedialog.askopenfilename())).pack(side="right")

    def get_embedding(self, pil_img, model, preprocess):
        img = preprocess(pil_img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            emb = model.encode_image(img)
        emb /= emb.norm(dim=-1, keepdim=True)
        return emb.cpu().numpy()[0]

    def is_blurry(self, image_bgr):
        # Calcula a variância do Laplaciano (método clássico de detecção de foco)
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        score = cv2.Laplacian(gray, cv2.CV_64F).var()
        return score < 100  # Se menor que 100, a imagem é considerada "borrada"

    def start_worker(self):
        self.btn_start.configure(state="disabled")
        threading.Thread(target=self.main_process, daemon=True).start()

    def main_process(self):
        start_time = time.time()
        try:
            output_dir = "dataset_extraido"
            os.makedirs(output_dir, exist_ok=True)

            # 1. Carregamento
            yolo = YOLO("yolov8m.pt")
            clip_model, preprocess = clip.load("ViT-L/14", device=self.device)

            # 2. Referência
            ref_path = Path(self.base_path.get())
            base_embs = [self.get_embedding(Image.open(f).convert("RGB"), clip_model, preprocess) 
                         for f in ref_path.glob("*.*") if f.suffix.lower() in [".jpg", ".png"]]
            ref_embedding = np.mean(base_embs, axis=0)
            ref_embedding /= np.linalg.norm(ref_embedding)

            # 3. Vídeo
            cap = cv2.VideoCapture(self.video_path.get())
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_idx, saved_count = 0, 0
            saved_embs = []

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break

                if frame_idx % self.frame_skip.get() == 0:
                    results = yolo(frame, verbose=False)[0]
                    for box in results.boxes:
                        if int(box.cls[0]) != 0: continue
                        
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        crop = frame[y1:y2, x1:x2]
                        if crop.size == 0 or self.is_blurry(crop): continue

                        pil_crop = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                        emb = self.get_embedding(pil_crop, clip_model, preprocess)
                        sim = float(np.dot(ref_embedding, emb))

                        if sim >= self.sim_threshold.get():
                            # Anti-duplicata usando a Janela Temporal
                            is_dup = False
                            if saved_embs:
                                window = saved_embs[-self.temporal_window.get():]
                                if max(float(np.dot(emb, s)) for s in window) > self.anti_dup_base.get():
                                    is_dup = True
                            
                            if not is_dup:
                                cv2.imwrite(f"{output_dir}/f_{frame_idx}.jpg", crop)
                                saved_embs.append(emb)
                                saved_count += 1

                frame_idx += 1
                if frame_idx % 50 == 0:
                    self.progress_bar.set(frame_idx / total_frames)
                    self.label_status.configure(text=f"Processando: {frame_idx}/{total_frames}")

            cap.release()

            # 4. Revisão Final (Remoção de duplicatas globais)
            self.revisar_final(output_dir, ref_embedding, clip_model, preprocess)
            
            # Relatório Final
            duration = time.time() - start_time
            self.show_report(duration, saved_count)

        except Exception as e:
            messagebox.showerror("Erro", str(e))
        finally:
            self.btn_start.configure(state="normal")

    def revisar_final(self, pasta, ref_emb, model, preprocess):
        self.label_status.configure(text="Iniciando revisão final...")
        caminho = Path(pasta)
        arquivos = sorted(list(caminho.glob("*.jpg")))
        embeddings = []
        
        for arq in arquivos:
            emb = self.get_embedding(Image.open(arq).convert("RGB"), model, preprocess)
            embeddings.append((arq, emb))

        # Compara todos contra todos para deletar duplicatas finas
        deletados = 0
        for i in range(len(embeddings)):
            arq_i, emb_i = embeddings[i]
            if not arq_i.exists(): continue
            for j in range(i + 1, len(embeddings)):
                arq_j, emb_j = embeddings[j]
                if not arq_j.exists(): continue
                
                if float(np.dot(emb_i, emb_j)) > self.anti_dup_result.get():
                    arq_j.unlink()
                    deletados += 1

    def show_report(self, duration, total_saved):
        relatorio = (
            f"=== RELATÓRIO DE EXTRAÇÃO ===\n\n"
            f"Tempo Total: {duration:.2f} segundos\n"
            f"Imagens Salvas: {total_saved}\n\n"
            f"--- Parâmetros Utilizados ---\n"
            f"Frame Skip: {self.frame_skip.get()}\n"
            f"Similarity Threshold: {self.sim_threshold.get():.2f}\n"
            f"Anti-Dup Base: {self.anti_dup_base.get():.2f}\n"
            f"Anti-Dup Result: {self.anti_dup_result.get():.2f}\n"
            f"Temporal Window: {self.temporal_window.get()}\n"
            f"Dispositivo: {self.device}"
        )
        messagebox.showinfo("Extração Concluída", relatorio)

if __name__ == "__main__":
    app = App()
    app.mainloop()
