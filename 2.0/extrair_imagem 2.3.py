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
from tqdm import tqdm
import threading

# Configurações de tema
ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")

class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Extrator PRO | YOLOv8 + CLIP")
        self.geometry("600x580")

        # Variáveis de controle
        self.base_path = ctk.StringVar()
        self.video_path = ctk.StringVar()
        self.sim_threshold = ctk.DoubleVar(value=0.85)
        [cite_start]self.device = "cuda" if torch.cuda.is_available() else "cpu" [cite: 1]
        
        self.setup_ui()

    def setup_ui(self):
        self.grid_columnconfigure(0, weight=1)

        self.label_title = ctk.CTkLabel(self, text="Análise de Vídeo Adaptativa", font=ctk.CTkFont(size=22, weight="bold"))
        self.label_title.pack(pady=20)

        # Seleção de Arquivos
        self.frame_files = ctk.CTkFrame(self)
        self.frame_files.pack(padx=20, pady=10, fill="x")

        # Base
        ctk.CTkLabel(self.frame_files, text="Pasta Base (Imagens de Referência):").grid(row=0, column=0, sticky="w", padx=10, pady=5)
        ctk.CTkEntry(self.frame_files, textvariable=self.base_path, width=350).grid(row=1, column=0, padx=10, pady=5)
        ctk.CTkButton(self.frame_files, text="Abrir", command=lambda: self.base_path.set(filedialog.askdirectory()), width=80).grid(row=1, column=1, padx=5)

        # Vídeo
        ctk.CTkLabel(self.frame_files, text="Vídeo para Analisar:").grid(row=2, column=0, sticky="w", padx=10, pady=5)
        ctk.CTkEntry(self.frame_files, textvariable=self.video_path, width=350).grid(row=3, column=0, padx=10, pady=5)
        ctk.CTkButton(self.frame_files, text="Abrir", command=lambda: self.video_path.set(filedialog.askopenfilename()), width=80).grid(row=3, column=1, padx=5)

        # Configurações de Sensibilidade
        ctk.CTkLabel(self.frame_files, text="Ajuste de Sensibilidade (Threshold):").grid(row=4, column=0, sticky="w", padx=10, pady=(15,0))
        self.slider = ctk.CTkSlider(self.frame_files, from_=0.5, to=1.0, variable=self.sim_threshold)
        self.slider.grid(row=5, column=0, padx=10, pady=5, sticky="ew")
        self.label_val = ctk.CTkLabel(self.frame_files, text="0.85")
        self.label_val.grid(row=5, column=1)
        self.sim_threshold.trace_add("write", lambda *args: self.label_val.configure(text=f"{self.sim_threshold.get():.2f}"))

        # Progresso
        self.progress_bar = ctk.CTkProgressBar(self, width=500)
        self.progress_bar.set(0)
        self.progress_bar.pack(pady=30)

        # Botão Ação
        self.btn_start = ctk.CTkButton(self, text="INICIAR PROCESSAMENTO", command=self.start_thread, fg_color="#1f538d", height=45, font=ctk.CTkFont(weight="bold"))
        self.btn_start.pack(pady=10)

        self.label_status = ctk.CTkLabel(self, text="Pronto para começar")
        self.label_status.pack()

    def get_embedding(self, pil_img, model, preprocess):
        [cite_start]img = preprocess(pil_img).unsqueeze(0).to(self.device) [cite: 1]
        with torch.no_grad():
            [cite_start]emb = model.encode_image(img) [cite: 1]
        [cite_start]emb /= emb.norm(dim=-1, keepdim=True) [cite: 1]
        [cite_start]return emb.cpu().numpy()[0] [cite: 1]

    def start_thread(self):
        if not self.base_path.get() or not self.video_path.get():
            messagebox.showwarning("Erro", "Por favor, selecione os caminhos primeiro!")
            return
        self.btn_start.configure(state="disabled")
        threading.Thread(target=self.process_video, daemon=True).start()

    def process_video(self):
        try:
            # 1. Setup Inicial
            OUTPUT_DIR = "output_analise"
            [cite_start]os.makedirs(OUTPUT_DIR, exist_ok=True) [cite: 1]
            
            self.label_status.configure(text="Carregando modelos de IA...")
            [cite_start]yolo = YOLO("yolov8l.pt") [cite: 1]
            [cite_start]clip_model, preprocess = clip.load("ViT-L/14", device=self.device) [cite: 1]

            # 2. Cálculo de Tempo Dinâmico
            [cite_start]cap = cv2.VideoCapture(self.video_path.get()) [cite: 1]
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps < 1: fps = 30
            
            FRAME_SKIP = max(1, int(fps * 0.2))  # Analisa a cada 0.2s
            TEMPORAL_WINDOW = max(1, int(0.5 / 0.2)) # Confirmação de 0.5s
            
            # 3. Processar Base
            self.label_status.configure(text="Processando imagens de referência...")
            base_embeddings = []
            files = [f for f in os.listdir(self.base_path.get()) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            for file in files:
                [cite_start]img = Image.open(os.path.join(self.base_path.get(), file)).convert("RGB") [cite: 1]
                base_embeddings.append(self.get_embedding(img, clip_model, preprocess))

            [cite_start]base_embeddings = np.array(base_embeddings) [cite: 1]
            [cite_start]base_mean_embedding = base_embeddings.mean(axis=0) [cite: 1]
            [cite_start]base_mean_embedding /= np.linalg.norm(base_mean_embedding) [cite: 1]

            # 4. Loop de Vídeo
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_id, saved_count = 0, 0
            saved_embeddings = []
            [cite_start]similarity_window = deque(maxlen=TEMPORAL_WINDOW) [cite: 1]

            self.label_status.configure(text="Analisando vídeo...")
            
            with tqdm(total=total_frames, desc="Processando") as pbar:
                while cap.isOpened():
                    [cite_start]ret, frame = cap.read() [cite: 1]
                    if not ret: break

                    frame_id += 1
                    pbar.update(1)
                    
                    if frame_id % 10 == 0:
                        self.progress_bar.set(frame_id / total_frames)

                    [cite_start]if frame_id % FRAME_SKIP != 0: continue [cite: 1]

                    [cite_start]results = yolo(frame, verbose=False)[0] [cite: 1]
                    for box in results.boxes:
                        [cite_start]if int(box.cls[0]) != 0: continue # Pessoas [cite: 1]

                        [cite_start]x1, y1, x2, y2 = map(int, box.xyxy[0]) [cite: 1]
                        [cite_start]crop = frame[y1:y2, x1:x2] [cite: 1]
                        if crop.size == 0: continue

                        [cite_start]pil_crop = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)) [cite: 1]
                        emb = self.get_embedding(pil_crop, clip_model, preprocess)

                        [cite_start]sim = float(np.dot(base_mean_embedding, emb)) [cite: 1]
                        similarity_window.append(sim)

                        [cite_start]if len(similarity_window) < TEMPORAL_WINDOW: continue [cite: 1]

                        if (sum(similarity_window)/TEMPORAL_WINDOW) >= self.sim_threshold.get():
                            # Anti-duplicata (Resultados recentes)
                            is_dup = False
                            if saved_embeddings:
                                [cite_start]if max(float(np.dot(emb, s)) for s in saved_embeddings[-5:]) > 0.93: [cite: 1]
                                    is_dup = True
                            
                            if not is_dup:
                                [cite_start]cv2.imwrite(f"{OUTPUT_DIR}/frame_{frame_id}.jpg", crop) [cite: 1]
                                saved_embeddings.append(emb)
                                saved_count += 1

            [cite_start]cap.release() [cite: 1]
            self.progress_bar.set(1)
            self.label_status.configure(text=f"Finalizado! {saved_count} capturas.")
            messagebox.showinfo("Sucesso", f"Processamento concluído.\nForam salvas {saved_count} imagens.")

        except Exception as e:
            messagebox.showerror("Erro Crítico", str(e))
        finally:
            self.btn_start.configure(state="normal")

if __name__ == "__main__":
    app = App()
    app.mainloop()
