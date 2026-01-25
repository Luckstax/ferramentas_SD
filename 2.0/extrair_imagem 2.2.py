<<<<<<< HEAD
<<<<<<< HEAD
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

# Configurações de tema da interface
ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")

class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Extrator de Imagens Semelhantes (YOLOv8 + CLIP)")
        self.geometry("600x450")

        # Variáveis de controle
        self.base_path = ctk.StringVar()
        self.video_path = ctk.StringVar()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.setup_ui()

    def setup_ui(self):
        self.grid_columnconfigure(0, weight=1)

        # Título
        self.label_title = ctk.CTkLabel(self, text="Processamento de Vídeo Inteligente", font=ctk.CTkFont(size=20, weight="bold"))
        self.label_title.pack(pady=20)

        # Frame de Seleção
        self.frame_files = ctk.CTkFrame(self)
        self.frame_files.pack(padx=20, pady=10, fill="x")

        # Pasta Base
        ctk.CTkLabel(self.frame_files, text="Pasta Base (Imagens de Referência):").grid(row=0, column=0, sticky="w", padx=10, pady=5)
        ctk.CTkEntry(self.frame_files, textvariable=self.base_path, width=350).grid(row=1, column=0, padx=10, pady=5)
        ctk.CTkButton(self.frame_files, text="Selecionar", command=self.browse_base, width=80).grid(row=1, column=1, padx=5)

        # Vídeo
        ctk.CTkLabel(self.frame_files, text="Vídeo para Análise:").grid(row=2, column=0, sticky="w", padx=10, pady=5)
        ctk.CTkEntry(self.frame_files, textvariable=self.video_path, width=350).grid(row=3, column=0, padx=10, pady=5)
        ctk.CTkButton(self.frame_files, text="Selecionar", command=self.browse_video, width=80).grid(row=3, column=1, padx=5)

        # Barra de Progresso
        self.progress_bar = ctk.CTkProgressBar(self, width=500)
        self.progress_bar.set(0)
        self.progress_bar.pack(pady=30)

        # Botão Iniciar
        self.btn_start = ctk.CTkButton(self, text="Iniciar Processamento", command=self.start_thread, fg_color="green", hover_color="darkgreen")
        self.btn_start.pack(pady=10)

        self.label_status = ctk.CTkLabel(self, text="Aguardando início...")
        self.label_status.pack()

    def browse_base(self):
        path = filedialog.askdirectory()
        if path: self.base_path.set(path)

    def browse_video(self):
        path = filedialog.askopenfilename(filetypes=[("Vídeos", "*.mp4 *.avi *.mkv")])
        if path: self.video_path.set(path)

    def get_embedding(self, pil_img, model, preprocess):
        img = preprocess(pil_img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            emb = model.encode_image(img)
        emb /= emb.norm(dim=-1, keepdim=True)
        return emb.cpu().numpy()[0]

    def cosine_sim(self, a, b):
        return float(np.dot(a, b))

    def start_thread(self):
        if not self.base_path.get() or not self.video_path.get():
            messagebox.showwarning("Erro", "Selecione a pasta base e o vídeo!")
            return
        
        self.btn_start.configure(state="disabled")
        threading.Thread(target=self.process_video, daemon=True).start()

    def process_video(self):
        try:
            # Configurações Internas
            FRAME_SKIP = 5
            SIMILARITY_THRESHOLD = 0.85
            ANTI_DUP_BASE_THRESHOLD = 0.95
            ANTI_DUP_RESULT_THRESHOLD = 0.93
            TEMPORAL_WINDOW = 5
            OUTPUT_DIR = "output_gui"
            os.makedirs(OUTPUT_DIR, exist_ok=True)

            # Carregar Modelos
            self.label_status.configure(text="Carregando modelos (IA)...")
            yolo = YOLO("yolov8l.pt")
            clip_model, preprocess = clip.load("ViT-L/14", device=self.device)

            # Gerar Embeddings da Base
            self.label_status.configure(text="Analisando imagens de base...")
            base_embeddings = []
            files = [f for f in os.listdir(self.base_path.get()) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            for file in files:
                img = Image.open(os.path.join(self.base_path.get(), file)).convert("RGB")
                base_embeddings.append(self.get_embedding(img, clip_model, preprocess))

            base_embeddings = np.array(base_embeddings)
            base_mean_embedding = base_embeddings.mean(axis=0)
            base_mean_embedding /= np.linalg.norm(base_mean_embedding)

            # Processar Vídeo
            cap = cv2.VideoCapture(self.video_path.get())
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            frame_id = 0
            saved_count = 0
            saved_embeddings = []
            similarity_window = deque(maxlen=TEMPORAL_WINDOW)

            self.label_status.configure(text="Processando frames...")
            
            # TQDM no console e progresso na GUI
            with tqdm(total=total_frames, desc="Analisando Vídeo") as pbar:
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret: break

                    frame_id += 1
                    pbar.update(1)
                    
                    # Atualiza barra da interface a cada 10 frames para performance
                    if frame_id % 10 == 0:
                        self.progress_bar.set(frame_id / total_frames)

                    if frame_id % FRAME_SKIP != 0:
                        continue

                    detections = yolo(frame, verbose=False)[0]

                    for box in detections.boxes:
                        if int(box.cls[0]) != 0: continue # Apenas pessoas (classe 0)

                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        crop = frame[y1:y2, x1:x2]
                        if crop.size == 0: continue

                        pil_crop = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                        emb = self.get_embedding(pil_crop, clip_model, preprocess)

                        sim = self.cosine_sim(base_mean_embedding, emb)
                        similarity_window.append(sim)

                        if len(similarity_window) < TEMPORAL_WINDOW: continue

                        avg_sim = sum(similarity_window) / TEMPORAL_WINDOW
                        if avg_sim < SIMILARITY_THRESHOLD: continue

                        if max(self.cosine_sim(emb, b) for b in base_embeddings) > ANTI_DUP_BASE_THRESHOLD:
                            continue

                        if saved_embeddings:
                            if max(self.cosine_sim(emb, s) for s in saved_embeddings[-5:]) > ANTI_DUP_RESULT_THRESHOLD:
                                continue

                        # Salvar
                        filename = f"{OUTPUT_DIR}/match_{frame_id}_{saved_count}.jpg"
                        cv2.imwrite(filename, crop)
                        saved_embeddings.append(emb)
                        saved_count += 1

            cap.release()
            self.progress_bar.set(1)
            self.label_status.configure(text=f"Concluído! {saved_count} imagens salvas.")
            messagebox.showinfo("Sucesso", f"Processamento finalizado!\n{saved_count} imagens salvas em: {OUTPUT_DIR}")

        except Exception as e:
            messagebox.showerror("Erro", f"Ocorreu um erro: {str(e)}")
        finally:
            self.btn_start.configure(state="normal")

if __name__ == "__main__":
    app = App()
=======
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

# Configurações de tema da interface
ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")

class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Extrator de Imagens Semelhantes (YOLOv8 + CLIP)")
        self.geometry("600x450")

        # Variáveis de controle
        self.base_path = ctk.StringVar()
        self.video_path = ctk.StringVar()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.setup_ui()

    def setup_ui(self):
        self.grid_columnconfigure(0, weight=1)

        # Título
        self.label_title = ctk.CTkLabel(self, text="Processamento de Vídeo Inteligente", font=ctk.CTkFont(size=20, weight="bold"))
        self.label_title.pack(pady=20)

        # Frame de Seleção
        self.frame_files = ctk.CTkFrame(self)
        self.frame_files.pack(padx=20, pady=10, fill="x")

        # Pasta Base
        ctk.CTkLabel(self.frame_files, text="Pasta Base (Imagens de Referência):").grid(row=0, column=0, sticky="w", padx=10, pady=5)
        ctk.CTkEntry(self.frame_files, textvariable=self.base_path, width=350).grid(row=1, column=0, padx=10, pady=5)
        ctk.CTkButton(self.frame_files, text="Selecionar", command=self.browse_base, width=80).grid(row=1, column=1, padx=5)

        # Vídeo
        ctk.CTkLabel(self.frame_files, text="Vídeo para Análise:").grid(row=2, column=0, sticky="w", padx=10, pady=5)
        ctk.CTkEntry(self.frame_files, textvariable=self.video_path, width=350).grid(row=3, column=0, padx=10, pady=5)
        ctk.CTkButton(self.frame_files, text="Selecionar", command=self.browse_video, width=80).grid(row=3, column=1, padx=5)

        # Barra de Progresso
        self.progress_bar = ctk.CTkProgressBar(self, width=500)
        self.progress_bar.set(0)
        self.progress_bar.pack(pady=30)

        # Botão Iniciar
        self.btn_start = ctk.CTkButton(self, text="Iniciar Processamento", command=self.start_thread, fg_color="green", hover_color="darkgreen")
        self.btn_start.pack(pady=10)

        self.label_status = ctk.CTkLabel(self, text="Aguardando início...")
        self.label_status.pack()

    def browse_base(self):
        path = filedialog.askdirectory()
        if path: self.base_path.set(path)

    def browse_video(self):
        path = filedialog.askopenfilename(filetypes=[("Vídeos", "*.mp4 *.avi *.mkv")])
        if path: self.video_path.set(path)

    def get_embedding(self, pil_img, model, preprocess):
        img = preprocess(pil_img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            emb = model.encode_image(img)
        emb /= emb.norm(dim=-1, keepdim=True)
        return emb.cpu().numpy()[0]

    def cosine_sim(self, a, b):
        return float(np.dot(a, b))

    def start_thread(self):
        if not self.base_path.get() or not self.video_path.get():
            messagebox.showwarning("Erro", "Selecione a pasta base e o vídeo!")
            return
        
        self.btn_start.configure(state="disabled")
        threading.Thread(target=self.process_video, daemon=True).start()

    def process_video(self):
        try:
            # Configurações Internas
            FRAME_SKIP = 5
            SIMILARITY_THRESHOLD = 0.85
            ANTI_DUP_BASE_THRESHOLD = 0.95
            ANTI_DUP_RESULT_THRESHOLD = 0.93
            TEMPORAL_WINDOW = 5
            OUTPUT_DIR = "output_gui"
            os.makedirs(OUTPUT_DIR, exist_ok=True)

            # Carregar Modelos
            self.label_status.configure(text="Carregando modelos (IA)...")
            yolo = YOLO("yolov8l.pt")
            clip_model, preprocess = clip.load("ViT-L/14", device=self.device)

            # Gerar Embeddings da Base
            self.label_status.configure(text="Analisando imagens de base...")
            base_embeddings = []
            files = [f for f in os.listdir(self.base_path.get()) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            for file in files:
                img = Image.open(os.path.join(self.base_path.get(), file)).convert("RGB")
                base_embeddings.append(self.get_embedding(img, clip_model, preprocess))

            base_embeddings = np.array(base_embeddings)
            base_mean_embedding = base_embeddings.mean(axis=0)
            base_mean_embedding /= np.linalg.norm(base_mean_embedding)

            # Processar Vídeo
            cap = cv2.VideoCapture(self.video_path.get())
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            frame_id = 0
            saved_count = 0
            saved_embeddings = []
            similarity_window = deque(maxlen=TEMPORAL_WINDOW)

            self.label_status.configure(text="Processando frames...")
            
            # TQDM no console e progresso na GUI
            with tqdm(total=total_frames, desc="Analisando Vídeo") as pbar:
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret: break

                    frame_id += 1
                    pbar.update(1)
                    
                    # Atualiza barra da interface a cada 10 frames para performance
                    if frame_id % 10 == 0:
                        self.progress_bar.set(frame_id / total_frames)

                    if frame_id % FRAME_SKIP != 0:
                        continue

                    detections = yolo(frame, verbose=False)[0]

                    for box in detections.boxes:
                        if int(box.cls[0]) != 0: continue # Apenas pessoas (classe 0)

                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        crop = frame[y1:y2, x1:x2]
                        if crop.size == 0: continue

                        pil_crop = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                        emb = self.get_embedding(pil_crop, clip_model, preprocess)

                        sim = self.cosine_sim(base_mean_embedding, emb)
                        similarity_window.append(sim)

                        if len(similarity_window) < TEMPORAL_WINDOW: continue

                        avg_sim = sum(similarity_window) / TEMPORAL_WINDOW
                        if avg_sim < SIMILARITY_THRESHOLD: continue

                        if max(self.cosine_sim(emb, b) for b in base_embeddings) > ANTI_DUP_BASE_THRESHOLD:
                            continue

                        if saved_embeddings:
                            if max(self.cosine_sim(emb, s) for s in saved_embeddings[-5:]) > ANTI_DUP_RESULT_THRESHOLD:
                                continue

                        # Salvar
                        filename = f"{OUTPUT_DIR}/match_{frame_id}_{saved_count}.jpg"
                        cv2.imwrite(filename, crop)
                        saved_embeddings.append(emb)
                        saved_count += 1

            cap.release()
            self.progress_bar.set(1)
            self.label_status.configure(text=f"Concluído! {saved_count} imagens salvas.")
            messagebox.showinfo("Sucesso", f"Processamento finalizado!\n{saved_count} imagens salvas em: {OUTPUT_DIR}")

        except Exception as e:
            messagebox.showerror("Erro", f"Ocorreu um erro: {str(e)}")
        finally:
            self.btn_start.configure(state="normal")

if __name__ == "__main__":
    app = App()
>>>>>>> bb35b62fd2af9ee484728e334798558a4dcc82cb
=======
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

# Configurações de tema da interface
ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")

class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Extrator de Imagens Semelhantes (YOLOv8 + CLIP)")
        self.geometry("600x450")

        # Variáveis de controle
        self.base_path = ctk.StringVar()
        self.video_path = ctk.StringVar()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.setup_ui()

    def setup_ui(self):
        self.grid_columnconfigure(0, weight=1)

        # Título
        self.label_title = ctk.CTkLabel(self, text="Processamento de Vídeo Inteligente", font=ctk.CTkFont(size=20, weight="bold"))
        self.label_title.pack(pady=20)

        # Frame de Seleção
        self.frame_files = ctk.CTkFrame(self)
        self.frame_files.pack(padx=20, pady=10, fill="x")

        # Pasta Base
        ctk.CTkLabel(self.frame_files, text="Pasta Base (Imagens de Referência):").grid(row=0, column=0, sticky="w", padx=10, pady=5)
        ctk.CTkEntry(self.frame_files, textvariable=self.base_path, width=350).grid(row=1, column=0, padx=10, pady=5)
        ctk.CTkButton(self.frame_files, text="Selecionar", command=self.browse_base, width=80).grid(row=1, column=1, padx=5)

        # Vídeo
        ctk.CTkLabel(self.frame_files, text="Vídeo para Análise:").grid(row=2, column=0, sticky="w", padx=10, pady=5)
        ctk.CTkEntry(self.frame_files, textvariable=self.video_path, width=350).grid(row=3, column=0, padx=10, pady=5)
        ctk.CTkButton(self.frame_files, text="Selecionar", command=self.browse_video, width=80).grid(row=3, column=1, padx=5)

        # Barra de Progresso
        self.progress_bar = ctk.CTkProgressBar(self, width=500)
        self.progress_bar.set(0)
        self.progress_bar.pack(pady=30)

        # Botão Iniciar
        self.btn_start = ctk.CTkButton(self, text="Iniciar Processamento", command=self.start_thread, fg_color="green", hover_color="darkgreen")
        self.btn_start.pack(pady=10)

        self.label_status = ctk.CTkLabel(self, text="Aguardando início...")
        self.label_status.pack()

    def browse_base(self):
        path = filedialog.askdirectory()
        if path: self.base_path.set(path)

    def browse_video(self):
        path = filedialog.askopenfilename(filetypes=[("Vídeos", "*.mp4 *.avi *.mkv")])
        if path: self.video_path.set(path)

    def get_embedding(self, pil_img, model, preprocess):
        img = preprocess(pil_img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            emb = model.encode_image(img)
        emb /= emb.norm(dim=-1, keepdim=True)
        return emb.cpu().numpy()[0]

    def cosine_sim(self, a, b):
        return float(np.dot(a, b))

    def start_thread(self):
        if not self.base_path.get() or not self.video_path.get():
            messagebox.showwarning("Erro", "Selecione a pasta base e o vídeo!")
            return
        
        self.btn_start.configure(state="disabled")
        threading.Thread(target=self.process_video, daemon=True).start()

    def process_video(self):
        try:
            # Configurações Internas
            FRAME_SKIP = 5
            SIMILARITY_THRESHOLD = 0.85
            ANTI_DUP_BASE_THRESHOLD = 0.95
            ANTI_DUP_RESULT_THRESHOLD = 0.93
            TEMPORAL_WINDOW = 5
            OUTPUT_DIR = "output_gui"
            os.makedirs(OUTPUT_DIR, exist_ok=True)

            # Carregar Modelos
            self.label_status.configure(text="Carregando modelos (IA)...")
            yolo = YOLO("yolov8l.pt")
            clip_model, preprocess = clip.load("ViT-L/14", device=self.device)

            # Gerar Embeddings da Base
            self.label_status.configure(text="Analisando imagens de base...")
            base_embeddings = []
            files = [f for f in os.listdir(self.base_path.get()) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            for file in files:
                img = Image.open(os.path.join(self.base_path.get(), file)).convert("RGB")
                base_embeddings.append(self.get_embedding(img, clip_model, preprocess))

            base_embeddings = np.array(base_embeddings)
            base_mean_embedding = base_embeddings.mean(axis=0)
            base_mean_embedding /= np.linalg.norm(base_mean_embedding)

            # Processar Vídeo
            cap = cv2.VideoCapture(self.video_path.get())
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            frame_id = 0
            saved_count = 0
            saved_embeddings = []
            similarity_window = deque(maxlen=TEMPORAL_WINDOW)

            self.label_status.configure(text="Processando frames...")
            
            # TQDM no console e progresso na GUI
            with tqdm(total=total_frames, desc="Analisando Vídeo") as pbar:
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret: break

                    frame_id += 1
                    pbar.update(1)
                    
                    # Atualiza barra da interface a cada 10 frames para performance
                    if frame_id % 10 == 0:
                        self.progress_bar.set(frame_id / total_frames)

                    if frame_id % FRAME_SKIP != 0:
                        continue

                    detections = yolo(frame, verbose=False)[0]

                    for box in detections.boxes:
                        if int(box.cls[0]) != 0: continue # Apenas pessoas (classe 0)

                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        crop = frame[y1:y2, x1:x2]
                        if crop.size == 0: continue

                        pil_crop = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                        emb = self.get_embedding(pil_crop, clip_model, preprocess)

                        sim = self.cosine_sim(base_mean_embedding, emb)
                        similarity_window.append(sim)

                        if len(similarity_window) < TEMPORAL_WINDOW: continue

                        avg_sim = sum(similarity_window) / TEMPORAL_WINDOW
                        if avg_sim < SIMILARITY_THRESHOLD: continue

                        if max(self.cosine_sim(emb, b) for b in base_embeddings) > ANTI_DUP_BASE_THRESHOLD:
                            continue

                        if saved_embeddings:
                            if max(self.cosine_sim(emb, s) for s in saved_embeddings[-5:]) > ANTI_DUP_RESULT_THRESHOLD:
                                continue

                        # Salvar
                        filename = f"{OUTPUT_DIR}/match_{frame_id}_{saved_count}.jpg"
                        cv2.imwrite(filename, crop)
                        saved_embeddings.append(emb)
                        saved_count += 1

            cap.release()
            self.progress_bar.set(1)
            self.label_status.configure(text=f"Concluído! {saved_count} imagens salvas.")
            messagebox.showinfo("Sucesso", f"Processamento finalizado!\n{saved_count} imagens salvas em: {OUTPUT_DIR}")

        except Exception as e:
            messagebox.showerror("Erro", f"Ocorreu um erro: {str(e)}")
        finally:
            self.btn_start.configure(state="normal")

if __name__ == "__main__":
    app = App()
>>>>>>> bb35b62fd2af9ee484728e334798558a4dcc82cb
    app.mainloop()