Extração de Personagem em Vídeo com IA (YOLO + CLIP)

Este projeto extrai automaticamente imagens de um vídeo sempre que uma personagem visual­mente semelhante a uma imagem de referência aparece.

A detecção é feita por visão computacional, sem regras manuais, usando embeddings e similaridade vetorial.

Visão geral da solução

Pipeline aplicado:

Extração de frames do vídeo

Detecção de pessoas com YOLO

Recorte das pessoas detectadas

Geração de embeddings visuais (CLIP)

Comparação com a imagem base (cosine similarity)

Salvamento das ocorrências acima do threshold definido

Estrutura do projeto
.
├── base.jpg            # imagem da personagem de referência
├── video.mp4           # vídeo bruto
├── script.py           # script principal
├── requirements.txt    # dependências
└── output/             # imagens extraídas (gerado automaticamente)

Requisitos

Python 3.9+

pip

(Opcional) GPU NVIDIA com CUDA para melhor desempenho

Instalação
1. Criar ambiente virtual (recomendado)
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

2. Instalar dependências
pip install -r requirements.txt


⚠️ GPU (opcional)
Para usar CUDA, instale o PyTorch correto antes:

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

Uso

Coloque a imagem da personagem como base.jpg

Coloque o vídeo bruto como video.mp4

Execute:

python script.py


As imagens detectadas serão salvas automaticamente na pasta output/.

Configurações principais (no código)
SIMILARITY_THRESHOLD = 0.85  # nível mínimo de similaridade
FRAME_SKIP = 5               # processa 1 frame a cada N

Ajustes recomendados

Threshold

0.90 → muito rigoroso

0.85 → equilíbrio

0.80 → mais falsos positivos

FRAME_SKIP

1 → máxima precisão (pesado)

5–10 → vídeos longos

Limitações conhecidas

Não é reconhecimento facial forense

Personagens visualmente parecidas podem gerar falso positivo

Blur, ângulos extremos ou baixa resolução afetam o resultado

Performance em CPU é limitada para vídeos longos

Tecnologias utilizadas

YOLOv8 (Ultralytics) – detecção de pessoas

CLIP – embeddings visuais

OpenCV – leitura e recorte de frames

NumPy – operações matemáticas

PyTorch – backend de inferência

Possíveis evoluções

Trocar CLIP por DINOv2

Uso de FAISS para acelerar comparações

Agrupamento por cena

Salvamento do frame completo ao invés do recorte

Interface CLI ou GUI

Observação final

Este projeto não usa regras manuais (cor de cabelo, roupas etc.).
A decisão é feita exclusivamente por similaridade visual, o que torna o sistema mais robusto e escalável.