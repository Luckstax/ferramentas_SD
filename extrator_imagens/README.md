README.md
# Extração de Personagem em Vídeo por Similaridade Visual – V2

Este projeto extrai automaticamente imagens de um vídeo sempre que uma personagem
visual­mente semelhante a um conjunto de imagens base é detectada.

A solução é totalmente automatizada, baseada em visão computacional e similaridade
vetorial, sem regras manuais (cor de cabelo, roupas etc.).

---

## Objetivo

Dado:
- Um conjunto de imagens base de uma personagem
- Um vídeo bruto

O sistema:
- Detecta personagens no vídeo
- Compara visualmente com as imagens base
- Salva apenas ocorrências relevantes
- Evita imagens duplicadas e imagens iguais às bases

Tudo isso ocorre **dentro do pipeline**, sem ferramentas externas.

---

## Pipeline técnico

1. Leitura do vídeo com OpenCV  
2. Detecção de pessoas/personagens com YOLO  
3. Recorte da região detectada  
4. Geração de embeddings visuais (CLIP ViT)  
5. Comparação com embedding médio das imagens base  
6. Validação por janela temporal de similaridade  
7. Remoção de duplicatas (base e resultados)  
8. Salvamento final das imagens  

---

## Estrutura do projeto

```text
.
├── base/
│   ├── img1.jpg
│   ├── img2.jpg
│   └── img3.jpg
├── video.mp4
├── output/
├── v2.py
├── requirements.txt
└── README.md

Requisitos

Python 3.9 ou superior

pip

(Opcional) GPU NVIDIA com CUDA para melhor desempenho

Instalação
Ambiente virtual (recomendado)
python -m venv venv
source venv/bin/activate   # Linux / Mac
venv\Scripts\activate      # Windows

Instalar dependências
pip install -r requirements.txt


Para GPU NVIDIA, instale o PyTorch compatível com CUDA antes.

Uso

Coloque as imagens da personagem na pasta base/

Coloque o vídeo bruto como video.mp4

Execute:

python v2.py


As imagens extraídas serão salvas automaticamente na pasta output/.

Configurações principais (no código)
FRAME_SKIP = 5
SIMILARITY_THRESHOLD = 0.85
ANTI_DUP_BASE_THRESHOLD = 0.95
ANTI_DUP_RESULT_THRESHOLD = 0.93
TEMPORAL_WINDOW = 5

Recomendações

Aumente TEMPORAL_WINDOW para reduzir ruído

Ajuste SIMILARITY_THRESHOLD conforme o vídeo

FRAME_SKIP maior reduz custo computacional

Tecnologias utilizadas

YOLOv8 (Ultralytics) – detecção de personagens

CLIP (ViT-B/32) – embeddings visuais

OpenCV – leitura e manipulação de vídeo

NumPy – operações matemáticas

PyTorch – inferência dos modelos

Limitações conhecidas

Não é reconhecimento facial forense

Personagens visualmente semelhantes podem gerar falso positivo

Qualidade do detector depende do modelo YOLO utilizado

Desempenho em CPU é limitado para vídeos longos

Próximas evoluções possíveis

Uso de YOLO treinado específico (anime / 3D)

Substituição do CLIP por DINOv2

Threshold adaptativo por vídeo

Tracking (DeepSORT / ByteTrack)

Agrupamento de resultados por cena

Observação final

Este projeto prioriza engenharia correta de pipeline, evitando soluções externas,
varreduras posteriores e decisões frágeis por frame isolado.

É uma base sólida para evolução, não um script demonstrativo.