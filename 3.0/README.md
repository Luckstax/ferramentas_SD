# Video Character Extractor V3

Extrai automaticamente imagens de um personagem específico a partir de um vídeo,
usando detecção (YOLO) + similaridade visual (CLIP ViT-L/14).

## Como funciona

1. YOLO detecta pessoas/personagens no vídeo
2. Cada detecção é recortada
3. CLIP gera embeddings das imagens base e dos recortes
4. Similaridade é calculada (cosine, top-K)
5. Threshold dinâmico reduz falso positivo
6. Hash perceptual remove duplicatas
7. Resultados são exportados com metadados

## Requisitos

- Python 3.10+
- GPU recomendada (opcional, mas acelera muito)

## Instalação

```bash
python -m venv venv
venv\Scripts\activate   # Windows
pip install -r requirements.txt
