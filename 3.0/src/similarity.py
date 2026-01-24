import torch


def is_similar(embedding, base_embeddings, threshold=0.85):
    if not base_embeddings:
        return False

    sims = [
        torch.cosine_similarity(embedding, b, dim=-1).item()
        for b in base_embeddings
    ]

    return max(sims) >= threshold
