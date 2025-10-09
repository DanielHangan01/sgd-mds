import torch
from .distances import full_square_distances, pairwise_distances


@torch.no_grad()
def kruskal_stress_full(X, D_full, weights=None):
    return torch.zeros


@torch.no_grad()
def kruskal_stress_pairs(X, D_full, i_idx, j_idx, weights=None):
    return torch.zeros
