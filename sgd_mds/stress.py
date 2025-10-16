import torch
from .distances import full_square_distances, pairwise_distances


@torch.no_grad()
def kruskal_stress_full(
    X: torch.Tensor,
    D_full: torch.Tensor,
    weights: torch.Tensor | None = None,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Calculate the exact Kruskal Stress value over all pairs.
    Stress(X) = sqrt( Σ₍i<j₎ w_ij (‖x_i - x_j‖ - δ_ij)²  /  Σ₍i<j₎ w_ij δ_ij² )

    - X: current embedding coordinates in R^(n * d)
    - D_full: target distance matrix δ_ij
    - weights: optional pair weights w_ij (default = 1)
    - eps: denominator clamp to avoid dividing by zero

    Used after optimization to evaluate the embedding.
    """
    D_hat = full_square_distances(X)

    tri = torch.triu(torch.ones_like(D_full, dtype=torch.bool), diagonal=1)

    r = (D_hat - D_full)[tri]
    d = D_full[tri]

    if weights is None:
        num = (r * r).sum()
        den = (d * d).sum().clamp_min(eps)
    else:
        if weights.dim() == 2:
            w = weights[tri]
        else:
            w = weights
            if w.numel() != int(tri.sum()):
                raise ValueError(
                    "weights length must match number of upper-tri pairs (i<j)."
                )
        num = (w * (r * r)).sum()
        den = (w * (d * d)).sum().clamp_min(eps)

    return torch.sqrt(num / den)


@torch.no_grad()
def kruskal_stress_pairs(
    X: torch.Tensor,
    D_full: torch.Tensor,
    i_idx: torch.Tensor,
    j_idx: torch.Tensor,
    weights: torch.Tensor | None = None,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Calculate a sampled approximation of the Kruskal Stress using selected pairs.
    Stress(X) = sqrt( Σ_k w_k (‖x_iₖ - x_jₖ‖ - δ_iₖjₖ)²  /  Σ_k w_k δ_iₖjₖ² )

    - X: current embedding coordinates in R^(n*d)
    - D_full: full target distance matrix δ_ij
    - i_idx, j_idx: sampled index tensors which define the pairs (i,j) to be evaluated
    - weights: optional pair weights (default = 1)
    - eps: denominator clamp to avoid dividing by zero

    Used on large datasets to estimate global stress efficiently
    """
    d_hat = pairwise_distances(X, i_idx, j_idx)
    deltas = D_full[i_idx, j_idx]

    if weights is None:
        w = torch.ones_like(deltas)
    else:
        w = weights
        if w.shape != deltas.shape:
            raise ValueError(
                "weights must have the same shape as the sampled pairs (B,)."
            )

    r = d_hat - deltas
    num = (w * (r * r)).sum()
    den = (w * (deltas * deltas)).sum().clamp_min(eps)

    return torch.sqrt(num / den)
