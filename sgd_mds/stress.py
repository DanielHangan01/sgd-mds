import torch
from .distances import full_square_distances, pairwise_distances

@torch.no_grad()
def _calculate_stress_from_components(
    residuals: torch.Tensor,
    deltas: torch.Tensor,
    weights: torch.Tensor | None,
    eps: float,
) -> torch.Tensor:
    """
    Core helper to compute Kruskal Stress from pre-calculated components.
    Stress = sqrt( Σ w*(res)² / Σ w*(delta)² )
    """
    if weights is None:
        numerator = (residuals * residuals).sum()
        denominator = (deltas * deltas).sum()
    else:
        numerator = (weights * (residuals * residuals)).sum()
        denominator = (weights * (deltas * deltas)).sum()

    return torch.sqrt(numerator / denominator.clamp_min(eps))


@torch.no_grad()
def kruskal_stress_full(
    X: torch.Tensor,
    D_full: torch.Tensor,
    weights: torch.Tensor | None = None,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Calculate the exact Kruskal Stress value over all pairs.
    Stress(X) = sqrt( Σ w_ij (‖x_i-x_j‖-δ_ij)² / Σ w_ij δ_ij² ) for i < j

    Parameters
    ----------
    X : current embedding coordinates in R^(n * d)
    D_full : target distance matrix δ_ij
    weights : optional pair weights w_ij (default = 1)
    eps : denominator clamp to avoid dividing by zero
    """
    D_hat = full_square_distances(X)

    triu_mask = torch.triu(torch.ones_like(D_full, dtype=torch.bool), diagonal=1)

    residuals = (D_hat - D_full)[triu_mask]
    deltas = D_full[triu_mask]
    
    w: torch.Tensor | None = None
    if weights is not None:
        if weights.dim() == 2:
            w = weights[triu_mask]
        else:
            if weights.numel() != deltas.numel():
                raise ValueError(
                    "1D weights length must match number of upper-tri pairs."
                )
            w = weights

    return _calculate_stress_from_components(residuals, deltas, w, eps)


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
    Stress(X) = sqrt( Σ w_k (‖x_ik-x_jk‖-δ_ikjk)² / Σ w_k δ_ikjk² )

    Parameters
    ----------
    X : current embedding coordinates in R^(n*d)
    D_full : full target distance matrix δ_ij
    i_idx, j_idx : sampled index tensors which define the pairs to be evaluated
    weights : optional pair weights (default = 1)
    eps : denominator clamp to avoid dividing by zero
    """
    d_hat = pairwise_distances(X, i_idx, j_idx)
    deltas = D_full[i_idx, j_idx]
    residuals = d_hat - deltas

    if weights is not None and weights.shape != deltas.shape:
        raise ValueError("Weights must have the same shape as the sampled pairs.")

    return _calculate_stress_from_components(residuals, deltas, weights, eps)
