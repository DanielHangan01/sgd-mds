import torch


@torch.no_grad()
def pairwise_distances(
    X: torch.Tensor, i_idx: torch.Tensor, j_idx: torch.Tensor
) -> torch.Tensor:
    xi = X.index_select(0, i_idx)
    xj = X.index_select(0, j_idx)
    diff = xi - xj
    sq = torch.clamp((diff * diff).sum(1), min=0.0)
    return torch.sqrt(sq)


@torch.no_grad()
def full_square_distances(X: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    G = X @ X.t()
    sq = torch.diag(G).unsqueeze(0)
    D2 = torch.clamp(sq.t() + sq - 2.0 * G, min=0.0)
    return torch.sqrt(D2)
