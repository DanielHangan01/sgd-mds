import torch


@torch.no_grad()
def pairwise_distances(
    X: torch.Tensor, i_idx: torch.Tensor, j_idx: torch.Tensor
) -> torch.Tensor:
    """
    Computes Euclidean distances for a specific list of pairs.

    This is a specialized function that avoids calculating the full distance
    matrix, making it efficient for batch-based updates or sampled stress
    calculations.

    Parameters
    ----------
    X : torch.Tensor of shape (n, d)
        The input coordinate matrix.
    i_idx : torch.Tensor of shape (B,)
        A tensor of indices for the first point in each pair.
    j_idx : torch.Tensor of shape (B,)
        A tensor of indices for the second point in each pair.

    Returns
    -------
    torch.Tensor of shape (B,)
        A tensor containing the Euclidean distance for each specified pair.
    """
    xi = X.index_select(0, i_idx)
    xj = X.index_select(0, j_idx)
    diff = xi - xj
    sq = torch.clamp((diff * diff).sum(1), min=0.0)
    return torch.sqrt(sq)


@torch.no_grad()
def full_square_distances(X: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Computes the full n x n matrix of pairwise Euclidean distances.

    This uses an optimized method based on the Gram matrix (X @ X.T) to avoid
    explicit loops, which is significantly faster for large inputs than naive
    methods.

    Parameters
    ----------
    X : torch.Tensor of shape (n, d)
        The input coordinate matrix.

    Returns
    -------
    torch.Tensor of shape (n, n)
        The full matrix of pairwise distances D, where D[i, j] is the
        distance between X[i] and X[j].
    """
    G = X @ X.t()
    sq = torch.diag(G).unsqueeze(0)
    # D^2 = diag(G) - 2G + diag(G)^T
    D2 = torch.clamp(sq.t() + sq - 2.0 * G, min=0.0)
    return torch.sqrt(D2)
