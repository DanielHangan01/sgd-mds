import torch
from typing import Tuple, Optional


def _upper_tri_unravel(n: int, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    if n < 2:
        raise ValueError("n must be >= 2 to have (i, j) pairs.")
    """
    Convert flattened upper-triangle indices into (i,j) matrix coordinates.

    Given k from [0, n*(n-1)/2], this maps each k to a unique pair (i,j) with i < j,
    corresponding to a position in the upper triangle of an n*n matrix.
    Used to reconstruct pair indices when sampling without replacement from all
    possible point pairs.
    """

    device = k.device
    dtype = torch.long

    counts = torch.arange(n - 1, 0, -1, device=device, dtype=dtype)
    offsets = torch.empty(n, device=device, dtype=dtype)
    offsets[0] = 0
    offsets[1:] = torch.cumsum(counts, dim=0)

    i = torch.bucketize(k, offsets[1:], right=False)

    pos_in_row = k - offsets[i]

    j = i + 1 + pos_in_row

    return i.to(dtype), j.to(dtype)


def random_pairs(
    n: int, B: int, device: Optional[torch.device] = None, *, allow_replace: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sample random index pairs (i,j) with i < j for stochastic MDS updates.

    - If allow_replace=True: draws B independent pairs with replacement.
    - If allow_replace=False: samples B unique pairs from the upper triangle.

    Used in SGD-MDS to form mini-batches of point pairs for distance updates,
    avoiding self-pairs (i != j) and ensuring i < j for consistent indexing.
    """
    if n < 2:
        raise ValueError("n must be >= 2 to have (i, j) pairs.")
    if B < 1:
        raise ValueError("B must be >= 1.")

    if device is None:
        device = torch.device("cpu")

    if allow_replace:
        i = torch.randint(0, n, (B,), device=device)
        j = torch.randint(0, n, (B,), device=device)

        same = i == j
        while same.any():
            j[same] = torch.randint(0, n, (int(same.sum()),), device=device)
            same = i == j

        i2 = torch.minimum(i, j)
        j2 = torch.maximum(i, j)
        return i2, j2

    M = n * (n - 1) // 2
    B_eff = min(B, M)

    k = torch.randperm(M, device=device, dtype=torch.long)[:B_eff]

    i, j = _upper_tri_unravel(n, k)
    return i, j
