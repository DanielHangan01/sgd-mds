import torch


@torch.no_grad()
def sgd_step(
    X: torch.Tensor,
    i_idx: torch.Tensor,
    j_idx: torch.Tensor,
    deltas: torch.Tensor,
    weights: torch.Tensor,
    h: float,
    *,
    cap_m: bool = True,
    eps: float = 1e-8,
    exact_max_update: bool = False,
) -> torch.Tensor:
    """
    Perform one SGD-MDS step and return a scalar tensor measuring the step size.

    Returns
    -------
    torch.Tensor (0-dim)
        If exact_max_update=False: max per-pair displacement norm (upper bound).
        If exact_max_update=True: max per-point displacement norm (exact).
    """
    xi = X.index_select(0, i_idx)
    xj = X.index_select(0, j_idx)

    diff = xi - xj
    sq = (diff * diff).sum(1)
    dist = torch.sqrt(torch.clamp(sq, min=0.0))

    res = dist - deltas

    # Combine weight and learning rate, optionally cap to <= 1 for stability
    m = weights * h
    if cap_m:
        m = torch.minimum(m, torch.ones_like(m))

    denom = dist.clamp_min(eps)
    scale = m * (res / denom)

    upd = scale.unsqueeze(1) * diff
    upd_i = -upd
    upd_j = +upd

    if exact_max_update:
        # Compute the exact per-point net displacement via an accumulator buffer,
        # then apply that buffer to X. This costs an extra (n x d) temp tensor.
        delta = torch.zeros_like(X)
        delta.index_add_(0, i_idx, upd_i)
        delta.index_add_(0, j_idx, upd_j)
        max_update = delta.norm(dim=1).max()
        X.add_(delta)
        return max_update
    
    else:
        # Fast upper bound: max per-pair movement (L2 norm of any single pair update)
        # This upper bound is the true per-point movement in this step.
        max_pair_move = upd.norm(dim=1).max()
        X.index_add_(0, i_idx, upd_i)
        X.index_add_(0, j_idx, upd_j)
        return max_pair_move
