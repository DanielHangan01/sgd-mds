import torch


@torch.no_grad()
def sgd_step(X, i_idx, j_idx, deltas, weights, h, cap_m=True, eps=1e-8):

    xi = X.index_select(0, i_idx)
    xj = X.index_select(0, j_idx)

    # Differences and true distances
    diff = xi - xj
    sq = (diff * diff).sum(1)
    dist = torch.sqrt(torch.clamp(sq, min=0.0))

    # Residuals between current and target distances
    res = dist - deltas

    # Combine weight and learning rate, optionally cap to <= 1 for stability
    m = weights * h
    if cap_m:
        m = torch.minimum(m, torch.ones_like(m))

    # Gradient scale: m * (res / max(dist, eps))
    denom = dist.clamp_min(eps)
    scale = m * (res / denom)

    # Full gradient for each pair (directional update for x_i)
    grad = scale.unsqueeze(1) * diff

    X.index_add_(0, i_idx, -grad)
    X.index_add_(0, j_idx, +grad)

    # Return the maximum absolute gradient magnitude
    return grad.abs().max()
