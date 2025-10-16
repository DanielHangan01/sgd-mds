import math
import pytest
import torch

from sgd_mds.stress import (
    kruskal_stress_full,
    kruskal_stress_pairs,
)
from sgd_mds.distances import full_square_distances, pairwise_distances


def available_devices():
    devs = [torch.device("cpu")]
    if torch.cuda.is_available():
        try:
            maj, _ = torch.cuda.get_device_capability(0)
            if maj >= 7:
                x = torch.ones(1, device="cuda")
                _ = (x + 1).item()
                devs.append(torch.device("cuda"))
        except Exception:
            pass
    return devs


@pytest.mark.parametrize("device", available_devices())
def test_stress_full_tiny_triangle(device):
    X = torch.tensor(
        [
            [0.0, 0.0],
            [3.0, 4.0],
            [3.0, 0.0],
        ],
        device=device,
    )

    D = full_square_distances(X)
    s = kruskal_stress_full(X, D)
    assert torch.isfinite(s)
    assert s.item() == pytest.approx(0.0, abs=1e-8)

    Xp = X.clone()
    Xp[1, 0] += 0.1
    D_true = D
    s2 = kruskal_stress_full(Xp, D_true)
    assert s2.item() > 0.0


@pytest.mark.parametrize("device", available_devices())
def test_stress_full_matches_reference(device):
    gen_device = "cuda" if device.type == "cuda" else "cpu"
    g = torch.Generator(device=gen_device).manual_seed(123)

    n, d = 8, 3
    X = torch.randn(n, d, generator=g, device=device)

    D_target = torch.cdist(X, X)
    Xp = X + 0.05 * torch.randn_like(X)

    s_full = kruskal_stress_full(Xp, D_target)

    D_hat = torch.cdist(Xp, Xp)
    tri = torch.triu(torch.ones_like(D_hat, dtype=torch.bool), diagonal=1)
    num = ((D_hat - D_target)[tri] ** 2).sum()
    den = (D_target[tri] ** 2).sum().clamp_min(1e-12)
    s_ref = torch.sqrt(num / den)

    assert torch.allclose(s_full, s_ref, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("device", available_devices())
def test_stress_pairs_approximates_full(device):
    gen_device = "cuda" if device.type == "cuda" else "cpu"
    g = torch.Generator(device=gen_device).manual_seed(7)

    n, d = 200, 2
    X0 = torch.randn(n, d, generator=g, device=device)
    D = torch.cdist(X0, X0)

    X = X0 + 0.1 * torch.randn_like(X0)

    s_full = kruskal_stress_full(X, D)

    B = 50_000
    i = torch.randint(0, n, (B,), generator=g, device=device)
    j = torch.randint(0, n, (B,), generator=g, device=device)
    same = i == j
    while same.any():
        j[same] = torch.randint(0, n, (int(same.sum()),), generator=g, device=device)
        same = i == j

    s_pairs = kruskal_stress_pairs(X, D, i, j)

    assert math.isfinite(s_full.item())
    assert math.isfinite(s_pairs.item())
    assert s_pairs.item() == pytest.approx(s_full.item(), rel=0.05, abs=5e-3)


@pytest.mark.parametrize("device", available_devices())
def test_stress_full_weight_mask_equals_subset(device):
    gen_device = "cuda" if device.type == "cuda" else "cpu"
    g = torch.Generator(device=gen_device).manual_seed(99)

    n, d = 12, 3
    X0 = torch.randn(n, d, generator=g, device=device)
    D = torch.cdist(X0, X0)

    X = X0 + 0.05 * torch.randn_like(X0)

    W = torch.zeros((n, n), device=device)
    tri = torch.triu(torch.ones((n, n), dtype=torch.bool, device=device), diagonal=1)
    keep = torch.rand(int(tri.sum()), device=device) > 0.5

    W[tri] = keep.float()
    W = W + W.t()

    s_w = kruskal_stress_full(X, D, weights=W)

    upper_idx = torch.nonzero(tri, as_tuple=False)
    kept_pairs = upper_idx[keep]
    i_kept = kept_pairs[:, 0]
    j_kept = kept_pairs[:, 1]

    s_subset = kruskal_stress_pairs(X, D, i_kept, j_kept)

    assert s_w.item() == pytest.approx(s_subset.item(), abs=1e-7, rel=1e-6)


@pytest.mark.parametrize("device", available_devices())
def test_zero_stress_identical_embedding(device):
    X = torch.tensor([[1.0, 2.0], [1.0, 2.0]], device=device)
    D = full_square_distances(X)
    s_full = kruskal_stress_full(X, D)
    s_pairs = kruskal_stress_pairs(
        X, D, torch.tensor([0, 0], device=device), torch.tensor([1, 1], device=device)
    )
    assert s_full.item() == pytest.approx(0.0, abs=1e-8)
    assert s_pairs.item() == pytest.approx(0.0, abs=1e-8)
