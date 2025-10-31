import math
import pytest
import torch

from sgd_mds.distances import (
    pairwise_distances,
    full_square_distances,
)


def available_devices():
    devs = [torch.device("cpu")]
    if torch.cuda.is_available():
        try:
            maj, _ = torch.cuda.get_device_capability(0)
            if maj >= 7:
                x = torch.ones(1, device="cuda")
                y = (x + 1).item()
                devs.append(torch.device("cuda"))
        except Exception:
            pass
    return devs


@pytest.mark.parametrize("device", available_devices())
def test_pairwise_distances_simple_correctness(device):
    X = torch.tensor([[0.0, 0.0], [3.0, 4.0], [3.0, 0.0]], device=device)

    i = torch.tensor([0, 0, 1], device=device)
    j = torch.tensor([1, 2, 2], device=device)

    d = pairwise_distances(X, i, j)

    expected = torch.tensor([5.0, 3.0, 4.0], device=device)
    assert torch.allclose(d, expected, atol=1e-6)


@pytest.mark.parametrize("device", available_devices())
def test_pairwise_distances_matches_torch_cdist(device):

    g = torch.Generator(device=device).manual_seed(0)
    X = torch.randn(7, 5, generator=g, device=device)

    i = torch.randint(0, X.size(0), (20,), device=device)
    j = torch.randint(0, X.size(0), (20,), device=device)

    mask = i == j
    while mask.any():
        j[mask] = torch.randint(0, X.size(0), (int(mask.sum()),), device=device)
        mask = i == j

    d_pairs = pairwise_distances(X, i, j)

    d_full = torch.cdist(X, X)
    d_ref = d_full[i, j]

    assert torch.allclose(d_pairs, d_ref, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("device", available_devices())
def test_full_squareform_properties(device):
    g = torch.Generator(device=device).manual_seed(1)
    X = torch.randn(6, 3, generator=g, device=device)

    D = full_square_distances(X)

    assert torch.allclose(D, D.T, atol=1e-6)
    assert torch.allclose(torch.diag(D), torch.zeros(6, device=device), atol=1e-7)
    assert (D >= -1e-9).all()


@pytest.mark.parametrize("device", available_devices())
def test_full_squareform_matches_cdist(device):
    g = torch.Generator(device=device).manual_seed(2)
    X = torch.randn(8, 4, generator=g, device=device)

    D_ours = full_square_distances(X)
    D_ref = torch.cdist(X, X)

    assert torch.allclose(D_ours, D_ref, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("device", available_devices())
def test_full_vs_pairs_consistency(device):
    g = torch.Generator(device=device).manual_seed(3)
    X = torch.randn(9, 2, generator=g, device=device)

    D_full = full_square_distances(X)

    i = torch.tensor([0, 1, 2, 3, 4, 5], device=device)
    j = torch.tensor([8, 7, 6, 5, 4, 3], device=device)

    d_pairs = pairwise_distances(X, i, j)
    d_ref = D_full[i, j]

    assert torch.allclose(d_pairs, d_ref, atol=1e-6)


@pytest.mark.parametrize("device", available_devices())
def test_numerical_stability_identical_points(device):
    X = torch.tensor([[1.2345, -6.789], [1.2345, -6.789]], device=device)
    D = full_square_distances(X)

    assert torch.isfinite(D).all()
    assert D[0, 1].abs() < 1e-7
    assert D[1, 0].abs() < 1e-7
