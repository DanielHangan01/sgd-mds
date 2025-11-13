import pytest
import torch

from sgd_mds import utils


def test_normalize_pair_weighting_accepts_aliases():
    assert utils.normalize_pair_weighting(None) == utils.PAIR_WEIGHTING_UNIFORM
    assert utils.normalize_pair_weighting("inverse_distance") == utils.PAIR_WEIGHTING_INVERSE_DISTANCE
    assert utils.normalize_pair_weighting("Inverse-Distance") == utils.PAIR_WEIGHTING_INVERSE_DISTANCE
    assert utils.normalize_pair_weighting("1/D") == utils.PAIR_WEIGHTING_INVERSE_DISTANCE


def test_normalize_pair_weighting_rejects_unknown():
    with pytest.raises(ValueError):
        utils.normalize_pair_weighting("mystery")


def test_compute_pair_weights_inverse_distance():
    deltas = torch.tensor([1.0, 2.0, 4.0])
    weights = utils.compute_pair_weights(
        deltas,
        utils.PAIR_WEIGHTING_INVERSE_DISTANCE,
        eps=1e-6,
    )
    expected = torch.tensor([1.0, 0.5, 0.25])
    assert torch.allclose(weights, expected, atol=1e-6)


def test_compute_pair_weights_inverse_distance_zero_delta():
    deltas = torch.tensor([0.0, 2.0])
    weights = utils.compute_pair_weights(
        deltas,
        utils.PAIR_WEIGHTING_INVERSE_DISTANCE,
        eps=1e-6,
        min_delta=1e-6,
    )
    expected = torch.tensor([1.0 / 1e-6, 0.5])
    assert torch.allclose(weights, expected, atol=1e-6)


def test_compute_full_weights_sets_zero_diagonal():
    D = torch.tensor([[0.0, 2.0], [2.0, 0.0]])
    weights = utils.compute_full_weights(D, utils.PAIR_WEIGHTING_INVERSE_DISTANCE, eps=1e-6)
    assert torch.allclose(weights, torch.tensor([[0.0, 0.5], [0.5, 0.0]]))


def test_compute_full_weights_uniform_returns_none():
    D = torch.ones((2, 2))
    assert utils.compute_full_weights(D, utils.PAIR_WEIGHTING_UNIFORM) is None
