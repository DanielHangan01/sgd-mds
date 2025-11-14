import numpy as np
import pytest
import torch

from sgd_mds.estimator import SGDMDS
from sgd_mds import utils, stress


@pytest.mark.parametrize(
    "pair_weighting",
    [utils.PAIR_WEIGHTING_UNIFORM, utils.PAIR_WEIGHTING_INVERSE_DISTANCE],
)
def test_sgdmds_stress_matches_helper(pair_weighting):
    D = np.array(
        [
            [0.0, 1.0, 2.0],
            [1.0, 0.0, 1.5],
            [2.0, 1.5, 0.0],
        ],
        dtype=np.float32,
    )

    model = SGDMDS(
        n_components=2,
        stopper="iterations",
        max_iter=2,
        batch_size=3,
        lr_init=0.01,
        scheduler="constant",
        pair_weighting=pair_weighting,
        random_state=0,
        device="cpu",
    )
    model.fit(D)

    X = torch.from_numpy(model.embedding_)
    D_t = torch.from_numpy(D)
    weighting_mode = utils.normalize_pair_weighting(pair_weighting)
    weight_matrix = utils.compute_full_weights(
        D_t,
        weighting_mode,
        min_delta=getattr(model, "pair_weight_min_delta_", None),
    )
    expected = stress.kruskal_stress_full(X, D_t, weights=weight_matrix).item()

    assert model.stress_ == pytest.approx(expected, rel=1e-6, abs=1e-6)


def test_inverse_distance_weights_handles_zero_deltas():
    D = np.array(
        [
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )

    model = SGDMDS(
        n_components=2,
        stopper="iterations",
        max_iter=2,
        batch_size=3,
        lr_init=0.01,
        scheduler="constant",
        pair_weighting=utils.PAIR_WEIGHTING_INVERSE_DISTANCE,
        random_state=0,
        device="cpu",
    )

    model.fit(D)
    assert np.isfinite(model.stress_)


def test_auto_learning_rate_uses_weight_extrema():
    D = np.array(
        [
            [0.0, 1.0, 2.0],
            [1.0, 0.0, 4.0],
            [2.0, 4.0, 0.0],
        ],
        dtype=np.float32,
    )

    model = SGDMDS(
        n_components=2,
        stopper="iterations",
        max_iter=1,
        batch_size=3,
        lr_init="auto",
        scheduler="constant",
        pair_weighting=utils.PAIR_WEIGHTING_INVERSE_DISTANCE,
        pair_weighting_min_delta=0.5,
        paper_lr_epsilon=0.2,
        random_state=0,
        device="cpu",
    )

    model.fit(D)

    # max delta = 4 -> w_min = 1/4; min delta clamp = 0.5 -> w_max = 2
    assert model.weight_min_ == pytest.approx(0.25, rel=1e-6)
    assert model.weight_max_ == pytest.approx(2.0, rel=1e-6)
    assert model.lr_init_ == pytest.approx(4.0, rel=1e-6)
