import numpy as np
import pytest
import torch
import sgd_mds.samplers as samplers

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


def test_convergence_tracking_records_history_and_final_iteration():
    D = np.array(
        [
            [0.0, 1.0, 2.0],
            [1.0, 0.0, 3.0],
            [2.0, 3.0, 0.0],
        ],
        dtype=np.float32,
    )

    model = SGDMDS(
        n_components=2,
        stopper="iterations",
        max_iter=3,
        batch_size=3,
        lr_init=0.01,
        scheduler="constant",
        track_convergence=True,
        convergence_log_every=2,
        random_state=0,
        device="cpu",
    )

    model.fit(D)

    history = model.convergence_history_
    assert len(history) == 2  # logged at iter 2 and final iter 3
    assert history[0]["iteration"] == 2
    assert history[-1]["iteration"] == model.n_iter_ == 3
    for point in history:
        assert set(point.keys()) == {"iteration", "stress", "elapsed_time"}
        assert np.isfinite(point["stress"])
        assert point["elapsed_time"] >= 0.0


def test_convergence_tracking_falls_back_to_sampling_with_replacement(monkeypatch):
    n = 1600  # force history logging to use sampled stress instead of full matrix
    coords = np.arange(n, dtype=np.float32)
    D = np.abs(coords[:, None] - coords[None, :])

    monkeypatch.setattr("sgd_mds.estimator._CONVERGENCE_UNIQUE_PAIR_THRESHOLD", 1)

    captured_calls: list[tuple[int, bool]] = []
    original_random_pairs = samplers.random_pairs

    def fake_random_pairs(n, B, device=None, *, allow_replace=True):
        captured_calls.append((B, allow_replace))
        return original_random_pairs(n, B, device=device, allow_replace=allow_replace)

    monkeypatch.setattr(samplers, "random_pairs", fake_random_pairs)

    model = SGDMDS(
        n_components=2,
        stopper="iterations",
        max_iter=1,
        batch_size=3,
        lr_init=0.01,
        scheduler="constant",
        track_convergence=True,
        convergence_log_every=1,
        convergence_sample_size=5,
        stress_sample_size=10,
        random_state=0,
        device="cpu",
    )

    model.fit(D)

    history_calls = [allow for (B, allow) in captured_calls if B == 5]
    assert history_calls, "Expected a convergence-sampling call to random_pairs."
    assert history_calls[0] is True
