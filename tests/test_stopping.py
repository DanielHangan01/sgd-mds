import pytest

from sgd_mds.stopping import (
    BaseStopper,
    MaxIterationsStopper,
    MovementThresholdStopper,
    create_stopper,
)

def run_steps(stopper, values, key="max_update"):
    stopper.reset()
    stopped_iter = None
    for t, v in enumerate(values, start=1):
        status = {key: v}
        if stopper.check(status):
            stopped_iter = t
            break
    return stopped_iter, stopper.current_iter


def test_base_stopper_never_stops_and_counts():
    s = BaseStopper()
    s.reset()
    for _ in range(5):
        assert s.check({}) is False
    assert s.current_iter == 5
    s.reset()
    assert s.current_iter == 0


def test_max_iterations_basic():
    s = MaxIterationsStopper(max_iter=3)
    s.reset()
    assert s.check({}) is False
    assert s.check({}) is False
    assert s.check({}) is True
    assert s.current_iter == 3

def test_max_iterations_invalid():
    with pytest.raises(ValueError):
        MaxIterationsStopper(max_iter=0)


def test_threshold_stops_immediately_when_below_and_patience0():
    s = MovementThresholdStopper(threshold=0.1, patience=0)
    stopped_iter, iters = run_steps(s, [0.2, 0.15, 0.09, 0.2])
    assert stopped_iter == 3
    assert iters == 3

def test_threshold_requires_consecutive_patience():
    s = MovementThresholdStopper(threshold=0.1, patience=2)
    seq = [0.2, 0.15, 0.09, 0.08, 0.2, 0.07, 0.06, 0.05]
    stopped_iter, _ = run_steps(s, seq)
    assert stopped_iter == 8

def test_threshold_resets_counter_on_spike():
    s = MovementThresholdStopper(threshold=0.5, patience=1)
    seq = [0.4, 0.6, 0.6, 0.4, 0.4, 0.6]
    stopped_iter, _ = run_steps(s, seq)
    assert stopped_iter == 5

def test_threshold_accepts_max_gradient_alias():
    s = MovementThresholdStopper(threshold=0.1, patience=0)
    s.reset()
    assert s.check({"max_gradient": 0.5}) is False
    assert s.check({"max_gradient": 0.05}) is True

def test_threshold_missing_value_means_never_stop():
    s = MovementThresholdStopper(threshold=0.1, patience=0)
    s.reset()
    for _ in range(3):
        assert s.check({}) is False
    assert s.current_iter == 3

def test_threshold_invalid_types_raise():
    s = MovementThresholdStopper(threshold=0.1, patience=0)
    with pytest.raises(TypeError):
        s.check({"max_update": object()})

def test_threshold_invalid_args():
    with pytest.raises(ValueError):
        MovementThresholdStopper(threshold=0.0)
    with pytest.raises(ValueError):
        MovementThresholdStopper(threshold=0.1, patience=-1)


def test_factory_iterations_ok_and_ignores_extras():
    st = create_stopper(
        name="iterations",
        max_iter=4,
        threshold=0.123,
        patience=7,
    )
    assert isinstance(st, MaxIterationsStopper)
    st.reset()
    assert st.check({}) is False
    assert st.check({}) is False
    assert st.check({}) is False
    assert st.check({}) is True

def test_factory_threshold_ok_with_patience():
    st = create_stopper(
        name="threshold",
        threshold=0.1,
        patience=2,
        max_iter=999,
    )
    assert isinstance(st, MovementThresholdStopper)
    seq = [0.2, 0.12, 0.09, 0.08, 0.07]
    stopped_iter, _ = run_steps(st, seq)
    assert stopped_iter == 5

def test_factory_unknown_raises():
    with pytest.raises(ValueError):
        create_stopper("nope", max_iter=10)
