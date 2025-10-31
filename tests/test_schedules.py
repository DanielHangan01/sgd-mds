import math
import pytest

from sgd_mds.schedules import (
    BaseScheduler,
    ConstantScheduler,
    ExponentialScheduler,
    ConvergenceScheduler,
    create_scheduler,
)

RTOL = 1e-12
ATOL = 1e-12


def step_many(sched, steps):
    vals = []
    for _ in range(steps + 1):
        vals.append(sched.get_lr())
        sched.step()
    return vals


def test_constant_scheduler_basic():
    s = ConstantScheduler(lr_init=0.5)
    vals = step_many(s, 10)
    assert all(abs(v - 0.5) < ATOL for v in vals)
    assert s.current_step == 11


def test_exponential_hits_final_without_warmup():
    h0 = 0.5
    hf = 0.05
    T = 10
    s = ExponentialScheduler(lr_init=h0, lr_final=hf, max_iter=T, warmup_steps=0)
    vals = step_many(s, T)

    assert math.isclose(vals[0], h0, rel_tol=RTOL, abs_tol=ATOL)
    assert math.isclose(vals[-1], hf, rel_tol=RTOL, abs_tol=ATOL)

    for t in range(1, len(vals)):
        assert vals[t] <= vals[t - 1] + 1e-15
        assert vals[t] >= hf - 1e-15


def test_exponential_with_warmup_then_decay_and_clamp():
    h0 = 0.4
    hf = 0.05
    T = 20
    W = 5
    s = ExponentialScheduler(lr_init=h0, lr_final=hf, max_iter=T, warmup_steps=W)
    vals = step_many(s, T)

    assert vals[0] > 0.0
    for t in range(1, W):
        assert vals[t] > vals[t - 1]

    for t in range(W + 1, len(vals)):
        assert vals[t] <= vals[t - 1] + 1e-15
        assert vals[t] >= hf - 1e-15


def test_exponential_guard_when_lr_final_ge_lr_init():
    s = ExponentialScheduler(lr_init=0.1, lr_final=0.2, max_iter=10, warmup_steps=0)
    vals = step_many(s, 10)
    assert all(math.isclose(v, 0.1, rel_tol=RTOL, abs_tol=ATOL) for v in vals)


def test_exponential_invalid_args():
    with pytest.raises(ValueError):
        ExponentialScheduler(lr_init=-0.1, lr_final=0.01, max_iter=10)
    with pytest.raises(ValueError):
        ExponentialScheduler(lr_init=0.1, lr_final=0.0, max_iter=10)
    with pytest.raises(ValueError):
        ExponentialScheduler(lr_init=0.1, lr_final=0.01, max_iter=0)


def test_convergence_two_phase_behavior():
    h0 = 0.5
    h1 = 0.1
    T1 = 5
    s = ConvergenceScheduler(lr_init=h0, lr_final_phase1=h1, phase1_iters=T1, lr_min=1e-4)
    vals = step_many(s, 20)

    p1 = vals[: T1 + 1]
    for t in range(1, len(p1)):
        assert p1[t] <= p1[t - 1] + 1e-15
        assert p1[t] >= h1 - 1e-15

    assert p1[-1] >= h1 - 1e-15

    p2 = vals[T1 + 1 :]
    assert all(v >= 1e-4 - 1e-15 for v in p2)
    for t in range(T1 + 2, len(vals)):
        assert vals[t] <= vals[t - 1] + 1e-15


def test_convergence_invalid():
    with pytest.raises(ValueError):
        ConvergenceScheduler(lr_init=0.5, lr_final_phase1=0.1, phase1_iters=0)


def test_factory_constant_ok():
    s = create_scheduler("constant", lr_init=0.3)
    assert isinstance(s, ConstantScheduler)
    vals = step_many(s, 3)
    assert all(math.isclose(v, 0.3, rel_tol=RTOL, abs_tol=ATOL) for v in vals)


def test_factory_exponential_ok_and_ignores_extra_kwargs():
    s = create_scheduler(
        "exponential",
        lr_init=0.5,
        lr_final=0.05,
        max_iter=10,
        warmup_steps=0,
        lr_min=123,
        phase1_iters=999,
        lr_final_phase1=777,
    )
    assert isinstance(s, ExponentialScheduler)
    vals = step_many(s, 10)
    assert math.isclose(vals[0], 0.5, rel_tol=RTOL, abs_tol=ATOL)
    assert math.isclose(vals[-1], 0.05, rel_tol=RTOL, abs_tol=ATOL)


def test_factory_convergence_ok():
    s = create_scheduler(
        "convergence",
        lr_init=0.5,
        lr_final_phase1=0.1,
        phase1_iters=5,
        lr_min=1e-4,
    )
    assert isinstance(s, ConvergenceScheduler)


def test_factory_unknown_raises():
    with pytest.raises(ValueError):
        create_scheduler("nope", lr_init=0.5)
