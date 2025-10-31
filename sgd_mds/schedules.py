from __future__ import annotations
import math
from typing import Any

class BaseScheduler:
    """Base class for learning rate schedulers."""
    def __init__(self, lr_init: float):
        self.lr_init = float(lr_init)
        self.current_step = 0
        self.lr = float(self.lr_init)

    def get_lr(self) -> float:
        """Return the current learning rate."""
        return self.lr

    def step(self) -> None:
        """Advance the scheduler by one step."""
        self.current_step += 1

class ConstantScheduler(BaseScheduler):
    """Always returns the same learning rate."""
    pass

class ExponentialScheduler(BaseScheduler):
    """
    η(t) = max(lr_final, lr_init * exp(-decay_rate * t)), with optional linear warmup.
    decay_rate chosen so that η(max_iter) ≈ lr_final (without warmup interval).
    """
    def __init__(self,
                 lr_init: float,
                 lr_final: float,
                 max_iter: int,
                 warmup_steps: int = 0):
        super().__init__(lr_init)
        if max_iter <= 0:
            raise ValueError("max_iter must > 0.")
        if lr_final <= 0 or lr_init <= 0:
            raise ValueError("lr_init and lr_final must be > 0.")
        self.lr_final = float(lr_final)
        self.max_iter = int(max_iter)
        self.warmup_steps = int(max(0, warmup_steps))

        self._clamp_min = self.lr_final < self.lr_init

        if not self._clamp_min:
            self.decay_rate = 0.0
        else:
            effective_span = max(1, self.max_iter - self.warmup_steps)
            self.decay_rate = math.log(self.lr_init / self.lr_final) / effective_span

        self.lr = self.lr_init if self.warmup_steps == 0 else 0.0

    def get_lr(self) -> float:
        t = self.current_step
        # Warmup: linearly ramp from 0 -> lr_init over warmup_steps.
        if self.warmup_steps > 0 and t < self.warmup_steps:
            w = (t + 1) / self.warmup_steps
            lr = self.lr_init * w
        else:
            # Exponential decay after warmup
            eff_t = max(0, t - self.warmup_steps)
            if self.decay_rate == 0.0:
                lr = self.lr_init
            else:
                lr = self.lr_init * math.exp(-self.decay_rate * eff_t)
            if self._clamp_min:
                lr = max(lr, self.lr_final)
        self.lr = lr
        return self.lr
    
    def step(self) -> None:
        super().step()


class ConvergenceScheduler(BaseScheduler):
    """
    Two-phase example:
      - Phase 1: exponential decay (fast coarse progress)
      - Phase 2: 1/t decay (gentle convergence)
    Parameters mirror ExponentialScheduler plus a switch point `phase1_iters`.
    """
    def __init__(self,
                 lr_init: float,
                 lr_final_phase1: float,
                 phase1_iters: int,
                 # optional final phase floor:
                 lr_min: float = 1e-4):
        super().__init__(lr_init)
        if phase1_iters <= 0:
            raise ValueError("phase1_iters must > 0.")
        self.phase1_iters = int(phase1_iters)
        self.lr_final_phase1 = float(lr_final_phase1)
        self.lr_min = float(lr_min)

        # Prepare phase1 exponential
        if self.lr_final_phase1 >= self.lr_init:
            self.decay_rate = 0.0
        else:
            self.decay_rate = math.log(self.lr_init / self.lr_final_phase1) / self.phase1_iters

    def get_lr(self) -> float:
        t = self.current_step
        if t < self.phase1_iters:
            # Phase 1: exponential decay
            if self.decay_rate == 0.0:
                lr = self.lr_init
            else:
                lr = self.lr_init * math.exp(-self.decay_rate * t)
            lr = max(lr, self.lr_final_phase1)
        else:
            # Phase 2: 1/t decay
            k = t - self.phase1_iters + 1
            lr = self.lr_final_phase1 / k
            lr = max(self.lr_min, lr)
        self.lr = lr
        return self.lr

    def step(self) -> None:
        super().step()

def _require(kwargs: dict[str, Any], *keys: str) -> None:
    missing = [k for k in keys if kwargs.get(k) is None]
    if missing:
        ks = ", ".join(missing)
        raise ValueError(f"Missing required scheduler argument(s): {ks}")

def create_scheduler(name: str, **kwargs) -> BaseScheduler:
    """
    Factory to create a learning-rate scheduler by name.

    Parameters
    ----------
    name : {"constant","exponential","convergence"}
        Scheduler type.
    kwargs : dict
        Generic argument bag. Only the relevant keys are used for each scheduler;
        extra keys are ignored so callers can pass a single dict safely.

    Returns
    -------
    BaseScheduler
    """
    name = (name or "constant").lower()
    if name == "constant":
        _require(kwargs, "lr_init")
        return ConstantScheduler(lr_init=float(kwargs.get('lr_init')))
    elif name == "exponential":
        _require(kwargs, "lr_init", "lr_final", "max_iter")
        return ExponentialScheduler(lr_init=float(kwargs.get('lr_init')),
                                    lr_final=float(kwargs.get('lr_final')),
                                    max_iter=int(kwargs.get('max_iter')),
                                    warmup_steps=int(kwargs.get('warmup_steps', 0)))
    elif name == "convergence":
        _require(kwargs, "lr_init", "lr_final_phase1", "phase1_iters")
        return ConvergenceScheduler(lr_init=float(kwargs.get('lr_init')),
                                     lr_final_phase1=float(kwargs.get('lr_final_phase1')),
                                     phase1_iters=int(kwargs.get('phase1_iters')),
                                     lr_min=float(kwargs.get('lr_min', 1e-4)))
    else:
        raise ValueError(f"Unknown scheduler name: {name!r}. "
                         "Valid: 'constant', 'exponential', 'convergence'.")
