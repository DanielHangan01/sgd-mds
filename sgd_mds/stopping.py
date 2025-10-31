from __future__ import annotations
from typing import Any, Dict, Optional
import time

class BaseStopper:
    """
    Base interface for stopping criteria.
    
    Usage pattern in training loop:
        stopper.reset()
        while True:
            # ... do one optimizer step, compute status dict ...
            if stopper.check(status):
                break
    """
    def __init__(self) -> None:
        self.current_iter: int = 0
    
    def reset(self) -> None:
        self.current_iter = 0
    
    def check(self, status: Dict[str, Any]) -> bool:
        """
        Inspect the current training status and decide whether to stop.
        Returns True to stop, False to continue. Default: never stop.
        Also increments self.current_iter by 1 to reflect that a step was made.
        """
        self.current_iter += 1
        return False
    
class MaxIterationsStopper(BaseStopper):
    """
    Stop after a fixed number of iterations.
    """
    def __init__(self, max_iter: int) -> None:
        super().__init__()
        if max_iter <= 0:
            raise ValueError("max_iter must be > 0.")
        self.max_iter = int(max_iter)
    
    def check(self, status: Dict[str, Any]) -> bool:
        stop = super().check(status)
        return self.current_iter >= self.max_iter
    
class MovementThresholdStopper(BaseStopper):
    """
    Stop when the maximum per-point movement (or max gradient proxy) falls 
    below a threshold for 'patience' consecutive iterations.

    Expect `status` to contain at least one of:
      - `max_update`: float
      - `max_gradient`: float
    """
    def __init__(self, threshold: float, patience: int = 0) -> None:
        super().__init__()
        if threshold <= 0.0:
            raise ValueError("threshold must be > 0.")
        if patience < 0:
            raise ValueError("patience must be >= 0.")
        self.threshold = float(threshold)
        self.patience = int(patience)
        self._below_threshold_count = 0

    def reset(self) -> None:
        super().reset()
        self._below_threshold_count = 0
    
    def check(self, status: Dict[str, Any]) -> bool:
        super().check(status)

        val = status.get("max_update", None)
        if val is None:
            val = status.get("max_gradient", None)

        if val is None:
            return False
        
        try:
            v = float(val)
        except Exception as e:
            raise TypeError(f"MovementThresholdStopper expected a numeric "
                            f"'max_update'/'max_gradient', got {type(val)}") from e
        
        if v < self.threshold:
            self._below_threshold_count += 1
        else:
            self._below_threshold_count = 0

        return self._below_threshold_count > self.patience
    

def _require(kwargs: Dict[str, Any], *keys: str) -> None:
    missing = [k for k in keys if kwargs.get(k) is None]
    if missing:
        ks = ", ".join(missing)
        raise TypeError(f"Missing required stopper argument(s): {ks}")
    

def create_stopper(name: str, **kwargs: Any) -> BaseStopper:
    """
    Factory to create a stopping criterion by name.

    name: 
      - "iterations"  -> MaxIterationsStopper(max_iter)
      - "threshold"   -> MovementThresholdStopper(threshold[, patience])

    kwargs (only relevant ones are used; extras are ignored):
      - max_iter: int
      - threshold: float
      - patience: int (optional, default 0)
    """
    name = (name or "iterations").lower().strip()
    if name in ("iterations", "max_iter", "fixed"):
        _require(kwargs, "max_iter")
        return MaxIterationsStopper(max_iter=int(kwargs.get("max_iter")))
    elif name in ("threshold", "movement", "convergence"):
        _require(kwargs, "threshold")
        patience = int(kwargs.get("patience", 0))
        return MovementThresholdStopper(
            threshold=float(kwargs.get("threshold")),
            patience=patience,
        )
    else:
        raise ValueError(f"Unknown stopper name: {name!r}. "
                         "Valid: 'iterations', 'threshold'.")
