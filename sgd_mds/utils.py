from __future__ import annotations
import random
from typing import Optional

import numpy as np
import numpy.random
import torch

PAIR_WEIGHTING_UNIFORM = "uniform"
PAIR_WEIGHTING_INVERSE_DISTANCE = "inverse_distance"
PAIR_WEIGHTING_CHOICES = (
    PAIR_WEIGHTING_UNIFORM,
    PAIR_WEIGHTING_INVERSE_DISTANCE,
)

def set_seed(seed: Optional[int]) -> np.random.Generator:
    """
    Set random seeds for all relevant libraries to ensure reproducibility.

    This function sets the seed for Python's built-in random module, NumPy,
    and PyTorch (for both CPU and CUDA). It also configures cuDNN to use
    deterministic algorithms, which is crucial for reproducible results on a GPU.

    Parameters
    ----------
    seed : int, optional
        The seed value. If None, the seeds will not be set.

    Returns
    -------
    np.random.Generator
        A NumPy random number generator instance, seeded for further use if needed.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    return np.random.default_rng(seed)


def resolve_device(device: str | torch.device) -> torch.device:
    """
    Resolve a device string into a torch.device object.

    Handles the special case 'auto', which selects a CUDA device if a
    capable one is available, otherwise defaults to CPU.

    Parameters
    ----------
    device : str or torch.device
        The device to use. Can be 'auto', 'cpu', 'cuda', 'cuda:0', etc.,
        or an existing torch.device object.

    Returns
    -------
    torch.device
        The resolved torch.device object.
    """
    if not isinstance(device, (str, torch.device)):
        raise TypeError(f"Device must be a string or torch.device, not {type(device)}")

    if isinstance(device, torch.device):
        return device

    if device == "auto":
        if torch.cuda.is_available():
            try:
                major_capability, _ = torch.cuda.get_device_capability(0)
                if major_capability >= 7:
                    return torch.device("cuda")
            except Exception:
                pass
        return torch.device("cpu")

    return torch.device(device)


def normalize_pair_weighting(weighting: Optional[str]) -> str:
    """
    Normalize a user-provided weighting name into one of the supported schemes.
    """
    if weighting is None:
        return PAIR_WEIGHTING_UNIFORM

    w = weighting.strip().lower()
    alias_map = {
        "inverse-distance": PAIR_WEIGHTING_INVERSE_DISTANCE,
        "inv_distance": PAIR_WEIGHTING_INVERSE_DISTANCE,
        "1/d": PAIR_WEIGHTING_INVERSE_DISTANCE,
        "1overd": PAIR_WEIGHTING_INVERSE_DISTANCE,
    }
    w = alias_map.get(w, w)

    if w not in PAIR_WEIGHTING_CHOICES:
        opts = ", ".join(PAIR_WEIGHTING_CHOICES)
        raise ValueError(f"Unknown pair weighting '{weighting}'. Supported: {opts}.")
    return w


def compute_pair_weights(
    deltas: torch.Tensor,
    weighting: str,
    eps: float = 1e-12,
    min_delta: float | None = None,
) -> torch.Tensor:
    """
    Compute per-pair weights according to the requested weighting scheme.
    """
    if weighting == PAIR_WEIGHTING_UNIFORM:
        return torch.ones_like(deltas)
    if weighting == PAIR_WEIGHTING_INVERSE_DISTANCE:
        floor = max(eps, float(min_delta) if min_delta is not None else eps)
        clamped = torch.clamp(deltas, min=floor)
        return torch.reciprocal(clamped)
    raise ValueError(f"Unsupported weighting scheme: {weighting}")


def compute_full_weights(
    D_full: torch.Tensor,
    weighting: str,
    eps: float = 1e-12,
    min_delta: float | None = None,
) -> torch.Tensor | None:
    """
    Build a full (n x n) weight matrix for stress evaluation.
    Returns None when weighting is uniform.
    """
    if weighting == PAIR_WEIGHTING_UNIFORM:
        return None
    weights = compute_pair_weights(D_full, weighting, eps, min_delta)
    if weights.dim() == 2 and weights.size(0) == weights.size(1):
        weights = weights.clone()
        weights.fill_diagonal_(0.0)
    return weights
