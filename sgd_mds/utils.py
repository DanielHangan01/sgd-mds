from __future__ import annotations
import random
from typing import Optional

import numpy as np
import numpy.random
import torch


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