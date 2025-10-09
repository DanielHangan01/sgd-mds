import random, numpy as np, torch


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return np.random.default_rng(seed)


def resolve_device(device):
    if device == "auto":
        if torch.cuda.is_available():
            try:
                maj, _ = torch.cuda.get_device_capability(0)
                if maj >= 7:  # sm_70 or newer
                    return torch.device("cuda")
            except Exception:
                pass
        return torch.device("cpu")

    return torch.device(device) if isinstance(device, str) else device
