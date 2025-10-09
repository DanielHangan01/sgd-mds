import numpy as np
import torch
from . import core, utils
from . import stress


class SGDMDS:
    """Sklearn-style SGD MDS estimator."""

    def __init__(
        self,
        n_components=2,
        max_iter=50,
        batch_size=50_000,
        lr_init=1.0,
        random_state=0,
        device="auto",
        stress_sample_size=200_000,
    ):
        self.n_components = n_components
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.lr_init = lr_init
        self.random_state = random_state
        self.device = device
        self.stress_sample_size = stress_sample_size

        self.embedding_ = None
        self.stress_ = float("nan")
        self.n_iter_ = 0

    def fit(self, D, y=None):
        rng = utils.set_seed(self.random_state)
        device = utils.resolve_device(self.device)

        D = np.asarray(D, dtype=np.float32)
        n = D.shape[0]
        D_t = torch.as_tensor(D, device=device)

        X = torch.randn(n, self.n_components, device=device)

        B = min(self.batch_size, max(1, n * (n - 1) // 2))
        i_idx = torch.randint(0, n, (B,), device=device)
        j_idx = torch.randint(0, n, (B,), device=device)

        mask = i_idx == j_idx
        while mask.any():
            j_idx[mask] = torch.randint(0, n, (int(mask.sum()),), device=device)
            mask = i_idx == j_idx
        i_idx, j_idx = torch.minimum(i_idx, j_idx), torch.maximum(i_idx, j_idx)

        deltas = D_t[i_idx, j_idx]
        weights = torch.ones_like(deltas)
        h = float(self.lr_init)

        for _ in range(self.max_iter):
            core.sgd_step(X, i_idx, j_idx, deltas, weights, h)

        self.embedding_ = X.detach().cpu().numpy()
        self.n_iter_ = self.max_iter

        if n <= 1500:
            self.stress_ = float(stress.kruskal_stress_full(X, D_t).item())
        else:
            ii = torch.zeros
            jj = torch.zeros
            self.stress_ = float(stress.kruskal_stress_pairs(X, D_t, ii, jj).item())

        return self

    def fit_transform(self, D, y=None):
        self.fit(D, y)
        return self.embedding_
