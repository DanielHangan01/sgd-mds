from __future__ import annotations
from typing import Optional, Dict, Any

import numpy as np
import torch
from . import core, utils, stress, samplers, schedules, stopping


class SGDMDS:
    """
    Multidimensional Scaling (MDS) using Stochastic Gradient Descent.
    """

    def __init__(
            self,
            n_components: int = 2,
            stopper: str = "threshold",
            stopper_params: Optional[Dict[str, Any]] = {'threshold': 0.03, 'patience': 3},
            max_iter: int = 500,  # Failsafe for convergence-based stoppers
            lr_init: float = 1.0,
            scheduler: str = "convergence",
            scheduler_params: Optional[Dict[str, Any]] = {'lr_final_phase1': 0.1, 'phase1_iters': 30},
            batch_size: int = 50_000,
            random_state: Optional[int] = 0,
            device: str = "auto",
            stress_sample_size: int = 200_000,
            pair_weighting: str = "uniform",
            pair_weighting_min_delta: float | None = None,
            pair_weighting_floor_quantile: float | None = 0.01,
            pair_weighting_max_step: float | None = 0.05,
    ):
        """
        Parameters
        ----------
        n_components : int, default=2
            The number of dimensions for the output embedding.

        stopper : str, default='threshold'
            The name of the stopping criterion to use. Common options:
            - 'threshold': Stop when points stop moving significantly.
            - 'iterations': Stop after a fixed number of iterations.

        stopper_params : dict, optional
            Parameters for the chosen stopper. For 'threshold', this can include:
            - 'threshold': float, e.g., 0.03 (the stopping threshold)
            - 'patience': int, e.g., 3 (steps to wait for stability)

        max_iter : int, default=500
            The maximum number of iterations to run. Acts as a failsafe
            to prevent infinite loops with convergence-based stoppers.

        lr_init : float, default=1.0
            The initial learning rate. A high value is recommended as the
            algorithm caps the effective step size.

        scheduler : str, default='convergence'
            The learning rate schedule. Common options:
            - 'convergence': A two-phase schedule ideal for threshold stopping.
            - 'exponential': A smooth decay schedule ideal for iteration stopping.

        scheduler_params : dict, optional
            Parameters for the chosen scheduler. For 'convergence', this can include:
            - 'lr_final_phase1': float (e.g., 0.1)
            - 'phase1_iters': int (e.g., 30)

        batch_size : int, default=50_000
            The number of pairs to sample in each SGD step.

        random_state : int, optional
            Seed for the random number generator for reproducibility.

        device : str, default='auto'
            The device to run computations on ('auto', 'cpu', 'cuda').

        stress_sample_size : int, default=200_000
            Number of pairs to sample for stress calculation on large datasets.

        pair_weighting : {"uniform", "inverse_distance"}, default="uniform"
            Weighting applied to every pair update and stress computation.
            "uniform" sets all w_ij = 1, while "inverse_distance"
            sets w_ij = 1 / δ_ij (clamped for numerical stability).

        pair_weighting_min_delta : float, optional
            Explicit clamp applied to δ_ij before inverting them. If None, a
            data-driven value derived from `pair_weighting_floor_quantile` is used.

        pair_weighting_floor_quantile : float, optional
            Quantile of the empirical distance distribution used as the clamp
            when `pair_weighting_min_delta` is not provided. Set to None to disable
            automatic flooring.

        pair_weighting_max_step : float, optional
            Maximum per-pair displacement factor allowed when using non-uniform
            weightings (default 0.05, i.e., at most 5% of the discrepancy).
        """
        self.n_components = n_components
        self.stopper = stopper
        self.stopper_params = stopper_params
        self.max_iter = max_iter
        self.lr_init = lr_init
        self.scheduler = scheduler
        self.scheduler_params = scheduler_params
        self.batch_size = batch_size
        self.random_state = random_state
        self.device = device
        self.stress_sample_size = stress_sample_size
        self.pair_weighting = utils.normalize_pair_weighting(pair_weighting)
        self.pair_weighting_min_delta = pair_weighting_min_delta
        self.pair_weighting_floor_quantile = pair_weighting_floor_quantile
        self.pair_weighting_max_step = pair_weighting_max_step
        self._weight_eps = 1e-12
        self.pair_weight_min_delta_: Optional[float] = None

        self.embedding_: Optional[np.ndarray] = None
        self.stress_: float = float("nan")
        self.n_iter_: int = 0

    def fit(self, D: np.ndarray, y: Any = None) -> SGDMDS:
        """
        Computes the MDS embedding from a dissimilarity matrix.

        Parameters
        ----------
        D : np.ndarray of shape (n_samples, n_samples)
            The input square, symmetric dissimilarity matrix.

        y : any, ignored
            Present for compatibility with scikit-learn pipelines.

        Returns
        -------
        self : SGDMDS
            The fitted estimator instance.
        """
        utils.set_seed(self.random_state)
        device: torch.device = utils.resolve_device(self.device)

        D = np.asarray(D, dtype=np.float32)
        assert D.ndim == 2 and D.shape[0] == D.shape[1], "D must be a square distance matrix."
        n: int = D.shape[0]
        D_t: torch.Tensor = torch.as_tensor(D, device=device)

        X: torch.Tensor = torch.randn(n, self.n_components, device=device)

        B: int = min(self.batch_size, max(1, n * (n - 1) // 2))

        s_params: Dict[str, Any] = self.scheduler_params or {}
        scheduler = schedules.create_scheduler(
            name=self.scheduler,
            lr_init=self.lr_init,
            max_iter=self.max_iter,
            **s_params,
        )

        st_params: Dict[str, Any] = self.stopper_params or {}
        user_stopper = stopping.create_stopper(
            name=self.stopper,
            max_iter=self.max_iter,
            **st_params,
        )

        failsafe_stopper = stopping.MaxIterationsStopper(max_iter=self.max_iter)

        use_exact: bool = (self.stopper.lower() in {"threshold", "movement", "convergence"})

        pair_weight_min_delta: Optional[float] = None
        pair_weight_max_step = None
        if self.pair_weighting != utils.PAIR_WEIGHTING_UNIFORM:
            candidate = self.pair_weighting_min_delta
            if candidate is None and self.pair_weighting_floor_quantile is not None:
                tri = torch.triu_indices(n, n, offset=1, device=device)
                sampled = D_t[tri[0], tri[1]]
                q = float(torch.quantile(
                    sampled,
                    float(self.pair_weighting_floor_quantile)
                ).item())
                candidate = q
            if candidate is None:
                candidate = float(self._weight_eps)
            pair_weight_min_delta = max(float(candidate), self._weight_eps)
            pair_weight_max_step = self.pair_weighting_max_step
        self.pair_weight_min_delta_ = pair_weight_min_delta

        user_stopper.reset()
        failsafe_stopper.reset()
        while True:
            h: float = scheduler.get_lr()
            
            i_idx, j_idx = samplers.random_pairs(
                n, B, device=device, allow_replace=True
            )
            deltas: torch.Tensor = D_t[i_idx, j_idx]
            if self.pair_weighting == utils.PAIR_WEIGHTING_UNIFORM:
                weights = torch.ones_like(deltas)
            else:
                weights = utils.compute_pair_weights(
                    deltas,
                    self.pair_weighting,
                    self._weight_eps,
                    min_delta=pair_weight_min_delta,
                )

            max_update: Optional[torch.Tensor] = core.sgd_step(
                X,
                i_idx,
                j_idx,
                deltas,
                weights,
                h,
                exact_max_update=use_exact,
                max_pair_step=pair_weight_max_step,
            )
            
            status: Dict[str, Any] = {}
            if max_update is not None:
                status["max_update"] = float(max_update.item())

            if user_stopper.check(status) or failsafe_stopper.check(status):
                break
            
            scheduler.step()

        self.n_iter_ = user_stopper.current_iter

        X -= X.mean(dim=0, keepdim=True)

        self.embedding_ = X.detach().cpu().numpy()

        if n <= 1500:
            weight_matrix = utils.compute_full_weights(
                D_t,
                self.pair_weighting,
                self._weight_eps,
                min_delta=pair_weight_min_delta,
            )
            self.stress_ = float(
                stress.kruskal_stress_full(X, D_t, weights=weight_matrix).item()
            )
        else:
            S = min(self.stress_sample_size, max(1, n * (n - 1) // 2))
            ii, jj = samplers.random_pairs(n, S, device=device, allow_replace=True)
            pair_weights = None
            if self.pair_weighting != utils.PAIR_WEIGHTING_UNIFORM:
                pair_weights = utils.compute_pair_weights(
                    D_t[ii, jj],
                    self.pair_weighting,
                    self._weight_eps,
                    min_delta=pair_weight_min_delta,
                )
            self.stress_ = float(
                stress.kruskal_stress_pairs(
                    X, D_t, ii, jj, weights=pair_weights
                ).item()
            )

        return self

    def fit_transform(self, D, y=None):
        """
        Fit the model and return the resulting embedding.
        """
        self.fit(D, y)
        return self.embedding_
