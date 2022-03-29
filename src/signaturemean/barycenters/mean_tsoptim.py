import numpy as np
import signatory
import torch
# Caution ! Requires PyManOpt from Git (main branch) :
# python3 -m pip install git+https://github.com/pymanopt/pymanopt.git@master
from pymanopt.manifolds import Euclidean
from pymanopt import Problem
from pymanopt.solvers import SteepestDescent  # ConjugateGradient, TrustRegions, ParticleSwarm
import pymanopt


class TSoptim():
    """
    Parameters
    ----------
    mean_stream : int
        Number of measurements of the average time series.
    n_init : int, default=1
        Number of time the averaging algorithm will be run with different
        initialization time series. The final results will be ??? (TO DO) of
        n_init consecutive runs in terms of ???.
    max_iter : int, default=200
        Maximum number of iterations of the averaging algorithm for a
        single run.

    Attributes
    ----------
    barycenter_ts : (mean_stream, channels) array
        The resulting barycenter. Caution: `barycenter_ts` is not a signature
        but a time series.
    """

    def __init__(self, depth, channels, random_state, n_init=1, mean_stream=10,
                 metric='euclidean', penalty=None, max_iter=200, verbosity=False):
        self.depth = depth
        self.channels = channels
        self.random_state = random_state
        self.n_init = n_init
        self.mean_stream = mean_stream
        self.metric = metric
        self.max_iter = max_iter
        self.verbosity = verbosity
        self.penalty = penalty

    def _check_params(self, X):
        # n_init
        if self.n_init <= 0:
            raise ValueError(f"n_init should be > 0, got {self.n_init} instead.")
        self._n_init = self.n_init

        # max_iter
        if self.max_iter <= 0:
            raise ValueError(f"max_iter should be > 0, got {self.max_iter} instead.")

        # shape of input dataset
        if len(X.shape) != 3:
            raise ValueError(
                f"Input data should have three dimensions (batch, stream, "
                f"channels), got {self.X.shape}"
            )

    def _create_cost(self):
        @pymanopt.function.PyTorch
        def cost(ts):
            ts = ts.reshape((1, ts.shape[0], ts.shape[1]))
            Sts = signatory.signature(ts, self.depth)
            # if self.penalty == "lasso":
            #    # (TO DO) do something
            return torch.sum(torch.linalg.norm(self.SX-Sts, dim=1))
        return cost

    def _fit_one_init(self, X, rs):
        idx_obs = rs.integers(X.shape[0], size=1)
        idx_stream = np.sort(
            rs.choice(
                range(1, X.shape[1]-1),
                size=self.mean_stream-2,
                replace=False
            )
        )

        idx_stream = np.concatenate(([0], idx_stream, [X.shape[1]-1]))
        self.init_ts_ = X[idx_obs][:, idx_stream, :].clone()
        self.init_ts_ = self.init_ts_[0]

        manifold = Euclidean(X.shape[2], X.shape[1])
        cost = self._create_cost()
        problem = Problem(manifold=manifold, cost=cost, verbosity=self.verbosity)
        solver = SteepestDescent(maxiter=self.max_iter, logverbosity=self.verbosity)

        self.init_ts_ = self.init_ts_.numpy()
        bary = solver.solve(problem, x=self.init_ts_)
        costbary = cost(bary)
        if costbary < self.cost_value_:
            bary = torch.from_numpy(bary)
            self.cost_value_ = costbary
            self.barycenter_ts = bary
        return self

    def fit(self, X):
        """
        Parameters
        ----------
        X : (batch, stream, channels) array
            Dataset consisting of `batch` time series. Each time series has
            `channels` dimensions and `stream` measurements.
        """
        self._check_params(X)
        self.init_ts_ = None
        self.barycenter_ts = None
        self.cost_value_ = np.inf
        rs = np.random.default_rng(self.random_state)
        self.SX = signatory.signature(X, self.depth)
        init_idx = 0
        while init_idx < self.n_init:
            bary = self._fit_one_init(X, rs)
            init_idx += 1
        return self
