import numpy as np
import signatory
import torch
# Caution ! Requires PyManOpt from Git (main branch) :
# python3 -m pip install git+https://github.com/pymanopt/pymanopt.git@master
from pymanopt.manifolds import Euclidean
from pymanopt import Problem
from pymanopt.solvers import SteepestDescent  # ConjugateGradient, TrustRegions, ParticleSwarm
import pymanopt
from signaturemean.utils import datashift


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

    def __init__(self, depth, random_state, n_init=1, mean_stream=10,
                 metric='euclidean', penalty=None, max_iter=200, verbosity=False):
        self.depth = depth
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
        if not torch.equal(X[:, 0, :], torch.zeros(X.shape[0], X.shape[2])):
            raise ValueError(
                f"Input data must be shifted and scaled beforehand"
                )

    def _cost_all_args(self, ts):
        return

    @pymanopt.function.PyTorch
    def _cost(self, path):
        return self._cost_all_args()

    def _fit_one_init(self, X, SX, rs):
        idx = rs.integers(X.shape[0], size=1)
        self.init_ts_ = X[idx].clone()

        manifold = Euclidean(X.shape[2], X.shape[1])
        problem = Problem(manifold=manifold, cost=self._cost, verbosity=1)
        solver = SteepestDescent(maxiter=self.max_iter, logverbosity=0)

        self.init_ts_ = self.init_ts_.numpy()  # (TO DO) needed ?
        self.barycenter_ts = solver.solve(problem, x=self.init_ts_)
        self.barycenter_ts = torch.from_numpy(self.barycenter_ts)
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
        # batch, stream, channels = self.X.shape
        SX = signatory.signature(X, self.depth)
        init_idx = 0
        while init_idx < self.n_init:
            self._fit_one_init(X, SX, self.random_state)
            init_idx += 1
        return self
