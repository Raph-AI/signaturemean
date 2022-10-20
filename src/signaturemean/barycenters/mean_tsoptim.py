import numpy as np
import signatory
import torch
import pymanopt  # tested under pymanopt==2.0.1
import pymanopt.manifolds
import pymanopt.optimizers


class TSoptim():
    """
    Parameters
    ----------
    depth : int
        Signature depth to use.

    random_state : int
        Seed for random initialization.

    stream_fixed : boolean, default=True
        Whether the stream (nb of time measurements) value of input dataset `X`
        take one (fixed) or various values.

    n_init : int, default=1
        Number of time the averaging algorithm will be run with different
        initializations. NB: not supported yet, should use default value.
        (TO DO)

    mean_stream : int
        Number of time measurements of the average time series.

    metric : str, default="euclidean"
        NB: not supported yet, should use default value.

    max_iter : int, default=200
        Maximum number of iterations of the averaging algorithm for a
        single run.

    weights : array-like
        Barycenter weights. Default is 1/batch.

    verbose : boolean, default=False
        If activated, print information.

    penalty : int
        NB: not supported yet.

    Attributes
    ----------
    barycenter_ts : (mean_stream, channels) array
        The resulting barycenter. Caution: `barycenter_ts` is not a signature
        but a time series.
    """

    def __init__(self,
                 depth,
                 random_state,
                 random_index=None,
                 stream_fixed=True,
                 n_init=1,
                 mean_stream=10,
                 metric='euclidean',
                 max_iter=200,
                 weights=None,
                 verbose=False,
                 penalty=None):
        self.depth = depth
        self.random_state = random_state
        self.random_index = random_index
        self.stream_fixed = stream_fixed
        self.n_init = n_init
        self.mean_stream = mean_stream
        self.metric = metric
        self.max_iter = max_iter
        self.weights = weights
        self.verbose = verbose
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
        if self.stream_fixed==True:
            if len(X.shape) != 3:
                raise ValueError(
                    f"Input data should have three-dimensional shape (batch, "
                    f"stream, channels), got shape {X.shape} instead."
                )
            if not torch.is_tensor(X):
                raise TypeError(
                    f"Input data type should be `torch.Tensor`, got type "
                    f"{type(X)} instead."
                )
            if self.mean_stream > X.shape[1]:
                raise ValueError(
                    f"Parameter mean_stream should be less than stream (second "
                    f"dimension of dataset X): got {self.mean_stream} > "
                    f"{X.shape[1]}."
                )

        elif self.stream_fixed==False:
            if not type(X)==list:
                raise TypeError(
                    f"Input data type should be `list`, got type {type(X)} "
                    f"instead."
                )
            if not torch.is_tensor(X[0]):
                raise TypeError(
                    f"Each time series should be stored as a "
                    f"`torch.Tensor` in a list. Got type {type(X[0])} instead."
                )
            if len(X[0].shape) != 2:
                raise ValueError(
                    f"Input data should have two-dimensional shape  "
                    f"(stream, channels), got shape {X[0].shape} instead."
                )

        if self.weights != None:
            if len(self.weights) != len(X):
                raise ValueError(
                    f"Length of weigths should be equal to len(X), got length "
                    f"{len(self.weights)} instead."
                )
        if self.random_index != None:
            if self.random_index > len(X) or self.random_index < 0:
                raise ValueError(
                    f"Random index for initialization should be less than number "
                    f"of obervations. Got {self.random_index} > {len(X)} instead."
                )

    def _create_cost(self):
        @pymanopt.function.pytorch(self.manifold)
        def cost(ts):
            ts = ts.reshape((1, ts.shape[0], ts.shape[1]))
            Sts = signatory.signature(ts, self.depth)
            if self.weights == None:
                return torch.sum(torch.linalg.norm(self.SX-Sts, dim=1))
            else:
                val = torch.mul(
                    self.weights,
                    torch.linalg.norm(self.SX-Sts, dim=1)
                )
                val = torch.sum(val)
                return val
        return cost


    def _fit_one_init(self, X, rs):
        if self.stream_fixed==True:
            if self.random_index == None:
                # draw observation index randomly
                idx_obs = rs.integers(X.shape[0], size=1)
            else:
                idx_obs = [self.random_index]
            # draw time indices randomly (to obtain subsample of sample)
            idx_stream = np.sort(rs.choice(
                range(1, X.shape[1]-1),
                size=self.mean_stream-2,
                replace=False
            ))
            idx_stream = np.concatenate(([0], idx_stream, [X.shape[1]-1]))
            self.init_ts_ = X[idx_obs][:, idx_stream, :].clone()
            self.init_ts_ = self.init_ts_[0]
        else:
            if self.random_index == None:
                # draw observation index randomly
                idx_obs = rs.integers(len(X), size=1)[0]
            else:
                idx_obs = self.random_index
            # draw time indices randomly (to obtain subsample of sample)
            idx_stream = np.sort(rs.choice(
                range(1, len(X[idx_obs])-1),
                size=self.mean_stream-2,
                replace=False
                ))
            idx_stream = np.concatenate(([0], idx_stream, [len(X[idx_obs])-1]))
            self.init_ts_ = X[idx_obs][idx_stream, :].clone()

        self.manifold = pymanopt.manifolds.Euclidean(self.channels, self.mean_stream)
        cost = self._create_cost()
        problem = pymanopt.Problem(manifold = self.manifold, cost = cost)
        optimizer = pymanopt.optimizers.SteepestDescent(
            max_iterations=self.max_iter,
            verbosity = self.verbose
        )

        self.init_ts_ = self.init_ts_.numpy()
        result = optimizer.run(problem = problem, initial_point = self.init_ts_)
        bary = result.point
        costbary = result.cost
        if costbary < self.cost_value_:  # runs with multiple inits
            self.cost_value_ = costbary
            self.barycenter_ts = torch.from_numpy(bary)
        return self

    def fit(self, X):
        """
        Parameters
        ----------
        X : (batch, stream, channels) array OR list of (stream, channels) array
            Type depends on parameter :attr:`stream_fixed`. Dataset consisting
            of `batch` time series. Each time series has `channels` dimensions
            and `stream` measurements.
        """
        self._check_params(X)
        self.init_ts_ = None
        self.barycenter_ts = None
        self.cost_value_ = np.inf
        rs = np.random.default_rng(self.random_state)
        if self.stream_fixed==True:
            self.SX = signatory.signature(X, self.depth)
            self.channels = X.shape[2]
        else:
            ts_temp = X[0].unsqueeze(0)
            self.SX = signatory.signature(ts_temp, self.depth)
            for i in range(1, len(X)):
                ts_temp = X[i].unsqueeze(0)
                self.SX = torch.cat(
                    (self.SX, signatory.signature(ts_temp, self.depth)), 0)
            self.channels = X[0].shape[1]
        init_idx = 0
        while init_idx < self.n_init:
            self._fit_one_init(X, rs)
            init_idx += 1
        return self
