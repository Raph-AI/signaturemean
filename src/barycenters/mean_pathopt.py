import numpy as np
import signatory
import torch
# Requires PyManOpt from Git (main branch) :
# python3 -m pip install git+https://github.com/pymanopt/pymanopt.git@master
from pymanopt.manifolds import Euclidean
from pymanopt import Problem
from pymanopt.solvers import SteepestDescent
# from pymanopt.solvers import ConjugateGradient
# from pymanopt.solvers import TrustRegions
# from pymanopt.solvers import ParticleSwarm
import pymanopt
import utils

"""

The main function of this script is `mean`.

"""


def cost(path, datasig, depth, stream, channels, groupdist):
    """
    Private function. Thus function is only used inside `mean_one_init_cost`
    function.
    """
    c = 0.
    path = torch.reshape(path, (1, path.shape[0], path.shape[1]))
    pathsig = signatory.signature(path, depth)
    for idx in range(datasig.shape[0]):
        if groupdist == torch.linalg.norm:
            c += torch.linalg.norm(pathsig - datasig[idx])
        else:
            c += groupdist(datasig[idx], pathsig, depth, channels)
    return(c)


def mean_one_init(data, depth, init_path, groupdist, niter):
    """
    Private function. Compute signature mean with path optimization for one
    initialization. This function is only used inside the `mean` function
    below.

    Paramaters
    ----------
    init_path : () torch.Tensor
                Initialization point for the optimization procedure.
    other : Refer to function `mean` below.

    Returns
    -------
    pathbarycenter : (stream, channels) torch.Tensor
                     A path such that its signature is the barycenter of the
                     signatures of paths in data.
    """
    batch, stream, channels = data.shape
    datasig = signatory.signature(data, depth)

    # (1) Instantiate a manifold
    manifold = Euclidean(channels, stream)

    @pymanopt.function.PyTorch
    def mean_one_init_cost(path):
        return(cost(path, datasig, depth, stream, channels, groupdist))

    problem = Problem(manifold=manifold, cost=mean_one_init_cost, verbosity=1)

    # (2) Instantiate a Pymanopt solver
    solver = SteepestDescent(maxiter=niter, logverbosity=0)  # maxiter default = 1000
    # solver = ConjugateGradient(maxiter=300)
    # solver = TrustRegions(maxiter=5)

    # let Pymanopt do the rest
    init = (torch.clone(init_path)).numpy()

    pathbarycenter = solver.solve(problem, x=init)
    pathbarycenter = torch.from_numpy(pathbarycenter)
    # filename = './saved_paths/pathbarycenter_{}.out'.format(np.datetime64('now'))
    # np.savetxt(filename, pathbarycenter.numpy())
    # print("Optimal path saved in {}".format(filename))
    # return(filename)
    return(pathbarycenter)


def mean(data, depth, n_init=1, init_len=10, init_paths=None,
         groupdist=torch.linalg.norm, niter=200):
    """
    Compute signature mean with optimization over path space (=matrix space).

    Parameters
    ----------
    data : (batch, stream, channels) torch.Tensor
           Dataset containing paths over which the mean is computed,
           along the batch axis. NB: data must NOT comprise a channel of
           timestamps.

    depth : int
            Maximum depth of the signature which is used to compute the mean.

    n_init : int
             Number of initializations. It will be paths randomly chosen from
             the data.

    init_len : int (2 < init_len < stream)
               Number of time steps of the initializations. Must be less than
               stream. This is a hyperparameter to tune. The time steps will be
               chosen randomly. Note that the first and last value are kept.

    init_paths : list of (s, channels) torch.Tensor
                 Initialization points (paths) for the optimization method.
                 Caution: the first channel must correspond to timestamps.
                 Caution 2: s can be < `stream`. This is a hyperparameter to
                 tune between 2 and `stream`.

    groupdist : function (default=torch.linalg.norm)
                The distance to use in cost function.

    niter : int
            Number of iterations in the optimization procedure.

    Returns
    -------
    pathbarycenters : a list of (stream, channels) torch.Tensor
                      A list of paths such that each path is obtained after an
                      optimization procedure starting from a
                      different initialization paths.

    Notes
    -----
    Optimization with gradient descent on matrix space using PyManOpt package
    with Torch backend. Gradient is computed with automatic differentation.
    Note that the channel for timestamps is not included in the optimization.

    Requirements
    ------------
    Requires PyManOpt from Git (main branch) :
    python3 -m pip install git+https://github.com/pymanopt/pymanopt.git@master
    """
    batch, stream, channels = data.shape

    # CHECK IF : DATA STARTS AT ZERO
    if not torch.all(torch.abs(data[0,0,:])<1e-3):
        data = utils.datashift(data)

    # INITIALIZATION
    if init_len > stream:
        raise ValueError(f"Parameter init_len ({init_len}) should be less "
                         f"than stream ({stream})"
                        )
    if init_paths is None:
        rng = np.random.default_rng(906)
        if n_init > batch:
            ids = rng.integers(batch, size=n_init)
        else:
            ids = rng.choice(batch, size=n_init, replace=False)
        idstream = np.sort(rng.choice(stream, init_len-2, replace=False))
        # keep first and last value
        idstream = np.concatenate((np.array([0]), idstream, np.array([stream-1])), 0)
        init_paths = []
        for idx in ids:
            path_init0 = data[idx, idstream, :]
            init_paths.append(path_init0)
    else:
        if not isinstance(init_paths, list):
            raise ValueError(
                f"init_paths must be a list of paths (got {type(init_paths)}). If "
                "multistart not desired, pass a list containing only one path"
            )
    # END INITIALIZATION

    pathbarycenters = []
    for i, init1 in enumerate(init_paths):
        m = mean_one_init(data, depth, init1, groupdist, niter)
        # print((f"Multistart: initialization #{i+1} / {len(init_paths)}"
        #        f" terminated."))
        if torch.sum(torch.isnan(m)) == 0:  # keep only clean means
            pathbarycenters.append(m)
    return(pathbarycenters)
