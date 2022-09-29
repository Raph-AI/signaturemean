import numpy as np
import torch
# from signaturemean.utils import dist_on_sigs
# from scipy.spatial.distance import cdist
from sklearn.utils import check_random_state
import signatory

# Personal libraries
from signaturemean.clustering.utils import EmptyClusterError
from signaturemean.clustering.utils import _check_no_empty_cluster
from ..cutils import depth_inds
from signaturemean.barycenters import mean_pennec
from signaturemean.barycenters import mean_le
from signaturemean.barycenters import mean_tsoptim
from signaturemean.barycenters import mean_group

"""
(TO DO)

    # 1. support CPU parallelization

    # 2. definir stratégie MiniBatchKMeans

    # 3. definir stratégie KMeans++

    _fit_one_init : _check_full_length
    fit : check_array , _check_initial_guess
"""


class KMeansSignature():
    """
    K-means clustering for time series with the signature.

    Parameters
    ----------
    depth : int
        Depth of signature.

    channels : int
        Number of space dimensions.

    random_state : int
        Seed to use for random numbers generation.

    metric : str
        Metric between samples. NB: currently, only 'euclidean' is supported.

    n_clusters : int (default: 3)
        Number of clusters to form.

    max_iter : int (default: 10)
        Maximum number of iterations of the k-means algorithm for a single run.

    n_init : int (default: 1)
        Number of time the k-means algorithm will be run with different
        centroid seeds. The final results will be the best output of n_init
        consecutive runs in terms of inertia. NB: not yet supported, thus
        n_init should be set to 1.

    init : str (default: "random")
        Method for the initialization of the cluster centers. NB: currently only
        "random" is supported.

    averaging : str (default: "LE")
        Averaging method. Should be one of "LE"
        (corresponding to :func:`signaturemean.barycenters.mean_le`), "pennec"
        (:func:`signaturemean.barycenters.mean_pennec`) or
        "tsoptim" (:func:`signaturemean.barycenters.mean_tsoptim`). Careful: if
        using "tsoptim", the input of `fit()` function should be a dataset of
        paths and not signatures.

    stream_fixed : bool (default: True)
        Relevant only when using `averaging='tsoptim'`.True if number of time
        recordings (stream) is the same for every path observation of the
        dataset.

    minibatch : boolean
        Apply MiniBatch strategy. NB: not yet supported.

    verbose : boolean (default: True)
        Print information about algorithm progress.

    tsoptim_mean_stream : int (default: 10)
        Only when using averaging="tsoptim". Path length (stream) to use for the
        optimization space.

    max_iter_pe : int (default: 20)
        See parameter `max_iter` in
        :func:`signaturemean.barycenters.mean_pennec.mean()`.

    Attributes
    ----------
    labels_ : numpy.ndarray
        Resulting cluster index for each observation.

    Example
    -------
    >>> from signaturemean.clustering.kmeans import KMeansSignature
    >>> batch = 5     # number of time series
    >>> stream = 30   # number of timestamps for each time series
    >>> channels = 3  # number of dimensions
    >>> depth = 4     # depth (order) of truncation of the signature
    >>> X = torch.rand(batch, stream, channels)   # simulate random numbers
    >>> SX = signatory.signature(X, depth=depth)

    >>> # log euclidean barycenter
    >>> km1 = KMeansSignature(n_clusters=2, max_iter=2, depth=depth,
                              channels=channels, random_state=1708,
                              metric='euclidean', averaging='LE')
    >>> km1.fit(SX)
    >>> print(km1.labels_)
    """
    def __init__(self,
                 depth,
                 channels,
                 random_state,
                 metric,
                 n_clusters=3,
                 max_iter=10,
                 n_init=1,
                 init='random',
                 averaging='LE',
                 stream_fixed=True,
                 minibatch=False,
                 verbose=False,
                 tsoptim_mean_stream=10,
                 max_iter_pe=20):
        self.depth = depth
        self.channels = channels
        self.random_state = random_state
        self.metric = metric
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.n_init = n_init
        self.init = init
        self.averaging = averaging
        self.stream_fixed = stream_fixed
        self.minibatch = minibatch
        self.verbose = verbose
        self.tsoptim_mean_stream = tsoptim_mean_stream
        self.max_iter_pe = max_iter_pe

    def _check_params(self, SX):
        # n_init
        if self.n_init <= 0:
            raise ValueError(f"n_init should be > 0, got {self.n_init} instead.")
        self._n_init = self.n_init

        # max_iter
        if self.max_iter <= 0:
            raise ValueError(f"max_iter should be > 0, got {self.max_iter} instead.")

        # n_clusters
        if self.batch_ < self.n_clusters:
            raise ValueError(
                f"The number of samples should be greater than the number of "
                f"clusters. Got {self.batch_} samples and "
                f"`n_clusters`={self.n_clusters}."
            )

        list_methods = ['LE', 'pennec', 'tsoptim', 'group']
        if not self.averaging in list_methods:
            raise ValueError(
                f"Parameter `averaging` should be one of {list_methods}, got "
                f"'{self.averaging}' instead."
            )

        if self.averaging in ['LE', 'pennec', 'group']:
            if len(SX.shape) != 2:
                raise ValueError(
                    f"fit input should have shape (batch, sig_coeffs), got"
                    f"{SX.shape} instead."
                )
        if self.averaging=='tsoptim':
            if self.stream_fixed==True:
                if len(SX.shape) != 3:
                    raise ValueError(
                        f"Input data should have three-dimensional shape (batch, "
                        f"stream, channels), got shape {SX.shape} instead."
                    )
            elif self.stream_fixed==False and len(SX[0].shape) != 2:
                raise ValueError(
                    f"Input list should have elements with two-dimensional "
                    f"shape (stream, channels). Got element with shape "
                    f"{SX[0].shape} instead."
                )
        if self.metric not in ['euclidean']:
            raise ValueError(
                f"Parameter metric should be 'euclidean', got '{self.metric}' "
                "instead."
            )

    def _fit_one_init(self, SX, rs):
        if self.init == "random":
            indices = rs.choice(self.batch_, self.n_clusters, replace=False)
            if self.averaging in ['LE', 'pennec', 'group']:
                self.cluster_centers_ = SX[indices].clone()
            elif self.averaging == 'tsoptim':
                if self.stream_fixed == False:
                    self.cluster_centers_ = torch.empty((self.n_clusters, self.tsoptim_mean_stream, self.channels))
                    for j, i in enumerate(indices):
                        obs = SX[i]
                        idx_stream = np.sort(rs.choice(
                            range(1, len(obs)-1),
                            size=self.tsoptim_mean_stream-2,
                            replace=False
                            ))
                        idx_stream = np.concatenate(([0], idx_stream, [len(obs)-1]))
                        self.cluster_centers_[j] = obs[idx_stream, :].clone()
                else:
                    idx_stream = np.sort(rs.choice(
                        range(1, len(SX[0])-1),
                        size=self.tsoptim_mean_stream-2,
                        replace=False
                        ))
                    idx_stream = np.concatenate(
                        (np.array([0]), idx_stream, np.array([SX.shape[1]-1])),
                        0
                    )
                    self.cluster_centers_ = SX[indices].clone()
                    self.cluster_centers_ = self.cluster_centers_[:, idx_stream, :]
        else:
            raise ValueError(
                f"Value for parameter 'init' is invalid : got '{self.init}'."
            )
        if self.verbose:
            print("kmeans iteration #0")
            print("cluster_centers : ")
            print(self.cluster_centers_)
        # self.cluster_centers_ = _check_full_length(self.cluster_centers_)  # should do this (check if NaNs)
        for it in range(self.max_iter):
            if self.verbose:
                print("")
                print(f"iteration #{it+1}")
            self._assign(SX)
            self._update_centroids(SX, rs)
            if self.verbose:
                print(f"clusters : {self.labels_}")
                print("cluster_centers : ")
                print(self.cluster_centers_)
        return self

    def _transform(self, SX):
        """
        Compute distance `metric` between each signature observation and every
        signature cluster centers.
        """
        if self.verbose:
            print("kmeans transform step")
        if self.metric == "euclidean":
            if self.averaging == 'tsoptim':
                # return torch.cdist(
                #     SX.reshape((SX.shape[0], -1)),
                #     self.cluster_centers_.reshape((self.n_clusters, -1)),
                #     p=2.0
                # )
                self.Scluster_centers_ = signatory.signature(
                    self.cluster_centers_,
                    depth=self.depth
                )
                self.SSX = self.SSX.double()
                self.Scluster_centers_ = self.Scluster_centers_.double()
                return torch.cdist(self.SSX, self.Scluster_centers_, p=2.0)
            else:
                # #########################################################################################
                # ssx = sigscaling(SX, depth=self.depth, channels=self.channels)
                # centers = sigscaling(self.cluster_centers_, depth=self.depth, channels=self.channels)
                # return torch.cdist(ssx, centers, p=2.0)
                # #########################################################################################
                return torch.cdist(SX, self.cluster_centers_, p=2.0)


    def _assign(self, SX):
        """
        Assign each observation to the cluster with nearest center.
        """
        if self.verbose:
            print("kmeans assign step")
        dists = self._transform(SX)
        if self.verbose:
            print(f"_assign : dists = \n{dists}")
        matched_labels = dists.argmin(dim=1)
        self.labels_ = matched_labels
        _check_no_empty_cluster(self.labels_, self.n_clusters)
        return matched_labels

    def _update_centroids(self, SX, rs):
        if self.verbose:
            print("kmeans update centroids step")
        for k in range(self.n_clusters):
            if self.averaging == 'LE':
                self.cluster_centers_[k] = mean_le.mean(
                    SX[self.labels_ == k],
                    depth=self.depth,
                    channels=self.channels
                )
            elif self.averaging == 'pennec':
                self.cluster_centers_[k] = mean_pennec.mean(
                    SX[self.labels_ == k],
                    depth=self.depth,
                    channels=self.channels,
                    max_iter=self.max_iter_pe
                )
            # elif self.averaging == 'group':  # for cython version
            #     m = mean_group.mean(
            #         self.SXnp[self.labels_ == k],
            #         depth=self.depth,
            #         channels=self.channels,
            #         inds=self.dinds
            #     )
            #     self.cluster_centers_[k] = torch.from_numpy(m)
            elif self.averaging == 'group':
                self.cluster_centers_[k] = mean_group.mean(
                    SX = SX[self.labels_ == k],
                    depth = self.depth,
                    channels = self.channels
                )
            elif self.averaging == 'tsoptim' and self.stream_fixed==True:
                tsoptim = mean_tsoptim.TSoptim(
                    self.depth,
                    random_state=self.random_state,
                    n_init=1,
                    mean_stream=self.tsoptim_mean_stream,
                    stream_fixed=self.stream_fixed
                )
                tsoptim.fit(SX[self.labels_ == k])
                self.cluster_centers_[k] = tsoptim.barycenter_ts
            elif self.averaging == 'tsoptim' and self.stream_fixed==False:
                tsoptim = mean_tsoptim.TSoptim(
                    self.depth,
                    random_state=self.random_state,
                    n_init=1,
                    mean_stream=self.tsoptim_mean_stream,
                    stream_fixed=self.stream_fixed
                )
                listtofit = []
                for i in range(self.batch_):
                    if self.labels_[i] == k:
                        listtofit.append(SX[i])
                tsoptim.fit(listtofit)
                self.cluster_centers_[k] = tsoptim.barycenter_ts

    def fit(self, SX):
        """
        Compute K-means algorithm.

        Parameters
        ----------
        SX : (batch, channels**1 + ... + channels**depth) torch.Tensor
            Array of signatures.
        """
        # self.times_pe = np.array([0., 0., 0., 0., 0.])
        self.batch_ = len(SX)
        self._check_params(SX)
        self.labels_ = None
        self.cluster_centers_ = None
        self.Scluster_centers_ = None
        self.n_iter_ = 0
        # if self.averaging == 'group':
        #     self.dinds = depth_inds(self.depth, self.channels, False)
        #     if type(SX) == torch.Tensor:
        #         self.SXnp = SX.numpy()
        #     elif type(SX) == np.ndarray:
        #         self.SXnp = SX.copy()
        #         SX = torch.from_numpy(SX)
        #     self.SXnp = (self.SXnp).astype('double')
        if self.averaging == 'tsoptim' and self.stream_fixed == True:
            self.SSX = signatory.signature(SX, depth=self.depth)
        elif self.averaging == 'tsoptim' and self.stream_fixed == False:
            ts_temp = SX[0].unsqueeze(0)
            self.SSX = signatory.signature(ts_temp, self.depth)
            for i in range(len(SX)):
                ts_temp = SX[i].unsqueeze(0)
                self.SSX = torch.cat(
                    (self.SSX, signatory.signature(ts_temp, self.depth)),
                    dim=0
                    )
        rs = check_random_state(self.random_state)
        # _check_initial_guess(self.init, self.n_clusters)
        init_idx = 0
        while init_idx < self.n_init:
            try:
                self._fit_one_init(SX, rs)
                init_idx += 1
            except EmptyClusterError:
                if self.verbose:
                    print("Interrupted because of empty cluster")
        return self


# def datascaling(data):
#     "Each observation is set to have total variation norm equals to 1."
#     batch, stream, channels = data.shape
#     print(f"stream = {stream}")
#     for i in range(batch):
#         tvnorm = np.array([(data[i, k, :] - data[i, k-1, :]).numpy() for k in range(1, stream)])
#         tvnorm = np.sum(tvnorm, axis=0)
#         tvnorm = np.linalg.norm(tvnorm)
#         print(f"tvnorm obs #{i} = {tvnorm}")
#         data[i] = data[i]/tvnorm
#         print(data[i])
#     return(data)


# def kmeans(n_clusters, signatures, depth, channels, max_iterations, seed,
#            mean_strat, dist, verbose=False, minibatch=False):
#     """
#     Perform k-means on a dataset of signatures.
#
#     Parameters
#     ----------
#     max_iterations : int
#         number of iterations before k-means stops
#     dist : string
#         Distance to use. One of 'euclidean', 'sigdist'.
#     mean_strat : string
#         Strategy to use for averaging. One of 'PE', 'LE'.
#
#     Notes
#     -----
#     Cluster associated to each observation for every k-means iteration is
#     written in the logs folder.
#     """
#     batch, siglen = signatures.shape
#     y_pred = -1*np.ones(batch, dtype='int')  # corresponding cluster index for each obs
#
#     # INITIALIZATION
#     # choose k obs randomly to start from
#     rng = np.random.default_rng(seed)
#     idx_init_centroid = rng.integers(batch, size=n_clusters)  # TO DO : replace with rng.choice
#     # # not random initialization
#     # idx_init_centroid = [7, 2, 1, 15, 3, 8, 5, 14, 0, 10]
#     centroids = signatures[idx_init_centroid]
#
#     n_iter = 0
#     while n_iter < max_iterations:
#         # if n_iter%2==0:
#         print(f"iteration #{n_iter}")
#
#         # 1. ASSIGN EACH OBS TO CLUSTER W/ NEAREST CENTROID
#         idx_obs = 0
#         while idx_obs < batch:
#             # print(f'{idx_obs} / {batch}')
#             sig = signatures[idx_obs]
#             id_cluster = 0
#             if dist=='euclidean':
#                 dist_sigc = torch.linalg.norm(sig-centroids[0])
#                 i = 0
#                 while i<n_clusters:
#                     dist_sigc_i = torch.linalg.norm(sig-centroids[i])
#                     if dist_sigc_i < dist_sigc:
#                         id_cluster = i
#                         dist_sigc = dist_sigc_i
#                     i += 1
#             # elif dist=='sigdist':
#             #     distlist = [utils.dist_on_sigs(sig, centroids[i], depth, channels) for i in range(k)]  # SIG DISTANCE
#             y_pred[idx_obs] = id_cluster
#             idx_obs += 1
#
#         # 2. UPDATE MEAN VALUES TO MEAN OF CLUSTER
#         i = 0
#         while i < n_clusters:
#             to_average = signatures[y_pred==i, :]  # select obs of cluster i
#             if not len(to_average)==0:
#                 if mean_strat=='PE':
#                     centroids[i] = mean_pennec.mean(to_average, depth, channels)
#                 elif mean_strat=='LE':
#                     centroids[i] = mean_le.mean(to_average, depth, channels)
#                 else:
#                     raise ValueError(f"Chosen mean strategy '{mean_strat}' do NOT exist.")
#             i += 1
#         if verbose:
#             # print(f"SUM  = {sum1}")
#             print(f"first cluster indices : {y_pred[:15]}")
#             print(f"mean values")
#             print((centroids[0])[14:24])
#             print((centroids[3])[14:24])
#             print((centroids[6])[14:24])
#         n_iter += 1
#     return(y_pred, centroids)


# def kmeansInertia(SX, y, centroids, dist, channels=None, sigdepth=None):
#     """
#     SX : signatures of the data
#     y : y_i is the cluster in which X_i has been assigned
#     centroids : center of each cluster
#     """
#     nclasses = len(centroids)
#     intracluster_inertia = {}
#     for key in range(nclasses):
#         intracluster_inertia[key] = 0.

#     for idx_obs, obs in enumerate(SX):
#         idx_cluster = y[idx_obs]
#         if dist == 'sigdist':
#             d_curr = utils.dist_on_sigs(obs, centroids[idx_cluster], sigdepth, channels)
#         intracluster_inertia[idx_cluster] += d_curr
#     inertia = sum(intracluster_inertia.values())
#     return(inertia, intracluster_inertia)
