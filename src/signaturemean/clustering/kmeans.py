import numpy as np
import torch
from signaturemean.barycenters import mean_pennec
from signaturemean.barycenters import mean_le
from signaturemean.barycenters import mean_pathopt
# from signaturemean.utils import dist_on_sigs
# from scipy.spatial.distance import cdist
from sklearn.utils import check_random_state
from signaturemean.clustering.utils import EmptyClusterError
from signaturemean.clustering.utils import _check_no_empty_cluster


"""
# (TO DO)

# S'inspirer de la classe `tslearn.clustering.TimeSeriesKMeans`

class KMeansSig():
    # definir KMeans pour stratégies signature :
    #   * LE (log euclidean)
    #   * PE (pennec)
    #   * PO (path optimization)

    # 1. support CPU parallelization

    # 2. definir stratégie MiniBatchKMeans

    # 3. definir stratégie KMeans++
# (END TO DO)
"""


class KMeansSignature():
    """
    K-means clustering for time series with the signature.

    Parameters
    ----------
    n_clusters : int (default: 3)
        Number of clusters to form.

    max_iter : int (default: 10)
        Maximum number of iterations of the k-means algorithm for a single run.

    seed : int
        Determines random number generation for centroid initialization.

    minibatch : boolean
        Apply MiniBatch strategy.

    pathlen_pe : int
        Only when using averaging='PE'. Path length (stream) to use for the
        optimization space.

    Attributes
    ----------
    labels_ : numpy.ndarray
        Label of each point.
    """

    """
    (TO DO)
    _fit_one_init : _check_full_length
    fit : check_array , _check_initial_guess
    """

    def __init__(self, depth, channels, random_state,metric, n_clusters=3, max_iter=10, n_init=1,
                 init='random', averaging='LE',
                 minibatch=False, verbose=False, pathlen_pe=10):
        self.depth = depth
        self.channels = channels
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.n_init = n_init
        self.init = init
        self.averaging = averaging
        self.metric = metric
        self.random_state = random_state
        self.minibatch = minibatch
        self.verbose = verbose
        self.pathlen_pe = pathlen_pe


    def _check_params(self, SX):
        # n_init
        if self.n_init <= 0:
            raise ValueError(f"n_init should be > 0, got {self.n_init} instead.")
        self._n_init = self.n_init

        # max_iter
        if self.max_iter <= 0:
            raise ValueError(f"max_iter should be > 0, got {self.max_iter} instead.")

        # n_clusters
        if SX.shape[0] < self.n_clusters:
            raise ValueError(
                f"n_samples={SX.shape[0]} should be >= n_clusters={self.n_clusters}."
            )

        if self.averaging=='LE' or self.averaging=='PE':
            if len(SX.shape) != 2:
                raise ValueError(
                    f"SX input shape is {SX.shape}, should have two dimensions : batch "
                    "and signature coefficients"
                )
        if self.averaging=='PO':
            if len(SX.shape) != 3:
                raise ValueError(
                    f"SX input shape is {SX.shape}, should have three dimensions : batch, "
                    "stream and channels"
                )


    def _fit_one_init(self, SX, rs):
        if self.init == "random":
            indices = rs.choice(SX.shape[0], self.n_clusters, replace=False)
            if self.averaging == 'LE' or self.averaging == 'PE':
                self.cluster_centers_ = SX[indices].clone()
            elif self.averaging == 'PO':
                stream_inds = np.sort(rs.choice(SX.shape[1]-2, self.pathlen_pe-2, replace=False))+1
                # keep first and last value
                stream_inds = np.concatenate((np.array([0]), stream_inds, np.array([SX.shape[1]-1])), 0)
                self.cluster_centers_ = SX[indices].clone()
                self.cluster_centers_ = self.cluster_centers_[:, stream_inds, :]

        else:
            raise ValueError(
                f"Value for parameter 'init' is invalid : got {self.init}"
            )
        # self.cluster_centers_ = _check_full_length(self.cluster_centers_)  # should do this (check if NaNs)

        for it in range(self.max_iter):
            self._assign(SX)
            self._update_centroids(SX)
            if self.verbose:
                print(f"iteration #{it} completed")
        return self


    def _transform(self, SX):
        if self.metric == "euclidean":
            if self.averaging == 'PO':
                return torch.cdist(SX.reshape((SX.shape[0], -1)),
                                   self.cluster_centers_.reshape((self.n_clusters, -1)),
                                   p=2.0)
            else:
                return torch.cdist(SX, self.cluster_centers_, p=2.0)
        else:
            raise ValueError(
                f"Incorrect metric : {self.metric} (should be one of 'euclidean')"
            )


    def _assign(self, SX):
        dists = self._transform(SX)
        # print(f"_assign : dists = \n{dists}")
        matched_labels = dists.argmin(dim=1)
        self.labels_ = matched_labels
        _check_no_empty_cluster(self.labels_, self.n_clusters)
        return matched_labels


    def _update_centroids(self, SX):
        for k in range(self.n_clusters):
            if self.averaging == 'LE':
                self.cluster_centers_[k] = mean_le.mean(
                    SX[self.labels_ == k],
                    depth=self.depth,
                    channels=self.channels)
            elif self.averaging == 'PE':
                self.cluster_centers_[k] = mean_pennec.mean(
                    SX[self.labels_ == k],
                    depth=self.depth,
                    channels=self.channels)
            elif self.averaging == 'PO':
                self.cluster_centers_[k] = mean_pathopt.mean(
                    SX[self.labels_ == k],
                    depth=self.depth,
                    n_init=1,
                    init_len=self.pathlen_pe)[0]  # (TO DO) output mean_pathopt.mean to change (list)
            else:
                raise ValueError(
                    f"Incorrect averaging method : {self.averaging} (should be one of 'LE', 'PE')."
                )


    def fit(self, SX):
        """
        Compute K-means algorithm.

        Parameters
        ----------
        SX : (batch, channels**1 + ... + channels**depth) torch.tensor
            Array of signatures.
        """
        self._check_params(SX)
        self.labels_ = None
        self.cluster_centers_ = None
        self.n_iter_ = 0
        max_attempts = 1  # to change later
        rs = check_random_state(self.random_state)
        # _check_initial_guess(self.init, self.n_clusters)
        n_attempts = 0
        while n_attempts < max_attempts:
            try:
                self._fit_one_init(SX, rs)
                n_attempts += 1
            except EmptyClusterError:
                if self.verbose:
                    print("Interrupted because of empty cluster")
        return self


def datascaling(data):
    "Each observation is set to have total variation norm equals to 1."
    batch, stream, channels = data.shape
    print(f"stream = {stream}")
    for i in range(batch):
        tvnorm = np.array([(data[i, k, :] - data[i, k-1, :]).numpy() for k in range(1, stream)])
        tvnorm = np.sum(tvnorm, axis=0)
        tvnorm = np.linalg.norm(tvnorm)
        print(f"tvnorm obs #{i} = {tvnorm}")
        data[i] = data[i]/tvnorm
        print(data[i])
    return(data)


def kmeans(n_clusters, signatures, depth, channels, max_iterations, seed,
           mean_strat, dist, verbose=False, minibatch=False):
    """
    Perform k-means on a dataset of signatures.

    Parameters
    ----------
    max_iterations : int
        number of iterations before k-means stops
    dist : string
        Distance to use. One of 'euclidean', 'sigdist'.
    mean_strat : string
        Strategy to use for averaging. One of 'PE', 'LE'.

    Notes
    -----
    Cluster associated to each observation for every k-means iteration is
    written in the logs folder.
    """
    batch, siglen = signatures.shape
    y_pred = -1*np.ones(batch, dtype='int')  # corresponding cluster index for each obs

    # INITIALIZATION
    # choose k obs randomly to start from
    rng = np.random.default_rng(seed)
    idx_init_centroid = rng.integers(batch, size=n_clusters)  # TO DO : replace with rng.choice
    # # not random initialization
    # idx_init_centroid = [7, 2, 1, 15, 3, 8, 5, 14, 0, 10]
    centroids = signatures[idx_init_centroid]

    n_iter = 0
    while n_iter < max_iterations:
        # if n_iter%2==0:
        print(f"iteration #{n_iter}")

        # 1. ASSIGN EACH OBS TO CLUSTER W/ NEAREST CENTROID
        idx_obs = 0
        while idx_obs < batch:
            # print(f'{idx_obs} / {batch}')
            sig = signatures[idx_obs]
            id_cluster = 0
            if dist=='euclidean':
                dist_sigc = torch.linalg.norm(sig-centroids[0])
                i = 0
                while i<n_clusters:
                    dist_sigc_i = torch.linalg.norm(sig-centroids[i])
                    if dist_sigc_i < dist_sigc:
                        id_cluster = i
                        dist_sigc = dist_sigc_i
                    i += 1
            # elif dist=='sigdist':
            #     distlist = [utils.dist_on_sigs(sig, centroids[i], depth, channels) for i in range(k)]  # SIG DISTANCE
            y_pred[idx_obs] = id_cluster
            idx_obs += 1

        # 2. UPDATE MEAN VALUES TO MEAN OF CLUSTER
        i = 0
        while i < n_clusters:
            to_average = signatures[y_pred==i, :]  # select obs of cluster i
            if not len(to_average)==0:
                if mean_strat=='PE':
                    centroids[i] = mean_pennec.mean(to_average, depth, channels)
                elif mean_strat=='LE':
                    centroids[i] = mean_le.mean(to_average, depth, channels)
                else:
                    raise ValueError(f"Chosen mean strategy '{mean_strat}' do NOT exist.")
            i += 1
        if verbose:
            # print(f"SUM  = {sum1}")
            print(f"first cluster indices : {y_pred[:15]}")
            print(f"mean values")
            print((centroids[0])[14:24])
            print((centroids[3])[14:24])
            print((centroids[6])[14:24])
        n_iter += 1
    return(y_pred, centroids)


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
