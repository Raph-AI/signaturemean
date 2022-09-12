import numpy as np
import iisignature  # logsig to sig
import torch
import signatory
# from ..utils import sigprod
# from ..utils import siginv
from ..cutils import sigprod as csigprod
from ..cutils import siginv as csiginv
from ..cutils import depth_inds as cdepth_inds


def mean(datasig, depth, channels, max_iter=20):
    """
    Compute the Group Exponential Mean from a dataset of signatures (see [1]),
    i.e. apply the following procedure:

    .. math::
        m_{(k+1)} = m_{(k)} \otimes \mathrm{Exp}\ \Bigg( \frac1n \sum_{i=1}^n \mathrm{Log}\ (m_{(k)}^{-1}\otimes \mathbb{X}_i)\Bigg)

    where :math:`m` is the mean that is returned, :math:`\mathrm{Exp}` and
    :math:`\mathrm{Log}` are the exponential and logarithm map on the Lie group
    of signatures.

    Parameters
    ----------
    datasig : (batch, stream, channels) torch.Tensor
        Dataset containing signatures over which the mean is computed,
        along the batch axis. For instance, an output of `signatory.signature` .

    depth : int
        Maximum depth of the signature which is used to compute the mean.

    max_iter : int (default=20)
        Number of iterations until the procedure defined as in the Equation
        above stops. The greater `channels` is, the greater `max_iter` must be
        set. Caution: if `max_iter` is too small, different runs might provides
        different means (due to randomness of the initialization).

    Returns
    -------
    sigbarycenter : (channels**1 + ... + channels**depth) torch.Tensor
        A signature which is the barycenter of the signatures in
        datasig.

    References
    ----------
    [1] Pennec, Xavier, and Vincent Arsigny. 2013. “Exponential Barycenters of the
    Canonical Cartan Connection and Invariant Means on Lie Groups.” In Matrix
    Information Geometry, edited by Frank Nielsen and Rajendra Bhatia, 123–66.
    Berlin, Heidelberg: Springer Berlin Heidelberg.
    https://doi.org/10.1007/978-3-642-30232-9_7
    """
    batch = datasig.shape[0]
    # lensig1 = datasig.shape[1]+1
    idx_rd = np.random.randint(batch)
    sigbarycenter = datasig[idx_rd]  # random initialization
    stoLogS = signatory.SignatureToLogSignature(
        channels, depth, mode='brackets'
        )
    # NB: brackets (= lyndon basis), because we then need LogSigtoSig
    logStoS = iisignature.prepare(channels, depth, "S2")
    # NB: S2 (= lyndon basis), corresponds to 'brackets' in Signatory
    datasig = datasig.numpy()
    sigbarycenter = sigbarycenter.numpy()
    sigbarycenter1 = np.concatenate(([1.], sigbarycenter))
    inds0 = cdepth_inds(depth, channels, scalar=True)
    n_iter = 0

    while n_iter < max_iter:
        inv_sigbarycenter = csiginv(
            sigbarycenter1, depth, channels, inds0
        )
        # inv_sigbarycenter1 = np.concatenate(([1.], inv_sigbarycenter)) # !!!
        mean_logsSX = 0.
        for idx in range(batch):
            obs = datasig[idx]
            obs1 = np.concatenate(([1.], obs))
            if len(inv_sigbarycenter) != len(obs1):
                raise ValueError(
                    f"Product between signatures of different sizes: got "
                    f"{len(inv_sigbarycenter)} and {len(obs1)}."
                )
            prod = csigprod(inv_sigbarycenter, obs1, depth, channels, inds0)
            prod = (torch.from_numpy(prod[1:])).unsqueeze(0)
            logSX = stoLogS.forward(prod)
            mean_logsSX += logSX
        mean_logsSX = mean_logsSX*1./batch
        exp_sigbarycenter = iisignature.logsigtosig(mean_logsSX[0], logStoS)
        exp_sigbarycenter1 = np.concatenate(([1.], exp_sigbarycenter))
        sigbarycenter_new = csigprod(
            sigbarycenter1, exp_sigbarycenter1, depth, channels, inds0
        )
        sigbarycenter1 = sigbarycenter_new
        n_iter += 1
    return torch.from_numpy(sigbarycenter1[1:])  # omit scalar value


# def mean(datasig, depth, channels, max_iter_pe=5):
#     """
#     Compute the Group Exponential Mean from a dataset of signatures.
#
#     Parameters
#     ----------
#     datasig : (batch, stream, channels) torch.Tensor
#         Dataset containing signatures over which the mean is computed,
#         along the batch axis.
#
#     depth : int
#         Maximum depth of the signature which is used to compute the mean.
#
#     max_iter_pe : int (default=5)
#         Number of iterations until the algorithm stops. Usually, requires
#         less than 10 iterations to converge to a solution.
#
#     Returns
#     -------
#     sigbarycenter : (channels**1 + ... + channels**depth) torch.Tensor
#         A signature which is the barycenter of the signatures in
#         datasig.
#
#     References
#     ----------
#     Pennec, Xavier, and Vincent Arsigny. 2013. “Exponential Barycenters of the
#     Canonical Cartan Connection and Invariant Means on Lie Groups.” In Matrix
#     Information Geometry, edited by Frank Nielsen and Rajendra Bhatia, 123–66.
#     Berlin, Heidelberg: Springer Berlin Heidelberg.
#     https://doi.org/10.1007/978-3-642-30232-9_7.
#     """
#     timesiginv = 0.
#     timesigprods = 0.
#     timestologs = 0.
#     timelogstos = 0.
#     totaltime = 0.
#     ts1 = time.time()
#     batch = datasig.shape[0]
#     idx_rd = np.random.randint(batch)
#     sigbarycenter = datasig[idx_rd]  # random initialization
#     # sigbarycenters = sigbarycenter.unsqueeze(0)
#     stoLogS = signatory.SignatureToLogSignature(
#         channels, depth, mode='brackets')
#     # brackets (= lyndon basis), because we then need LogSigtoSig
#     logStoS = iisignature.prepare(channels, depth, "S2")
#     # S2 (= lyndon basis), corresponds to 'brackets' in Signatory
#     n_iter = 0
#     while n_iter < max_iter_pe:
#         ts = time.time()
#         inv_sigbarycenter = siginv(sigbarycenter, depth, channels)
#         timesiginv += time.time() - ts
#         mean_logsSX = 0.
#         for idx in range(batch):
#             ts = time.time()
#             prod = sigprod(inv_sigbarycenter, datasig[idx], depth, channels)
#             timesigprods += time.time() - ts
#             prod = prod.unsqueeze(0)
#             ts = time.time()
#             logSX = stoLogS.forward(prod)
#             timestologs += time.time() - ts
#             mean_logsSX += logSX
#         mean_logsSX = mean_logsSX*1./batch
#         ts = time.time()
#         exp_sigbarycenter = torch.tensor(iisignature.logsigtosig(
#                                          mean_logsSX[0], logStoS))
#         timelogstos += time.time()-ts
#         ts = time.time()
#         sigbarycenter_new = sigprod(sigbarycenter, exp_sigbarycenter, depth,
#                                     channels)
#         timesigprods += time.time() - ts
#         sigbarycenter = sigbarycenter_new
#         n_iter += 1
#     totaltime = time.time()-ts1
#     # print(f"time sig inv: {timesiginv:.2f} sec")
#     # print(f"time sig prods: {timesigprods:.2f} sec")
#     # print(f"time stologs: {timestologs:.2f} sec")
#     # print(f"time logstos: {timelogstos:.2f} sec")
#     # print(f"total time : {totaltime:.2f} sec")
#     times = [timesiginv, timesigprods, timestologs, timelogstos, totaltime]
#     return sigbarycenter, times


# def mean_from_paths(data, depth, max_iter_pe=5):
#     """
#     (TO DO) : make consistent with `mean_pennec.mean`.
#     Compute the Group Exponential Mean from a dataset of paths.
#
#     Parameters
#     ----------
#     data : (batch, stream, channels) torch.Tensor
#            Dataset containing paths over which the mean is computed,
#            along the batch axis.
#
#     depth : int
#             Maximum depth of the signature which is used to compute the mean.
#
#     max_iter_pe : int (default=5)
#              Number of iterations until the algorithm stops. Usually, requires
#              less than 10 iterations to converge to a solution.
#
#     Returns
#     -------
#     sigbarycenter : (channels**1 + ... + channels**depth) torch.Tensor
#                     A signature which is the barycenter of the signatures of
#                     paths in dataset.
#
#     References
#     ----------
#     Pennec, Xavier, and Vincent Arsigny. 2013. “Exponential Barycenters of the
#     Canonical Cartan Connection and Invariant Means on Lie Groups.” In Matrix
#     Information Geometry, edited by Frank Nielsen and Rajendra Bhatia, 123–66.
#     Berlin, Heidelberg: Springer Berlin Heidelberg.
#     https://doi.org/10.1007/978-3-642-30232-9_7.
#     """
#     n_obs = data.shape[0]
#     channels = data.shape[2]
#     data_sig = signatory.signature(data, depth)
#     idx_rd = np.random.randint(n_obs)
#     sigbarycenter = data_sig[idx_rd]        # random initialization
#     sigbarycenters = sigbarycenter.unsqueeze(0)
#     # median = sigbarycenter
#     # medians = sigbarycenters
#     n_iter = 0
#     while n_iter < max_iter_pe:
#         inv_sigbarycenter = siginv(sigbarycenter, depth, channels)
#         # inv_med = siginv(median, depth, channels)
#         m_logs, logs = 0., []
#         for idx in range(n_obs):
#             prod = sigprod(inv_sigbarycenter, data_sig[idx], depth, channels)
#             prod = prod.unsqueeze(0)
#             log = signatory.signature_to_logsignature(
#                   prod, channels, depth, mode="brackets")  # lyndon basis
#             # log = signatory.SignatureToLogSignature()  # using signatory
#             logs.append((log[0]).numpy())
#             m_logs += log
#         m_logs = m_logs*1./n_obs
#         logs_np = logs[0]
#         for obs in range(1, n_obs):
#             logs_np = np.vstack((logs_np, logs[obs]))
#         logs_np = torch.tensor(np.median(logs_np, axis=0), dtype=torch.float64)
#         s = iisignature.prepare(channels, depth, "S2")  # lyndon basis
#         exp_sigbarycenter = torch.tensor(iisignature.logsigtosig(m_logs[0], s))
#         sigbarycenter_new = sigprod(sigbarycenter, exp_sigbarycenter, depth,
#                                     channels)
#         # exp_median = torch.tensor(iisignature.logsigtosig(logs_np, s))
#         # median_new = sigprod(median, exp_median, depth, channels)
#         iter += 1
#         sigbarycenter = sigbarycenter_new
#         # median = median_new
#     return(sigbarycenter)
