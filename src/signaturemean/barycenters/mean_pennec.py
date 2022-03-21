import numpy as np
import iisignature  # logsig to sig
import torch
import signatory
from signaturemean.utils import sigprod
from signaturemean.utils import siginv


def mean(datasig, depth, channels, max_iter_pe=5):
    """
    Compute the Group Exponential Mean from a dataset of signatures.

    Parameters
    ----------
    datasig : (batch, stream, channels) torch.Tensor
        Dataset containing signatures over which the mean is computed,
        along the batch axis.

    depth : int
        Maximum depth of the signature which is used to compute the mean.

    max_iter_pe : int (default=5)
        Number of iterations until the algorithm stops. Usually, requires
        less than 10 iterations to converge to a solution.

    Returns
    -------
    sigbarycenter : (channels**1 + ... + channels**depth) torch.Tensor
        A signature which is the barycenter of the signatures in
        datasig.

    References
    ----------
    Pennec, Xavier, and Vincent Arsigny. 2013. “Exponential Barycenters of the
    Canonical Cartan Connection and Invariant Means on Lie Groups.” In Matrix
    Information Geometry, edited by Frank Nielsen and Rajendra Bhatia, 123–66.
    Berlin, Heidelberg: Springer Berlin Heidelberg.
    https://doi.org/10.1007/978-3-642-30232-9_7.
    """
    batch = datasig.shape[0]
    idx_rd = np.random.randint(batch)
    sigbarycenter = datasig[idx_rd]  # random initialization
    sigbarycenters = sigbarycenter.unsqueeze(0)
    stoLogS = signatory.SignatureToLogSignature(channels, depth, mode='brackets')
    # brackets (= lyndon basis), because we then need LogSigtoSig
    logStoS = iisignature.prepare(channels, depth, "S2")
    # S2 (= lyndon basis), corresponds to 'brackets' in Signatory
    n_iter = 0
    while n_iter < max_iter_pe:
        inv_sigbarycenter = siginv(sigbarycenter, depth, channels)
        mean_logsSX = 0.
        for idx in range(batch):
            prod = sigprod(inv_sigbarycenter, datasig[idx], depth, channels)
            prod = prod.unsqueeze(0)
            logSX = stoLogS.forward(prod)
            mean_logsSX += logSX
        mean_logsSX = mean_logsSX*1./batch
        exp_sigbarycenter = torch.tensor(iisignature.logsigtosig(
                                         mean_logsSX[0], logStoS))
        sigbarycenter_new = sigprod(sigbarycenter, exp_sigbarycenter, depth,
                                    channels)
        sigbarycenter = sigbarycenter_new
        n_iter += 1
    return(sigbarycenter)


def mean_from_paths(data, depth, max_iter_pe=5):
    """
    (TO DO) : make consistent with `mean_pennec.mean`.
    Compute the Group Exponential Mean from a dataset of paths.

    Parameters
    ----------
    data : (batch, stream, channels) torch.Tensor
           Dataset containing paths over which the mean is computed,
           along the batch axis.

    depth : int
            Maximum depth of the signature which is used to compute the mean.

    max_iter_pe : int (default=5)
             Number of iterations until the algorithm stops. Usually, requires
             less than 10 iterations to converge to a solution.

    Returns
    -------
    sigbarycenter : (channels**1 + ... + channels**depth) torch.Tensor
                    A signature which is the barycenter of the signatures of
                    paths in dataset.

    References
    ----------
    Pennec, Xavier, and Vincent Arsigny. 2013. “Exponential Barycenters of the
    Canonical Cartan Connection and Invariant Means on Lie Groups.” In Matrix
    Information Geometry, edited by Frank Nielsen and Rajendra Bhatia, 123–66.
    Berlin, Heidelberg: Springer Berlin Heidelberg.
    https://doi.org/10.1007/978-3-642-30232-9_7.
    """
    n_obs = data.shape[0]
    channels = data.shape[2]
    data_sig = signatory.signature(data, depth)
    idx_rd = np.random.randint(n_obs)
    sigbarycenter = data_sig[idx_rd]        # random initialization
    sigbarycenters = sigbarycenter.unsqueeze(0)
    # median = sigbarycenter
    # medians = sigbarycenters
    n_iter = 0
    while n_iter < max_iter_pe:
        inv_sigbarycenter = siginv(sigbarycenter, depth, channels)
        # inv_med = siginv(median, depth, channels)
        m_logs, logs = 0., []
        for idx in range(n_obs):
            prod = sigprod(inv_sigbarycenter, data_sig[idx], depth, channels)
            prod = prod.unsqueeze(0)
            log = signatory.signature_to_logsignature(
                  prod, channels, depth, mode="brackets")  # lyndon basis
            # log = signatory.SignatureToLogSignature()  # using signatory
            logs.append((log[0]).numpy())
            m_logs += log
        m_logs = m_logs*1./n_obs
        logs_np = logs[0]
        for obs in range(1, n_obs):
            logs_np = np.vstack((logs_np, logs[obs]))
        logs_np = torch.tensor(np.median(logs_np, axis=0), dtype=torch.float64)
        s = iisignature.prepare(channels, depth, "S2")  # lyndon basis
        exp_sigbarycenter = torch.tensor(iisignature.logsigtosig(m_logs[0], s))
        sigbarycenter_new = sigprod(sigbarycenter, exp_sigbarycenter, depth,
                                    channels)
        # exp_median = torch.tensor(iisignature.logsigtosig(logs_np, s))
        # median_new = sigprod(median, exp_median, depth, channels)
        iter += 1
        sigbarycenter = sigbarycenter_new
        # median = median_new
    return(sigbarycenter)