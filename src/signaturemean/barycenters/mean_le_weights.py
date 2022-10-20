import iisignature  # logsig to sig in Lyndon basis
import torch
import signatory
from signaturemean.utils import sigscaling


def mean(datasig, depth, channels, weights=None):
    """
    Compute the Log Euclidean mean from a dataset of signatures, that is

    .. math::
        \mathrm{Exp}\ \Bigg( \frac1n \sum_{i=1}^n \mathrm{Log}\ \mathbb{X}_i\Bigg)

    where :math:`\mathrm{Exp}` and :math:`\mathrm{Log}` are the exponential and
    logarithm map on the Lie group of signatures and :math:`(X_i)_i` is the
    dataset of signatures to average.

    Parameters
    ----------
    datasig : (batch, stream, channels) torch.Tensor
              dataset containing signatures over which the mean is computed,
              along the batch axis.

    depth : int
            Corresponding depth of the signatures stored in `datasig`.

    Returns
    -------
    sigbarycenter : (channels**1 + ... + channels**depth) torch.Tensor
                    A signature which is the barycenter of the signatures
                    in datasig.
    """
    stoLogS = signatory.SignatureToLogSignature(channels, depth, mode='brackets')
    datalogsig = stoLogS.forward(datasig)

    if weights != None:
        val = torch.mul(weights, datalogsig)
        logbarycenter = torch.sum(val)
    else:
        logbarycenter = torch.mean(datalogsig, dim=0)
    # Logsig to sig is not implemented in SIGNATORY
    # thus we use IISIGNATURE
    s_lyndon = iisignature.prepare(channels, depth, "S2")
    sigbarycenter = iisignature.logsigtosig(logbarycenter, s_lyndon)
    sigbarycenter = torch.from_numpy(sigbarycenter)
    return(sigbarycenter)


def mean_from_paths(data, depth, rescaling=False):
    """
    Compute the Log Euclidean mean from a dataset of paths.

    Parameters
    ----------
    data : (batch, stream, channels) torch.Tensor
           dataset containing paths over which the mean is computed,
           along the batch axis.

    depth : int
            max depth of the signature which is used to compute the mean.

    rescaling : boolean (default=False)
                utils.sigscaling is applied on signatures before computing
                the mean.

    Returns
    -------
    sigbarycenter : (channels**1 + ... + channels**depth) torch.Tensor
                    A signature which is the barycenter of the signatures of
                    paths in dataset.
    """
    channels = data.shape[2]
    datasig = signatory.signature(data, depth)
    if rescaling:  # normalization
        datasig = sigscaling(datasig, depth, channels)
    datalogsig = signatory.signature_to_logsignature(
                 datasig, channels, depth, mode="brackets")
    logbarycenter = torch.mean(datalogsig, dim=0)
    # Logsig to sig is not implemented in SIGNATORY
    # thus we use IISIGNATURE
    s_lyndon = iisignature.prepare(channels, depth, "S2")
    sigbarycenter = iisignature.logsigtosig(logbarycenter, s_lyndon)
    sigbarycenter = torch.from_numpy(sigbarycenter)
    # if rescaling:
    #     sigbarycenter = utils.sigscaling_reverse(sigbarycenter, depth, channels)
    return(sigbarycenter)


# def median(data, depth, rescaling=False):
#     """
#     Compute the Log Euclidean median.
#     """
#     channels = data.shape[2]
#     datasig = signatory.signature(data, depth)
#     if rescaling:
#         datasig = utils.sigrescaling(datasig, depth, channels)
#     datalogsig = signatory.signature_to_logsignature(
#         datasig, channels, depth, mode="brackets")
#     # datalogsig = signatory.logsignature(data, depth, mode="brackets")
#     logbarycenter_med, inds = torch.median(datalogsig, dim=0)
#     # Logsig to sig is not implemented in SIGNATORY
#     # thus we use IISIGNATURE
#     s_lyndon = iisignature.prepare(channels, depth, "S2")
#     sigbarycenter_med = iisignature.logsigtosig(logbarycenter_med, s_lyndon)
#     # sigbarycenter_med = torch.tensor(sigbarycenter_med, dtype=torch.float64)
#     sigbarycenter_med = torch.from_numpy(sigbarycenter_med)
#     return(sigbarycenter_med)
