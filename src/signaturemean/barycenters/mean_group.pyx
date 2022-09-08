#cython: language_level=3

cimport cython
import numpy as np
cimport numpy as cnp
from scipy.linalg.cython_blas cimport dger
import torch
import iisignature  # logsigtosig
import signatory  # SignatureToLogSignature
from ..utils import depth_inds
from ..cutils import depth_inds as cdepth_inds
from ..cutils import siginv as csiginv


cpdef cnp.ndarray[double, ndim=1] mean(
    cnp.ndarray[double, ndim=2] datasig,
    unsigned int depth,
    unsigned int channels,
    cnp.ndarray[int, ndim=1] inds
    ):
    r"""
    Compute the signature tensor :math:`m` (called empirical group mean in
    [1, Definition 11]) that verifies

    .. math::
        \sum_{i=1}^N \log (m^{-1}\boxtimes x_i)=0

    where :math:`(x_i)_{i=1,\dots,N}` is a batch of signatures.

    Parameters
    ----------
    datasig: np.array
        Batch of signatures to average.

    depth: int
        Depth of signature.

    channels: int
        Number of space dimensions.

    inds: array like
        Starting indices of each signature tensor. Example: if depth=4 and
        channels=2: [0, 2, 6, 14, 30]. See
        :func:`signaturemean.utils.depth_inds`. Careful: first value has to be
        0.

    Returns
    -------
    m: np.array
        Signature group mean of `datasig`.

    References
    ----------
    [1] Pennec, X. and Lorenzi, M. (2020) ‘Beyond Riemannian geometry’, in
    Riemannian Geometric Statistics in Medical Image Analysis. Elsevier,
    pp. 169–229.
    """
    cdef unsigned int batch = len(datasig)
    cdef:
        cnp.ndarray[double, ndim=1] \
            a = np.empty(inds[-1], dtype=cnp.dtype("d"))
        cnp.ndarray[double, ndim=1] \
            a_aug = np.empty(inds[-1]+1, dtype=cnp.dtype("d"))
        cnp.ndarray[double, ndim=1] \
            m = np.empty(inds[-1], dtype=cnp.dtype("d"))
        cnp.ndarray[double, ndim=2] \
            p = np.empty((batch, inds[-1]), dtype=cnp.dtype("d"))
        cnp.ndarray[double, ndim=2] \
            q = np.empty((batch, inds[-1]), dtype=cnp.dtype("d"))
        cnp.ndarray[double, ndim=3] \
            v = np.zeros((depth, batch, inds[-1]), dtype=cnp.dtype("d"))
        cnp.ndarray[double, ndim=1] \
            p_coeffs = np.empty(depth-1, dtype=cnp.dtype("d"))
        cnp.ndarray[int, ndim=1] \
            dinds = np.empty(depth+1, dtype=np.int32)

    p_coeffs = np.array([np.power(-1, k+1)*1/k for k in range(2, depth+1)])
    dinds = cdepth_inds(depth, channels, scalar=False)

    # CASE K = 1
    p[:, dinds[0]:dinds[1]] = np.zeros((batch, dinds[1]-dinds[0]))
    q[:, dinds[0]:dinds[1]] = np.zeros((batch, dinds[1]-dinds[0]))
    a[dinds[0]:dinds[1]] = -np.mean(datasig[:, :channels], axis=0)

    # CASE K = 2, 3, ...
    cdef int K, i, j, depth1, lenleft, lenright, lenleft2, lenright2
    cdef int inc=1#, inc2=depth
    cdef double one = 1.0
    for K in range(2, depth+1):
        for i in range(batch):
            v[0, i, dinds[K-2]:dinds[K-1]] = q[i, dinds[K-2]:dinds[K-1]] \
                + a[dinds[K-2]:dinds[K-1]] \
                + datasig[i, dinds[K-2]:dinds[K-1]]
            q[i, dinds[K-1]:dinds[K]] = np.zeros(dinds[K]-dinds[K-1])
            p[i, dinds[K-1]:dinds[K]] = np.zeros(dinds[K]-dinds[K-1])
            for j in range(K-1):
                lenright = dinds[K-(j+1)]-dinds[K-(j+2)]
                lenleft = dinds[j+1]-dinds[j]
                dger(
                    &lenright,
                    &lenleft,
                    &one,
                    &datasig[i, dinds[K-(j+2)]],
                    &inc,
                    &a[dinds[j]],
                    &inc,
                    &q[i, dinds[K-1]],
                    &lenright
                )
                for depth1 in range(K-(j+1)):
                    lenright2 = dinds[K-(j+1)] - dinds[K-(j+2)]
                    lenleft2 = dinds[depth1+1] - dinds[depth1]
                    dger(
                        &lenright2,
                        &lenleft2,
                        &one,
                        &v[0, i, dinds[K-(j+2)]],
                        &inc,
                        &v[j, i, dinds[depth1]],
                        &inc,
                        &v[j+1, i, dinds[K-1]],
                        &lenright2
                    )
                p[i, dinds[K-1]:dinds[K]] += p_coeffs[j]*v[j+1, i, dinds[K-1]:dinds[K]]
        a[dinds[K-1]:dinds[K]] = -np.mean(
            datasig[:, dinds[K-1]:dinds[K]] \
            + q[:, dinds[K-1]:dinds[K]] \
            + p[:, dinds[K-1]:dinds[K]],
            axis=0
        )
    cdef cnp.ndarray[int, ndim=1] inds0 = \
        np.empty(depth+2, dtype=np.int32)
    inds0[0] = 0
    inds0[1:] = dinds + 1
    a_aug[0] = 1.
    a_aug[1:] = a
    m = csiginv(a_aug, depth, channels, inds0)[1:]
    return(m)
