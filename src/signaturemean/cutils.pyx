#cython: language_level=3

cimport cython
import numpy as np
cimport numpy as cnp
from scipy.linalg.cython_blas cimport dger
from scipy.linalg.cython_blas cimport ddot
# from libc.string cimport memcpy
from libc.math cimport pow
from libc.math cimport sqrt
# from libcpp cimport bool


# cpdef cnp.ndarray[double, ndim=1] sigprod():

def sigprod(cnp.ndarray[double, ndim=1] sigA,
            cnp.ndarray[double, ndim=1] sigB,
            unsigned int depth,
            unsigned int channels,
            cnp.ndarray[int, ndim=1] inds,
            bint check_params = True
           ):
    if check_params:
        if sigA[0] != 1. or sigB[0] != 1.:
            raise ValueError(
                "First value of sigA and sigB should be 1."
            )
    if len(sigA) != len(sigB):
        raise ValueError(
            f"Signatures should be of same truncated order (i.e. have the "
            f"same number of signature coefficients. Got {len(sigA)} "
            f"and {len(sigB)} signature coefficients"
        )
    # cdef double[:] prod = np.zeros(len(sigA)-1, dtype=np.dtype("d"))
    cdef cnp.ndarray[double, ndim=1] prod = np.zeros(len(sigA)-1, dtype=cnp.dtype("d"))
    cdef int sh1 = 0, sh2 = 0, idx_depth, i, inc = 1, lenleft = 0, lenright = 0
    # cdef int[:] cinds = inds
    cdef double one = 1.0
    for idx_depth in range(2, depth+2):
        sh2 += channels**(idx_depth-1) # sh2 += (int)(pow(channels, idx_depth-1))
        for i in range(0, idx_depth):
            lenleft = inds[i+1]-inds[i]
            lenright = inds[idx_depth-i]-inds[idx_depth-i-1]
            dger(&lenright, &lenleft, &one,
                 &sigB[inds[idx_depth-i-1]], &inc,
                 &sigA[inds[i]], &inc,
                 &prod[sh1], &lenright)
            # Careful! fortran order (column major): sigA and sigB switched !
        sh1 = sh2
    return prod


cpdef cnp.ndarray[double, ndim=1] sigrprod_inplace(
    cnp.ndarray[double, ndim=1] sigA,
    cnp.ndarray[double, ndim=1] sigB,
    unsigned int depth,
    unsigned int channels,
    cnp.ndarray[int, ndim=1] inds,
    bint check_params = True
    ):
    """
    Right inplace product in signature space: a <- a*b
    """
    if check_params:  # checks are to be removed
        if len(sigA) != inds[-1]:
            raise ValueError(
                f"Signatures should be of length {inds[-1]}. Got {len(sigA)} "
            )
        if len(sigA) != len(sigB):
            raise ValueError(
                f"Signatures should be of same truncated order (i.e. have the "
                f"same number of signature coefficients. Got {len(sigA)} "
                f"and {len(sigB)} signature coefficients"
            )
    cdef cnp.ndarray[double, ndim=1] sigAcopy = np.empty(inds[-1], dtype=cnp.dtype("d"))
    sigAcopy = sigA.copy()
    cdef double one = 1.0
    cdef int start, idx_depth, i, inc = 1, lenleft = 0, lenright = 0
    for idx_depth in range(depth+1, 0, -1):
        start = inds[idx_depth-1]
        for i in range(idx_depth-1, -1, -1):  # Careful! Must make use of sigA[start:] in first iteration
            lenleft = inds[i+1]-inds[i]
            lenright = inds[idx_depth-i]-inds[idx_depth-i-1]
            dger(&lenright, &lenleft, &one,
                 &sigB[inds[idx_depth-i-1]], &inc,
                 &sigA[inds[i]], &inc,
                 &sigA[start], &lenright)  # Careful! In place modif of `sigA`
    sigA -= sigAcopy
    return sigA


cpdef cnp.ndarray[double, ndim=1] siglprod_inplace(
    cnp.ndarray[double, ndim=1] sigA,
    cnp.ndarray[double, ndim=1] sigB,
    unsigned int depth,
    unsigned int channels,
    cnp.ndarray[int, ndim=1] inds,
    bint check_params = True
    ):
    """
    Same as previous function, but the result product is stored in `sigB`
    instead of `sigA`.
    """
    if check_params:  # checks are to be removed
        if len(sigA) != inds[-1]:
            raise ValueError(
                f"Signatures should be of length {inds[-1]}. Got {len(sigA)} "
            )
        if len(sigA) != len(sigB):
            raise ValueError(
                f"Signatures should be of same truncated order (i.e. have the "
                f"same number of signature coefficients. Got {len(sigA)} "
                f"and {len(sigB)} signature coefficients"
            )
    cdef cnp.ndarray[double, ndim=1] sigBcopy = np.empty(inds[-1], dtype=cnp.dtype("d"))
    sigBcopy = sigB.copy()
    cdef double one = 1.0
    cdef int start, idx_depth, i, inc = 1, lenleft = 0, lenright = 0
    for idx_depth in range(depth+1, 0, -1):
        start = inds[idx_depth-1]
        for i in range(idx_depth):  # Careful! Must make use of sigB[start:] in first iteration
            lenleft = inds[i+1]-inds[i]
            lenright = inds[idx_depth-i]-inds[idx_depth-i-1]
            dger(&lenright, &lenleft, &one,
                 &sigB[inds[idx_depth-i-1]], &inc,
                 &sigA[inds[i]], &inc,
                 &sigB[start], &lenright)  # Careful! In place modif of `sigB`
    sigB -= sigBcopy
    return sigB


def siginv(cnp.ndarray[double, ndim=1] sig,
           unsigned int depth,
           unsigned int channels,
           cnp.ndarray[int, ndim=1] inds,
           unsigned int lensig
          ):  # don't need lensig: lensig = inds[-1] ??
    r"""
    Compute the inverse of an element a of the signature Lie group with formula
    $a^{-1} = \sum_{k=0}^m(1-a)^{\otimes k}$ with m signature depth.

    Parameters
    ----------
    sig : torch.tensor
        Signature to inverse.
    """
    cdef:
        cnp.ndarray[double, ndim=1] right = np.empty(lensig, dtype=cnp.dtype("d"))
        cnp.ndarray[double, ndim=1] summand = np.empty(lensig, dtype=cnp.dtype("d"))
        cnp.ndarray[double, ndim=1] inv = np.empty(lensig-1, dtype=cnp.dtype("d"))
    right = sig.copy()
    right[0] = 0.  # change first value from 1. to 0.
    right = -right   # 1-a
    summand = right.copy()
    inv = right[1:].copy()
#     memcpy(right, summand, lensig)
#     memcpy(right, inv, lensig)
    cdef unsigned int k
    for k in range(1, depth+1):
        summand[1:] = sigprod(summand, right, depth, channels, inds, False)
        inv += summand[1:]
    # return inv[1:]
    return inv


cpdef cnp.ndarray[double, ndim=1] siginv_inplace(
    cnp.ndarray[double, ndim=1] sig,
    unsigned int depth,
    unsigned int channels,
    cnp.ndarray[int, ndim=1] inds
    ):
    """
    Inverse computation using this trick: 1+x+x^2+x^3 = 1+x(1+x(1+x))
    where we replace x with (1-a).
    I need supplementary memory for (2-a) only.
    """
    cdef cnp.ndarray[double, ndim=1] right = np.empty(inds[-1], dtype=cnp.dtype("d"))
    sigbis = sig.copy()
    cdef unsigned int i = 0
    for i in range(inds[-1]):
        sigbis[i] = -sigbis[i]
        sig[i] = -sig[i]
    sigbis[0] += 1.  # sigbis is (1-a)
    sig[0] += 2.  # sig is (2-a)
    for i in range(depth):
        sig = siglprod_inplace(sigbis, sig, depth, channels, inds, True)
        sig[0] += 1.
    return sig


def sigdist(cnp.ndarray[double, ndim=1] sigA,
            cnp.ndarray[double, ndim=1] sigB,
            unsigned int depth,
            unsigned int channels,
            cnp.ndarray[int, ndim=1] inds,
            unsigned int lensig
           ):
    cdef:
        cnp.ndarray[double, ndim=1] invA = np.zeros(lensig, dtype=cnp.dtype("d"))
        cnp.ndarray[double, ndim=1] prod = np.zeros(lensig, dtype=cnp.dtype("d"))
        int i
        int ind1, ind2
        int n = len(inds)-2
        int inc = 1
        double current_dist, dist = 0.
        cnp.ndarray[int, ndim=1] indsdiff = inds[1:] - inds[:-1]

    invA[0] = 1.
    invA[1:] = siginv(sigA, depth, channels, inds, lensig)
    prod[0] = 1.
    prod[1:] = sigprod(invA, sigB, depth, channels, inds, False)
    for i in range(1, n+1):
        ind1 = inds[i]
        ind2 = inds[i+1]
        current_dist = pow(
            ddot(&indsdiff[i], &prod[ind1], &inc, &prod[ind1], &inc),
            1./(2*i)
        )
        if current_dist > dist:
            dist = current_dist
    return dist


# cdef int ipow(int base, int exp)
#     """Power function for integers."""
#     int result = 1;
#     for (;;)
#     {
#         if (exp & 1)
#             result *= base;
#         exp >>= 1;
#         if (!exp)
#             break;
#         base *= base;
#     }
#     return result;
# }


cpdef cnp.ndarray[int, ndim=1] depth_inds(int depth, int channels):
    """
    Most libraries computing the signature transform output the signature as a
    vector. This function outputs the indices corresponding to first value of
    each signature depth in this vector.
    Example: with depth=4 and channels=2, returns [0, 1, 3, 7, 15, 31].
    """
    cdef cnp.ndarray[int, ndim=1] inds = np.empty(depth+2, dtype=cnp.dtype("i"))
    cdef int i
    cdef int sum = 0
    inds[0] = 0
    for i in range(depth+1):
        sum +=  channels**i
        inds[i+1] = sum
    return(inds)
