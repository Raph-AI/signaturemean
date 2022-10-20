#cython: language_level=3

cimport cython
import numpy as np
cimport numpy as cnp

# personal libraries
# from ..cutils import depth_inds
from ..cutils import siginv
from ..cutils import sigprodlastlevel


"""

`mean_group_dger.py` but in cython

"""


cpdef cnp.ndarray[double, ndim=1] mean(
    cnp.ndarray[double, ndim=2] SX,
    unsigned int depth,
    unsigned int channels,
    cnp.ndarray[int, ndim=1] dinds
    ):

    cdef unsigned int batch = len(SX)

    cdef cnp.ndarray[double, ndim=1] \
        a = np.empty(SX.shape[1]+1, dtype=cnp.dtype("d"))
    a[0] = 1.
    cdef cnp.ndarray[double, ndim=2] \
        b = np.empty((batch, SX.shape[1]+1), dtype=cnp.dtype("d"))
    b = np.concatenate((np.ones((batch, 1)), SX), 1, dtype=cnp.dtype("d"))
    cdef cnp.ndarray[double, ndim=2] \
        p = np.zeros((batch, SX.shape[1]+1), dtype=cnp.dtype("d"))
    cdef cnp.ndarray[double, ndim=2] \
        q = np.zeros((batch, SX.shape[1]+1), dtype=cnp.dtype("d"))
    cdef cnp.ndarray[double, ndim=3] \
        v = np.zeros((batch, depth, SX.shape[1]+1), dtype=cnp.dtype("d"))
    # add dimension to store powers of vi.
    # vi shape =(batch, powers, sigterms)
    # /!\ careful to not add it as 3rd dim (but 2nd dim)

    cdef cnp.ndarray[double, ndim=1] \
        p_coeffs = np.array([np.power(-1, k+1)*1/k for k in range(2, depth+1)])

    # CASE K=1
    a[dinds[1]:dinds[2]] = -np.mean(b[:, dinds[1]:dinds[2]], axis=0)

    # CASE K = 2, 3, ...
    cdef int K, left, right, l2, r2, i, j, power
    for K in range(2, depth+1):
        left, right = dinds[K], dinds[K+1]
        for i in range(batch):
            l2, r2 = dinds[K-1], dinds[K]
            v[i, 0, l2:r2] = q[i, l2:r2] + a[l2:r2] + b[i, l2:r2]

            # update q
            q[i, left:right] = sigprodlastlevel(
                a[:right],
                b[i, :right],
                K,
                channels,
                dinds[:K+2],
                leading_zeros_sigA=1,
                leading_zeros_sigB=1
            )

            for j in range(K-1):
                # compute powers of v
                power = j+1
                v[i, power, left:right] = sigprodlastlevel(
                    v[i, 0, :right],
                    v[i, power-1, :right],
                    depth=K,
                    channels=channels,
                    inds=dinds[:K+2],
                    leading_zeros_sigA = 1,
                    leading_zeros_sigB = power
                )

                # update p
                p[i, left:right] += p_coeffs[j]*v[i, power, left:right]

        # update a
        a[left:right] = -np.mean(b[:, left:right]
                                    + q[:, left:right]
                                    + p[:, left:right],
                                    axis=0)

    mp = siginv(a, depth, channels, dinds)
    return(mp[1:])
