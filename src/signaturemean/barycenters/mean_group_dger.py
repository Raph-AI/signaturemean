# import torch
import numpy as np
from ..cutils import depth_inds
from ..cutils import siginv
from ..cutils import sigprodlastlevel


"""

mean_group but with LAPACK's dger function 

"""


def mean(SX, depth, channels):
    batch = len(SX)
    a = np.empty(SX.shape[1]+1)
    a[0] = 1.
    b = SX.numpy()
    b = np.concatenate((np.ones((batch, 1)), b), 1)
    p = np.zeros((batch, SX.shape[1]+1)) # p[0] = ??
    q = np.zeros((batch, SX.shape[1]+1)) # same
    v = np.zeros((batch, depth, SX.shape[1]+1))
    # add dimension to store powers of vi.
    # vi shape =(batch, powers, sigterms)
    # /!\ careful to not add it as 3rd dim (but 2nd dim)
    p_coeffs = [np.power(-1, k+1)*1/k for k in range(2, depth+1)]

    dinds = depth_inds(depth, channels, scalar=True)

    # CASE K=1
    left, right = dinds[1], dinds[2]
    a[left:right] = -np.mean(b[:, left:right], axis=0)

    # CASE K = 2, 3, ...
    for K in range(2, depth+1):
        left, right = dinds[K], dinds[K+1]
        for i in range(batch):
            l3, r3 = dinds[K-1], dinds[K]
            v[i, 0, l3:r3] = q[i, l3:r3] + a[l3:r3] + b[i, l3:r3]

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
