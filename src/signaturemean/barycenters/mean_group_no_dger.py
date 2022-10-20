import torch
import numpy as np
from ..utils import depth_inds
from ..utils import siginv


def mean(SX, depth, channels):
    batch = len(SX)
    a = torch.empty(SX[0].shape)
    b = SX.clone()
    p = torch.empty(SX.shape)
    q = torch.empty(SX.shape)
    v = torch.zeros(SX.shape + (depth,))
    # added 3rd dimension to store powers of vi
    # shape: (batch, sigterms, powers)
    p_coeffs = [np.power(-1, k+1)*1/k for k in range(2, depth+1)]
    # print(p_coeffs)

    dinds = depth_inds(depth, channels, scalar=False)
    dinds = np.concatenate(([0], dinds))

    # CASE K=1
    left, right = dinds[0], dinds[1]
    p[:, left:right] = torch.zeros((batch, right-left))
    q[:, left:right] = torch.zeros((batch, right-left))
    # v[:, left:right] = torch.zeros((batch, right-left))
    a[left:right] = -torch.mean(b[:, :channels], dim=0)

    # CASE K = 2, 3, ...
    for K in range(2, depth+1):
        left, right = dinds[K-1], dinds[K]
        dinds_sub = dinds[:K]
        for i in range(batch):
            l3, r3 = dinds[K-2], dinds[K-1]
            v[i, l3:r3, 0] = q[i, l3:r3] + a[l3:r3] + b[i, l3:r3]
            q[i, left:right] = torch.zeros(right-left)
            p[i, left:right] = torch.zeros(right-left)
            for j in range(K-1):
                l1, r1 = dinds_sub[j], dinds_sub[j+1]
                l2, r2 = dinds_sub[-(j+2)], dinds_sub[-(j+1)]
                outer = torch.outer(a[l1:r1], b[i, l2:r2]).flatten()
                q[i, left:right] += outer
                power = j+1
                for depth1 in range(K-power):
                    l4, r4 = dinds_sub[depth1], dinds_sub[depth1+1]
                    l5, r5 = dinds_sub[-(depth1+2)], dinds_sub[-(depth1+1)]  # to check
                    l_outer = v[i, l4:r4, 0]
                    r_outer = v[i, l5:r5, power-1]
                    val_temp = torch.outer(l_outer, r_outer).flatten()
                    v[i, left:right, power] += val_temp
                p[i, left:right] += p_coeffs[j]*v[i, left:right, power]

        # update a
        a[left:right] = -torch.mean(b[:, left:right]
                                    + q[:, left:right]
                                    + p[:, left:right],
                                    dim=0)
    mp = siginv(a, depth, channels)
    return(mp)
