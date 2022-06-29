#cython: language_level=3

cimport cython
import iisignature  # logsigtosig
import signatory  # SignatureToLogSignature
cimport numpy as cnp
import numpy as np
import torch
from ..cutils import sigprod
from ..cutils import siginv
from ..cutils import depth_inds


# cpdef cnp.ndarray[double, ndim=1] mean(
#     cnp.ndarray[double, ndim=2] datasig,
#     unsigned int depth,
#     unsigned int channels,
#     cnp.ndarray[int, ndim=1] inds,
#     unsigned int max_iter=5):
#     """
#     Compute the Group Exponential Mean from a dataset of signatures.
#     """
#     logStoS = iisignature.prepare(channels, depth, "S2")  # S2 (= lyndon basis)
#     stoLogS = signatory.SignatureToLogSignature(
#         channels, depth, mode='brackets'
#         )
#     cdef int lenlogsig = len(signatory.lyndon_brackets(channels, depth))
#     cdef int batch = datasig.shape[0]
#     cdef:
#         cnp.ndarray[double, ndim=1] sigmean = np.empty(inds[-1], dtype=cnp.dtype("d"))
#         cnp.ndarray[double, ndim=1] sigmeaninv = np.empty(inds[-1], dtype=cnp.dtype("d"))
#         cnp.ndarray[double, ndim=2] logsigbatch = np.empty((batch, lenlogsig), dtype=cnp.dtype("d"))
#         cnp.ndarray[double, ndim=2] sigbatchtemp = np.empty((batch, inds[-1]), dtype=cnp.dtype("d"))
#
#     idx_rd = np.random.randint(batch)
#     sigmean = datasig[idx_rd].copy()
#
#     cdef it = 0
#     for it in range(max_iter):
#         sigmeaninv = siginv(sigmean, depth, channels, inds)
#         for obs in range(batch):
#             sigbatchtemp[obs] = sigprod(sigmeaninv, datasig[obs], depth, channels, inds)
# """
#         sigbatchtemp = torch.from_numpy(sigbatchtemp)
#         logsigbatch = (stoLogS.forward(sigbatchtemp[:, 1:])).numpy()
# """
#         logsigbatch = np.mean(logsigbatch, axis=0)
#         sigmeaninv[1:] = iisignature.logsigtosig(logsigbatch, logStoS)
#         sigmean = sigprod(sigmean, sigmeaninv, depth, channels, inds)
#     return sigmean
