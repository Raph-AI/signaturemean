import numpy as np
import torch
import math

# def to_tens_alg(sig, sig_depth, channels):
#     """
#     Needed for sigprod function.
#     Transform a signature output of signatory to an element of the truncated
#     tensor algebra T^N((E)) i.e. subdivide the output tensor into a list of
#     tensors, each one corresponding to a different signature order.
#     """
#     inds = np.cumsum([channels**k for k in range(1,sig_depth+1)])
#     free_tensor = [1.]
#     ind_prev = 0
#     for it,ind in enumerate(inds):
#         shape = tuple([channels]*(it+1))
#         tens = sig[ind_prev:ind].numpy()
#         tens = tens.reshape(shape)
#         free_tensor.append(tens)
#         ind_prev = ind
#     return(free_tensor)


def to_tens_alg(sig, sig_depth, channels):
    """
    Needed for sigprod function.
    Transform a signature output of signatory to an element of the truncated
    tensor algebra T^N((E)) i.e. subdivide the output tensor into a list of
    tensors, each one corresponding to a different signature order.

    Parameters
    ----------
    sig : torch.Tensor
        A signature as obtained with `signatory.signature` module.
    """
    inds = np.cumsum([channels**k for k in range(1, sig_depth+1)])
    free_tensor = [1.]
    ind_prev = 0
    for it, ind in enumerate(inds):
        shape = tuple([channels]*(it+1))
        # tens = sig[ind_prev:ind].numpy()
        tens = sig[ind_prev:ind]
        # tens = tens.reshape(shape)
        tens = torch.reshape(tens, shape)
        free_tensor.append(tens)
        ind_prev = ind
    return(free_tensor)


# def from_tens_alg(free_tensor):
#     """
#     Needed for sigprod function.
#     Reverse the operation of to_tens_alg function.
#     Danger: presence of 1 or not as first value of the tensor.
#     """
#     sig = []
#     for elem in free_tensor[1:]:
#         flat = elem.flatten()
#         sig = np.hstack((sig,flat))
#     sig = torch.tensor(sig)
#     return(sig)


def from_tens_alg(free_tensor):
    """
    Needed for sigprod function.
    Reverse the operation of to_tens_alg function.
    Danger: presence of 1 or not as first value of the tensor.
    """
    sig = torch.empty(0)
    for elem in free_tensor[1:]:
        # print("elem", elem)
        flat = elem.flatten()
        sig = torch.hstack((sig, flat))
    # sig = torch.tensor(sig)
    return(sig)


def sigprod(sig1, sig2, sig_depth, channels):
    r"""
    Product between two signatures (in other words: two elements of the tensor
    algebra).
    $a \otimes b = (c_0, ..., c_N)$
    Parameters
    ----------
    sig1 : torch.Tensor
        Signature, e.g. the output of :func:`signatory.signature()`.
    """
    if len(sig1) != len(sig2):
        raise ValueError(
            f"Signatures should be of same truncated order (i.e. have the "
            f"same number of signature coefficients. Got {len(sig1)} "
            f"and {len(sig2)} signature coefficients"
        )
    tens1 = to_tens_alg(sig1, sig_depth, channels)
    tens2 = to_tens_alg(sig2, sig_depth, channels)
    prod = []
    for idx_coord in range(sig_depth+1):
        coord = 0.
        for k in range(idx_coord+1):
            A = tens1[k]
            B = tens2[idx_coord-k]
            # coord += np.multiply.outer(A, B)

            if isinstance(A, float):
                A = torch.tensor(A)
            if isinstance(B, float):
                B = torch.tensor(B)
            a = torch.flatten(A)
            b = torch.flatten(B)
            d = torch.outer(a, b)
            d = torch.reshape(d, list(A.shape)+list(B.shape))
            coord += d
        prod.append(coord)
    prod = from_tens_alg(prod)
    return(prod)


# def sigprod_old(sig1, sig2, sig_depth, channels):
#     r"""
#     Product between two signatures (in other words: two elements of the tensor
#     algebra).
#     $a \otimes b = (c_0, ..., c_N)$
#     """
#     if len(sig1) != len(sig2):
#         raise ValueError(
#             f"Signatures must be of same truncated order : got {len(sig1)} "
#             f"and {len(sig2)} signature coordinates"
#             )
#     tens1 = to_tens_alg(sig1, sig_depth, channels)
#     tens2 = to_tens_alg(sig2, sig_depth, channels)
#     prod = []
#     for idx_coord in range(sig_depth+1):
#         coord = 0.
#         for k in range(idx_coord+1):
#             A = tens1[k]
#             B = tens2[idx_coord-k]
#             coord += np.multiply.outer(A, B)
#         prod.append(coord)
#     prod = from_tens_alg(prod)
#     return(prod)


def to_tens_alg_inv(sig, sig_depth, channels):
    """
    Needed for sigprod_inv function.
    Transform a signature output of signatory to an element of the truncated
    tensor algebra T^N((E)) i.e. subdivide the output tensor into a list of
    tensors, each one corresponding to a different signature order.
    """
    inds = np.cumsum([channels**k for k in range(1, sig_depth+1)])
    free_tensor = [sig[0].numpy()]
    sig = sig[1:]
    ind_prev = 0
    for it, ind in enumerate(inds):
        shape = tuple([channels]*(it+1))
        tens = sig[ind_prev:ind].numpy()
        # print(tens.shape)
        tens = tens.reshape(shape)
        free_tensor.append(tens)
        ind_prev = ind
    return(free_tensor)


def from_tens_alg_inv(free_tensor):
    """
    Needed for sigprod_inv function.
    Reverse the operation of to_tens_alg function.
    Danger: presence of 1 or not as first value of the tensor.
    """
    sig = []
    for elem in free_tensor:
        flat = elem.flatten()
        sig = np.hstack((sig, flat))
    sig = torch.tensor(sig)
    return(sig)


def sigprod_inv(sig1, sig2, sig_depth, channels):
    """
    Variant of sigprod, ONLY FOR siginv FUNCTION.
    Here sig1 and sig2 first item is a scalar and not a vector (vector is the
    second item).
    """
    if len(sig1) != len(sig2):
        raise ValueError(
            f"Signatures must be of same truncated order : got {len(sig1)} "
            f"and {len(sig2)} signature coordinates"
            )
    tens1 = to_tens_alg_inv(sig1, sig_depth, channels)
    tens2 = to_tens_alg_inv(sig2, sig_depth, channels)
    prod = []
    for idx_coord in range(sig_depth+1):
        coord = 0.
        for k in range(idx_coord+1):
            A = tens1[k]
            B = tens2[idx_coord-k]
            coord += np.multiply.outer(A, B)
        prod.append(coord)
    prod = from_tens_alg_inv(prod)
    return(prod)


def siginv(sig, sig_depth, channels):
    r"""
    Compute the inverse of an element :math:`a` of the signature Lie group with
    formula:

    .. math::
        a^{-1} = \sum_{k=0}^m(1-a)^{\otimes k}

    with :math:`m` is the signature depth.
    Computation uses that: :math:`1+x+x^2+x^3=1+x(1+x(1+x))` where we replace
    :math:`x` with :math:`(1-a)` (Horner's method).

    Input
    -----
    sig : torch.Tensor
        Signature to inverse. Caution: the signature must NOT comprise the
        scalar value `1.` as it first value.
    """
    if len(sig.shape) > 1:
        raise ValueError(
            f"Signature input must be one dimensional : got shape = "
            f"{sig.shape}"
            )
    idty = torch.zeros(len(sig))
    idty = torch.cat((torch.tensor([1.]), idty))
    inv = idty
    sig = torch.cat((torch.tensor([1.]), sig))
    rprod = idty-sig
    summand = rprod
    inv += summand
    for k in range(1, sig_depth+1):
        summand = sigprod_inv(summand, rprod, sig_depth, channels)
        inv += summand
    inv = inv[1:]  # remove scalar 1 at first position
    return(inv)


def dist_on_sigs(g, h, depth, channels):
    """
    Compute the distance induced by the norm introduced in [1, Example 7.37]
    It is defined as :
    $$d(g,h) := \max_{i=1, \dots, N} ||\pi_i(g^{-1} \otimes h)||^{1/i}$$
    with
      * $\pi_i$ is the projection on the tensor space of dimension i
      * ||.|| is the Frobenius norm on tensors i.e. sqrt(sum of squared values)

    Parameters
    ----------
    g : (channels**1 + ... + channels**depth) torch.Tensor
        The first signature tensor used to compute the distance.
    h : same as g
    depth : int
            Depth of the signature tensors g and h.
    channels : int
               Number of channels of the data x such that g = S(x).

    Returns
    -------
    distance : float
               The distance in signature space as defined above.

    References
    ----------
    [1] Friz, P.K. and Victoir, N.B. (2010) Multidimensional Stochastic
    Processes as Rough Paths (672 p.)
    """
    g_inv = siginv(g, depth, channels)
    prod = sigprod(g_inv, h, depth, channels)
    prod = to_tens_alg(prod, depth, channels)
    tensor_norms = []
    # print(prod)
    prod = prod[1:]  # skip scalar value (since i start at 1 in the formula)
    # print("prod : ", prod)
    for i, tensor in enumerate(prod):
        tensor = tensor.flatten()
        norm = torch.pow(torch.sqrt(torch.sum(torch.pow(tensor, 2))), 1./(i+1))
        # norm = np.power(np.sqrt(np.sum(np.power(tensor, 2))), 1./(i+1))
        tensor_norms.append(norm)
    # print(tensor_norms)
    distance = np.max(tensor_norms)
    return(distance)
    # return(torch.max(tensor_norms))


def dist_on_sigsL2(g, h, depth, channels):
    g_inv = siginv(g, depth, channels)
    prod = sigprod(g_inv, h, depth, channels)
    prod = prod[1:]  # skip scalar value (since i start at 1 in the formula)
    norm = torch.sum(torch.pow(prod, 2))
    return(norm)


# def dist_on_sigs2(g, h, depth, channels):
#     """
#     d(g,h) := \max_{i=1, \dots, N} ||\pi_i(g - h)||
#     where
#         * \pi_i is the projection on the tensor space of dimension i
#         * ||.|| is the classical norm on tensors sqrt(sum of squared values)
#     """
#     diff = g - h
#     diff = to_tens_alg(diff, depth, channels)
#     tensor_norms = []
#     # print(prod)
#     diff = diff[1:] #skip scalar value (since i start at 1 in the formula)
#     # print("prod : ", prod)
#     for i,tensor in enumerate(diff):
#         tensor = tensor.flatten()
#         norm = np.sqrt(np.sum(np.power(tensor, 2)))
#         tensor_norms.append(norm)
#     # print(tensor_norms)
#     return(np.max(tensor_norms))


def dist_to_sigdataset(g, datasig, depth, channels):
    dist = 0.
    for sig in datasig:
        dist += dist_on_sigs(g, sig, depth, channels)
    return(dist)


def dist_to_sigdatasetL2(g, datasig, depth, channels):
    dist = 0.
    for sig in datasig:
        dist += dist_on_sigsL2(g, sig, depth, channels)
    return(dist)


def order_along_time(path):
    """
    Sort a path along its first row (which is supposed to be the timestamps).

    Input is a 2D matrix stream x channels

    Example
    -------
    Given path = [[1,0,3],[-3.,5,-1]] returns [[0.,1.,3.],[5.,-3.,-1.]]
    """
    if len(path.shape) == 3:
        path = path[0, :, :]
    val, ind = torch.sort(path[:, 0])
    ordered_path = torch.index_select(path, 0, ind)
    ordered_path = torch.unsqueeze(ordered_path, 0)
    return(ordered_path)


def augment_stream(path, stream):
    """
    Add points to increase the stream (length) of path. A timestamp is chosen.
    Then the value of the corresponding point is taken on the line joining the
    two points it fits between.

    Input is 3D tensor batch x stream x channels
    """
    nb_pts_to_add = stream - path.shape[1]
    channels = path.shape[2]
    if nb_pts_to_add < 1:
        raise ValueError(('Path can not be augmented since it is larger that'
                         f'stream: {stream}≤{path.shape[1]}.'))
    time_beg, time_end = path[0, 0, 0], path[0, -1, 0]
    timepts = path[0, :, 0]
    timepts_to_add = np.linspace(time_beg, time_end, nb_pts_to_add+2)[1:-1]
    # print(time_beg,time_end)
    # print(timepts_to_add)
    pts_to_add = -1.*torch.ones((1, nb_pts_to_add, channels-1),
                                dtype=torch.float64)
    for i, timept in enumerate(timepts_to_add):
        idx = np.where(timept > timepts)[0][-1]
        a = path[0, idx, 1:]
        b = path[0, idx+1, 1:]
        ta, tb = path[0, idx, 0], path[0, idx+1, 0]
        val = (b-a)/(tb-ta)*timept + a-(b-a)/(tb-ta)*ta  # pt on line [a,b]
        pts_to_add[0, i, :] = val
    timepts_to_add = torch.from_numpy(timepts_to_add)
    timepts_to_add = torch.reshape(timepts_to_add, (1, nb_pts_to_add, 1))
    newpts = torch.cat((timepts_to_add, pts_to_add), dim=2)
    #
    # print(path.shape, newpts.shape)
    newpath = torch.cat((path, newpts), dim=1)
    # newpath = torch.squeeze(newpath)
    # print(newpath.shape)
    newpath = order_along_time(newpath)
    # newpath = torch.unsqueeze(newpath, 0)
    return(newpath)


def datashift(data):
    """
    Shift the data so that it starts at zero.

    Parameters
    ----------
    data : array-like or list
        Type should be list if stream value is not the same for every
        observation.
    """
    if type(data) == torch.Tensor:
        batch, stream, channels = data.shape
        datashifted = data.clone()
        for idx_obs in range(batch):
            data0 = data[idx_obs, 0, :]
            for idx_stream in range(stream):
                # print(data[idx_obs, idx_stream, :])
                # print(data0)
                datashifted[idx_obs, idx_stream, :] = data[idx_obs, idx_stream, :] - data0
    elif type(data) == list:
        datashifted = []
        for obs in data:
            obsshift = obs.clone()
            for idx_stream in range(obs.shape[0]):
                obsshift[idx_stream, :] = obs[idx_stream, :] - obs[0, :]
            datashifted.append(obsshift)
    return(datashifted)


def datascaling(data):
    """
    Each observation is set to have total variation norm equals to 1.

    Parameters
    ----------
    data : array-like or list
        Type should be list if stream value is not the same for every
        observation.
    """
    if type(data) == torch.Tensor:
        batch, stream, channels = data.shape
        datascaled = data.clone()
        for i in range(batch):
            variations = [np.linalg.norm(data[i, k, :]-data[i, k-1, :])
                          for k in range(1, stream)]
            tvnorm = np.sum(np.array(variations), axis=0)
            if tvnorm != 0.:  # case: data[i] is constant
                datascaled[i] = data[i]/tvnorm
    elif type(data) == list:
        datascaled = []
        for obs in data:
            stream = obs.shape[0]
            variations = [np.linalg.norm(obs[k, :]-obs[k-1, :])
                for k in range(1, stream)]
            tvnorm = np.sum(np.array(variations), axis=0)
            if tvnorm != 0.:  # case: obs is constant
                obsscaled = obs/tvnorm
                datascaled.append(obsscaled)
            else:
                datascaled.append(obs)
    return(datascaled)


def sigscaling(SX, depth, channels):
    """
    Each signature tensor of depth m is multiplied by factorial m.
    Careful: should use `utils.datascaling` before computation of signature and
    use of `utils.sigscaling`.

    Parameters
    ----------
    SX : torch.Tensor
        An array of signature to scale.

    depth : int
        Depth of the signature.

    channels : int
        Number of channels (space dimensions).

    Returns
    -------
    SX : torch.Tensor
        Scaled signatures.
    """
    if len(SX.shape) != 2:
        raise ValueError(
            "Input SX should be a batch of signatures, i.e. 2-dim array,"
            f" got {SX.shape} instead. If only one signature to rescale, "
            "reshape your input using `SX.reshape((1,)+SX.shape)`."
        )
    # indices of each signature tensor
    inds = np.cumsum([0]+[channels**k for k in range(1, depth+1)])
    SX_scaled = torch.empty((SX.shape))
    for i in range(len(SX)):
        for d in range(1, depth+1):
            idx1, idx2 = inds[d-1], inds[d]
            SX_scaled[i, idx1:idx2] = SX[i, idx1:idx2]*math.factorial(d)
    return(SX_scaled)


# def datasigscaling(datasig, depth, channels):
#     """
#     Same as :sigscaling: but for a batch of signatures.
#     Careful: use datascaling before computation of signature and use of sigscaling.
#     """
#     datasigcopy = datasig.clone()
#     for sig in enumerate(datasigcopy):
#         # indices of each signature tensor
#         inds = np.cumsum([0]+[channels**k for k in range(1, depth+1)])
#         for d in range(1, depth+1):
#             idx1, idx2 = inds[d-1], inds[d]
#             sig[idx1:idx2] *= math.factorial(d)
#     return(sig)

def sigscaling_reverse(sig, depth, channels):
    """
    each signature tensor of depth m is divided by factorial m
    """
    # indices of each signature tensor
    inds = np.cumsum([0]+[channels**k for k in range(1, depth+1)])
    for d in range(1, depth+1):
        idx1, idx2 = inds[d-1], inds[d]
        sig[idx1:idx2] /= math.factorial(d)
    return(sig)


# def sigdownscaling(sig, depth, channels):
#     """
#     Do the exact opposite of function `sigrescaling` i.e.
#     each signature tensor of order m is divided by factorial m
#     """
#
#     if len(sig.shape) == 2:
#         sigs = []
#         for s in sig:
#             sigs.append(sigdownscaling(s, depth, channels))
#         sigstorch = torch.empty(size=(len(sigs), len(sigs[0])))
#         for i, elem in enumerate(sigs):
#             sigstorch[i] = elem
#         return(sigstorch)
#     # print("0.",sig)
#     sigres = np.copy(sig)
#     # print("1.",sigres)
#     tmp1 = channels*np.ones(depth, dtype=np.int8)
#     tmp2 = np.arange(1, depth+1, 1)
#     tmp3 = np.power(tmp1, tmp2)
#     idx_orders = np.cumsum(tmp3)
#     idx1 = 0
#     # print(sig)
#     for order in range(1, depth+1):
#         idx2 = idx_orders[order-1]
#         # print("icite",order, idx1, idx2)
#         # print(sig[idx1:idx2])
#         sigres[idx1:idx2] /= math.factorial(order)
#         # print(sig[idx1:idx2])
#         idx1 = idx2
#     # print(sig)
#     # print(sigres)
#     sigres = torch.from_numpy(sigres)
#     return(sigres)


# def leadlag_embedding(X, lag=1):
#     """
#     Apply lead-lag transform to paths dataset.
#
#     Parameters
#     ----------
#     lag : int
#         If lag=1, a new channel is created with lag 1 data. If lag=2, 2 new
#         channels are created with the first channel lag 1 data and the second
#         lag 2 data.
#     """
#
#     if torch.is_tensor(X):
#         if len(X.shape) != 3:
#             raise ValueError(
#                 "Input data should be 3-dimensional (batch, stream, channels), "
#                 f"got {X.shape} instead."
#             )
#         batch, stream, channels = X.shape
#         X_ll = torch.empty(batch, stream+lag, channels+lag)
#
#         X_ll[:, :channels, :stream] =
#         for l in range(lag):
#
#
#     elif type(X) is list:


def leadlag_embedding(X):
    """
    Apply lead-lag transform to paths dataset. Caution: if used,
    :func:`time_embedding` should be applied AFTER and not BEFORE.

    Parameters
    ----------
    X : torch.Tensor or list
        Data to embed with lead-lag.
    """
    if type(X) == torch.Tensor:
        if len(X.shape) != 3:
            raise ValueError(
                "Input data should be 3-dimensional (batch, stream, channels), "
                f"got {X.shape} instead."
            )
        batch, stream, channels = X.shape
        X_ll = torch.empty(batch, stream+1, 2*channels)

        X_ll[:, :-1, :channels] = torch.clone(X)
        X_ll[:, -1, :channels] = X_ll[:, -2, :channels]  # repeat last value

        X_ll[:, 1:, channels:] = torch.clone(X)
        X_ll[:, 0, channels:] = X_ll[:, 1, channels:]

    elif type(X) == list:
        X_ll = []
        for obs in X:
            obs_extended = obs.reshape((1,)+obs.shape)
            if type(obs_extended) == np.ndarray:
                obs_extended = torch.from_numpy(obs_extended)
            obs_extended = leadlag_embedding(obs_extended)
            obs_ = obs_extended[0]
            X_ll.append(obs_)

    return(X_ll)


def time_embedding(X):
    """
    Add a new dimension which is time: [-1,3,4] becomes [[-1, 3, 4], [0, 1, 2]].
    Should not be performed before using :func:`leadlag_embedding`.
    """
    if type(X) != torch.Tensor and type(X) != list:
        raise ValueError(
            f"Input data type should be list or torch.Tensor, got {type(X)} "
            "instead."
        )

    if type(X) == torch.Tensor:
        if len(X.shape) != 3:
            raise ValueError(
                "Input data should be 3-dimensional (batch, stream, channels), "
                f"got {X.shape} instead."
            )
        batch, stream, channels = X.shape
        time_ = torch.Tensor(range(stream)).reshape((1, stream, 1))
        time_ = time_.repeat((batch, 1, 1))
        X_time = torch.cat((X, time_), axis=2)
    elif type(X) == list:
        X_time = []
        for obs in X:
            obs_extended = obs.reshape((1,)+obs.shape)
            if type(obs_extended) == np.ndarray:
                obs_extended = torch.from_numpy(obs_extended)
            obs_extended = time_embedding(obs_extended)
            obs_ = obs_extended[0]
            X_time.append(obs_)

    return X_time


def nsigterms(depth, channels):
    """
    Number of signature terms.
    """
    return channels*(channels**depth-1)//(channels-1)


def depth_inds(depth, channels, scalar=False):
    """
    Most libraries computing the signature transform output the signature as a
    vector. This function outputs the indices corresponding to first value of
    each signature depth in this vector. Example: with depth=4 and channels=2,
    returns [2, 6, 14, 30] (or [1, 3, 7, 15, 31] if scalar=True).
    Parameters
    ----------
    scalar : boolean
        Presence of scalar as first value of the signature coordinates. By
        default, `signatory` returns signature vectors without the scalar
        value.
    """
    if scalar:
        return np.cumsum([channels**k for k in range(0, depth+1)])
    return np.cumsum([channels**k for k in range(1, depth+1)])
