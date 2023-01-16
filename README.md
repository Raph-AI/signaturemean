# <p align='center'> signaturemean </p>

A toolbox for signature averaging.

## Signature barycenters

This repository contains approaches to compute a barycenter of _iterated integrals signatures_.

From a dataset of multivariate time series $(X_i)_{1\leq i \leq n}$ we define the following barycenters with weights $w_i$ and denoting $\mathbb{X} = S_{[0,1]}^{(\leq m)}(X)$ the signature up to order m :


-   group mean `signaturemean.barycenters.mean_group` : solution $m$ of $$\sum_{i=1}^n w_i \mathrm{Log}(m^{-1}\boxtimes\mathbb{X}_i)=0$$
-   `signaturemean.barycenters.mean_tsoptim` : Optimization on time series space method $$\min_{X\in\mathbb{R}^{D\times L}} \sum_{i=1}^n d(\mathbb{X}, \mathbb{X}_i) .$$

For an introduction on the iterated integrals signature transform, the reader can refer itself to [1].

## Quick example

```python
import numpy as np
import iisignature
from signaturemean.barycenters import mean_group
from signaturemean.barycenters import mean_tsoptim
from signaturemean import utils

batch = 5     # number of time series
stream = 30   # number of timestamps for each time series
channels = 3  # number of dimensions
depth = 4     # depth (order) of truncation of the signature

# Simulate random data
X = np.random.rand(batch, stream, channels)   # simulate random numbers
X = utils.datashift(X)    # paths start at zero
X = utils.datascaling(X)  # paths have total variation = 1

# Compute signature
SX = iisignature.sig(X, depth)
weights = 1./batch*np.ones(batch)

# Averaging, method 1
SX = np.concatenate((np.ones((batch,1)), SX), axis=1)  # level 0 of signature
m = mean_group.mean(SX, depth, channels, weights)
print(m)

# Averaging, method 2 (requires torch)
import torch
tso = mean_tsoptim.TSoptim(depth=depth, random_state=1641)
tso.fit(torch.from_numpy(X))
print(tso.barycenter_ts)  # returns a path
```

**Remarks**

-   Note that `tso.barycenter_ts` is a list of paths and not a signature.
-   In the example above, data is shifted and scaled. This step is not necessary, only required.

## Requirements

```sh
python3 -m pip install -r requirements.txt
```

Note that `torch` and `pymanopt` are used only in `mean_tsoptim`.

<!-- 1.  `python3 -m pip install -r requirements.txt`.
2.  Requires `signatory`. [How to install signatory](https://signatory.readthedocs.io/en/latest/pages/usage/installation.html). NB: verify that your `signatory` package version is compatible with your `torch` package version. For instance, use this installation: torch 1.9.0 and signatory 1.2.6.1.9.0
    ```
    pip install torch==1.9.0
    pip install signatory==1.2.6.1.9.0 --no-cache-dir --force-reinstall
    ```-->



<!-- ## Example -->

<!-- <img src="./figures/gaussian_process.png" width="50%" />

Figure: Representation in three-dimensional path space of various barycenters. Inputs are gaussian processes with RBF kernel. Parameters: depth is the truncation order of the signature; obs is the number of inputs; length is the number of timestamps. -->

## References

[1] Chevyrev, I. and Kormilitzin, A. (2016) ‘A Primer on the Signature Method in Machine Learning’, arXiv:1603.03788 [cs, stat] [Preprint]. Available at: http://arxiv.org/abs/1603.03788 (Accessed: 9 August 2021).
