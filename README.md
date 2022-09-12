# <p align='center'> signaturemean </p>

A toolbox for signature averaging.

## Signature barycenters

This repository contains three approaches to compute a barycenter of _iterated integrals signatures_ of paths. Let $X:[0,1]\to\mathbb{R}^D$

be a path and denote $\mathbb{X} = S_{[0,1]}^{(\leq m)}(X)$ its associated signature up to order m.

From a dataset $(X_i)_{1\leq i \leq n}$ we define the following barycenters:

-   `signaturemean.barycenters.mean_le` : Log Euclidean mean method

$$\bar{\mathbb{X}} = \mathrm{Exp}\ \Bigg( \frac1n \sum_{i=1}^n \mathrm{Log}\ \mathbb{X}_i\Bigg) .$$

-   `signaturemean.barycenters.mean_pennec` : Group Exponential mean method [2, Algorithm 1]

$$m_{(k+1)} = m_{(k)} \otimes \mathrm{Exp}\ \Bigg( \frac1n \sum_{i=1}^n \mathrm{Log}\ (m_{(k)}^{-1}\otimes \mathbb{X}_i)\Bigg) .$$

-   `signaturemean.barycenters.mean_tsoptim` : Optimization on time series space method

$$\min_{X\in\mathbb{R}^{D\times L}} \sum_{i=1}^n d(\mathbb{X}, \mathbb{X}_i) .$$


<!-- - `mean_pathopt_proj.py` **WIP** : method of Nozomi Sugiura (see Appendix B in [3]). -->

For an introduction on the iterated integrals signature transform, the reader can refer itself to [1].

## Usage

``` python
import torch
import numpy as np
import signatory

from signaturemean.barycenters import mean_le
from signaturemean.barycenters import mean_pennec
from signaturemean.barycenters import mean_tsoptim
from signaturemean import utils

batch = 5     # number of time series
stream = 30   # number of timestamps for each time series
channels = 3  # number of dimensions
depth = 4     # depth (order) of truncation of the signature

# Simulate random data
X = torch.rand(batch, stream, channels)   # simulate random numbers
X = utils.datashift(X)    # paths start at zero
X = utils.datascaling(X)  # paths have total variation = 1
SX = signatory.signature(X, depth=depth)

## Compute barycenter with each approach
# Approach 1: log euclidean mean
print(mean_le.mean(SX, depth, channels))      # returns a signature
# Approach 2: group exponential method
print(mean_pennec.mean(SX, depth, channels))  # returns a signature
# Approach 3: optimization on path space
tso = mean_tsoptim.TSoptim(depth=depth, random_state=1641)
tso.fit(X)
print(tso.barycenter_ts)  # returns a path
```

**Remarks**

-   Note that `tso.barycenter_ts` is a list of paths and not a signature.
-   In the example above, data is shifted and scaled. This step is not necessary, only required.

## Requirements

1.  `python3 -m pip install -r requirements.txt`.
2.  Requires `signatory`. [How to install signatory](https://signatory.readthedocs.io/en/latest/pages/usage/installation.html). NB: verify that your `signatory` package version is compatible with your `torch` package version. For instance, use this installation: torch 1.7.1 and signatory 1.2.6.1.7.1
    ```
    pip install torch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2
    pip install signatory==1.2.6.1.7.1 --no-cache-dir --force-reinstall
    ```



<!-- ## Example -->

<!-- <img src="./figures/gaussian_process.png" width="50%" />

Figure: Representation in three-dimensional path space of various barycenters. Inputs are gaussian processes with RBF kernel. Parameters: depth is the truncation order of the signature; obs is the number of inputs; length is the number of timestamps. -->

## References

[1] Chevyrev, I. and Kormilitzin, A. (2016) ‘A Primer on the Signature Method in Machine Learning’, arXiv:1603.03788 [cs, stat] [Preprint]. Available at: http://arxiv.org/abs/1603.03788 (Accessed: 9 August 2021).

[2] Pennec, Xavier, and Vincent Arsigny. 2013. “Exponential Barycenters of the Canonical Cartan Connection and Invariant Means on Lie Groups.” In Matrix Information Geometry, edited by Frank Nielsen and Rajendra Bhatia, 123–66. Berlin, Heidelberg: Springer Berlin Heidelberg. https://doi.org/10.1007/978-3-642-30232-9_7.

<!-- [3] Sugiura, Nozomi. 2021. “Clustering Global Ocean Profiles According to Temperature-Salinity Structure.” ArXiv:2103.14165 [Physics], March. http://arxiv.org/abs/2103.14165. -->
