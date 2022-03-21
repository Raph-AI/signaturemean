#!/usr/bin/env python
# coding: utf-8
import os
import sktime
from sktime.datasets import load_from_tsfile_to_dataframe
from tslearn.clustering import TimeSeriesKMeans
# from tslearn.preprocessing import TimeSeriesScalerMeanVariance
# from tslearn.preprocessing import TimeSeriesResampler
import matplotlib.pyplot as plt
import numpy as np
import signatory
import torch
# from .. import mean_pennec
from signature_mean import mean_pennec
import utils
import pandas as pd
import time



X_train, y_train = load_from_tsfile_to_dataframe("./data/PenDigits/PenDigits_TRAIN.ts")
X_test, y_test = load_from_tsfile_to_dataframe("./data/PenDigits/PenDigits_TEST.ts")

X = pd.concat([X_train, X_test], ignore_index=True)
y_true = np.concatenate((y_train, y_test))
# np.unique(y_true, return_counts=True)
# **Remark.** Each class represent an equal proportion of the data.
X = X.to_numpy()
batch = X.shape[0]
stream = len(X[0,0])
channels = X.shape[1]
Xtemp = np.empty((batch, stream, channels))
for obs in range(batch):
    for channel in range(channels):
        Xtemp[obs, :, channel] = X[obs, channel]
Xtemp = torch.from_numpy(Xtemp)
X = Xtemp


def datascaling(data):
    "Each observation is set to have total variation norm equals to 1."
    batch, stream, channels = data.shape
    print(f"stream = {stream}")
    for i in range(batch):
        tvnorm = np.array([(data[i, k, :] - data[i, k-1, :]).numpy() for k in range(1, stream)])
        tvnorm = np.sum(tvnorm, axis=0)
        tvnorm = np.linalg.norm(tvnorm)
        print(f"tvnorm obs #{i} = {tvnorm}")
        data[i] = data[i]/tvnorm
        print(data[i])
    return(data)


# **STEP 1.**  dataset of paths `X` already normalized.
depth = 5
SX = signatory.signature(X, depth=depth)

timestr = time.strftime("%Y%m%d_%H%M%S")
logsfilename = f'./logs/log_clusteridx_{timestr}.txt'

def kmeans(k, signatures, depth, channels, max_iterations, dist='l2',
           verbose=False):
    """
    Perform k-means on a dataset of signatures.

    Parameters
    ----------
    max_iterations : int
                     number of iterations before k-means stops
    dist : string
           Distance to use. One of 'l2', 'sigdist'.

    Notes
    -----
    Cluster associated to each observation for every k-means iteration is
    written in the logs folder.
    """
    batch, siglen = signatures.shape
    y = -1*np.ones(batch, dtype='int')  # corresponding cluster index for each obs
    with open(logsfilename, 'w') as the_file:
        the_file.write('')
    # INITIALIZATION
    # choose k obs randomly to start from
    rd_idx = np.random.randint(batch, size=k)
    means = signatures[rd_idx]
    for iteration in range(max_iterations):
        # if iteration%5==0:
        print(f"iteration #{iteration}")
        # ASSIGN OBS TO CLUSTER W/ NEAREST CENTROID
        for idx, sig in enumerate(signatures):
            if dist=='l2':
                distlist = [torch.norm(sig-means[i]) for i in range(k)]  # L2 DISTANCE
            elif dist=="sigdist":
                distlist = [utils.dist_on_sigs(sig, means[i], depth, channels) for i in range(k)]  # SIG DISTANCE
            id_cluster = np.argmin(distlist)
            y[idx] = id_cluster
        # UPDATE MEAN VALUES TO MEAN OF CLUSTER
        sum1 = 0
        for i in range(k):
            to_average = signatures[y==i, :]  # select obs of cluster i
            sum1 += to_average.shape[0]
            if not len(to_average)==0:
                means[i] = mean_pennec.mean(to_average, depth, channels)
        if verbose:
            print(f"SUM  = {sum1}")
            print(f"first cluster indices : {y[:15]}")
            print(f"mean values")
            print((means[0])[14:24])
            print((means[3])[14:24])
            print((means[6])[14:24])
        with open(logsfilename, 'a') as the_file:
            the_file.write(f'{list(y)}\n')
    return(y)


start = time.time()


### PARAMETERS TO SET FOR K-MEANS ALGORITHM ###################################

y = kmeans(k=10, signatures=SX, depth=depth, channels=channels,
           max_iterations=2, #10, 30
           dist="sigdist")

###############################################################################


end = time.time()
dura = f"duration of algorithm : {np.around((start-end)/60, 3)} min"
with open(logsfilename, 'a') as the_file:
    the_file.write(dura)
print(dura)

for i in range(10):
    output = np.unique(y[y_true==f'{i}'], return_counts=True)
    with open(logsfilename, 'a') as the_file:
        the_file.write(output)
