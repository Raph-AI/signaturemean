import torch

class EmptyClusterError(Exception):
    def __init__(self, message=""):
        super().__init__()
        self.message = message

    def __str__(self):
        if len(self.message) > 0:
            suffix = " (%s)" % self.message
        else:
            suffix = ""
        return "Cluster assignments lead to at least one empty cluster" + \
               suffix


def _check_no_empty_cluster(labels, n_clusters):
    """Check that all clusters have at least one sample assigned.
    """
    for k in range(n_clusters):
        # print(f"_check_no_empty_cluster : labels = {labels}")
        if torch.sum(labels == k) == 0:
            raise EmptyClusterError
            
def _check_full_length(centroids):
    """Check that provided centroids are full-length (ie. not padded with
    nans).
    If some centroids are found to be padded with nans, TimeSeriesResampler is
    used to resample the centroids.
    """
    resampler = TimeSeriesResampler(sz=centroids.shape[1])
    return resampler.fit_transform(centroids)            

def _check_initial_guess(init, n_clusters):
    if hasattr(init, '__array__'):
        if not init.shape[0] == n_clusters:
            raise ValueError(
                f"Initial guess index array must contain {n_clusters} samples,"
                f" {init.shape[0]} given."
            )
