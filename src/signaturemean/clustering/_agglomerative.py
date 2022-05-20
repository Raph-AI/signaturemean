import numpy as np


class AgglomerativeClustering():
    """
    Hierarchical Agglomerative Clustering: recursively merges pair of clusters
    of sample data; uses linkage distance.

    Attributes
    ----------
    labels_ : ndarray of shape (n_samples)
        Cluster labels for each point.

    """
    def __init__(
        self,
        n_clusters=2,
        metric='euclidean',
        linkage='ward'
    ):
        self.n_clusters = n_clusters
        self.metric = metric
        self.linkage = linkage

    def fit(self, SX):
        if self.linkage == "ward" and self.metric != "euclidean":
            raise ValueError(
                f"{self.metric} was provided as metric. Ward can only "
                "work with euclidean distances."
            )
        
