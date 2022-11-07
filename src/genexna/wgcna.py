from sklearn.cluster import AgglomerativeClustering
import numpy as np


def calc_tom(A):
    """
    Calculate the Topological Overlap Measure for matrix A.
    :param A: target matrix
    """
    d = A.shape[0]  # dimension
    A[np.arange(d), np.arange(d)] = 0
    L = A @ A
    K = A.sum(axis=1)

    # Initialize results.
    A_tom = np.zeros_like(A)

    # Calculate TOM according to definition.
    for i in range(d):
        for j in range(i + 1, d):
            numerator = L[i, j] + A[i, j]
            denominator = min(K[i], K[j]) + 1 - A[i, j]
            A_tom[i, j] = numerator / denominator
    A_tom += A_tom.T

    # Set diagonal to 1 by default.
    A_tom[np.arange(d), np.arange(d)] = 1
    return A_tom


class WGCNA:
    def __init__(self, n_networks, alpha=0.5, beta=6):
        self.n_networks = n_networks
        self.alpha = alpha
        self.beta = beta

    def _calc_sim_measure(self, X):
        """
        Calculate the similarity measure between columns of X.
        :param X: target matrix
        """
        # Initialise similarity measure.
        corr = np.corrcoef(X)

        # Calculate the similarity measure.
        S = self.alpha + (1 - self.alpha) * corr
        return S

    def _calc_adj_matrix(self, S):
        """
        Calculate Adjacency Matrix.
        """
        return np.power(S, self.beta)

    def fit_transform(self, X):
        """
        Apply weighted gene coexpression network analysis on X.
        """
        S = self._calc_sim_measure(X)  # similarity measure
        A = self._calc_adj_matrix(S)  # adjacency matrix
        tom = calc_tom(A)  # topological overlap measure

        # Apply Average Linkage Agglomerative Clustering with TOM.
        clusterer = AgglomerativeClustering(n_clusters=self.n_networks,
                                            affinity='precomputed',
                                            linkage='average')
        cluster_labels = clusterer.fit_predict(tom)
        return np.eye(self.n_networks)[cluster_labels]
