"""
Weighted Gene Coexpression Network Analysis
"""

from sklearn.cluster import AgglomerativeClustering
import numpy as np


def calc_tom(mat):
    """
    Calculate the Topological Overlap Measure for matrix A.
    :param A: target matrix
    """
    dim = mat.shape[0]  # dimension
    mat[np.arange(dim), np.arange(dim)] = 0
    l = mat @ mat
    k = mat.sum(axis=1)

    # Initialize results.
    mat_tom = np.zeros_like(mat)

    # Calculate TOM according to definition.
    for i in range(dim):
        for j in range(i + 1, dim):
            numerator = l[i, j] + mat[i, j]
            denominator = min(k[i], k[j]) + 1 - mat[i, j]
            mat_tom[i, j] = numerator / denominator
    mat_tom += mat_tom.T

    # Set diagonal to 1 by default.
    mat_tom[np.arange(dim), np.arange(dim)] = 1
    return mat_tom


def label_networks(mat, n_networks, alpha=0.5, beta=6):
    """
    Apply weighted gene coexpression network analysis on X.
    """
    sim = alpha + (1 - alpha) * np.corrcoef(mat)  # similarity measure
    adj = np.power(sim, beta)  # adjacency matrix
    tom = calc_tom(adj)  # topological overlap measure

    # Apply Average Linkage Agglomerative Clustering with TOM.
    clusterer = AgglomerativeClustering(n_clusters=n_networks,
                                        affinity='precomputed',
                                        linkage='average')
    return clusterer.fit_predict(tom)
