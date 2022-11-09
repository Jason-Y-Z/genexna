"""
Decompose a gene expression profile to obtain network composition
"""

import abc
from dataclasses import dataclass
from typing import Optional, Sequence

from sklearn import decomposition
from sklearn.preprocessing import normalize
import numpy as np
import pandas as pd

from .lnmf import factorize
from .wgcna import label_networks


class DecomposeAlgorithm(abc.ABC):
    """
    Interface for an algorithm used to decompose the gene expression profile.
    """
    @abc.abstractmethod
    def __call__(self, mat: np.array, labels: Optional[Sequence[int]]) -> np.array:
        """
        The algorithm should decompose the gene expression `mat`, optionally using
        the class labels of the subjects `labels`, to produce the gene network
        composition as a probablity distribution.
        """


@dataclass
class WGCNA(DecomposeAlgorithm):
    """
    Weighted gene coexpression network analysis based decomposer.
    """
    n_networks: int
    alpha: float = 0.5
    beta: float = 6

    def __call__(self, mat: np.array, _: Optional[Sequence[int]]) -> np.array:
        cluster_labels = label_networks(
            mat.T, self.n_networks, self.alpha, self.beta)
        return np.eye(self.n_networks)[cluster_labels].T


@dataclass
class LNMF(DecomposeAlgorithm):
    """
    Labelled nonnegative matrix factorization based decomposer.
    """
    w_init: np.array = None
    h_init: np.array = None
    n_components: Optional[int] = None
    alpha: float = 0.1
    beta: float = 0.1
    gamma: float = 0.1
    max_iters: int = 1000

    def __call__(self, mat: np.array, labels: Optional[Sequence[int]]) -> np.array:
        h = factorize(mat, labels, self.w_init, self.h_init, self.n_components, self.alpha,
                      self.beta, self.gamma, self.max_iters, return_w=False)
        return normalize(np.nan_to_num(h), norm='l1', axis=0)


class NMF(DecomposeAlgorithm):
    """
    Nonnegative matrix factorization based decomposer.
    """

    def __init__(self, *args, **kwargs):
        self._nmf = decomposition.NMF(*args, **kwargs)

    def __call__(self, mat: np.array, _: Optional[Sequence[int]]) -> np.array:
        self._nmf.fit_transform(mat)
        h = self._nmf.components_
        return normalize(np.nan_to_num(h), norm='l1', axis=0)


def decompose(
    gene_expr: pd.DataFrame,
    traits: Optional[Sequence[int]] = None,
    algo: DecomposeAlgorithm = NMF(),
):
    """
    Decompose the gene expression matrix to
    obtain the composition of each gene in terms of
    the gene networks.
    :param gene_expr: gene expression profile of shape [n_subject, n_genes]
    """
    prob_vals = algo(
        gene_expr.to_numpy(),
        traits if traits else np.zeros(gene_expr.shape[0])
    )
    return pd.DataFrame(data=prob_vals, index=range(prob_vals.shape[0]), columns=gene_expr.columns)
