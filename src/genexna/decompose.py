"""
TODO: Docstring
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
    TODO
    """
    @abc.abstractmethod
    def __call__(self, mat: np.array, _: Optional[Sequence[int]]) -> np.array:
        """
        An algorithm class must be callable.
        """


@dataclass
class WGCNA(DecomposeAlgorithm):
    """
    TODO
    """
    n_networks: int
    alpha: float = 0.5
    beta: float = 6

    def __call__(self, mat: np.array, _: Optional[Sequence[int]]) -> np.array:
        cluster_labels = label_networks(
            mat, self.n_networks, self.alpha, self.beta)
        return np.eye(self.n_networks)[cluster_labels]


@dataclass
class LNMF(DecomposeAlgorithm):
    """
    TODO
    """
    w_init: np.array = None
    h_init: np.array = None
    n_components: Optional[int] = None
    alpha: float = 0.1
    beta: float = 0.1
    gamma: float = 0.1
    max_iters: int = 1000

    def __call__(self, mat: np.array, y: Optional[Sequence[int]]) -> np.array:
        h = factorize(mat, y, self.w_init, self.h_init, self.n_components, self.alpha,
                      self.beta, self.gamma, self.max_iters, return_w=False)
        return normalize(np.nan_to_num(h), norm='l1', axis=0)


class NMF(DecomposeAlgorithm):
    """
    TODO
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
    """
    prob_vals = algo(gene_expr.to_numpy(), traits)
    return pd.DataFrame(data=prob_vals, index=range(prob_vals.shape[0]), columns=gene_expr.columns)
