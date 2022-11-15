from sklearn.decomposition import PCA
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
import scipy.stats


def _pearson_corr_coeff(
    mat: pd.DataFrame, traits: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    pcc = pd.DataFrame(0, index=mat.index, columns=traits.columns)
    p_value = pd.DataFrame(0, index=mat.index, columns=traits.columns)
    for col in traits:
        for index, eigengene in mat.iterrows():
            pearson_r = scipy.stats.pearsonr(eigengene, traits[col])
            pcc.loc[index, col] = pearson_r[0]
            p_value.loc[index, col] = pearson_r[1]
    return pcc, p_value


def _soft_net_eigengenes(
    gene_expr: pd.DataFrame, gene_network_prob: pd.DataFrame
) -> pd.DataFrame:
    """
    Find the eigengenes for each network with soft boundaries,
    which means each gene is not assigned to a single cluster
    but to multiple clusters with probability distribution given.

    Args:
        gene_expr: gene expression levels, [n_subjects, n_genes]
        gene_network_prob: probability of gene A in network B, [n_networks, n_genes]

    Returns:
        pd.DataFrame: soft eigengenes, [n_networks, n_subjects]
    """
    eigengenes = {
        network: np.zeros(gene_expr.shape[0])
        for network in gene_network_prob.index
    }
    for gene in gene_network_prob:
        for network, prob in gene_network_prob[gene].items():
            eigengenes[network] += gene_expr.loc[:, gene] * prob
    return pd.DataFrame.from_dict(eigengenes, orient="index")


def _segregate_2d(mat, network_labels):
    networks = {i: [] for i in set(network_labels)}

    # Put the vectors into corresponding networks.
    for i in range(len(network_labels)):
        networks[network_labels[i]].append(mat[i, :])

    # Stack the vectors into matrices.
    networks = {k: np.vstack(v) for k, v in networks.items()}
    return networks


def _hard_net_eigengenes(
    gene_expr: pd.DataFrame, network_labels: List[int]
) -> pd.DataFrame:
    """
    Find the eigengenes for each network. Hard thresholds mean that
    each gene can only belong to one network.

    Args:
        gene_expr: gene expression levels, [n_subjects, n_genes]
        network_labels: which network the gene belongs to

    Returns:
        pd.DataFrame: hard eigengenes, [n_networks, n_subjects]
    """
    
    # Separate the genes into networks.
    x_network = _segregate_2d(gene_expr.to_numpy().T, network_labels)
    eigengenes = {}

    # Apply PCA to find first principal component.
    for network_i, X_network_i in x_network.items():
        # Calculate the first principle component as the eigengene.
        pc1_components = PCA(n_components=1).fit_transform(X_network_i.T)
        eigengenes[network_i] = pc1_components.flatten()
    return pd.DataFrame.from_dict(eigengenes, orient='index')


def calc_eigengene_pccs(
    gene_expr, traits, gene_network_rel, aggregate=np.max, soft=True,
) -> Dict[int, float]:
    """
    Calculate the aggregated Pearson Correlation Coefficients
    for all the network eigengenes.

    Args:
        gene_expr: gene expression levels, [n_subjects, n_genes]
        traits: the traits or labels we are correlating with
        gene_network_rel: relationship between gene and networks, 
            [n_networks, n_genes] if soft else [n_genes]
        aggregate: method to aggregate the PCC's over traits
        soft: whether to use soft thresholds for network division

    Returns:
        Dict[int, float]: aggregated PCC for each gene network
    """
    net_eigengenes = _soft_net_eigengenes if soft else _hard_net_eigengenes
    eigengenes = net_eigengenes(gene_expr, gene_network_rel)
    eigen_corr, _ = _pearson_corr_coeff(eigengenes, traits)
    return {
        i: aggregate(np.abs(eigen_corr.loc[i, :]))
        for i in eigen_corr.index
    }
