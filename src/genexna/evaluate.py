from typing import Dict, Tuple
import numpy as np
import pandas as pd
import scipy.stats


def _pearson_corr_coeff(
    mat: pd.DataFrame, labels: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    pcc_df = pd.DataFrame(index=mat.columns, columns=labels.columns)
    p_value_df = pd.DataFrame(index=mat.columns, columns=labels.columns)
    for label in labels:
        for feature in mat.columns:
            pearson_r = scipy.stats.pearsonr(mat.loc[:, feature], labels[label])
            pcc_df.loc[feature, label] = pearson_r[0]
            p_value_df.loc[feature, label] = pearson_r[1]
    return pcc_df, p_value_df


def calc_soft_pcc(
    gene_expr, traits, gene_network_prob, aggregate=np.max
) -> Dict[int, float]:
    """
    Calculate the Pearson Correlation Coefficients for all the networks.

    Args:
        gene_network_prob: probability of gene A in network B, [n_networks, n_genes]
        aggregate: method to aggregate the PCCs over traits

    Returns:
        Dict[int, float]: aggregated PCC for each network
    """
    pcc_df, _ = _pearson_corr_coeff(gene_expr, traits)  # [n_genes, n_traits]
    avg_val = np.zeros((gene_network_prob.shape[0], pcc_df.shape[1]))

    # calculate the average value for each network
    total_network_probs = gene_network_prob.sum(axis=1)
    for gene in gene_network_prob:
        for network, prob in gene_network_prob[gene].items():
            avg_val[network, :] = np.add(
                avg_val[network, :],
                np.nan_to_num(
                    pcc_df.loc[gene, :].to_numpy() * prob / total_network_probs[network]
                )
            )

    # average metric value for all the soft networks
    avg_pcc = pd.DataFrame(
        data=avg_val,
        index=range(gene_network_prob.shape[0])
    ) # [n_networks, n_traits]
    return {i: aggregate(np.abs(avg_pcc.loc[i, :])) for i in avg_pcc.index}
