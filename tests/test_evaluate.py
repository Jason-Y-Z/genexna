import random
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
from genexna.evaluate import calc_soft_pcc


def test_calc_soft_pcc():
    # given
    n_subjects = random.randint(10, 20)
    n_traits = random.randint(10, 20)
    n_genes = random.randint(100, 200)
    n_networks = random.randint(3, 7)
    gene_expr = pd.DataFrame(np.abs(np.random.rand(n_subjects, n_genes)))
    traits = pd.DataFrame(np.abs(np.random.rand(n_subjects, n_traits)))
    gene_network_prob = pd.DataFrame(normalize(
        np.nan_to_num(np.abs(np.random.rand(n_networks, n_genes))), norm='l1', axis=0
    ))

    # when
    soft_pcc = calc_soft_pcc(gene_expr, traits, gene_network_prob)

    # then
    for network in range(n_networks):
        assert soft_pcc[network] <= 1
