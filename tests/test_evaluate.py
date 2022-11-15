import random
import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import normalize
from genexna.evaluate import calc_eigengene_pccs


@pytest.mark.parametrize("soft", [False, True])
def test_calc_eigengene_pccs(soft):
    # given
    n_subjects = random.randint(10, 20)
    n_traits = random.randint(10, 20)
    n_genes = random.randint(100, 200)
    n_networks = random.randint(3, 7)
    gene_expr = pd.DataFrame(np.abs(np.random.rand(n_subjects, n_genes)))
    traits = pd.DataFrame(np.abs(np.random.rand(n_subjects, n_traits)))
    gene_network_prob = pd.DataFrame(normalize(
        np.abs(np.random.rand(n_networks, n_genes)), norm='l1', axis=0
    )) if soft else random.choices(range(n_networks), k=n_genes)

    # when
    pcc = calc_eigengene_pccs(gene_expr, traits, gene_network_prob, soft=soft)

    # then
    for network in range(n_networks):
        assert pcc[network] <= 1
