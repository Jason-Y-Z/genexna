import random
import numpy as np
import pandas as pd
import pytest
from genexna import decompose


@pytest.mark.parametrize(
    "algo", [decompose.NMF(), decompose.LNMF(), decompose.WGCNA(n_networks=5)]
)
def test_decompose_a_random_matrix(algo):
    # Generate random input to WGCNA.
    n = random.randint(10, 20)
    m = random.randint(100, 200)
    gene_expr = pd.DataFrame(np.abs(np.random.rand(n, m)))

    # Perform clustering.
    probs = decompose.decompose(gene_expr=gene_expr, algo=algo)

    # Check test results.
    for col in probs:
        assert probs[col].sum() == pytest.approx(1)
