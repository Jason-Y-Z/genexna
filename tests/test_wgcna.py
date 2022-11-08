import random
import numpy as np
from genexna.wgcna import label_networks


def test_wgcna_on_a_random_matrix():
    # Generate random input to WGCNA.
    n = random.randint(10, 20)
    m = random.randint(100, 200)
    n_networks = random.randint(1, 5)
    X = np.abs(np.random.rand(n, m))

    # Perform clustering.
    y = label_networks(X, n_networks=n_networks)

    # Check test results.
    for y_i in y:
        assert y_i < n_networks
