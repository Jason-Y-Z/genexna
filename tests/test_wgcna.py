from genexna.wgcna import WGCNA
import numpy as np
import random


def test_wgcna_on_a_random_matrix():
    # Generate random input to WGCNA.
    n = random.randint(10, 20)
    m = random.randint(100, 200)
    n_networks = random.randint(1, 5)
    X = np.abs(np.random.rand(n, m))

    # Initialise the algorithm.
    wgcna = WGCNA(n_networks=n_networks)

    # Perform clustering.
    y = wgcna.fit_transform(X)

    # Check test results.
    for n_i in range(n):
        assert np.argmax(y[n_i]) < n_networks
