import pandas as pd
from typing import List

def decompose(gene_expr: pd.DataFrame, traits: List[int] = None):
    """
    Decompose the gene expression matrix to 
    obtain the composition of each gene in terms of
    the gene networks.
    """
