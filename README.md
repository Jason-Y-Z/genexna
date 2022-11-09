# genexna
[![Python package](https://github.com/Jason-Y-Z/genexna/actions/workflows/python-package.yml/badge.svg)](https://github.com/Jason-Y-Z/genexna/actions/workflows/python-package.yml)

genexna is a Python library for GENe EXpression Network Analysis.

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install genexna.

```bash
pip install genexna
```

## Usage

```python
import random
import numpy as np
import pandas as pd
from genexna import decompose

n = random.randint(10, 20)
m = random.randint(100, 200)
gene_expr = pd.DataFrame(np.abs(np.random.rand(n, m)))
algo = decompose.NMF()

probs = decompose.decompose(gene_expr=gene_expr, algo=algo)
```

## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License

[MIT](https://choosealicense.com/licenses/mit/)