from benchopt import BaseDataset
from benchopt import safe_import_context

from benchopt.datasets.simulated import make_correlated_data

with safe_import_context() as import_ctx:
    import numpy as np


class Dataset(BaseDataset):
    name = "Simulated"

    parameters = {
        'n_samples, n_features': [(200, 500), (1000, 2000)],
        'groups': [5, 10]
    }

    def get_data(self):
        rng = np.random.RandomState(46)
        X, y, _ = make_correlated_data(
            self.n_samples, self.n_features, random_state=rng
        )

        return dict(X=X, y=y, groups=self.groups)
