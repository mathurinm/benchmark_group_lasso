from benchopt import BaseDataset
from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import pandas as pd
    from sklearn.datasets import fetch_openml


class Dataset(BaseDataset):
    name = "Molecular Biology Promoters"

    requirements = ["scikit-learn"]

    parameters = {'groups': [4]}

    def __init__(self, groups=4):
        self.groups = groups

    def get_data(self):
        X, y = fetch_openml(
            "molecular-biology_promoters",
            version=1,
            parser='auto',
            return_X_y=True
        )

        # clean/cast to the right dtype
        X = pd.get_dummies(X).to_numpy(dtype=float)
        y = y.replace({'+': 1, '-': 0}).to_numpy(dtype=float)

        return dict(X=X, y=y, groups=self.groups)
