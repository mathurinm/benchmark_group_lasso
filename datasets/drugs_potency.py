from benchopt import BaseDataset
from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    from sklearn.datasets import fetch_openml


class Dataset(BaseDataset):
    name = "Drugs Potency"

    requirements = ["scikit-learn"]

    parameters = {'groups': [4]}

    def __init__(self, groups):
        self.groups = groups

    def get_data(self):
        X, y = fetch_openml(
            "QSAR_TID_11",
            parser='auto',
            return_X_y=True
        )

        X = X.to_numpy(dtype=float)
        y = y.to_numpy(dtype=float)
        return dict(X=X, y=y, groups=self.groups)
