from benchopt import BaseObjective
from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    from numpy.linalg import norm

    from skglm.datafits import QuadraticGroup
    from skglm.penalties import WeightedGroupL2
    from skglm.utils.data import grp_converter


class Objective(BaseObjective):
    name = "Group Lasso objective"

    requiremens = ["pip:skglm"]

    parameters = {
        'reg': [1., 1e-1, 1e-2, 1e-3],
    }

    def __init__(self, reg):
        self.reg = reg

    def set_data(self, X, y, groups):
        _, n_features = X.shape

        grp_indices, grp_ptr = grp_converter(groups, n_features)
        n_groups = len(grp_ptr) - 1
        weights = np.ones(n_groups)

        lmbd_max = self._compute_lmbd_max(X, y, grp_indices, grp_ptr)
        self.lmbd = self.reg * lmbd_max

        self.X, self.y = X, y
        self.grp_indices, self.grp_ptr = grp_indices, grp_ptr

        self.datafit = QuadraticGroup(grp_ptr, grp_indices)
        self.penalty = WeightedGroupL2(
            self.lmbd, weights, grp_ptr, grp_indices
        )

    def evaluate_result(self, w):
        X, y = self.X, self.y
        datafit, penalty = self.datafit, self.penalty

        return dict(
            value=datafit.value(y, w, X @ w) + penalty.value(w)
        )

    def get_objective(self):
        return dict(
            X=self.X, y=self.y, alpha=self.alpha, groups=self.groups,
            grp_indices=self.grp_indices, grp_ptr=self.grp_ptr
        )

    def _compute_lmbd_max(self, X, y, grp_indices, grp_ptr):
        lmbd_max = 0.
        n_samples = X.shape[0]
        n_groups = len(grp_ptr) - 1

        for g in range(n_groups):
            grp_g_indices = grp_indices[grp_ptr[g]: grp_ptr[g+1]]
            lmbd_max = max(
                lmbd_max,
                norm(X[:, grp_g_indices].T @ y) / n_samples
            )

        return lmbd_max
