from benchopt import BaseObjective
from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    from numpy.linalg import norm

    from skglm.utils.data import grp_converter

    from gsroptim.sgl import build_lambdas


class Objective(BaseObjective):
    name = "Sparse Group Lasso objective"

    requirements = [
        "numpy'<2'",
        "pip::git+https://github.com/scikit-learn-contrib/skglm",
        "pip::git+https://github.com/EugeneNdiaye/Gap_Safe_Rules"
    ]
    # tau = 0 is Group Lasso
    parameters = {
        'tau': [0, 0.5, 0.9],
        'reg': [1., 1e-1, 1e-2]
    }

    def value_penalty(self, w):
        """Value of penalty at vector ``w``."""

        sum_penalty = 0.
        for g in range(self.n_groups):
            grp_g_indices = self.grp_indices[
                self.grp_ptr[g]: self.grp_ptr[g+1]]
            w_g = w[grp_g_indices]

            sum_penalty += norm(w_g)
        sum_penalty *= (1 - self.tau)
        sum_penalty += self.tau * np.sum(np.abs(w))

        return self.lmbd * sum_penalty

    def set_data(self, X, y, groups):
        self.n_samples, self.n_features = X.shape

        self.grp_indices, self.grp_ptr = grp_converter(groups, self.n_features)
        self.n_groups = len(self.grp_ptr) - 1

        self.X, self.y = X, y
        self.groups = groups

        lmbd_max = self._compute_lmbd_max()
        self.lmbd = self.reg * lmbd_max

    def evaluate_result(self, beta):
        diff = self.y - self.X @ beta
        datafit = .5 * diff @ diff / self.n_samples
        pen = self.value_penalty(beta)

        return dict(
            value=datafit+pen,
        )

    def get_objective(self):
        return dict(
            X=self.X, y=self.y, lmbd=self.lmbd, groups=self.groups,
            grp_indices=self.grp_indices, grp_ptr=self.grp_ptr, tau=self.tau
        )

    def _compute_lmbd_max(self):
        size_groups = self.groups * np.ones(
            self.n_features // self.groups, dtype=np.int32)

        # omega stores the square root of the size of each group
        # it is used to scale the regularization parameters for each group
        omega = np.ones(self.n_features // self.groups)

        # g_start contains the starting index for each group, computed using
        # the cumulative sum of group size. It is used to identify the columns
        # in X that belong to each group
        g_start = np.cumsum(size_groups, dtype=np.int32) - size_groups[0]

        # Compute the maximum lambda value for regularization
        lambda_max = build_lambdas(self.X, self.y, omega, size_groups,
                                   g_start, n_lambdas=1, tau=self.tau)[0]
        return lambda_max / self.n_samples

    def get_one_result(self):
        return dict(w=np.zeros(self.n_features))
