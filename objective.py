from benchopt import BaseObjective
from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    from numpy.linalg import norm

    from skglm.utils.data import grp_converter

    from gsroptim.sgl import build_lambdas


class Objective(BaseObjective):
    name = "Sparse Group Lasso"

    # skglm is needed here to create group partition and indices
    requirements = [
        "numpy'<2'",
        "pip::git+https://github.com/scikit-learn-contrib/skglm",
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
        # Size of each group, might be non-contiguous
        size_groups = np.array(self.groups, dtype=np.int32)

        # Omega stores the square root of the size of each group
        # It is used to scale the regularization parameters for each group
        omega = np.sqrt(size_groups)

        # g_start contains the starting index for each group, computed using
        # the cumulative sum of group size.
        g_start = np.zeros(len(size_groups), dtype=np.int32)
        g_start[1:] = np.cumsum(size_groups[:-1])

        # Compute the maximum lambda value for regularization
        lambda_max, _ = build_lambdas(
            self.X, self.y, omega, size_groups, g_start, n_lambdas=1,
            tau=self.tau)

        return lambda_max / self.n_samples

    def get_one_result(self):
        return dict(w=np.zeros(self.n_features))
