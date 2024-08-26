from benchopt import BaseObjective
from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    from numpy.linalg import norm

    from skglm.utils.data import grp_converter


class Objective(BaseObjective):
    name = "Sparse Group Lasso"

    # skglm is needed here to create group partition and indices
    requirements = [
        "pip::git+https://github.com/scikit-learn-contrib/skglm"
    ]
    # tau = 0 is Group Lasso, tau = 1 is Lasso
    parameters = {
        'tau': [0, 0.5, 0.9],
        'reg': [1., 1e-1, 1e-2]
    }

    def value_penalty(self, beta):
        """Value of penalty at vector ``beta``."""

        sum_penalty = 0.
        for g in range(self.n_groups):
            grp_g_indices = self.grp_indices[
                self.grp_ptr[g]: self.grp_ptr[g+1]]
            beta_g = beta[grp_g_indices]

            sum_penalty += norm(beta_g)
        sum_penalty *= (1 - self.tau)
        sum_penalty += self.tau * np.sum(np.abs(beta))

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

    # this is a heuristic to compute lambda_max
    # when tau == 0, it provides the exact value of lambda_max
    # otherwise, it returns a cheaper to compute upper bound
    def _compute_lmbd_max(self):
        lmbd_max_l1 = 0.
        lmbd_max_group = 0.
        n_groups = len(self.grp_ptr) - 1

        # Compute lambda max for the L1 part
        lmbd_max_l1 = norm(self.X.T @ self.y, ord=np.inf) / self.n_samples

        # Compute lambda max for the group lasso part
        for g in range(n_groups):
            grp_g_indices = self.grp_indices[
                self.grp_ptr[g]: self.grp_ptr[g+1]]
            lmbd_max_group = max(
                lmbd_max_group,
                norm(self.X[:, grp_g_indices].T @ self.y) / self.n_samples
            )

        # Combine the two parts using the tau parameter
        lmbd_max = self.tau * lmbd_max_l1 + (1 - self.tau) * lmbd_max_group

        return lmbd_max

    def get_one_result(self):
        return dict(beta=np.zeros(self.n_features))
