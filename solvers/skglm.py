from benchopt import BaseSolver
from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np

    from skglm.datafits.group import QuadraticGroup
    from skglm.penalties.block_separable import WeightedGroupL2
    from skglm.solvers import GroupBCD


class Solver(BaseSolver):

    name = 'skglm'

    requirements = ["pip:skglm"]

    def __init__(self, use_acc, K):
        self.use_acc = use_acc
        self.K = K

    def set_objective(self, X, y, lmbd, groups, grp_indices, grp_ptr):
        self.X, self.y = X, y
        n_groups = len(grp_ptr) - 1

        self.lmbd = lmbd
        weights = np.ones(n_groups)
        self.grp_indices, self.grp_ptr = grp_indices, grp_ptr

        self.datafit = QuadraticGroup(grp_ptr, grp_indices)
        self.penalty = WeightedGroupL2(lmbd, weights, grp_ptr, grp_indices)

        self.solver = GroupBCD(fit_intercept=False, max_iter=1, tol=1e-12)

    def run(self, n_iter):
        self.solver.max_iter = n_iter

        self.w, _, _ = self.solver.solve(
            self.X, self.y, self.datafit, self.penalty,
            max_iter=n_iter, tol=1e-12, p0=10
        )

    def get_result(self):
        return dict(w=self.w)

    def warm_up(self):
        # cache numba compilation
        self.run(n_iter=10)
