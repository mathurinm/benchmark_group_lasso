from benchopt import BaseSolver
from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np

    from skglm.solvers import GroupBCD
    from skglm.datafits import QuadraticGroup
    from skglm.penalties import WeightedL1GroupL2
    from skglm.utils.jit_compilation import compiled_clone


class Solver(BaseSolver):
    """Q. Bertrand and Q. Klopfenstein and P.-A. Bannier and G. Gidel and
    M. Massias, "Beyond L1: Faster and Better Sparse Models with skglm",
    NeurIPS 2022.
    """
    name = 'skglm'

    # Sparse Group Lasso is new in version skglm=0.4 which is not released yet
    # Therefore it is installed directly from the repo
    # TODO:remove when skglm=0.4 is released
    requirements = [
        "numpy'<2'",
        "pip::skglm@git+https://github.com/scikit-learn-contrib/skglm"
    ]

    def set_objective(self, X, y, lmbd, groups, grp_indices, grp_ptr, tau):
        self.X, self.y, self.lmbd, self.groups = X, y, lmbd, groups
        self.tau = tau

        n_groups = len(grp_ptr) - 1

        weights_g = (1 - self.tau) * np.ones(n_groups, dtype=np.float64)
        weights_f = self.tau * np.ones(self.X.shape[1])

        self.penalty = compiled_clone(WeightedL1GroupL2(
            alpha=self.lmbd, weights_groups=weights_g,
            weights_features=weights_f, grp_indices=grp_indices,
            grp_ptr=grp_ptr))

        self.datafit = compiled_clone(QuadraticGroup(grp_ptr, grp_indices))
        self.solver = GroupBCD(
            fit_intercept=False, ws_strategy='fixpoint', verbose=0, tol=1e-12)

    def run(self, n_iter):
        self.solver.max_iter = n_iter

        self.w, _, _ = self.solver.solve(
            self.X, self.y, self.datafit, self.penalty
        )

    def get_result(self):
        return dict(beta=self.w)

    def warm_up(self):
        # cache numba compilation
        self.run(n_iter=4)
