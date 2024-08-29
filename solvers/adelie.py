from benchopt import BaseSolver
from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    import adelie as ad
    from skglm.utils.data import grp_converter


class Solver(BaseSolver):
    name = "adelie"
    sampling_strategy = "iteration"

    install_cmd = 'conda'
    requirements = [
        'pip::adelie'
    ]

    def skip(self, X, y, lmbd, groups, grp_indices, grp_ptr, tau):
        if tau != 0:
            return True, "Only Group Lasso is supported by adelie"
        return False, None

    def set_objective(self, X, y, lmbd, groups, grp_indices, grp_ptr, tau):
        self.X, self.y, self.lmbd, self.grp_size = X, y, lmbd, groups

        _, grp_ptr = grp_converter(self.grp_size, self.X.shape[1])

        self.groups = np.array(grp_ptr[:-1], dtype=np.int32)
        self.omega = np.ones(len(self.groups), dtype=np.float64)

    def run(self, n_iter):
        if n_iter == 0:
            self.coef = np.zeros([self.X.shape[1]])
        else:
            state = ad.grpnet(
                X=self.X, glm=ad.glm.gaussian(y=self.y), groups=self.groups,
                tol=1e-12, intercept=False, penalty=self.omega,
                lmda_path=[self.lmbd], progress_bar=False
            )
            self.coef = state.betas.toarray().flatten()

    def get_result(self):
        return dict(beta=self.coef)
