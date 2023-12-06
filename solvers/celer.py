from benchopt import BaseSolver
from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import warnings
    import numpy as np
    from celer import GroupLasso


class Solver(BaseSolver):

    name = 'celer'

    requirements = ["pip:celer"]

    def set_objective(self, X, y, lmbd, groups, grp_indices, grp_ptr):
        self.X, self.y = X, y

        warnings.filterwarnings('ignore')
        self.model = GroupLasso(
            groups, lmbd, max_iter=1, max_epochs=100,
            tol=1e-12, fit_intercept=False,
        )

    def run(self, n_iter):
        if n_iter == 0:
            self.w = np.zeros(self.X.shape[1])
            return

        X, y = self.X, self.y

        self.model.max_iter = n_iter
        self.model.fit(X, y)

        self.w = self.model.coef_

    def get_result(self):
        return dict(w=self.w)
