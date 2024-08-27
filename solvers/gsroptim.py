from benchopt import BaseSolver
from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    from gsroptim.sgl import sgl_path


class Solver(BaseSolver):
    """E. Ndiaye, O. Fercoq, A. Gramfort, and J. Salmon, "GAP safe screening
    rules for Sparse-Group Lasso", NeurIPS 2016.
    """
    name = "gsroptim"
    sampling_strategy = "iteration"

    install_cmd = 'conda'
    requirements = [
        'cython',
        "pip::git+https://github.com/EugeneNdiaye/Gap_Safe_Rules"
    ]

    def set_objective(self, X, y, lmbd, groups, grp_indices, grp_ptr, tau):
        self.X, self.y, self.lmbd, self.groups = X, y, lmbd, groups
        self.grp_ptr, self.tau = grp_ptr, tau

        self.n_groups = len(self.grp_ptr) - 1

        self.weights_g = np.ones(self.n_groups, dtype=np.float64)

    def run(self, n_iter):
        if n_iter == 0:
            self.coef = np.zeros([self.X.shape[1]])
        else:
            # no datafit normalization: need to multiply lmbd by n_samples
            res = sgl_path(
                self.X, self.y, size_groups=[self.groups] * self.n_groups,
                lambdas=[self.lmbd*self.X.shape[0]], omega=self.weights_g,
                eps=1e-12, tau=self.tau
            )[0]

            self.coef = res.flatten()

    def get_result(self):
        return dict(beta=self.coef)
