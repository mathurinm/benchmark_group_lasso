from benchopt import BaseSolver
from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    from gsroptim.sgl import sgl_path


class Solver(BaseSolver):
    """Eugene Ndiaye, Olivier Fercoq, Alexandre Gramfort, and Joseph Salmon.
    2016. GAP safe screening rules for Sparse-Group Lasso. In Proceedings
    of the 30th International Conference on Neural Information Processing
    Systems (NIPS'16). Curran Associates Inc., Red Hook, NY, USA, 388–396.
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
        self.grp_indices, self.grp_ptr, self.tau = grp_indices, grp_ptr, tau

        self.n_samples = self.X.shape[0]
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

    def warm_up(self):
        # cache numba compilation
        self.run(n_iter=4)
