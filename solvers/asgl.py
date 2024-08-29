from benchopt import BaseSolver
from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np

    from asgl import Regressor

    from benchmark_utils.funcs import convert_to_group_index


class Solver(BaseSolver):
    """Mendez-Civieta, A., Aguilera-Morillo, M.C. & Lillo, R.E. Adaptive sparse
    group LASSO in quantile regression. Adv Data Anal Classif 15, 547–573
    (2021).
    """
    name = "asgl"
    sampling_strategy = "iteration"

    install_cmd = 'conda'
    requirements = [
        'pip::asgl'
    ]
    parameters = {
        'penalty': [
            'gl',
            'sgl',
            'agl',
            'asgl'
        ],
    }

    def skip(self, X, y, lmbd, groups, grp_indices, grp_ptr, tau):
        if self.penalty in ['gl', 'agl'] and tau != 0:
            return True, f"Group Lasso needs tau=0, currently : tau={tau}"

        return False, None

    def set_objective(self, X, y, lmbd, groups, grp_indices, grp_ptr, tau):
        self.X, self.y, self.lmbd, self.groups = X, y, lmbd, groups
        self.grp_ptr, self.tau = grp_ptr, tau

        self.n_groups = len(self.grp_ptr) - 1

        weights_g = np.ones(self.n_groups, dtype=np.float64)
        weights_f = np.ones(self.X.shape[1])

        self.model = Regressor(
            model='lm', penalization=self.penalty, fit_intercept=False,
            lambda1=2*self.lmbd, alpha=self.tau,
            individual_weights=weights_f, group_weights=weights_g
        )
        self.group_index = convert_to_group_index(self.groups, self.X.shape[1])
        print('len(self.group_index)=', len(self.group_index))
        print('n_features=', self.X.shape[1])

    def run(self, n_iter):
        self.model.max_iter = n_iter + 1
        self.model.fit(self.X, self.y, **{'group_index': self.group_index})

    def get_result(self):
        return dict(beta=self.model.coef_.flatten())
