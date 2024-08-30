from skglm.utils.data import grp_converter
from gsroptim.sgl import build_lambdas
import numpy as np


def compute_lmbd_max(X, y, tau, groups):
    n_samples, n_features = X.shape
    _, grp_ptr = grp_converter(groups, n_features)

    size_groups = np.diff(grp_ptr).astype(np.int32)

    # omega stores the penalization of the size of each group
    omega = np.ones(size_groups.shape)

    # Start indices are just the pointers except the last one
    g_start = grp_ptr[:-1]

    lambda_max = build_lambdas(
        X, y, omega,
        size_groups, g_start, n_lambdas=1, tau=tau)[0]

    return lambda_max / n_samples
