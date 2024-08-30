import numpy as np
from benchmark_utils.helper import compute_lmbd_max
from benchopt.datasets.simulated import make_correlated_data


def shuffle_and_test_lambda_max(X, y, tau, groups, n_shuffles=5,
                                random_state=42):
    initial_lambda_max = compute_lmbd_max(X, y, tau, groups)
    print(f"Initial lambda_max: {initial_lambda_max}")

    rng = np.random.RandomState(random_state)
    for i in range(n_shuffles):
        # Shuffle features of X
        shuffled_indices = rng.permutation(X.shape[1])
        X_shuffled = X[:, shuffled_indices]

        # Recompute lambda_max after shuffling
        shuffled_lambda_max = compute_lmbd_max(X_shuffled, y, tau, groups)
        print(f"Lambda_max after shuffle {i+1}: {shuffled_lambda_max}")

        # Check if lambda_max remains the same
        assert np.isclose(initial_lambda_max,
                          shuffled_lambda_max), "Lambda max changed "
        "after shuffling!"

    print("All lambda_max values remained unchanged after shuffling.")


if __name__ == "__main__":
    n_samples = 200
    n_features = 500
    tau = 1.0
    random_state = 42
    groups = 5

    # Generate data
    rng = np.random.RandomState(random_state)
    X, y, _ = make_correlated_data(
        n_samples, n_features, random_state=rng
    )

    # Test if lambda_max stays unchanged after shuffling features
    shuffle_and_test_lambda_max(
        X, y, tau, groups, n_shuffles=5, random_state=random_state)
