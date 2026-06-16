from typing import Optional

import numpy as np
from sklearn.kernel_approximation import Nystroem
from sklearn.metrics import euclidean_distances

from dowhy.gcm.util.general import shape_into_2d


def apply_rbf_kernel(X: np.ndarray, precision: Optional[float] = None) -> np.ndarray:
    """
    Estimates the RBF (Gaussian) kernel for the given input data.

    :param X: Input data.
    :param precision: Specific precision matrix for the RBF kernel. If None is given, this is inferred from the data.
    :return: The outcome of applying a RBF (Gaussian) kernel on the data.
    """
    X = shape_into_2d(X)

    distance_matrix = euclidean_distances(X, squared=True)

    if precision is None:
        precision = _median_based_precision(distance_matrix)

    return np.exp(-precision * distance_matrix)


def apply_rbf_kernel_with_adaptive_precision(X: np.ndarray) -> np.ndarray:

    X = shape_into_2d(X)

    result = np.ones((X.shape[0], X.shape[0]))
    for i in range(X.shape[1]):
        distance_matrix = euclidean_distances(X, squared=True)
        result *= np.exp(-_median_based_precision(distance_matrix) * distance_matrix)

    return result


def apply_delta_kernel(X: np.ndarray) -> np.ndarray:

    X = shape_into_2d(X)
    return np.array(list(map(lambda value: value == X, X))).reshape(X.shape[0], X.shape[0]).astype(float)


def approximate_rbf_kernel_features(
    X: np.ndarray, num_random_components: int, precision: Optional[float] = None
) -> np.ndarray:

    X = shape_into_2d(X)

    if precision is None:
        precision = _median_based_precision(euclidean_distances(X, squared=True))

    return Nystroem(kernel="rbf", gamma=precision, n_components=num_random_components).fit_transform(X)


def approximate_delta_kernel_features(X: np.ndarray, num_random_components: int) -> np.ndarray:

    X = shape_into_2d(X)

    def delta_function(x, y) -> float:
        return float(x == y)

    for i, unique_element in enumerate(np.unique(X)):
        X[X == unique_element] = i

    result = Nystroem(kernel=delta_function, n_components=num_random_components).fit_transform(X.astype(int))
    result[result != 0] = 1

    return result


def _median_based_precision(distances: np.ndarray) -> float:
    tmp = np.sqrt(distances)
    tmp = tmp - np.tril(tmp, -1)
    tmp = tmp.reshape(-1, 1)

    return 1 / np.median(tmp[tmp > 0])