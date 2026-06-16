from typing import Callable, Optional, Union

import numpy as np
from scipy import stats

from dowhy.gcm.auto import AssignmentQuality, select_model
from dowhy.gcm.ml import PredictionModel
from dowhy.gcm.util.general import is_categorical, shape_into_2d


def generalised_cov_based(
    X: np.ndarray,
    Y: np.ndarray,
    Z: Optional[np.ndarray] = None,
    prediction_model_X: Union[AssignmentQuality, Callable[[], PredictionModel]] = AssignmentQuality.BETTER,
    prediction_model_Y: Union[AssignmentQuality, Callable[[], PredictionModel]] = AssignmentQuality.BETTER,
):

    X, Y = shape_into_2d(X, Y)

    if is_categorical(X) or is_categorical(Y):
        raise ValueError("X and Y need to be continuous variables!")

    if X.shape[1] > 1 or Y.shape[1] > 1:
        raise ValueError("X and Y need to be one dimensional!")

    if X.shape[0] != Y.shape[0]:
        raise ValueError("X and Y need to have the same number of rows!")

    if Z is None:
        residuals_xz = X - np.mean(X)
        residuals_yz = Y - np.mean(Y)
    else:
        if Z.shape[0] != X.shape[0]:
            raise ValueError("Z, X and Y need to have the same number of rows!")

        model_x = _create_model(Z, X, prediction_model_X)
        model_y = _create_model(Z, Y, prediction_model_Y)

        model_x.fit(Z, X)
        model_y.fit(Z, Y)

        residuals_xz = X - model_x.predict(Z)
        residuals_yz = Y - model_y.predict(Z)

    if np.var(residuals_yz) == 0 or np.var(residuals_xz) == 0:
        raise ValueError("Residuals cannot be constant!")

    # Calculate Ri, the product of the residuals
    residual_products = np.multiply(residuals_xz, residuals_yz)

    # Standard deviation of the residuals
    residual_std = np.std(residual_products)

    if residual_std == 0:
        return 1

    test_statistic = (np.sum(residual_products) / np.sqrt(X.shape[0])) / residual_std

    return stats.norm.sf(abs(test_statistic)) * 2


def _create_model(
    input_features: np.ndarray, target: np.ndarray, model: Union[str, Callable[[], PredictionModel]]
) -> PredictionModel:
    if not isinstance(model, AssignmentQuality):
        return model()
    else:
        return select_model(input_features, target, model)[0]