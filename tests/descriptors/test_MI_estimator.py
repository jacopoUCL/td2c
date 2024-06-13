import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

from d2c.descriptors.estimators import MutualInformationEstimator, LOWESS, mse

@pytest.fixture
def default_mi_estimator():
    return MutualInformationEstimator()

@pytest.fixture
def custom_mi_estimator():
    proxy_params = {'alpha': 0.1}
    return MutualInformationEstimator(proxy='Ridge', proxy_params=proxy_params)

def test_initialization_default(default_mi_estimator):
    assert default_mi_estimator.proxy == 'Ridge'
    assert default_mi_estimator.proxy_params == {}

def test_initialization_custom(custom_mi_estimator):
    assert custom_mi_estimator.proxy == 'Ridge'
    assert custom_mi_estimator.proxy_params == {'alpha': 0.1}

def test_get_regression_model_ridge(default_mi_estimator):
    model = default_mi_estimator.get_regression_model()
    assert isinstance(model, Ridge)

def test_get_regression_model_lowess():
    mi_estimator = MutualInformationEstimator(proxy='LOWESS')
    model = mi_estimator.get_regression_model()
    assert isinstance(model, LOWESS)

def test_mse_method():
    X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    Y = np.array([10, 20, 30])
    cv = 3
    actual_mse = mse(X, Y, cv)
    model = Ridge(alpha=1e-3).fit(X, Y)  # Assuming Ridge is the proxy model being used
    expected_mse = mean_squared_error(Y, model.predict(X))
    assert np.isclose(actual_mse, max(1e-3, expected_mse))

def test_estimate_method_x2_none(default_mi_estimator):
    y = pd.Series([1, 2, 3, 4, 5])
    x1 = pd.DataFrame({'A': [1, 2, 3, 4, 5]})
    mi = default_mi_estimator.estimate(y.values, x1.values)
    actual_mse = mse(x1.values, y.values, cv=5)  # Ensure this matches the `estimate` method's internal cv if set
    assert mi == 1 - actual_mse / np.var(y)

def test_estimate_method_x2_not_none(default_mi_estimator):
    y = pd.Series([1, 2, 3, 4, 5])
    x1 = pd.DataFrame({'A': [1, 2, 3, 4, 5]})
    x2 = pd.DataFrame({'B': [6, 7, 8, 9, 10]})
    mi = default_mi_estimator.estimate(y.values, x1.values, x2.values)
    x1x2 = pd.concat([x1, x2], axis=1)
    mi_expected = 1 - mse(x1x2.values, y.values, cv=5) / mse(x2.values, y.values, cv=5)  # Adjust `cv` as needed
    assert np.isclose(mi, mi_expected)

def test_estimate_method_empty_y(default_mi_estimator):
    y = pd.Series([], dtype=float)
    x1 = pd.DataFrame({'A': [1, 2, 3, 4, 5]})
    x2 = pd.DataFrame({'B': [6, 7, 8, 9, 10]})
    mi = default_mi_estimator.estimate(y.values, x1.values, x2.values)
    assert mi == 0

def test_estimate_method_empty_x1(default_mi_estimator):
    y = pd.Series([1, 2, 3, 4, 5])
    x1 = pd.DataFrame({})
    x2 = pd.DataFrame({'B': [6, 7, 8, 9, 10]})
    mi = default_mi_estimator.estimate(y.values, x1.values, x2.values)
    assert mi == 0

def test_estimate_method_empty_x2(default_mi_estimator):
    y = pd.Series([1, 2, 3, 4, 5])
    x1 = pd.DataFrame({'A': [1, 2, 3, 4, 5]})
    x2 = pd.DataFrame({})
    mi = default_mi_estimator.estimate(y.values, x1.values, x2.values)
    actual_mse = mse(x1.values, y.values, cv=5)
    assert mi == 1 - actual_mse / np.var(y)

def test_estimate_method_negative_mi(default_mi_estimator):
    y = pd.Series([1, 2, 3, 4, 5])
    x1 = pd.DataFrame({'A': [1, 2, 3, 4, 5]})
    x2 = pd.DataFrame({'B': [6, 7, 8, 9, 10]})
    mi = default_mi_estimator.estimate(y.values, x1.values, x2.values)
    assert mi >= 0


