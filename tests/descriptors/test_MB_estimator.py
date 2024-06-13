# test_markov_blanket_estimator.py
import numpy as np
import pytest

from d2c.descriptors.estimators import MarkovBlanketEstimator

def test_initialization():
    estimator = MarkovBlanketEstimator(size=5, verbose=True)
    assert estimator.size == 5
    assert estimator.verbose is True

def test_column_based_correlation():
    estimator = MarkovBlanketEstimator(size=5, verbose=False)
    X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    Y = np.array([1, 2, 3])
    expected_correlations = np.array([1, 1, 1])  # Perfect correlation
    np.testing.assert_array_almost_equal(estimator.column_based_correlation(X, Y), expected_correlations)

def test_rank_features():
    estimator = MarkovBlanketEstimator(size=5, verbose=False)
    X = np.array([[7, 2, 2], [1, 5, 5], [4, 8, 9]])
    Y = np.array([2, 5, 8])
    ranked_indices = estimator.rank_features(X, Y, regr=False)
    assert np.array_equal(ranked_indices, np.array([1, 2, 0]))  # Assuming correlation ranking

def test_estimate_markov_blanket():
    estimator = MarkovBlanketEstimator(size=2, verbose=False)
    dataset = np.random.rand(100, 5)  # Random dataset
    node = 0
    markov_blanket = estimator.estimate(dataset, node)
    assert len(markov_blanket) == 2
    assert node not in markov_blanket  # Ensure the node itself is not in its Markov Blanket

@pytest.mark.parametrize("size,expected", [(1, 1), (3, 3), (5, 5)])
def test_estimate_markov_blanket_sizes(size, expected):
    estimator = MarkovBlanketEstimator(size=size, verbose=False)
    dataset = np.random.rand(100, 10)  # Larger dataset
    node = 0
    markov_blanket = estimator.estimate(dataset, node)
    assert len(markov_blanket) == expected
    assert node not in markov_blanket
