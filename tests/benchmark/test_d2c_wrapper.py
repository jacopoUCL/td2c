import pytest
import pandas as pd
from unittest.mock import MagicMock

# Assuming your class file is named d2c.py
from d2c import D2C

class MockModel:
    def predict_proba(self, X):
        # Simulate predict_proba method of a model. Customize as needed.
        return [[0.1, 0.9] for _ in range(len(X))]

def test_d2c_initialization():
    with pytest.raises(ValueError):
        D2C()  # No model passed
    
    model = MockModel()
    d2c = D2C(model=model)
    assert d2c.n_variables == 6  # Default value
    assert d2c.model is not None
    assert d2c.returns_proba == True

def test_d2c_infer():
    model = MockModel()
    d2c = D2C(model=model, n_variables=3, maxlags=2)
    
    # Mocking DataLoader and D2C_ dependency behaviors
    d2c.DataLoader = MagicMock()
    d2c.D2C_ = MagicMock()
    d2c.DataLoader._create_lagged_single_ts.return_value = pd.DataFrame()
    d2c_instance_mock = d2c.D2C_.return_value
    d2c_instance_mock.compute_descriptors_without_dag.return_value = [
        {'graph_id': 1, 'edge_source': 'X', 'edge_dest': 'Y', 'is_causal': True}
    ]
    
    single_ts = pd.Series([1, 2, 3, 4, 5])
    results = d2c.infer(single_ts)
    
    # Check if the results dataframe is formed correctly
    assert 'probability' in results.columns
    assert 'is_causal' in results.columns
    assert results.iloc[0]['edge_source'] == 'X'
    assert results.iloc[0]['edge_dest'] == 'Y'
    assert results.iloc[0]['is_causal'] == True

def test_d2c_build_causal_df():
    model = MockModel()
    d2c = D2C(model=model)
    results = pd.DataFrame({
        'edge_source': ['X'],
        'edge_dest': ['Y'],
        'probability': [0.9],
        'is_causal': [True]
    })
    
    causal_df = d2c.build_causal_df(results)
    
    # Check if the causal dataframe is built correctly
    assert 'from' in causal_df.columns and 'to' in causal_df.columns
    assert 'effect' in causal_df.columns and causal_df['effect'].isnull().all()
    assert 'p_value' in causal_df.columns and causal_df['p_value'].isnull().all()
    assert len(causal_df) == len(results)

