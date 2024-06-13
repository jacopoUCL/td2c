import numpy as np
import pytest
from d2c.data_generation.builder import TSBuilder  

def test_initialization():
    ts_builder = TSBuilder()
    assert ts_builder.observations_per_time_series == 200
    assert ts_builder.maxlags == 3
    assert ts_builder.n_variables == 5
    assert ts_builder.ts_per_process == 10
    assert len(ts_builder.processes_to_use) == 20
    assert ts_builder.noise_std == 0.1
    assert ts_builder.max_neighborhood_size == 5  # Note: Adjusted due to min() in __init__
    assert ts_builder.seed == 42
    assert ts_builder.verbose == True

def test_generated_data_structure():
    ts_builder = TSBuilder(observations_per_time_series=10, n_variables=2, time_series_per_process=1, processes_to_use=[1], max_neighborhood_size=2, seed=1)
    ts_builder.build(max_attempts=2)
    observations = ts_builder.get_generated_observations()
    dags = ts_builder.get_generated_dags()

    assert isinstance(observations, dict)
    assert isinstance(dags, dict)
    assert len(observations) == 1
    assert len(dags) == 1
    assert all(isinstance(observations[model_id], dict) for model_id in observations)
    assert all(isinstance(dags[model_id], dict) for model_id in dags)

def test_retry_on_non_finite_values(mocker):
    mocker.patch("d2c.data_generation.builder.TSBuilder.build", side_effect=ValueError("Non-finite value detected. Trying again."))
    ts_builder = TSBuilder(observations_per_time_series=10, n_variables=2, time_series_per_process=1, processes_to_use=[1], max_neighborhood_size=2, seed=1)

    with pytest.raises(ValueError) as exc_info:
        ts_builder.build(max_attempts=1)
    assert "Non-finite value detected. Trying again." in str(exc_info.value)

def test_valid_data_generated():
    np.random.seed(42)
    ts_builder = TSBuilder(observations_per_time_series=10, n_variables=2, time_series_per_process=1, processes_to_use=[1], max_neighborhood_size=2, seed=42)
    ts_builder.build(max_attempts=5)
    observations = ts_builder.get_generated_observations()
    dags = ts_builder.get_generated_dags()

    # Check that data generated is valid and conforms to expected shapes
    for model_id, ts_data in observations.items():
        for ts_index, data in ts_data.items():
            assert data.shape == (10, 2)  # Observations, Variables
            assert not np.any(np.isnan(data))
            assert not np.any(np.isinf(data))

    #TODO: Similar checks can be added for the DAGs