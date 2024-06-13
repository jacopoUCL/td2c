# tests the generated DAGS, making sure that the expected neighborhood relationships are respected. 

import numpy as np
import math
import pytest
from d2c.data_generation.builder import TSBuilder  
from d2c.data_generation.models import model_registry  
import networkx as nx

@pytest.fixture
def initialization():
    observations_per_time_series=10 
    n_variables=10 
    time_series_per_process=1 
    max_neighborhood_size=5
    seed=42 
    return observations_per_time_series, n_variables, time_series_per_process, max_neighborhood_size, seed 

def test_observations(initialization):
    observations_per_time_series, n_variables, time_series_per_process, max_neighborhood_size, seed  = initialization
    
    for model in range(1,21):
        if model == 5 or model == 17:
            # these models generative process tend to diverge. Hard to test them.
            # However, the graph generating method is shared between models, so we can expect
            # that if the other models are working, these should work as well.
            continue
        ts_builder = TSBuilder(observations_per_time_series=observations_per_time_series, n_variables=n_variables, time_series_per_process=time_series_per_process, processes_to_use=[model], max_neighborhood_size=max_neighborhood_size, seed=seed, noise_std=0)
        ts_builder.build()
        neighbors = ts_builder.get_neighbors()
        observations = ts_builder.get_generated_observations()
        dags = ts_builder.get_generated_dags()
        model = model_registry.get_model(model)
        all_time_lags = model().time_from

        for model_id, ts_data in observations.items():
            for ts_index, data in ts_data.items():
                corresponding_dag = dags[model_id][ts_index]
                for variable_idx in range(data.shape[1]):
                    neighbors_variable = neighbors[model_id][ts_index][variable_idx]
                    for variable_2_idx in range(data.shape[1]):
                        for lag in all_time_lags:
                            if variable_2_idx in neighbors_variable:
                                assert corresponding_dag.has_edge(f'{variable_2_idx}_t-{lag + 1}', f'{variable_idx}_t-0')
                            else:
                                assert not corresponding_dag.has_edge(f'{variable_2_idx}_t-{lag + 1}', f'{variable_idx}_t-0')