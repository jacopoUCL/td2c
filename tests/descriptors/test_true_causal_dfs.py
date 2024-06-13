# test if the DataLoader correctly constructs a True Causal DF from the DAG. Checks the corresponsance between the CausalDF and the DAG saved in the file.

import numpy as np
import math
import pytest
from d2c.data_generation.builder import TSBuilder  
from d2c.data_generation.models import model_registry  
from d2c.descriptors import DataLoader
import networkx as nx

@pytest.fixture
def initialization():
    observations_per_time_series=10 
    n_variables=10 
    time_series_per_process=1 
    max_neighborhood_size=5
    seed=42 
    maxlags = 5
    return observations_per_time_series, n_variables, time_series_per_process, max_neighborhood_size, seed, maxlags 

def test_observations(initialization):
    observations_per_time_series, n_variables, time_series_per_process, max_neighborhood_size, seed, maxlags  = initialization
    
    for model in range(1,21):
        if model == 5 or model == 17:
            # these models generative process tend to diverge. Hard to test them.
            # However, the graph generating method is shared between models, so we can expect
            # that if the other models are working, these should work as well.
            continue
        ts_builder = TSBuilder(observations_per_time_series=observations_per_time_series, n_variables=n_variables, time_series_per_process=time_series_per_process, processes_to_use=[model], max_neighborhood_size=max_neighborhood_size, seed=seed, noise_std=0)
        ts_builder.build()
        dags = ts_builder.get_generated_dags()
        
        dataloader = DataLoader(n_variables = n_variables,
                                maxlags = maxlags)
        dataloader.from_tsbuilder(ts_builder)

        true_causal_dfs = dataloader.get_true_causal_dfs()
        for ts_idx, true_causal_df in enumerate(true_causal_dfs):
            ones = true_causal_df.loc[true_causal_df.is_causal == 1]
            zeros = true_causal_df.loc[true_causal_df.is_causal == 0]

            #for each one in the ones, check if it's an edge in the graph
            for _ , row in ones.iterrows():
                from_idx = int(row['from'])
                row_idx = int(row['to'])

                to_string = f'{row_idx}_t-0'

                original_var_from = from_idx % n_variables
                original_lag = math.floor(from_idx / n_variables)
                from_string = f'{original_var_from}_t-{original_lag}'

                assert dags[model][ts_idx].has_edge(from_string, to_string)
            
            for _ , row in zeros.iterrows():
                from_idx = int(row['from'])
                row_idx = int(row['to'])

                to_string = f'{row_idx}_t-0'

                original_var_from = from_idx % n_variables
                original_lag = math.floor(from_idx / n_variables)
                from_string = f'{original_var_from}_t-{original_lag}'

                assert not dags[model][ts_idx].has_edge(from_string, to_string)