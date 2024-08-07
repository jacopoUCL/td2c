import numpy as np
import pandas as pd
import networkx as nx
import pickle
import pytest
import os
from unittest.mock import Mock
from d2c.descriptors.loader import DataLoader  # Adjust import path as needed

from d2c.data_generation.builder import TSBuilder # Could be mocked

def test_initialization():
    loader = DataLoader(maxlags=2, n_variables=4)
    assert loader.maxlags == 2
    assert loader.n_variables == 4


# def test_from_pickle():
#         # Define the test data
#         data_path = 'test_data.pkl'
#         loader = DataLoader()
#         observations = {1: {0: np.array([1, 2, 3]), 1: np.array([4, 5, 6])},
#                     2: {0: np.array([7, 8, 9]), 1: np.array([10, 11, 12])}}
#         # Save the test data to a pickle file

#         graphs = {1: {0: nx.DiGraph(), 1: nx.DiGraph()}, 2: {0: nx.DiGraph()}} 
#         with open(data_path, 'wb') as f:
#             pickle.dump((observations, graphs), f)

#         # Call the from_pickle method
#         loader.from_pickle(data_path)

#         # Check that the observations and dags are loaded correctly
#         assert np.array_equal(loader.observations, loader._flatten(observations))
#         assert (loader.dags == loader._flatten(graphs)).all()

#         # Clean up the test data file
#         os.remove(data_path)

# def test_flatten():
#     loader = DataLoader()
#     input_dict = {'1': {0: np.array([1, 2, 3]), 1: np.array([4, 5, 6])},
#                   '2': {0: np.array([7, 8, 9]), 1: np.array([10, 11, 12])}}
#     result = loader._flatten(input_dict)
#     assert result.shape == (4, 3)
#     np.testing.assert_array_equal(result[0], np.array([1, 2, 3]))

def test_rename_dags():
    loader = DataLoader(n_variables=3)
    dag1 = nx.DiGraph()
    dag1.add_edges_from([("0_t-0", "1_t-1"), ("2_t-2", "1_t-0")])
    dag2 = nx.DiGraph()
    dag2.add_edges_from([("1_t-1", "0_t-0"), ("2_t-0", "0_t-2")])
    result = loader._rename_dags([dag1, dag2], loader.n_variables)
    assert result[0].has_edge(0, 4)
    assert result[1].has_edge(4, 0)

def test_create_lagged_multiple_ts():

    observations = [np.array([[1, 2], [3, 4]]), np.array([[5, 6], [7, 8]])]
    maxlags = 1 
    expected_shape =  [(2, 4)]

    loader = DataLoader(maxlags=maxlags)
    result = loader._create_lagged_multiple_ts(observations, maxlags)
    shapes = [obs.shape for obs in result]
    assert shapes == [(1, 4),(1, 4)]
    assert np.array_equal(result[0],np.array([[3, 4, 1, 2]]))
    assert np.array_equal(result[1],np.array([[7, 8, 5, 6]]))

    observations = [np.array([[1, 2], [3, 4], [5, 6]])]
    maxlags = 2
    expected_shape =  [(1, 6)]

    loader = DataLoader(maxlags=maxlags)
    result = loader._create_lagged_multiple_ts(observations, maxlags)
    shapes = [obs.shape for obs in result]
    assert shapes == expected_shape
    assert np.array_equal(result[0],np.array([[5,6,3,4,1,2]]))

    observations = [np.array([[1, 2], [3, 4], [5, 6], [7, 8]])]
    maxlags = 2
    expected_shape =  [(2, 6)]

    loader = DataLoader(maxlags=maxlags)
    result = loader._create_lagged_multiple_ts(observations, maxlags)
    shapes = [obs.shape for obs in result]
    assert shapes == expected_shape
    assert np.array_equal(result[0],np.array([[5,6,3,4,1,2], [7,8,5,6,3,4]]))
 
# Mock TSBuilder for loading data test
def test_from_tsbuilder(mocker):
    mocker.patch('d2c.data_generation.builder.TSBuilder', autospec=True)
    mock_builder = Mock()
    mock_builder.get_generated_observations.return_value = {'1': {0: np.array([1, 2, 3])}}
    mock_builder.get_generated_dags.return_value = {'1': {0: nx.DiGraph()}}

    loader = DataLoader()
    loader.from_tsbuilder(mock_builder)
    assert len(loader.observations) == 1
    assert len(loader.dags) == 1


def test_causal_df():

    n_variables = 5
    maxlags = 5
    process = 1
    noise_std = 0.1
    max_neighborhood_size = 2

    tsbuilder = TSBuilder(observations_per_time_series=250, 
                              maxlags=5, 
                              n_variables=n_variables, 
                              time_series_per_process=10, 
                              processes_to_use=[process], 
                              noise_std=noise_std, 
                              max_neighborhood_size=max_neighborhood_size, 
                              seed=42, 
                              max_attempts=10,
                              verbose=True)

    tsbuilder.build()

    dataloader = DataLoader(n_variables = n_variables,
                        maxlags = maxlags)
    dataloader.from_tsbuilder(tsbuilder)
    observations = dataloader.get_original_observations()
    true_causal_dfs = dataloader.get_true_causal_dfs()
    dags = dataloader.get_dags()

    for i in range(len(observations)):    
        current_df = true_causal_dfs[i]
        current_dag = dags[i]

        #first we check that all the causal edges are ones in the causal df 
        for edge in current_dag.edges(): 
            edge_from = edge[0]
            edge_to = edge[1]
            if edge_to < n_variables: #in the graph there is also past to past values but we don't care about these
                print('Checking edge', edge_from, edge_to)
                corresponding_row = current_df.loc[(current_df['from'] == edge_from) & 
                                    (current_df['to'] == edge_to)]
                assert corresponding_row['is_causal'].values == 1

        #then we check that all the remaining edges are zero in the causal df
        true_causal_dfs_copy = current_df.copy()
        for edge in current_dag.edges(): 
            edge_from = edge[0]
            edge_to = edge[1]
            if edge_to < n_variables: #in the graph there is also past to past values but we don't care about these
                #remove from true_causal_dfs_copy
                true_causal_dfs_copy = true_causal_dfs_copy.loc[~((true_causal_dfs_copy['from'] == edge_from) & 
                                    (true_causal_dfs_copy['to'] == edge_to))]
            
        assert true_causal_dfs_copy.is_causal.sum() == 0