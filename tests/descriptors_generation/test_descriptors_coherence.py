# # test_descriptors_coherence.py
# import os
# import pickle
# from typing import Any, Tuple
# import pandas as pd
# import pytest

# #TODO: handle this better! 
# import sys
# sys.path.append("../..")
# from src.d2c.utils import from_dict_of_lists_to_list, rename_dags

# @pytest.fixture
# def load_data():
#     """
#     Load data from specified paths and return the loaded data.

#     Returns:
#         n_processes (int): The number of generative processes.
#         observations (list): List of loaded observations.
#         dags (list): List of loaded directed acyclic graphs.
#         causal_dfs (list): List of loaded causal dataframes.
#         descriptors (object): Loaded descriptors.
#     """
#     data_path = '/home/gpaldino/D2CPY/small_data_lag3_variables5/'
#     descriptors_path = '/home/gpaldino/D2CPY/small_data_lag3_variables5/descriptors_Ridge.pkl'

#     loaded_observations = {}
#     loaded_dags = {}
#     loaded_causal_dfs = {}
#     for file in os.listdir(data_path):
#         if file.startswith('data'):
#             index = file.split('_')[1].split('.')[0]
#             with open(data_path+file, 'rb') as f:
#                 loaded_observations[index], loaded_dags[index], loaded_causal_dfs[index] = pickle.load(f)
    
#     n_processes = len(loaded_observations) #how many generative processes did we have? 
#     #TODO: add the case of single data file, or consider moving always to single data file
#     #TODO: add the 'process' column directly in the descriptor generation

#     observations = from_dict_of_lists_to_list(loaded_observations)
#     dags = from_dict_of_lists_to_list(loaded_dags)
#     causal_dfs = from_dict_of_lists_to_list(loaded_causal_dfs)

#     with open(descriptors_path, 'rb') as f:
#         descriptors = pickle.load(f)

#     return n_processes, observations, dags, causal_dfs, descriptors

# def test_dags_and_causal_dfs_coherence(load_data: tuple[int, list, list, list, Any]):
#     """
#     Test the coherence between DAGs and causal dataframes.

#     Args:
#         load_data (tuple[int, list, list, list, Any]): A tuple containing the loaded data.

#     Raises:
#         AssertionError: If the coherence between DAGs and causal dataframes is not maintained.
#     """
#     _, observations, dags, causal_dfs, _ = load_data
#     n_variables = observations[0].shape[1]
#     dags = rename_dags(dags, n_variables)
#     for index, dag in enumerate(dags):
#         current_causal_df = causal_dfs[index]
#         edges_df = pd.DataFrame(columns=['is_causal'], index=pd.MultiIndex.from_tuples([(i,j) for i,j in list(dag.edges())], names=['source', 'target']), data=1)
#         to_compare = current_causal_df.join(edges_df, how='inner', lsuffix='_causal_df', rsuffix='_graph_edges')
#         assert to_compare['is_causal_causal_df'].equals(to_compare['is_causal_graph_edges'])

# def test_descriptors_coherence(load_data: tuple[int, list, list, list, Any]):
#     """
#     Test the coherence between descriptors and causal dataframes.

#     Args:
#         load_data (tuple[int, list, list, list, Any]): A tuple containing the loaded data.

#     Raises:
#         AssertionError: If the coherence between descriptors and causal dataframes is not maintained.
#     """
#     _, _, _, causal_dfs, descriptors = load_data
#     for graph_index in descriptors.graph_id.unique():
#         to_compare = descriptors.loc[descriptors['graph_id'] == graph_index][['graph_id','edge_source','edge_dest','is_causal']].sort_values(by=['edge_source', 'edge_dest'])
#         to_compare.index = pd.MultiIndex.from_tuples([(i,j) for i,j in zip(to_compare['edge_source'], to_compare['edge_dest'])], names=['source', 'target'])
#         to_compare.drop(['graph_id','edge_source','edge_dest'], axis=1, inplace=True)

#         to_compare = to_compare.join(causal_dfs[graph_index], how='inner', lsuffix='_descriptors', rsuffix='_true')

#         assert to_compare['is_causal_descriptors'].equals(to_compare['is_causal_true'])

# def test_signal_in_descriptors(load_data: tuple[int, list, list, list, Any]):
#     """
#     Test the presence of signal in descriptors. 
#     The presence of signal means that a classifier 
#     is able to do better than random guessing.

#     Args:
#         load_data (tuple[int, list, list, list, Any]): A tuple containing the loaded data.

#     Raises:
#         AssertionError: If the median AUC is not above 0.5.
#     """
#     n_processes,_,_,_, descriptors = load_data
#     n_graphs = len(descriptors.graph_id.unique())
#     n_graphs_per_process = n_graphs // n_processes
#     descriptors['process'] = descriptors['graph_id'] // n_graphs_per_process + 1

#     from sklearn.model_selection import cross_val_score, LeaveOneGroupOut
#     from imblearn.ensemble import BalancedRandomForestClassifier
#     import numpy as np

#     X = descriptors.drop(['is_causal','edge_source','edge_dest','graph_id','process'], axis=1)
#     y = descriptors['is_causal']

#     #cross validation LOGO 'graph_id
#     rf = BalancedRandomForestClassifier(n_estimators=50, max_depth=10, random_state=0, sampling_strategy='all',replacement=True)
#     # rf = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=0)
#     logo = LeaveOneGroupOut()
#     groups = descriptors['process']
#     cv = logo.split(X, y, groups)

#     cvs = cross_val_score(rf, X, y, cv=cv, scoring='roc_auc', n_jobs=n_processes)
    
#     # assert median auc is above 0.5
#     assert np.median(cvs) > 0.5
#     assert np.mean(cvs) > 0.5



# if __name__ == '__main__':
#     pytest.main()