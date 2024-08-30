import pickle
import numpy as np
import networkx as nx
import pandas as pd

class DataLoaderIteriative():
    """
    A class for loading and processing data for descriptors computation.

    Attributes:
    - observations: The flattened array of time series observations.
    - dags: The flattened array of directed acyclic graphs (DAGs).
    - n_variables: The number of variables in the data.
    - maxlags: The maximum number of lags to create for the lagged time series.

    Methods:
    - __init__(self, maxlags=3, n_variables=3): Initializes the DataLoader object.
    - _flatten(self, dict_of_dicts): Flattens a dictionary of dictionaries into a single list.
    - _rename_dags(self, dags, n_variables): Renames the nodes of the DAGs to match the descriptor convention.
    - _create_lagged_multiple_ts(self, observations, maxlags): Creates lagged time series from the given observations.
    - _extend_graph_with_past_states(self, dag): Extends the DAG to include past states.
    - from_pickle(self, data_path): Loads data from a pickle file.
    - from_tsbuilder(self, ts_builder): Loads data from a TimeSeriesBuilder object.
    - get_observations(self): Returns the lagged time series observations.
    - get_dags(self): Returns the renamed DAGs.
    """

    def __init__(self, maxlags=3, n_variables=3):
        """
        Initializes the DataLoader object.

        Parameters:
        - maxlags (int): The maximum number of lags to create for the lagged time series.
        - n_variables (int): The number of variables in the data.
        """
        self.observations = None
        self.dags = None
        self.n_variables = n_variables
        self.maxlags = maxlags

    def _flatten(self, dict_of_dicts):
        """
        Convert a dictionary of dictionaries to a single list.
        """
        list_of_arrays = [dict_of_dicts[process][ts] for process in sorted(dict_of_dicts.keys()) for ts in sorted(dict_of_dicts[process].keys())]
        return list_of_arrays

    @staticmethod
    def _rename_dags(dags, n_variables):
        """
        Rename the nodes of the DAGs to use the same convention as the descriptors.
        """
        updated_dags = []
        for dag in dags:
            mapping = {node: int(node.split('_')[0]) + int(node.split('-')[1]) * n_variables for node in dag.nodes()}
            dag = nx.relabel_nodes(dag, mapping)
            updated_dags.append(dag)
        return updated_dags

    def _extend_graph_with_past_states(self, dag):
        """
        Extend the given DAG to include past states reflecting the progression from past to present.

        Parameters:
        - dag (nx.DiGraph): The directed acyclic graph to be extended.

        Returns:
        - extended_dag (nx.DiGraph): The extended directed acyclic graph including past states.
        """
        extended_dag = dag.copy()
        for lag in range(1, self.maxlags + 1):
            for node in dag.nodes():
                new_node = node + lag * self.n_variables  # New node for the past state
                extended_dag.add_node(new_node)  # Add past state node
                extended_dag.add_edge(new_node, node)  # Connect past state to current node
        return extended_dag

    @staticmethod
    def _create_lagged_multiple_ts(observations, maxlags):
        """
        Create lagged multiple time series from the given observations.
        """
        lagged_observations = []
        for obs in observations:
            lagged = obs.copy()
            for i in range(1, maxlags + 1):
                lagged = np.concatenate((lagged, np.roll(obs, i, axis=0)), axis=1)
            lagged_observations.append(lagged[maxlags:])
        return lagged_observations

    def from_pickle(self, data_path):
        """
        Data loader from a data file.
        """
        with open(data_path, 'rb') as f:
            loaded_observations, loaded_dags, _ = pickle.load(f)
        self.observations = self._flatten(loaded_observations)
        self.dags = self._flatten(loaded_dags)

    def from_tsbuilder(self, ts_builder):
        """
        Data loader from a TimeSeriesBuilder object.
        """
        loaded_observations = ts_builder.get_generated_observations()
        loaded_dags = ts_builder.get_generated_dags()
        self.observations = self._flatten(loaded_observations)
        self.dags = self._flatten(loaded_dags)

    def get_observations(self):
        """
        Get the observations after having created the lagged time series.
        """
        return self._create_lagged_multiple_ts(self.observations, self.maxlags)
    
    def get_dags(self):
        """
        Get the DAGs after having renamed the nodes and extended to include past states.
        """
        renamed_dags = self._rename_dags(self.dags, self.n_variables)
        extended_dags = [self._extend_graph_with_past_states(dag) for dag in renamed_dags]
        return extended_dags
    
    def get_true_causal_dfs(self):
        """
        Get causal dataframes from the renamed and extended DAGs.
        """
        causal_dataframes = []
        for dag in self.get_dags():
            pairs = [(source, effect) for source in range(self.n_variables, self.n_variables * self.maxlags + self.n_variables) for effect in range(self.n_variables)]
            multi_index = pd.MultiIndex.from_tuples(pairs, names=['from', 'to'])
            causal_dataframe = pd.DataFrame(index=multi_index, columns=['is_causal'])
            causal_dataframe['is_causal'] = 0
            for edge in dag.edges:
                source = edge[0]
                effect = edge[1]
                causal_dataframe.loc[(source, effect), 'is_causal'] = 1

            causal_dataframe.reset_index(inplace=True)   
            causal_dataframe = causal_dataframe.loc[causal_dataframe.to < self.n_variables]
            causal_dataframes.append(causal_dataframe)

        return causal_dataframes
